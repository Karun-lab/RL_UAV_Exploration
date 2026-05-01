"""
iris_exploration_env.py
=======================
Clean, simple indoor exploration environment.

Design philosophy:
  - Extrinsic rewards teach NAVIGATION (wall avoidance, gap seeking)
  - ICM teaches EXPLORATION (visit new places)
  - Both must work independently — play uses policy only, no ICM

Actions:  [vx, yaw_rate]  in [-1, 1]
Obs:      flat tensor = depth_flat (H*W) + state (7)
          state = [lin_vel(3), ang_vel_z(1), yaw_sin(1), yaw_cos(1), coverage(1)]

Depth:    full frame (H, W) normalised to [0,1]
          used for: proximity penalty, gap reward, ICM input, policy obs

Rewards:
  EXTRINSIC (always active, teaches navigation):
    + gap_reward:      positive when a large depth opening is ahead
    - proximity:       graded penalty when depth < warn threshold
    - time:            small per-step penalty (forces efficiency)
    + success:         large bonus at coverage threshold
    - stagnation:      penalty for not moving

  INTRINSIC / ICM (teaches exploration, bonus on top):
    + icm_reward:      prediction error of forward model

No debug visualisers (no red cubes).
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCamera, TiledCameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots.iris import IRIS_CFG


# ---------------------------------------------------------------------------
# Maze layout
# ---------------------------------------------------------------------------
WALL_DEFS: List[Tuple[float, float, float, float]] = [
    # Outer perimeter
    ( 0.0,  4.0, 12.4,  0.4),   # north wall
    ( 0.0, -4.0, 12.4,  0.4),   # south wall
    (-6.0,  0.0,  0.4,  8.4),   # west wall
    ( 6.0,  0.0,  0.4,  8.4),   # east wall
    # Room 1 dividers
    (-5.5,  0.0,  0.8,  0.4),
    (-1.8,  0.0,  3.5,  0.4),
    (-1.5,  2.0,  0.4,  4.0),
]

# AABB for each wall: [x_min, x_max, y_min, y_max]
WALL_AABBS = np.array(
    [[cx - sx/2, cx + sx/2, cy - sy/2, cy + sy/2]
     for cx, cy, sx, sy in WALL_DEFS],
    dtype=np.float32,
)

# Drone spawn — inside the open corridor area
SPAWN_XYZ: Tuple[float, float, float] = (-4.0, -2.0, 0.9)


# ---------------------------------------------------------------------------
# ICM
# ---------------------------------------------------------------------------
class ICM(nn.Module):
    """
    Intrinsic Curiosity Module.

    Encodes depth frames into features, then:
      - Forward model:  phi(s_t) + a_t  →  predicted phi(s_{t+1})
      - Inverse model:  phi(s_t) + phi(s_{t+1})  →  predicted a_t
    Intrinsic reward = eta * ||phi(s_{t+1}) - predicted||^2

    The inverse model forces the encoder to capture only action-relevant
    features, ignoring irrelevant visual noise.

    Input depth: (N, 1, H, W) normalised full frame
    """

    def __init__(
        self,
        img_h: int,
        img_w: int,
        action_dim: int = 2,
        feature_dim: int = 64,
        eta: float = 0.02,
        beta: float = 0.2,
    ):
        super().__init__()
        self.eta  = eta
        self.beta = beta

        # Small CNN encoder: (1, H, W) → feature_dim
        # Designed to be cheap — 3 conv layers with aggressive downsampling
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=8, stride=4, padding=2),  # /4
            nn.ELU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),  # /8
            nn.ELU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),  # /16
            nn.ELU(),
            nn.Flatten(),
        )

        # Compute flat size
        with torch.no_grad():
            dummy     = torch.zeros(1, 1, img_h, img_w)
            flat_size = self.encoder(dummy).shape[-1]

        self.enc_fc = nn.Sequential(
            nn.Linear(flat_size, feature_dim),
            nn.ELU(),
        )

        # Forward model: phi(s) + a → phi(s')
        self.forward_model = nn.Sequential(
            nn.Linear(feature_dim + action_dim, 128),
            nn.ELU(),
            nn.Linear(128, feature_dim),
        )

        # Inverse model: phi(s) + phi(s') → a
        self.inverse_model = nn.Sequential(
            nn.Linear(feature_dim * 2, 128),
            nn.ELU(),
            nn.Linear(128, action_dim),
        )

    def encode(self, depth: torch.Tensor) -> torch.Tensor:
        """depth: (N, 1, H, W) → (N, feature_dim)"""
        return self.enc_fc(self.encoder(depth))

    def forward(
        self,
        depth_t:  torch.Tensor,
        depth_t1: torch.Tensor,
        action:   torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            intrinsic_reward: (N,)
            forward_loss:     scalar
            inverse_loss:     scalar
        """
        phi_t  = self.encode(depth_t)
        phi_t1 = self.encode(depth_t1)

        # Forward prediction
        phi_t1_pred  = self.forward_model(torch.cat([phi_t, action], dim=-1))
        intrinsic    = self.eta * 0.5 * F.mse_loss(
            phi_t1_pred, phi_t1.detach(), reduction="none"
        ).mean(dim=-1)
        forward_loss = F.mse_loss(phi_t1_pred, phi_t1.detach())

        # Inverse prediction
        action_pred  = self.inverse_model(torch.cat([phi_t, phi_t1], dim=-1))
        inverse_loss = F.mse_loss(action_pred, action)

        return intrinsic, forward_loss, inverse_loss

    def loss(self, fwd: torch.Tensor, inv: torch.Tensor) -> torch.Tensor:
        return (1.0 - self.beta) * inv + self.beta * fwd


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@configclass
class IrisExploreEnvCfg(DirectRLEnvCfg):

    # Episode
    episode_length_s = 60.0
    decimation       = 2       # 100 Hz physics / 2 = 50 Hz policy

    # Spaces — flat vector: depth (H*W) + state (7)
    # Default camera: 160x120 → 19200 + 7 = 19207
    # Adjust if you change cam_height / cam_width
    cam_height: int = 120
    cam_width:  int = 160
    action_space      = 2
    observation_space = 120 * 160 + 7   # updated automatically if you change res
    state_space       = 0

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=16,
        env_spacing=20.0,
        replicate_physics=True,
    )

    robot: ArticulationCfg = IRIS_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )

    # Camera — synthetic pinhole, no hardware warnings
    camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/quadrotor/body/DepthCam",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.1, 0.0, 0.0),
            rot=(-0.5, -0.5, 0.5, 0.5),
            convention="opengl",
        ),
        data_types=["distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.15, 10.0),
        ),
        width=160,
        height=120,
    )

    cam_fov_deg:   float = 90.0
    cam_min_depth: float = 0.15
    cam_max_depth: float = 6.0

    wall_height: float = 2.0

    # Motion
    max_forward_vel:  float = 1.0
    max_yaw_rate:     float = 1.5
    hover_height:     float = 0.9
    altitude_kp:      float = 2.0
    max_altitude_vel: float = 1.0

    # Occupancy grid: 12 x 8 m at 0.1 m/cell
    cell_size: float = 0.1
    grid_cols: int   = 120
    grid_rows: int   = 80

    # ICM
    icm_feature_dim: int   = 64
    icm_eta:         float = 0.02   # intrinsic reward scale
    icm_beta:        float = 0.2    # forward/inverse balance
    icm_lr:          float = 3e-4

    # Rewards
    # Extrinsic (navigation — active during play too)
    time_penalty:          float = -0.02   # per step — small but constant
    prox_warn_m:           float = 0.8    # start penalising at this depth
    prox_max_penalty:      float = -1.5   # penalty at zero distance
    gap_reward_scale:      float =  1.0   # reward for open space ahead
    success_threshold:     float =  0.75  # coverage fraction for success
    success_bonus:         float =  150.0
    stagnation_penalty:    float = -1.0   # if not moved 0.5 m in 3 s

    # Logging
    grid_save_path:  str = "/tmp/iris_explore_grid.npy"
    grid_save_every: int = 500


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
class IrisExploreEnv(DirectRLEnv):
    cfg: IrisExploreEnvCfg

    def __init__(self, cfg: IrisExploreEnvCfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        N  = self.num_envs
        H  = self.cfg.cam_height
        W  = self.cfg.cam_width

        # Occupancy grid
        self._grid            = np.zeros((N, self.cfg.grid_rows, self.cfg.grid_cols), dtype=np.uint8)
        self._prev_free_count = np.zeros(N, dtype=np.int32)

        # Depth buffers: (N, 1, H, W) for ICM
        self._depth_t  = torch.zeros(N, 1, H, W, device=self.device)
        self._depth_t1 = torch.zeros(N, 1, H, W, device=self.device)

        # Actions
        self._actions = torch.zeros(N, 2, device=self.device)

        # ICM
        self._icm = ICM(
            img_h=H, img_w=W,
            action_dim=2,
            feature_dim=self.cfg.icm_feature_dim,
            eta=self.cfg.icm_eta,
            beta=self.cfg.icm_beta,
        ).to(self.device)
        self._icm_opt = torch.optim.Adam(self._icm.parameters(), lr=self.cfg.icm_lr)

        # Stagnation: ring buffer of XY positions, 3 s at 50 Hz = 150 steps
        self._stag_n   = 150
        self._stag_buf = torch.zeros(N, self._stag_n, 2, device=self.device)
        self._stag_idx = 0
        self._stag_rdy = False

        # Success flag (set in _get_rewards, read in _get_dones)
        self._succeeded = torch.zeros(N, dtype=torch.bool, device=self.device)

        self._step_counter = 0

        # Precompute camera ray angles for grid update
        self._precompute_rays()

    def _precompute_rays(self):
        W    = self.cfg.cam_width
        hfov = math.radians(self.cfg.cam_fov_deg)
        fx   = (W / 2.0) / math.tan(hfov / 2.0)
        cols = np.arange(W, dtype=np.float32)
        self._ray_offsets = np.arctan2(cols - W / 2.0, fx)   # (W,)

    # ------------------------------------------------------------------
    # Scene
    # ------------------------------------------------------------------
    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self._camera = TiledCamera(self.cfg.camera)
        self.scene.sensors["depth"] = self._camera

        self.cfg.terrain.num_envs    = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # Spawn walls in env_0 before clone
        self._spawn_walls()
        self.scene.clone_environments(copy_from_source=False)

        sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)).func(
            "/World/Light",
            sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
        )

    def _spawn_walls(self):
        import omni.usd
        from pxr import UsdGeom
        stage = omni.usd.get_context().get_stage()
        UsdGeom.Xform.Define(stage, "/World/envs/env_0/Maze")
        for i, (cx, cy, sx, sy) in enumerate(WALL_DEFS):
            sim_utils.CuboidCfg(
                size=(sx, sy, self.cfg.wall_height),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=True, disable_gravity=True
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=1000.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.6, 0.6, 0.65), roughness=0.9
                ),
            ).func(
                f"/World/envs/env_0/Maze/Wall_{i:03d}",
                sim_utils.CuboidCfg(
                    size=(sx, sy, self.cfg.wall_height),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        kinematic_enabled=True, disable_gravity=True
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1000.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.6, 0.6, 0.65), roughness=0.9
                    ),
                ),
                translation=(cx, cy, self.cfg.wall_height / 2.0),
            )

    # ------------------------------------------------------------------
    # Depth helpers
    # ------------------------------------------------------------------
    def _fetch_depth(self) -> torch.Tensor:
        """
        Returns (N, 1, H, W) normalised depth tensor, values in [0, 1].
        0 = at min_depth (very close), 1 = at max_depth or invalid (open).
        Invalid/NaN pixels are set to 1.0 (treat as open space).
        """
        raw = self._camera.data.output.get("distance_to_image_plane")
        N, H, W = self.num_envs, self.cfg.cam_height, self.cfg.cam_width

        if raw is None:
            return torch.ones(N, 1, H, W, device=self.device)

        if isinstance(raw, torch.Tensor):
            d = raw.float().to(self.device)
        else:
            d = torch.tensor(raw, dtype=torch.float32, device=self.device)

        # Normalise shape → (N, H, W)
        if d.ndim == 2:
            d = d.unsqueeze(0).expand(N, -1, -1)
        elif d.ndim == 3 and d.shape[-1] == 1:
            d = d.squeeze(-1)
        elif d.ndim == 4:
            d = d.squeeze(-1)

        # Clamp and normalise to [0, 1]
        d = d.clamp(self.cfg.cam_min_depth, self.cfg.cam_max_depth)
        d = (d - self.cfg.cam_min_depth) / (self.cfg.cam_max_depth - self.cfg.cam_min_depth)
        d = torch.nan_to_num(d, nan=1.0, posinf=1.0, neginf=0.0)

        return d.unsqueeze(1)   # (N, 1, H, W)

    def _update_grid(self, depth: torch.Tensor):
        """Update occupancy grid from depth frame using ray projection."""
        cs   = self.cfg.cell_size
        NR   = self.cfg.grid_rows
        NC   = self.cfg.grid_cols
        hx   = (NC * cs) / 2.0
        hy   = (NR * cs) / 2.0

        # Middle row of depth frame → 1D horizontal scan
        mid_row = self.cfg.cam_height // 2
        # (N, W) — denormalised to metres
        scan = (
            depth[:, 0, mid_row, :].cpu().numpy() *
            (self.cfg.cam_max_depth - self.cfg.cam_min_depth) +
            self.cfg.cam_min_depth
        )

        pos_np  = self._robot.data.root_pos_w.cpu().numpy()
        origins = self._terrain.env_origins.cpu().numpy()
        quat    = self._robot.data.root_state_w[:, 3:7].cpu().numpy()
        w, x, y, z = quat[:,0], quat[:,1], quat[:,2], quat[:,3]
        yaw = np.arctan2(2.0*(w*z + x*y), 1.0 - 2.0*(y*y + z*z))

        for i in range(self.num_envs):
            ox = pos_np[i, 0] - origins[i, 0]
            oy = pos_np[i, 1] - origins[i, 1]
            angles = yaw[i] + self._ray_offsets
            depths = scan[i]
            valid  = (depths > self.cfg.cam_min_depth) & \
                     (depths < self.cfg.cam_max_depth) & \
                     np.isfinite(depths)
            if not valid.any():
                continue

            cos_a = np.cos(angles[valid])
            sin_a = np.sin(angles[valid])
            dv    = depths[valid]

            # Free space midpoints
            fx_   = ox + dv * 0.5 * cos_a
            fy_   = oy + dv * 0.5 * sin_a
            fc    = np.clip(np.floor((fx_ + hx) / cs).astype(int), 0, NC-1)
            fr    = np.clip(np.floor((hy - fy_) / cs).astype(int), 0, NR-1)
            mask  = self._grid[i, fr, fc] != 2
            self._grid[i, fr[mask], fc[mask]] = 1

            # Occupied endpoints
            hx_  = ox + dv * cos_a
            hy_  = oy + dv * sin_a
            hc   = np.clip(np.floor((hx_ + hx) / cs).astype(int), 0, NC-1)
            hr   = np.clip(np.floor((hy - hy_) / cs).astype(int), 0, NR-1)
            self._grid[i, hr, hc] = 2

            # Drone's own cell
            dc = np.clip(int((ox + hx) / cs), 0, NC-1)
            dr = np.clip(int((hy - oy) / cs), 0, NR-1)
            self._grid[i, dr, dc] = 1

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone().clamp(-1.0, 1.0)

        lin_b = torch.zeros(self.num_envs, 3, device=self.device)
        lin_b[:, 0] = self._actions[:, 0] * self.cfg.max_forward_vel

        target_z = self.cfg.hover_height + self._terrain.env_origins[:, 2]
        vz = (self.cfg.altitude_kp * (target_z - self._robot.data.root_pos_w[:, 2])).clamp(
            -self.cfg.max_altitude_vel, self.cfg.max_altitude_vel
        )

        q         = self._robot.data.root_state_w[:, 3:7]
        lin_w     = _quat_rotate(q, lin_b)
        lin_w[:, 2] = vz

        ang_w        = torch.zeros(self.num_envs, 3, device=self.device)
        ang_w[:, 2]  = self._actions[:, 1] * self.cfg.max_yaw_rate

        self._lin_w = lin_w
        self._ang_w = ang_w

    def _apply_action(self):
        self._robot.write_root_velocity_to_sim(
            torch.cat([self._lin_w, self._ang_w], dim=-1)
        )
        jv = torch.zeros_like(self._robot.data.joint_vel)
        jv[:, 0], jv[:, 1] =  200.0, -200.0
        jv[:, 2], jv[:, 3] =  200.0, -200.0
        self._robot.set_joint_velocity_target(jv)

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------
    def _get_observations(self) -> dict:
        # Shift buffers
        self._depth_t  = self._depth_t1.clone()
        self._depth_t1 = self._fetch_depth()   # (N, 1, H, W)

        self._update_grid(self._depth_t1)
        self._step_counter += 1

        if self._step_counter % self.cfg.grid_save_every == 0:
            np.save(self.cfg.grid_save_path, self._grid[0])

        # Yaw (sin, cos — avoids angle discontinuity at ±π)
        q    = self._robot.data.root_state_w[:, 3:7]
        w_q, x_q, y_q, z_q = q[:,0], q[:,1], q[:,2], q[:,3]
        yaw  = torch.atan2(2.0*(w_q*z_q + x_q*y_q), 1.0 - 2.0*(y_q*y_q + z_q*z_q))
        yaw_sin = torch.sin(yaw).unsqueeze(1)
        yaw_cos = torch.cos(yaw).unsqueeze(1)

        # Coverage
        tc  = self.cfg.grid_rows * self.cfg.grid_cols
        cov = torch.tensor(
            (self._grid == 1).sum(axis=(1,2)).astype(np.float32) / tc,
            device=self.device,
        ).unsqueeze(1)

        # State: (N, 7)
        state = torch.cat([
            self._robot.data.root_lin_vel_b,           # 3
            self._robot.data.root_ang_vel_b[:, 2:3],   # 1 yaw rate
            yaw_sin,                                   # 1
            yaw_cos,                                   # 1
            cov,                                       # 1
        ], dim=-1)

        # Flatten depth: (N, 1, H, W) → (N, H*W)
        depth_flat = self._depth_t1.reshape(self.num_envs, -1)

        # Full flat observation
        obs = torch.cat([depth_flat, state], dim=-1)   # (N, H*W + 7)
        return {"policy": obs}

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:

        # ── ICM intrinsic ──────────────────────────────────────────────
        if self._step_counter > 1:
            with torch.no_grad():
                icm_r, _, _ = self._icm(
                    self._depth_t, self._depth_t1, self._actions
                )

            # Isaac Lab wraps _get_rewards in torch.inference_mode() which
            # kills autograd. We explicitly escape it for the ICM training step.
            with torch.inference_mode(False):
                # Re-clone to get fresh tensors with grad tracking enabled
                depth_t_in  = self._depth_t.clone()
                depth_t1_in = self._depth_t1.clone()
                actions_in  = self._actions.detach().clone()

                self._icm.train()
                _, fwd, inv = self._icm(depth_t_in, depth_t1_in, actions_in)
                icm_loss = self._icm.loss(fwd, inv)
                self._icm_opt.zero_grad()
                icm_loss.backward()
                torch.nn.utils.clip_grad_norm_(self._icm.parameters(), 1.0)
                self._icm_opt.step()
        else:
            icm_r = torch.zeros(self.num_envs, device=self.device)

        # ── Depth-based proximity penalty ─────────────────────────────
        # Use minimum depth across the full frame
        depth_m = (
            self._depth_t1[:, 0] *
            (self.cfg.cam_max_depth - self.cfg.cam_min_depth) +
            self.cfg.cam_min_depth
        )   # (N, H, W) in metres
        min_d = depth_m.reshape(self.num_envs, -1).min(dim=-1).values  # (N,)
        warn  = self.cfg.prox_warn_m
        prox_factor = ((warn - min_d) / warn).clamp(0.0, 1.0)
        prox_r = prox_factor * self.cfg.prox_max_penalty   # (N,) negative

        # ── Gap reward: reward open space directly ahead ───────────────
        # Take the central column strip (middle 30% of width)
        H, W   = self.cfg.cam_height, self.cfg.cam_width
        c_lo   = W * 35 // 100
        c_hi   = W * 65 // 100
        ahead  = depth_m[:, :, c_lo:c_hi]                 # (N, H, centre_W)
        gap_d  = ahead.reshape(self.num_envs, -1).mean(dim=-1)  # mean ahead depth
        # Normalise: 0 at cam_min, 1 at cam_max — reward larger gaps
        gap_norm = (gap_d - self.cfg.cam_min_depth) / (
            self.cfg.cam_max_depth - self.cfg.cam_min_depth
        )
        gap_r  = gap_norm.clamp(0.0, 1.0) * self.cfg.gap_reward_scale

        # ── Time penalty ───────────────────────────────────────────────
        time_r = torch.full(
            (self.num_envs,), self.cfg.time_penalty, device=self.device
        )

        # ── Stagnation ─────────────────────────────────────────────────
        pos_xy = self._robot.data.root_pos_w[:, :2].detach()
        self._stag_buf[:, self._stag_idx] = pos_xy
        self._stag_idx = (self._stag_idx + 1) % self._stag_n
        if not self._stag_rdy and self._stag_idx == 0:
            self._stag_rdy = True

        if self._stag_rdy:
            oldest    = self._stag_buf[:, self._stag_idx]
            disp      = torch.linalg.norm(pos_xy - oldest, dim=1)
            stag_r    = (disp < 0.5).float() * self.cfg.stagnation_penalty
        else:
            stag_r = torch.zeros(self.num_envs, device=self.device)

        # ── Success ─────────────────────────────────────────────────────
        tc       = self.cfg.grid_rows * self.cfg.grid_cols
        free_frac = torch.tensor(
            (self._grid == 1).sum(axis=(1,2)).astype(np.float32) / tc,
            device=self.device,
        )
        self._succeeded = free_frac >= self.cfg.success_threshold
        success_r = self._succeeded.float() * self.cfg.success_bonus

        # ── Total ────────────────────────────────────────────────────────
        reward = icm_r + prox_r + gap_r + time_r + stag_r + success_r

        # ── Diagnostic printout every 200 steps ─────────────────────────
        if self._step_counter % 200 == 0:
            cov = float((self._grid == 1).mean() * 100)
            print(
                f"[step {self._step_counter:6d}]  "
                f"cov={cov:5.1f}%  "
                f"icm={icm_r.mean():.3f}  "
                f"prox={prox_r.mean():.3f}  "
                f"gap={gap_r.mean():.3f}  "
                f"stag={stag_r.mean():.3f}  "
                f"total={reward.mean():.3f}"
            )

        self.extras["log"] = {
            "coverage_pct":   float((self._grid == 1).mean() * 100),
            "icm_reward":     float(icm_r.mean()),
            "prox_penalty":   float(prox_r.mean()),
            "gap_reward":     float(gap_r.mean()),
            "success_rate":   float(self._succeeded.float().mean()),
            "min_depth_mean": float(min_d.mean()),
        }

        return reward

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        alt_fail = (
            (self._robot.data.root_pos_w[:, 2] < 0.1) |
            (self._robot.data.root_pos_w[:, 2] > 3.5)
        )

        hx   = (self.cfg.grid_cols * self.cfg.cell_size) / 2.0
        hy   = (self.cfg.grid_rows * self.cfg.cell_size) / 2.0
        p    = self._robot.data.root_pos_w.cpu().numpy()
        orig = self._terrain.env_origins[:, :2].cpu().numpy()
        loc  = p[:, :2] - orig
        oob  = torch.tensor(
            (np.abs(loc[:,0]) > hx) | (np.abs(loc[:,1]) > hy),
            device=self.device,
        )

        # Collision: min depth < danger threshold (half of prox_warn)
        depth_m = (
            self._depth_t1[:, 0] *
            (self.cfg.cam_max_depth - self.cfg.cam_min_depth) +
            self.cfg.cam_min_depth
        )
        min_d   = depth_m.reshape(self.num_envs, -1).min(dim=-1).values
        col     = min_d < (self.cfg.prox_warn_m * 0.4)

        died = alt_fail | oob | col
        return died | self._succeeded, time_out

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if len(env_ids) == self.num_envs:
            self.episode_length_buf = torch.randint_like(
                self.episode_length_buf, high=int(self.max_episode_length),
            )

        ids_np = env_ids.cpu().numpy()
        self._grid[ids_np]            = 0
        self._prev_free_count[ids_np] = 0
        self._depth_t[env_ids]        = 0.0
        self._depth_t1[env_ids]       = 0.0
        self._stag_buf[env_ids]       = 0.0
        self._succeeded[env_ids]      = False

        for eid in ids_np:
            t     = torch.tensor([eid], device=self.device)
            orig  = self._terrain.env_origins[eid].cpu().numpy()
            state = self._robot.data.default_root_state[t].clone()
            state[0, 0] = orig[0] + SPAWN_XYZ[0]
            state[0, 1] = orig[1] + SPAWN_XYZ[1]
            state[0, 2] = SPAWN_XYZ[2]
            self._robot.write_root_pose_to_sim(state[:, :7], t)
            self._robot.write_root_velocity_to_sim(state[:, 7:], t)
            jp = self._robot.data.default_joint_pos[t]
            jv = self._robot.data.default_joint_vel[t]
            self._robot.write_joint_state_to_sim(jp, jv, None, t)

    # ------------------------------------------------------------------
    # No debug visualiser (as requested)
    # ------------------------------------------------------------------
    def _set_debug_vis_impl(self, debug_vis: bool):
        pass

    def _debug_vis_callback(self, event):
        pass


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
def _quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    w   = q[:, 0:1]
    xyz = q[:, 1:]
    t   = 2.0 * torch.linalg.cross(xyz, v)
    return v + w * t + torch.linalg.cross(xyz, t)