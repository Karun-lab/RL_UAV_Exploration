"""
iris_icm_map_env.py
======================
ICM-driven office exploration with pose-based visited-area memory.

Design principle
----------------
The drone's XY pose (from Isaac sim during training, from VIO over ROS 2
during deployment) feeds a per-env 2-D occupancy grid.  Every step the
drone's current cell is looked up; the novelty of that cell modulates the
reward the ICM reward receives before being handed to PPO.

  total_reward = novelty_weight(cell) * icm_r
               + vel_r + smooth_r + yaw_r + col_r + time_r

novelty_weight decays each visit:
    weight = 1.0  on first visit       → full ICM reward
    weight = 0.5  on second visit
    weight = 1 / (visit_count + 1)     → asymptotes to 0

The policy therefore learns to maximise coverage naturally: revisiting
kills the reward, so it has to keep moving to new areas.

The observation adds a LOCAL map crop centred on the drone: a small
patch of the visit-count grid (normalised) rendered as a 2nd image
channel alongside depth.  This gives the CNN a spatial memory it can
read without any recurrent state.

  Observation: (T=3, H=64, W=80, C=2)
      channel 0: normalised depth   [0,1]  1=open 0=wall
      channel 1: local novelty map  [0,1]  1=never visited 0=visited many times

Sim-to-real pose interface
--------------------------
All pose consumption is isolated in _get_drone_pos_local() which returns
(N, 2) XY positions in env-local frame.  During training it reads from
the Isaac articulation state.  During real-world deployment replace ONLY
this method — subscribe to your VIO ROS 2 topic and return the same shape.
A drop-in deployment adapter is provided at the bottom of this file.

Everything else — ICM, rewards, spawns, action smoothing — is unchanged
from the working exploration version.
"""

from __future__ import annotations

import math
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
import gymnasium as gym
from rl_WorkSpace.models.drone.iris import IRIS_CFG


# =============================================================================
# OFFICE
# =============================================================================

OFFICE_USD_PATH = (
    "/workspace/isaaclab/rl_WorkSpace/models/environments/TestEnvOfficeB.usd"
    # "/workspace/isaaclab/rl_WorkSpace/models/environments/TrainEnvOffice4.usd"
    # "/workspace/isaaclab/rl_WorkSpace/models/environments/TrainEnvOffice3.usd"
    # "/workspace/isaaclab/rl_WorkSpace/models/environments/TrainEnvOffice2.usd"
    # "/workspace/isaaclab/rl_WorkSpace/models/environments/TrainEnvOffice1.usd"
)

SPAWN_TABLE = [
    # officeB
    (-2.0, 56.0, 1.0,   0.0),
    (-3.0, 39.5, 1.0,   0.0),
    ( 4.0, 39.5, 1.0, 180.0),
    (-3.0, 50.0, 1.0, -90.0),
    ( 0.0, 58.0, 1.0, -90.0),
    # Office4 (uncomment to use)
    # (5.0, 16.5, 1.0,   0.0),
    # (5.0, 16.5, 1.0, -90.0),
    # (5.0, 7.5, 1.0,    0.0),
    # (5.0, 7.5, 1.0,   90.0),
    # (12.0, 12.0, 1.0,  0.0),
    # Office3 (uncomment to use)
    # (10.0, 22.0, 1.0, 180.0),
    # (4.0, 22.0, 1.0,  -90.0),
    # (2.0, 2.0, 1.0,    90.0),
    # (7.0, 8.0, 1.0,   -90.0),
    # (7.0, 10.0, 1.0,   90.0),
]

SPAWN_XY_NOISE:  float = 0.3   # metres
SPAWN_YAW_NOISE: float = 30.0  # degrees


# =============================================================================
# OCCUPANCY GRID PARAMETERS
# =============================================================================
# The grid is env-local: origin at env_origin, X/Y in metres.
# Every env gets its own independent grid (no cross-env sharing).

GRID_CELL_M:    float = 0.25   # metres per cell — 25 cm resolution
GRID_EXTENT_M:  float = 30.0   # grid covers ±15 m from env origin
GRID_N:         int   = int(GRID_EXTENT_M / GRID_CELL_M)  # cells per side = 120

# Local map crop rendered into the observation
# Must be odd so the drone is always at the exact centre pixel
LOCAL_MAP_PX:   int   = 21     # 21×21 cells → 5.25×5.25 m window at 25 cm/cell

# visit_count is stored as float32 for GPU-friendly operations.
# novelty = 1 / (visit_count + 1)  ∈ (0, 1]
# Reward weight = novelty  → halved each visit, asymptotes to 0.


# =============================================================================
# ICM MODULE  (unchanged from working version)
# =============================================================================

class DepthEncoder(nn.Module):
    def __init__(self, h: int, w: int, feature_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2), nn.ELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.ELU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1), nn.ELU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            flat_dim = self.net(torch.zeros(1, 1, h, w)).shape[1]
        self.fc = nn.Sequential(nn.Linear(flat_dim, feature_dim), nn.ELU())

    def forward(self, x):
        return self.fc(self.net(x))


class ICM(nn.Module):
    def __init__(self, h, w, action_dim=2, feature_dim=64, eta=0.1, beta=0.2):
        super().__init__()
        self.eta  = eta
        self.beta = beta
        self.encoder      = DepthEncoder(h, w, feature_dim)
        self.forward_model = nn.Sequential(
            nn.Linear(feature_dim + action_dim, 256), nn.ELU(),
            nn.Linear(256, feature_dim),
        )
        self.inverse_model = nn.Sequential(
            nn.Linear(feature_dim * 2, 256), nn.ELU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, depth_t, depth_t1, action):
        phi_t        = self.encoder(depth_t)
        phi_t1       = self.encoder(depth_t1)
        phi_t1_pred  = self.forward_model(torch.cat([phi_t, action], dim=-1))
        reward       = self.eta * 0.5 * F.mse_loss(
            phi_t1_pred, phi_t1.detach(), reduction="none").mean(dim=-1)
        fwd_loss     = F.mse_loss(phi_t1_pred, phi_t1.detach())
        action_pred  = self.inverse_model(torch.cat([phi_t, phi_t1], dim=-1))
        inv_loss     = F.mse_loss(action_pred, action.detach())
        return reward, fwd_loss, inv_loss

    def icm_loss(self, fwd, inv):
        return (1.0 - self.beta) * inv + self.beta * fwd


# =============================================================================
# CONFIG
# =============================================================================

@configclass
class IrisICMOfficeEnvCfg(DirectRLEnvCfg):

    episode_length_s = 60.0
    decimation       = 2

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

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=16,
        env_spacing=80.0,
        replicate_physics=True,
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

    robot: ArticulationCfg = IRIS_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )

    cam_h:         int   = 64
    cam_w:         int   = 80
    cam_min_depth: float = 0.2
    cam_max_depth: float = 6.0

    camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/quadrotor/body/ICMCam",
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
            clipping_range=(0.2, 10.0),
        ),
        width=80,
        height=64,
    )

    # Motion — unchanged
    max_forward_vel:  float = 1.5
    max_yaw_rate:     float = 1.0
    hover_height:     float = 1.0
    altitude_kp:      float = 3.0
    max_altitude_vel: float = 1.5

    # Observation: (T=3, H=64, W=80, C=2)
    #   channel 0: depth
    #   channel 1: local novelty map crop
    history_len:  int = 3
    num_channels: int = 2   # ← was 1, now 2 (depth + novelty map)

    observation_space = gym.spaces.Box(
        low=0.0, high=1.0, shape=(3, 64, 80, 2), dtype=float)
    action_space  = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,))
    state_space   = gym.spaces.Box(
        low=-float("inf"), high=float("inf"), shape=(0,))

    # ICM — unchanged
    icm_feature_dim: int   = 64
    icm_eta:         float = 0.1
    icm_beta:        float = 0.2
    icm_lr:          float = 3e-4

    # Rewards — unchanged except novelty_icm_weight
    novelty_icm_weight: float =  1.0   # base multiplier on icm_r (novelty scales this)
    velocity_bonus:     float =  1.0
    yaw_penalty_scale:  float = -0.3
    smooth_vel_scale:   float =  0.5
    collision_penalty:  float = -8.0
    time_penalty:       float = -0.01
    danger_depth:       float =  0.20

    # Action smoothing — unchanged
    action_alpha: float = 0.6


# =============================================================================
# ENVIRONMENT
# =============================================================================

class IrisICMOfficeEnv(DirectRLEnv):

    cfg: IrisICMOfficeEnvCfg

    def __init__(self, cfg: IrisICMOfficeEnvCfg, render_mode=None, **kwargs):
        self._depth_hist: torch.Tensor | None = None
        self._prev_depth: torch.Tensor | None = None
        super().__init__(cfg, render_mode, **kwargs)

        N = self.num_envs

        # ── Unchanged from working version ────────────────────────────────────
        self._actions        = torch.zeros(N, 2, device=self.device)
        self._smooth_actions = torch.zeros(N, 2, device=self.device)
        self._last_depth     = torch.ones(N, cfg.cam_h, cfg.cam_w, device=self.device)
        self._collided       = torch.zeros(N, dtype=torch.bool, device=self.device)
        self._step_count     = 0

        self._icm = ICM(
            h=cfg.cam_h, w=cfg.cam_w,
            action_dim=2, feature_dim=cfg.icm_feature_dim,
            eta=cfg.icm_eta, beta=cfg.icm_beta,
        ).to(self.device)
        self._icm_opt    = torch.optim.Adam(self._icm.parameters(), lr=cfg.icm_lr)
        self._ep_icm_sum = torch.zeros(N, device=self.device)
        self._ep_steps   = torch.zeros(N, device=self.device)

        # ── Occupancy / visit-count grid ──────────────────────────────────────
        # Shape: (N, GRID_N, GRID_N)  float32 on GPU.
        # visit_count[i, r, c] = number of times env i visited cell (r, c).
        # Kept on GPU so the novelty lookup and local-crop are fully batched.
        self._visit_count = torch.zeros(
            N, GRID_N, GRID_N, dtype=torch.float32, device=self.device)

        # Half-extent of the local map crop in cells (integer)
        self._half_crop = LOCAL_MAP_PX // 2

        # Pad size needed when the drone is near a grid edge
        self._pad = self._half_crop

    # ─────────────────────────────────────────────────────────────────────────
    # POSE INTERFACE — the ONLY method that reads drone position.
    # Replace this during deployment with your VIO ROS 2 subscriber.
    # ─────────────────────────────────────────────────────────────────────────
    def _get_drone_pos_local(self) -> torch.Tensor:
        """
        Returns (N, 2) XY position of each drone in its env-local frame.

        During training  : reads from Isaac articulation state.
        During deployment: replace with VIO pose from ROS 2 topic.
                          Must return a (1, 2) tensor on self.device.

        Local frame convention:
            origin = env_origin (world XY of env tile)
            X = East,  Y = North  (matches Isaac world frame)
        """
        pos_w    = self._robot.data.root_pos_w[:, :2]            # (N, 2) world XY
        origin   = self._terrain.env_origins[:, :2].to(self.device)  # (N, 2)
        return pos_w - origin                                    # (N, 2) local XY

    # ─────────────────────────────────────────────────────────────────────────
    # GRID HELPERS
    # ─────────────────────────────────────────────────────────────────────────
    def _pos_to_cell(self, pos_local: torch.Tensor) -> torch.Tensor:
        """
        Convert (N, 2) local XY metres → (N, 2) integer grid indices (row, col).
        Grid origin is at the centre: cell (GRID_N//2, GRID_N//2) = (0, 0) m.
        Row increases with -Y (north up), col increases with +X (east right).
        """
        half  = GRID_N // 2
        col   = (pos_local[:, 0] / GRID_CELL_M + half).long()
        row   = (-pos_local[:, 1] / GRID_CELL_M + half).long()
        col   = col.clamp(0, GRID_N - 1)
        row   = row.clamp(0, GRID_N - 1)
        return torch.stack([row, col], dim=-1)   # (N, 2)

    def _update_visit_count(self, cells: torch.Tensor) -> torch.Tensor:
        """
        Increment visit count for each env's current cell.
        Returns (N,) novelty weights = 1 / (visit_count_before + 1).
        Higher on first visit, decays with revisits.

        cells: (N, 2) int64 [row, col]
        """
        env_idx  = torch.arange(self.num_envs, device=self.device)
        rows     = cells[:, 0]
        cols     = cells[:, 1]

        # novelty BEFORE incrementing — first visit gets weight 1.0
        counts_before = self._visit_count[env_idx, rows, cols]
        novelty       = 1.0 / (counts_before + 1.0)   # (N,) in (0, 1]

        # Increment
        self._visit_count[env_idx, rows, cols] += 1.0

        return novelty   # (N,)

    def _get_local_novelty_map(self, cells: torch.Tensor) -> torch.Tensor:
        """
        Extract a LOCAL_MAP_PX × LOCAL_MAP_PX crop of the novelty grid
        centred on each drone's current cell.

        Returns (N, LOCAL_MAP_PX, LOCAL_MAP_PX) float32 in [0, 1].
        Value 1.0 = never visited (high novelty).
        Value → 0 = visited many times.

        The crop is then resized to (cam_h, cam_w) to match the depth
        observation shape, so the two channels align pixel-to-pixel.
        """
        N    = self.num_envs
        pad  = self._pad
        half = self._half_crop
        L    = LOCAL_MAP_PX

        # Novelty from visit count: 1/(count+1), normalised to [0,1]
        # Pad the full grid with value 1.0 (unvisited) on all edges
        # Shape after padding: (N, GRID_N+2*pad, GRID_N+2*pad)
        padded = F.pad(
            self._visit_count,
            (pad, pad, pad, pad),
            mode="constant",
            value=0.0,   # pad with 0 visits (unvisited = high novelty)
        )

        # Gather crops for each env
        crops = torch.zeros(N, L, L, device=self.device)
        rows  = cells[:, 0]
        cols  = cells[:, 1]

        for i in range(N):
            r0 = int(rows[i].item())           # grid row (before padding)
            c0 = int(cols[i].item())
            # After padding, the original cell (r0, c0) is at (r0+pad, c0+pad)
            pr = r0 + pad
            pc = c0 + pad
            crops[i] = padded[i, pr - half: pr + half + 1,
                                  pc - half: pc + half + 1]

        # Convert visit counts → novelty in [0, 1]
        novelty_crops = 1.0 / (crops + 1.0)   # (N, L, L)

        # Resize to (cam_h, cam_w) so it aligns with the depth channel
        # Use bilinear for smooth gradients into the CNN
        novelty_resized = F.interpolate(
            novelty_crops.unsqueeze(1),                     # (N, 1, L, L)
            size=(self.cfg.cam_h, self.cfg.cam_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)                                        # (N, H, W)

        return novelty_resized

    # ─────────────────────────────────────────────────────────────────────────
    # SCENE SETUP — unchanged
    # ─────────────────────────────────────────────────────────────────────────
    def _setup_scene(self):
        self._robot  = Articulation(self.cfg.robot)
        self._camera = TiledCamera(self.cfg.camera)
        self.scene.articulations["robot"] = self._robot
        self.scene.sensors["icm_cam"]     = self._camera
        self.cfg.terrain.num_envs    = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        office_cfg = sim_utils.UsdFileCfg(usd_path=OFFICE_USD_PATH)
        office_cfg.func("/World/envs/env_0/Office", office_cfg,
                        translation=(0.0, 0.0, 0.0))
        self.scene.clone_environments(copy_from_source=False)
        sim_utils.DomeLightCfg(intensity=1500.0, color=(1.0, 1.0, 1.0)).func(
            "/World/Light",
            sim_utils.DomeLightCfg(intensity=1500.0, color=(1.0, 1.0, 1.0)))

    # ─────────────────────────────────────────────────────────────────────────
    # DEPTH HELPER — unchanged
    # ─────────────────────────────────────────────────────────────────────────
    def _fetch_depth_norm(self) -> torch.Tensor:
        """Returns (N, H, W) normalised depth in [0,1]. 1=open, 0=wall."""
        raw = self._camera.data.output.get("distance_to_image_plane")
        N, H, W = self.num_envs, self.cfg.cam_h, self.cfg.cam_w
        if raw is None:
            return torch.ones(N, H, W, device=self.device)
        d = raw.float() if isinstance(raw, torch.Tensor) else \
            torch.tensor(raw, dtype=torch.float32, device=self.device)
        if d.ndim == 2: d = d.unsqueeze(0)
        if d.ndim == 4: d = d.squeeze(-1)
        d = d.clamp(self.cfg.cam_min_depth, self.cfg.cam_max_depth)
        d = (d - self.cfg.cam_min_depth) / (
            self.cfg.cam_max_depth - self.cfg.cam_min_depth)
        return torch.nan_to_num(d, nan=1.0, posinf=1.0, neginf=0.0)

    # ─────────────────────────────────────────────────────────────────────────
    # ACTIONS — unchanged
    # ─────────────────────────────────────────────────────────────────────────
    def _pre_physics_step(self, actions: torch.Tensor):
        raw = actions.clone().clamp(-1.0, 1.0)
        self._smooth_actions = (self.cfg.action_alpha * self._smooth_actions +
                                (1.0 - self.cfg.action_alpha) * raw)
        self._actions = self._smooth_actions

        lin_b       = torch.zeros(self.num_envs, 3, device=self.device)
        lin_b[:, 0] = self._actions[:, 0] * self.cfg.max_forward_vel

        vz = (self.cfg.altitude_kp *
              (self.cfg.hover_height + self._terrain.env_origins[:, 2]
               - self._robot.data.root_pos_w[:, 2])
              ).clamp(-self.cfg.max_altitude_vel, self.cfg.max_altitude_vel)

        lin_w       = _quat_rotate(self._robot.data.root_state_w[:, 3:7], lin_b)
        lin_w[:, 2] = vz
        ang_w       = torch.zeros(self.num_envs, 3, device=self.device)
        ang_w[:, 2] = self._actions[:, 1] * self.cfg.max_yaw_rate

        self._robot.write_root_velocity_to_sim(torch.cat([lin_w, ang_w], dim=-1))

        jv = torch.zeros_like(self._robot.data.joint_vel)
        jv[:, 0], jv[:, 1] =  200.0, -200.0
        jv[:, 2], jv[:, 3] =  200.0, -200.0
        self._robot.set_joint_velocity_target(jv)

    def _apply_action(self):
        pass

    # ─────────────────────────────────────────────────────────────────────────
    # OBSERVATIONS
    # ─────────────────────────────────────────────────────────────────────────
    def _get_observations(self) -> dict:
        """
        Returns {"policy": (N, T=3, H=64, W=80, C=2)}
            C=0: normalised depth
            C=1: local novelty map  (1=unvisited, 0=heavily revisited)
        """
        depth = self._fetch_depth_norm()   # (N, H, W)

        # Shift depth history for ICM
        if self._prev_depth is None:
            self._prev_depth = depth.clone()
        else:
            self._prev_depth = self._last_depth.clone()
        self._last_depth = depth

        self._step_count += 1

        # Get drone cell and novelty map from pose
        pos_local   = self._get_drone_pos_local()          # (N, 2)
        cells       = self._pos_to_cell(pos_local)         # (N, 2) int
        novelty_map = self._get_local_novelty_map(cells)   # (N, H, W)

        # Stack depth + novelty → (N, H, W, 2)
        frame = torch.stack([depth, novelty_map], dim=-1)

        if self._depth_hist is None:
            self._depth_hist = frame.unsqueeze(1).repeat(
                1, self.cfg.history_len, 1, 1, 1).contiguous()
        else:
            self._depth_hist = torch.cat(
                [self._depth_hist[:, 1:], frame.unsqueeze(1)], dim=1)

        return {"policy": self._depth_hist}

    # ─────────────────────────────────────────────────────────────────────────
    # REWARDS
    # ─────────────────────────────────────────────────────────────────────────
    def _get_rewards(self) -> torch.Tensor:

        # ── ICM intrinsic reward (depth only — channel 0) ─────────────────────
        with torch.inference_mode(False):
            depth_t  = self._prev_depth.unsqueeze(1).clone()
            depth_t1 = self._last_depth.unsqueeze(1).clone()
            actions  = self._actions.clone()

            with torch.no_grad():
                icm_r, _, _ = self._icm(depth_t, depth_t1, actions)

            self._icm.train()
            _, fwd, inv = self._icm(depth_t, depth_t1, actions)
            loss = self._icm.icm_loss(fwd, inv)
            self._icm_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._icm.parameters(), 1.0)
            self._icm_opt.step()

        # ── Novelty weight from pose-based visit count ─────────────────────────
        # This is called AFTER _get_observations() so pose is fresh.
        pos_local = self._get_drone_pos_local()    # (N, 2)
        cells     = self._pos_to_cell(pos_local)   # (N, 2)
        novelty   = self._update_visit_count(cells)  # (N,) in (0, 1]
        # Scale ICM reward by novelty: first visit = full reward, decays fast
        icm_r_weighted = icm_r * novelty * self.cfg.novelty_icm_weight

        # ── Velocity rewards — unchanged ──────────────────────────────────────
        fwd_vel    = self._robot.data.root_lin_vel_b[:, 0].clamp(min=0.0)
        yaw_rate   = self._robot.data.root_ang_vel_b[:, 2].abs()
        vel_r      = fwd_vel * self.cfg.velocity_bonus
        not_moving = (fwd_vel < 0.2).float()
        yaw_r      = yaw_rate * not_moving * self.cfg.yaw_penalty_scale
        smooth_r   = (fwd_vel > 0.5).float() * self.cfg.smooth_vel_scale

        # ── Collision — unchanged ─────────────────────────────────────────────
        min_d          = self._last_depth.reshape(self.num_envs, -1).min(dim=-1).values
        self._collided = min_d < self.cfg.danger_depth
        col_r          = self._collided.float() * self.cfg.collision_penalty

        # ── Time penalty — unchanged ──────────────────────────────────────────
        time_r = torch.full((self.num_envs,), self.cfg.time_penalty, device=self.device)

        total = icm_r_weighted + vel_r + yaw_r + smooth_r + col_r + time_r

        self._ep_icm_sum += icm_r_weighted.detach()
        self._ep_steps   += 1.0

        if self._step_count % 200 == 0:
            avg_icm    = float(self._ep_icm_sum.mean() /
                               self._ep_steps.mean().clamp(min=1.0))
            total_vis  = int((self._visit_count > 0).sum().item())
            cells_each = total_vis // max(self.num_envs, 1)
            area_m2    = cells_each * (GRID_CELL_M ** 2)
            print(f"\n{'='*55}  step={self._step_count}")
            print(f"  icm_r     = {float(icm_r.mean()):+.5f}  weighted={float(icm_r_weighted.mean()):+.5f}  ep_avg={avg_icm:.5f}")
            print(f"  novelty   = {float(novelty.mean()):.4f}  "
                  f"visited_cells/env≈{cells_each}  area≈{area_m2:.1f}m²")
            print(f"  icm_loss  = {float(loss):+.5f}")
            print(f"  fwd_vel   = {float(fwd_vel.mean()):+.3f} m/s  "
                  f"vel_r={float(vel_r.mean()):+.4f}")
            print(f"  yaw_r     = {float(yaw_r.mean()):+.4f}  "
                  f"smooth_r={float(smooth_r.mean()):+.4f}")
            print(f"  min_depth = {float(min_d.mean()):.3f}m  "
                  f"col={float(self._collided.float().mean()*100):.0f}%")
            print(f"  total_r   = {float(total.mean()):+.4f}")
            print(f"  alt       = {float(self._robot.data.root_pos_w[:,2].mean()):.2f}m")

        self.extras["log"] = {
            "icm_reward":      float(icm_r.mean()),
            "icm_r_weighted":  float(icm_r_weighted.mean()),
            "icm_loss":        float(loss),
            "novelty_mean":    float(novelty.mean()),
            "fwd_vel":         float(fwd_vel.mean()),
            "min_depth":       float(min_d.mean()),
            "col_rate":        float(self._collided.float().mean()),
            "visited_cells":   int((self._visit_count > 0).float().mean().item()
                                   * GRID_N * GRID_N),
        }
        return total

    # ─────────────────────────────────────────────────────────────────────────
    # TERMINATION — unchanged
    # ─────────────────────────────────────────────────────────────────────────
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        alt_fail = (
            (self._robot.data.root_pos_w[:, 2] < 0.2) |
            (self._robot.data.root_pos_w[:, 2] > 3.0)
        )
        return alt_fail | self._collided, time_out

    # ─────────────────────────────────────────────────────────────────────────
    # RESET
    # ─────────────────────────────────────────────────────────────────────────
    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        ids_np     = env_ids.cpu().numpy()
        N_reset    = len(ids_np)
        spawn_idxs = np.random.randint(0, len(SPAWN_TABLE), size=N_reset)

        for i, eid in enumerate(ids_np):
            t = torch.tensor([eid], device=self.device)
            sx, sy, sz, syaw_deg = SPAWN_TABLE[spawn_idxs[i]]
            sx       += float(np.random.uniform(-SPAWN_XY_NOISE, SPAWN_XY_NOISE))
            sy       += float(np.random.uniform(-SPAWN_XY_NOISE, SPAWN_XY_NOISE))
            syaw_deg += float(np.random.uniform(-SPAWN_YAW_NOISE, SPAWN_YAW_NOISE))
            syaw_rad  = math.radians(syaw_deg)
            orig      = self._terrain.env_origins[eid].cpu().numpy()

            s = self._robot.data.default_root_state[t].clone()
            s[0, 0]  = float(orig[0]) + sx
            s[0, 1]  = float(orig[1]) + sy
            s[0, 2]  = sz
            half     = syaw_rad * 0.5
            s[0, 3]  = float(math.cos(half))
            s[0, 4]  = 0.0
            s[0, 5]  = 0.0
            s[0, 6]  = float(math.sin(half))
            s[0, 7:] = 0.0

            self._robot.write_root_pose_to_sim(s[:, :7], t)
            self._robot.write_root_velocity_to_sim(s[:, 7:], t)
            self._robot.write_joint_state_to_sim(
                self._robot.data.default_joint_pos[t],
                self._robot.data.default_joint_vel[t],
                None, t,
            )

        # Clear visit grids for reset envs — each episode starts with a blank map
        self._visit_count[env_ids] = 0.0

        if self._depth_hist is not None:
            self._depth_hist[env_ids] = 0.5
        self._last_depth[env_ids]     = 1.0
        self._prev_depth              = None
        self._collided[env_ids]       = False
        self._smooth_actions[env_ids] = 0.0
        self._ep_icm_sum[env_ids]     = 0.0
        self._ep_steps[env_ids]       = 0.0

    def _set_debug_vis_impl(self, debug_vis: bool): pass
    def _debug_vis_callback(self, event):           pass


# =============================================================================
# UTILITY
# =============================================================================

def _quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    w, xyz = q[:, 0:1], q[:, 1:]
    t = 2.0 * torch.linalg.cross(xyz, v)
    return v + w * t + torch.linalg.cross(xyz, t)


# =============================================================================
# DEPLOYMENT ADAPTER  —  VIO pose over ROS 2
# =============================================================================
# During real-world deployment, subclass IrisICMOfficeEnv and override
# _get_drone_pos_local() with this adapter.  Everything else is identical.
#
# Usage:
#   from iris_icm_office_env import IrisICMOfficeEnv, IrisICMOfficeEnvCfg
#   from iris_icm_office_env import VIOPoseAdapter
#   env = VIOPoseAdapter(cfg)          # drop-in replacement
#
# The adapter subscribes to /vio/pose (geometry_msgs/PoseStamped) and
# returns the XY position in env-local frame.  For a single real drone
# num_envs=1, so the tensor is always (1, 2).

class VIOPoseAdapter(IrisICMOfficeEnv):
    """
    Subclass that replaces Isaac pose reads with VIO pose from ROS 2.
    Only _get_drone_pos_local() changes — the entire training/reward/
    observation pipeline is inherited unchanged.

    Requires:
        pip install rclpy geometry_msgs
    Run ROS 2 before instantiating:
        rclpy.init()
    """

    def __init__(self, cfg: IrisICMOfficeEnvCfg, vio_topic: str = "/vio/pose",
                 render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # VIO state — updated by ROS callback, read by _get_drone_pos_local()
        self._vio_x: float = 0.0
        self._vio_y: float = 0.0
        self._vio_lock = __import__("threading").Lock()

        # ROS 2 subscriber (lazy import so training env works without ROS)
        try:
            import rclpy
            from rclpy.node import Node
            from geometry_msgs.msg import PoseStamped

            class _VIONode(Node):
                def __init__(inner_self):
                    super().__init__("icm_vio_listener")
                    inner_self.create_subscription(
                        PoseStamped, vio_topic,
                        inner_self._cb, 10)

                def _cb(inner_self, msg: PoseStamped):
                    with self._vio_lock:
                        self._vio_x = msg.pose.position.x
                        self._vio_y = msg.pose.position.y

            self._vio_node = _VIONode()
            import threading
            threading.Thread(
                target=rclpy.spin,
                args=(self._vio_node,),
                daemon=True,
            ).start()
            print(f"[VIOPoseAdapter] subscribed to {vio_topic}")

        except ImportError:
            print("[VIOPoseAdapter] rclpy not found — VIO pose will be (0, 0).")

    def _get_drone_pos_local(self) -> torch.Tensor:
        """Return VIO pose as (1, 2) tensor in env-local frame."""
        with self._vio_lock:
            x, y = self._vio_x, self._vio_y
        return torch.tensor([[x, y]], dtype=torch.float32, device=self.device)