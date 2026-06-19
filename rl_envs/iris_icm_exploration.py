"""
iris_icm_office_env.py
======================
ICM-driven office exploration with visited position heatmap memory.

WHAT CHANGED FROM PREVIOUS VERSION
------------------------------------
Added: Visited heatmap — lightweight 2D grid tracking where the drone
       has physically been. Decays slowly so recent visits matter more
       than old ones. Cropped around drone, resized to match depth frame,
       fed as channel 1 of observation.

Observation: (T=3, H=64, W=80, C=2)
    channel 0: normalised depth      [0,1]  — 1=open, 0=wall
    channel 1: visited heatmap crop  [0,1]  — 1=visited recently, 0=never been

WHY THIS COMBINATION WORKS
---------------------------
ICM rewards novel depth observations → drives toward unseen space.
Heatmap shows visited space → policy learns to avoid revisiting.
Together: explore new space AND remember where you've been.

The policy network in iris_icm_office_agent.py already handles C=2.
Only change needed there: num_channels = 2 (already set if you used
the updated agent). No architecture change — same CNN, same MLP.

HEATMAP DESIGN
--------------
Grid: 200×200 cells at 0.25m/cell = 50m×50m coverage
      Large enough for any office environment.

Update: drone's current cell set to 1.0 every step.
Decay:  all cells multiplied by 0.995 every step.
        At 50Hz: half-life ≈ 140 steps ≈ 2.8 seconds.
        Drone needs to leave and return within ~3s to see it as visited.
        This is intentionally short — we want recent path memory,
        not permanent history (that would discourage revisiting even
        when necessary, e.g. backtracking to find a new corridor).

Crop: 20×20 cells around drone (5m radius) → resize to 64×80 to match depth.
      Only local neighbourhood is shown — policy doesn't need to know
      about visits 20m away, only immediate surroundings.

DEPLOYMENT NOTE 
---------------
On the Jetson Orin, run this at 50Hz alongside the policy:
    - Get drone pose from SLAM (x, y in metres)
    - Mark pose in visit_grid numpy array
    - Multiply grid by decay factor
    - Crop 20×20 window around pose
    - Resize to (64, 80) using cv2.resize
    - Normalise to [0,1]
    - Concatenate with depth frame as channel 1
Total compute: ~0.5ms per step. Negligible.
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
# from isaaclab_assets.robots.iris import IRIS_CFG
from rl_WorkSpace.models.drone.iris import IRIS_CFG




# OFFICE ENVIRONMENT
OFFICE_USD_PATH = (
    # "/workspace/isaaclab/rl_WorkSpace/models/environments/TestEnvOfficeB.usd"
    # "/workspace/isaaclab/rl_WorkSpace/models/environments/TrainEnvOffice4.usd"
    #  "/workspace/isaaclab/rl_WorkSpace/models/environments/TrainEnvOffice3.usd"
     "/workspace/isaaclab/rl_WorkSpace/models/environments/TrainEnvOffice2.usd"
    # "/workspace/isaaclab/rl_WorkSpace/models/environments/TrainEnvOffice1.usd"

)

SPAWN_TABLE = [
    #officeB
    # (-2.0, 56.0, 1.0,   0.0),
    # (-3.0, 39.5, 1.0,   0.0),
    # ( 4.0, 39.5, 1.0, 180.0),
    # (-3.0, 50.0, 1.0, -90.0),
    # ( 0.0, 58.0, 1.0, -90.0),

    #Office4
    # (5.0, 16.5, 1.0,   0.0),
    # (5.0, 16.5, 1.0, -90.0),
    # (5.0, 7.5, 1.0,    0.0),
    # (5.0, 7.5, 1.0,   90.0),
    # (12.0, 12.0, 1.0,  0.0),

    #Office3
    # (10.0, 22.0, 1.0,  180.0),
    # (4.0, 22.0, 1.0, -90.0),
    # (2.0, 2.0, 1.0,   90.0),
    # (7.0, 8.0, 1.0,   -90.0),
    # (7.0, 10.0, 1.0,  90.0),

    #Office2
    (2.0, 12.0, 1.0,  -90.0),
    (1.5, 1.5, 1.0,    45.0),
    (5.0, 4.0, 1.0,     0.0),
    (13.0, 14.0, 1.0, -135.0),
    (10.5, 14.0, 1.0, -90.0),

    #Office1
    # (4.0, 15.0, 1.0,   0.0),
    # (9.0, 14.0, 1.0, -90.0),
    # (5.0, 4.0, 1.0,   -90.0),
    # (5.0, 4.0, 1.0,   180.0),
    # (2.0, 7.0, 1.0,  0.0),

]

SPAWN_XY_NOISE  = 0.3   # metres
SPAWN_YAW_NOISE = 30.0  # degrees
HEAT_ORIGIN_X = 0.0    # world x at grid centre
HEAT_ORIGIN_Y = 48.0   # world y at grid centre — midpoint of your spawns
# Heatmap grid parameters
# 200×200 cells at 0.25m = 50m×50m — covers any office
HEAT_COLS     = 200
HEAT_ROWS     = 200
HEAT_CS       = 0.25    # metres per cell
HEAT_CROP     = 20      # cells to crop around drone (5m radius)
HEAT_DECAY    = 0.995   # per-step decay: half-life ≈ 138 steps ≈ 2.8s at 50Hz



# ICM MODULE
class DepthEncoder(nn.Module):
    def __init__(self, h: int, w: int, feature_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.ELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
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
        self.encoder = DepthEncoder(h, w, feature_dim)
        self.forward_model = nn.Sequential(
            nn.Linear(feature_dim + action_dim, 256), nn.ELU(),
            nn.Linear(256, feature_dim),
        )
        self.inverse_model = nn.Sequential(
            nn.Linear(feature_dim * 2, 256), nn.ELU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, depth_t, depth_t1, action):
        phi_t       = self.encoder(depth_t)
        phi_t1      = self.encoder(depth_t1)
        phi_t1_pred = self.forward_model(torch.cat([phi_t, action], dim=-1))
        reward      = self.eta * 0.5 * F.mse_loss(
            phi_t1_pred, phi_t1.detach(), reduction="none").mean(dim=-1)
        fwd_loss    = F.mse_loss(phi_t1_pred, phi_t1.detach())
        action_pred = self.inverse_model(torch.cat([phi_t, phi_t1], dim=-1))
        inv_loss    = F.mse_loss(action_pred, action.detach())
        return reward, fwd_loss, inv_loss

    def icm_loss(self, fwd, inv):
        return (1.0 - self.beta) * inv + self.beta * fwd



# CONFIG
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

    # Motion
    max_forward_vel:  float = 1.5   # raised from 1.0
    max_yaw_rate:     float = 1.0   # lowered from 1.5 — reduces shiver
    hover_height:     float = 1.0
    altitude_kp:      float = 3.0
    max_altitude_vel: float = 1.5

    # ── Observation: (T=3, H=64, W=80, C=2)
    # C=2: channel 0 = depth, channel 1 = visited heatmap
    history_len:  int = 3
    num_channels: int = 2   # was 1

    observation_space = gym.spaces.Box(
        low=0.0, high=1.0, shape=(3, 64, 80, 2), dtype=float  # was (3,64,80,1)
    )
    action_space  = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,))
    state_space   = gym.spaces.Box(
        low=-float("inf"), high=float("inf"), shape=(0,))

    # ICM
    icm_feature_dim: int   = 64
    icm_eta:         float = 0.1
    icm_beta:        float = 0.2
    icm_lr:          float = 3e-4

    # Rewards (updated from previous version)
    velocity_bonus:    float =  1.0    # raised from 0.3 — forward motion dominant
    yaw_penalty_scale: float = -0.3    # new — penalise yaw without forward motion
    smooth_vel_scale:  float =  0.5    # new — reward sustained forward velocity
    collision_penalty: float = -8.0    # raised from -5.0
    time_penalty:      float = -0.01   # lowered from -0.02
    danger_depth:      float =  0.20   # lowered from 0.25

    # Action smoothing — kills shivering
    action_alpha:      float =  0.6    # EMA factor: 0=no smooth, 1=never update



# ENVIRONMENT
class IrisICMOfficeEnv(DirectRLEnv):

    cfg: IrisICMOfficeEnvCfg

    def __init__(self, cfg: IrisICMOfficeEnvCfg, render_mode=None, **kwargs):
        self._depth_hist: torch.Tensor | None = None
        self._prev_depth: torch.Tensor | None = None
        super().__init__(cfg, render_mode, **kwargs)

        N = self.num_envs

        self._actions        = torch.zeros(N, 2, device=self.device)
        self._smooth_actions = torch.zeros(N, 2, device=self.device)
        self._last_depth     = torch.ones(N, cfg.cam_h, cfg.cam_w, device=self.device)
        self._collided       = torch.zeros(N, dtype=torch.bool, device=self.device)
        self._step_count     = 0

        # Visited heatmap 
        # One grid per env: (N, HEAT_ROWS, HEAT_COLS) float32
        # Stored on CPU — updated via numpy, no GPU needed
        self._visit_grid = np.zeros((N, HEAT_ROWS, HEAT_COLS), dtype=np.float32)

        # ICM
        self._icm = ICM(
            h=cfg.cam_h, w=cfg.cam_w,
            action_dim=2, feature_dim=cfg.icm_feature_dim,
            eta=cfg.icm_eta, beta=cfg.icm_beta,
        ).to(self.device)
        self._icm_opt = torch.optim.Adam(self._icm.parameters(), lr=cfg.icm_lr)

        self._ep_icm_sum = torch.zeros(N, device=self.device)
        self._ep_steps   = torch.zeros(N, device=self.device)

    #Scene
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

    #Depth
    def _fetch_depth_norm(self) -> torch.Tensor:
        """(N, H, W) normalised depth in [0,1]. 1=open, 0=wall."""
        raw = self._camera.data.output.get("distance_to_image_plane")
        N, H, W = self.num_envs, self.cfg.cam_h, self.cfg.cam_w
        if raw is None:
            return torch.ones(N, H, W, device=self.device)
        d = raw.float() if isinstance(raw, torch.Tensor) else \
            torch.tensor(raw, dtype=torch.float32, device=self.device)
        if d.ndim == 2:   d = d.unsqueeze(0)
        if d.ndim == 4:   d = d.squeeze(-1)
        d = d.clamp(self.cfg.cam_min_depth, self.cfg.cam_max_depth)
        d = (d - self.cfg.cam_min_depth) / (
            self.cfg.cam_max_depth - self.cfg.cam_min_depth)
        return torch.nan_to_num(d, nan=1.0, posinf=1.0, neginf=0.0)

    # Heatmap update and crop
    def _update_heatmap(self) -> torch.Tensor:
        """
        1. Mark drone's current cell as visited (set to 1.0)
        2. Decay all cells by HEAT_DECAY
        3. Crop HEAT_CROP×HEAT_CROP window around drone
        4. Resize crop to (cam_h, cam_w) to match depth frame
        5. Return (N, cam_h, cam_w) tensor on GPU

        The resize step is important — it lets the CNN process the heatmap
        and depth at the same spatial resolution without any architecture
        changes. The policy sees "where I've been" at the same resolution
        as "what I see ahead."
        """
        pos_np  = self._robot.data.root_pos_w.cpu().numpy()
        origins = self._terrain.env_origins.cpu().numpy()

        # Grid centre offsets — heatmap covers ±25m from grid origin
        ghx = (HEAT_COLS * HEAT_CS) / 2.0
        ghy = (HEAT_ROWS * HEAT_CS) / 2.0

        H, W = self.cfg.cam_h, self.cfg.cam_w
        crops = np.zeros((self.num_envs, H, W), dtype=np.float32)

        for i in range(self.num_envs):
            # Local position relative to env origin
            ox = pos_np[i, 0] - origins[i, 0]
            oy = pos_np[i, 1] - origins[i, 1]

            # With these — offset by the actual office centre:
            world_x = pos_np[i, 0]   # use absolute world position, not local
            world_y = pos_np[i, 1]   # (env origins are (0,0) for env_0 anyway)


            ghx = (HEAT_COLS * HEAT_CS) / 2.0   # = 25.0
            ghy = (HEAT_ROWS * HEAT_CS) / 2.0   # = 25.0

            gc = int(np.clip((world_x - HEAT_ORIGIN_X + ghx) / HEAT_CS, 0, HEAT_COLS - 1))
            gr = int(np.clip((ghy - (world_y - HEAT_ORIGIN_Y)) / HEAT_CS, 0, HEAT_ROWS - 1))

            # Mark visited and decay
            self._visit_grid[i, gr, gc] = 1.0
            self._visit_grid[i] *= HEAT_DECAY

            # Crop window around drone
            r0 = max(0, gr - HEAT_CROP)
            r1 = min(HEAT_ROWS, gr + HEAT_CROP)
            c0 = max(0, gc - HEAT_CROP)
            c1 = min(HEAT_COLS, gc + HEAT_CROP)
            crop = self._visit_grid[i, r0:r1, c0:c1]

            # Pad if drone is near edge of grid
            pad_top    = max(0, HEAT_CROP - gr)
            pad_bottom = max(0, (gr + HEAT_CROP) - HEAT_ROWS)
            pad_left   = max(0, HEAT_CROP - gc)
            pad_right  = max(0, (gc + HEAT_CROP) - HEAT_COLS)
            if any([pad_top, pad_bottom, pad_left, pad_right]):
                crop = np.pad(crop, ((pad_top, pad_bottom),
                                     (pad_left, pad_right)),
                              mode='constant', constant_values=0.0)

            # Resize crop to match depth frame dimensions (H, W)
            # Use simple numpy resize — bilinear would be nicer but numpy
            # doesn't have it; scipy adds a dependency. Step-resize is fine
            # because the heatmap is smooth from the decay anyway.
            crop_h, crop_w = crop.shape
            row_idx = (np.linspace(0, crop_h - 1, H)).astype(int)
            col_idx = (np.linspace(0, crop_w - 1, W)).astype(int)
            crops[i] = crop[np.ix_(row_idx, col_idx)]

            if self._step_count < 10:
                print(f"[HEAT] env0: world=({world_x:.1f},{world_y:.1f}) "
                    f"→ grid=({gc},{gr})  grid_valid={0<=gc<HEAT_COLS and 0<=gr<HEAT_ROWS}")

        return torch.tensor(crops, dtype=torch.float32, device=self.device)

    #Actions
    def _pre_physics_step(self, actions: torch.Tensor):
        # Exponential moving average smoothing — kills shiver
        raw = actions.clone().clamp(-1.0, 1.0)
        self._smooth_actions = (self.cfg.action_alpha * self._smooth_actions +
                                (1.0 - self.cfg.action_alpha) * raw)
        self._actions = self._smooth_actions

        lin_b        = torch.zeros(self.num_envs, 3, device=self.device)
        lin_b[:, 0]  = self._actions[:, 0] * self.cfg.max_forward_vel

        vz = (self.cfg.altitude_kp *
              (self.cfg.hover_height + self._terrain.env_origins[:, 2]
               - self._robot.data.root_pos_w[:, 2])
              ).clamp(-self.cfg.max_altitude_vel, self.cfg.max_altitude_vel)

        lin_w        = _quat_rotate(self._robot.data.root_state_w[:, 3:7], lin_b)
        lin_w[:, 2]  = vz
        ang_w        = torch.zeros(self.num_envs, 3, device=self.device)
        ang_w[:, 2]  = self._actions[:, 1] * self.cfg.max_yaw_rate

        self._robot.write_root_velocity_to_sim(torch.cat([lin_w, ang_w], dim=-1))

        jv = torch.zeros_like(self._robot.data.joint_vel)
        jv[:, 0], jv[:, 1] =  200.0, -200.0
        jv[:, 2], jv[:, 3] =  200.0, -200.0
        self._robot.set_joint_velocity_target(jv)

    def _apply_action(self):
        pass

    # Observations
    def _get_observations(self) -> dict:
        """
        Returns {"policy": (N, T=3, H=64, W=80, C=2)}

        channel 0: normalised depth
        channel 1: visited heatmap crop

        The two channels are stacked before building the history.
        Each frame in the T=3 stack contains both channels — the CNN
        sees how both depth AND visit history change across time.
        """
        depth = self._fetch_depth_norm()   # (N, H, W)

        # Previous depth for ICM (stored before update)
        if self._prev_depth is None:
            self._prev_depth = depth.clone()
        else:
            self._prev_depth = self._last_depth.clone()
        self._last_depth = depth

        self._step_count += 1

        # Heatmap crop at current position: (N, H, W)
        heatmap = self._update_heatmap()

        # Stack as 2-channel frame: (N, H, W, 2)
        frame = torch.stack([depth, heatmap], dim=-1)

        # Build T=3 history stack
        if self._depth_hist is None:
            self._depth_hist = frame.unsqueeze(1).repeat(
                1, self.cfg.history_len, 1, 1, 1).contiguous()
        else:
            self._depth_hist = torch.cat(
                [self._depth_hist[:, 1:], frame.unsqueeze(1)], dim=1)

        return {"policy": self._depth_hist}

    # Rewards
    def _get_rewards(self) -> torch.Tensor:

        # ── ICM intrinsic ──
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

        # ── Velocity reward — forward motion dominant ─────────────────────────
        fwd_vel  = self._robot.data.root_lin_vel_b[:, 0].clamp(min=0.0)
        yaw_rate = self._robot.data.root_ang_vel_b[:, 2].abs()

        vel_r    = fwd_vel * self.cfg.velocity_bonus

        # Yaw penalty: only when not moving forward
        # Kills the spinning-in-place exploit
        not_moving = (fwd_vel < 0.2).float()
        yaw_r      = yaw_rate * not_moving * self.cfg.yaw_penalty_scale

        # Sustained forward motion bonus
        smooth_r = (fwd_vel > 0.5).float() * self.cfg.smooth_vel_scale

        # ── Collision ──────
        min_d          = self._last_depth.reshape(self.num_envs, -1).min(dim=-1).values
        self._collided = min_d < self.cfg.danger_depth
        col_r          = self._collided.float() * self.cfg.collision_penalty

        # ── Time penalty ───
        time_r = torch.full(
            (self.num_envs,), self.cfg.time_penalty, device=self.device)

        total = icm_r + vel_r + yaw_r + smooth_r + col_r + time_r

        self._ep_icm_sum += icm_r.detach()
        self._ep_steps   += 1.0

        # ── Health printout 
        if self._step_count % 200 == 0:
            avg_icm     = float(self._ep_icm_sum.mean() /
                                self._ep_steps.mean().clamp(min=1.0))
            heat_pct    = float((self._visit_grid[0] > 0.1).mean() * 100)
            heat_max    = float(self._visit_grid[0].max())

            print(f"\n{'='*55}  step={self._step_count}")
            print(f"  icm_r     = {float(icm_r.mean()):+.5f}  ep_avg={avg_icm:.5f}")
            print(f"  icm_loss  = {float(loss):+.5f}")
            print(f"  fwd_vel   = {float(fwd_vel.mean()):+.3f} m/s  "
                  f"vel_r={float(vel_r.mean()):+.4f}")
            print(f"  yaw_r     = {float(yaw_r.mean()):+.4f}  "
                  f"smooth_r={float(smooth_r.mean()):+.4f}")
            print(f"  min_depth = {float(min_d.mean()):.3f}m  "
                  f"col={float(self._collided.float().mean()*100):.0f}%")
            # print(f"  heatmap   = {heat_pct:.1f}% cells visited  "
                #   f"max={heat_max:.3f}")
            print(f"  total_r   = {float(total.mean()):+.4f}")
            print(f"  alt       = {float(self._robot.data.root_pos_w[:,2].mean()):.2f}m")

            # Health checks
            # if float(icm_r.mean()) < 1e-6:
            #     print("  ⚠ ICM near zero — check depth camera")
            # if heat_pct < 1.0 and self._step_count > 500:
            #     print("  ⚠ Heatmap barely filling — drone may not be moving")
            # if heat_pct > 50.0:
            #     print("  ✓ Drone is covering significant area")
            # else:
            #     print(f"  · Heatmap filling: {heat_pct:.1f}% (grows as drone explores)")

        self.extras["log"] = {
            "icm_reward":    float(icm_r.mean()),
            "icm_loss":      float(loss),
            "fwd_vel":       float(fwd_vel.mean()),
            "min_depth":     float(min_d.mean()),
            "col_rate":      float(self._collided.float().mean()),
            "heat_coverage": float((self._visit_grid > 0.1).mean() * 100),
        }
        return total

    # ── Termination ────────
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        alt_fail = (
            (self._robot.data.root_pos_w[:, 2] < 0.2) |
            (self._robot.data.root_pos_w[:, 2] > 3.0)
        )
        return alt_fail | self._collided, time_out

    # ── Reset ──────────────
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

        # Reset all buffers for resetting envs
        if self._depth_hist is not None:
            self._depth_hist[env_ids] = 0.5
        self._last_depth[env_ids]       = 1.0
        self._prev_depth                = None
        self._collided[env_ids]         = False
        self._smooth_actions[env_ids]   = 0.0   # reset smoothing buffer
        self._ep_icm_sum[env_ids]       = 0.0
        self._ep_steps[env_ids]         = 0.0

        # Clear heatmap for resetting envs — fresh memory each episode
        self._visit_grid[ids_np] = 0.0

    def _set_debug_vis_impl(self, debug_vis: bool): pass
    def _debug_vis_callback(self, event):           pass



# UTILITY
def _quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    w, xyz = q[:, 0:1], q[:, 1:]
    t = 2.0 * torch.linalg.cross(xyz, v)
    return v + w * t + torch.linalg.cross(xyz, t)