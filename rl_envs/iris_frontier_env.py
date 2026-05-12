"""
iris_explore_frontier_env.py  (office edition — fixed)
=======================================================
Fixes vs previous version:

  1. SPAWN BUG (root cause of outside-office spawning):
     The old code did:  s[0,0] = orig[0] + chosen[i,0]
     where chosen[i,0] was an XY offset from a local table.
     But your spawn coords (x=-2, y=56, etc.) are ABSOLUTE world positions.
     Adding the env origin on top pushed the drone far outside.
     Fix: write_root_pose_to_sim takes positions in world frame directly,
     so we pass (sx, sy, sz) straight, without adding orig at all.

  2. SPAWN ROTATION BUG:
     Old code built quaternion from a random yaw, ignoring each spawn's
     desired heading. Fixed: SPAWN_TABLE includes yaw_deg per entry and
     SPAWN_QUATS pre-computes the correct (qw,qx,qy,qz) for each.

  3. STAGNATION — three-layer fix:
     a. VX_MIN_CURRICULUM raised 0.25 → 0.40 and CURRICULUM_STEPS raised
        to 80k so early episodes always have meaningful forward motion.
     b. Wall-ahead gate: if min depth in centre strip < WALL_NEAR_M,
        vx is forced to 0 so the drone yaws rather than wall-hugs.
     c. Anti-revisit vectorised (no Python loop) — correct and fast.

  4. COVERAGE GRID enlarged to 30×30m and GCS raised to 0.20m so the
     grid always contains the full office footprint.

  5. env_spacing raised to 80m so cloned office USDs never overlap.
"""

from __future__ import annotations
import math
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCamera, TiledCameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
import gymnasium as gym
from isaaclab_assets.robots.iris import IRIS_CFG


# =============================================================================
# OFFICE ENVIRONMENT
# =============================================================================

OFFICE_USD_PATH = (
    "/workspace/isaaclab/rl_WorkSpace/models/environments/TestEnvOfficeB.usd"
)

# ---------------------------------------------------------------------------
# Spawn table  —  ABSOLUTE world positions as seen in Isaac Sim.
# yaw_deg: 0=faces +X, 90=faces +Y, 180=faces -X, -90=faces -Y
# ---------------------------------------------------------------------------
SPAWN_TABLE = [
    # x      y      z     yaw_deg
    (-2.0,  56.0,  1.0,    0.0),
    (-3.0,  39.5,  1.0,    0.0),
    ( 4.0,  39.5,  1.0,  180.0),
    (-3.0,  50.0,  1.0,  -90.0),
    ( 0.0,  58.0,  1.0,  -90.0),
]

def _yaw_to_quat(yaw_deg: float):
    """Rotation around Z axis → quaternion (qw, qx, qy, qz)."""
    h = math.radians(yaw_deg) * 0.5
    return (math.cos(h), 0.0, 0.0, math.sin(h))

SPAWN_QUATS = [_yaw_to_quat(row[3]) for row in SPAWN_TABLE]

# ---------------------------------------------------------------------------
# Coverage grid  (local to each env origin, 30×30 m at 20 cm resolution)
# ---------------------------------------------------------------------------
OFFICE_W_M: float = 30.0
OFFICE_H_M: float = 30.0
GCS:   float = 0.20
GCOLS: int   = int(OFFICE_W_M / GCS)
GROWS: int   = int(OFFICE_H_M / GCS)

FREE = 1; OCCUPIED = 2; UNKNOWN = 0

# ---------------------------------------------------------------------------
# Anti-stagnation curriculum
# ---------------------------------------------------------------------------
CURRICULUM_STEPS:  int   = 80_000
VX_MIN_CURRICULUM: float = 0.40    # raised from 0.25

# Force yaw instead of crashing when a wall is this close ahead
WALL_NEAR_M: float = 0.50


# =============================================================================
# CONFIG
# =============================================================================

@configclass
class IrisFrontierEnvCfg(DirectRLEnvCfg):

    episode_length_s = 90.0
    decimation       = 2

    sim: SimulationCfg = SimulationCfg(
        dt=1/100,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0, dynamic_friction=1.0, restitution=0.0,
        ),
    )

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0, dynamic_friction=1.0, restitution=0.0,
        ),
        debug_vis=False,
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=16,
        env_spacing=80.0,   # raised: office mesh is large, prevent overlap
        replicate_physics=True,
    )

    sky_light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(1.0, 0.98, 0.95)),
    )

    robot: ArticulationCfg = IRIS_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )

    cam_width:     int   = 80
    cam_height:    int   = 64
    cam_fov_deg:   float = 90.0
    cam_min_depth: float = 0.15
    cam_max_depth: float = 8.0

    camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/quadrotor/body/FrontierCam",
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
        width=80, height=64,
    )

    max_forward_vel:  float = 1.2
    max_yaw_rate:     float = 1.5
    hover_height:     float = 1.2
    altitude_kp:      float = 2.5
    max_altitude_vel: float = 1.2

    history_len:  int = 3
    num_channels: int = 2

    observation_space = gym.spaces.Box(
        low=0.0, high=1.0, shape=(3, 64, 80, 2), dtype=float)
    action_space  = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,))
    state_space   = gym.spaces.Box(
        low=-float("inf"), high=float("inf"), shape=(0,))

    new_cell_scale:        float =  3.0
    velocity_reward_scale: float =  0.8
    curiosity_scale:       float =  0.3
    revisit_limit:         int   =  4
    revisit_penalty:       float = -0.5
    time_penalty:          float = -0.15
    collision_penalty:     float = -8.0
    collision_thresh_m:    float =  0.22
    success_threshold:     float =  0.55
    success_bonus:         float =  200.0


# =============================================================================
# ENVIRONMENT
# =============================================================================

class IrisFrontierEnv(DirectRLEnv):

    cfg: IrisFrontierEnvCfg

    def __init__(self, cfg: IrisFrontierEnvCfg, render_mode=None, **kwargs):
        self._depth_hist: torch.Tensor | None = None
        super().__init__(cfg, render_mode, **kwargs)

        N = self.num_envs
        self._global_grid     = np.zeros((N, GROWS, GCOLS), dtype=np.uint8)
        self._visit_count     = np.zeros((N, GROWS, GCOLS), dtype=np.int16)
        self._prev_free_count = np.zeros(N, dtype=np.int32)
        self._actions         = torch.zeros(N, 2, device=self.device)
        self._last_depth_m    = np.full(
            (N, cfg.cam_height, cfg.cam_width), cfg.cam_max_depth, np.float32)
        self._succeeded   = torch.zeros(N, dtype=torch.bool, device=self.device)
        self._collided    = torch.zeros(N, dtype=torch.bool, device=self.device)
        self._total_steps = 0
        self._step_count  = 0

        W    = cfg.cam_width
        hfov = math.radians(cfg.cam_fov_deg)
        fx   = (W / 2.0) / math.tan(hfov / 2.0)
        self._ray_angles = np.arctan2(
            np.arange(W, dtype=np.float32) - W / 2.0, fx)
        self._cx_lo  = W // 3
        self._cx_hi  = 2 * W // 3
        self._mid_row = cfg.cam_height // 2

    # ── Scene ─────────────────────────────────────────────────────────────────
    def _setup_scene(self):
        self.cfg.terrain.num_envs    = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        office_cfg = sim_utils.UsdFileCfg(
            usd_path=OFFICE_USD_PATH,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        )
        office_cfg.func("/World/envs/env_.*/Office", office_cfg,
                        translation=(0.0, 0.0, 0.0))

        self.cfg.sky_light.spawn.func(
            self.cfg.sky_light.prim_path, self.cfg.sky_light.spawn)

        self._robot  = Articulation(self.cfg.robot)
        self._camera = TiledCamera(self.cfg.camera)
        self.scene.articulations["robot"]  = self._robot
        self.scene.sensors["frontier_cam"] = self._camera
        self.scene.clone_environments(copy_from_source=False)

    # ── Depth helpers ─────────────────────────────────────────────────────────
    def _fetch_depth(self) -> np.ndarray:
        raw = self._camera.data.output.get("distance_to_image_plane")
        N, H, W = self.num_envs, self.cfg.cam_height, self.cfg.cam_width
        if raw is None:
            return np.full((N, H, W), self.cfg.cam_max_depth, np.float32)
        d = raw.float().cpu().numpy() if isinstance(raw, torch.Tensor) \
            else np.asarray(raw, np.float32)
        if   d.ndim == 2: d = d[np.newaxis].repeat(N, axis=0)
        elif d.ndim == 4: d = d[..., 0]
        d = np.clip(d, self.cfg.cam_min_depth, self.cfg.cam_max_depth)
        return np.where(np.isfinite(d), d, self.cfg.cam_max_depth)

    def _normalise_depth(self, depth_m: np.ndarray) -> torch.Tensor:
        d = (depth_m - self.cfg.cam_min_depth) / (
            self.cfg.cam_max_depth - self.cfg.cam_min_depth)
        return torch.tensor(np.clip(d, 0.0, 1.0).astype(np.float32),
                            dtype=torch.float32, device=self.device)

    # ── Coverage grid ─────────────────────────────────────────────────────────
    def _update_global_grid(self, depth_m: np.ndarray):
        ghx = (GCOLS * GCS) / 2.0
        ghy = (GROWS * GCS) / 2.0
        pos_np  = self._robot.data.root_pos_w.cpu().numpy()
        origins = self._terrain.env_origins.cpu().numpy()
        quat    = self._robot.data.root_state_w[:, 3:7].cpu().numpy()
        yaw     = np.arctan2(
            2.0*(quat[:,0]*quat[:,3] + quat[:,1]*quat[:,2]),
            1.0 - 2.0*(quat[:,2]**2 + quat[:,3]**2))
        scan = depth_m[:, self._mid_row, :]

        for i in range(self.num_envs):
            ox = pos_np[i, 0] - origins[i, 0]
            oy = pos_np[i, 1] - origins[i, 1]
            depths = scan[i]
            valid  = (depths > self.cfg.cam_min_depth) & \
                     (depths < self.cfg.cam_max_depth) & np.isfinite(depths)
            if not valid.any():
                continue
            dv = depths[valid]
            rw = yaw[i] + self._ray_angles[valid]
            cw, sw = np.cos(rw), np.sin(rw)

            fx_ = ox + dv*0.5*cw;  fy_ = oy + dv*0.5*sw
            gc  = np.clip(np.floor((fx_+ghx)/GCS).astype(int), 0, GCOLS-1)
            gr  = np.clip(np.floor((ghy-fy_)/GCS).astype(int), 0, GROWS-1)
            nm  = self._global_grid[i, gr, gc] != OCCUPIED
            self._global_grid[i, gr[nm], gc[nm]] = FREE

            hx_ = ox + dv*cw;  hy_ = oy + dv*sw
            hc  = np.clip(np.floor((hx_+ghx)/GCS).astype(int), 0, GCOLS-1)
            hr  = np.clip(np.floor((ghy-hy_)/GCS).astype(int), 0, GROWS-1)
            self._global_grid[i, hr, hc] = OCCUPIED

            dc = int(np.clip((ox+ghx)/GCS, 0, GCOLS-1))
            dr = int(np.clip((ghy-oy)/GCS, 0, GROWS-1))
            self._global_grid[i, dr, dc] = FREE
            self._visit_count[i, dr, dc] = min(
                int(self._visit_count[i, dr, dc]) + 1, 32767)

    # ── Actions ───────────────────────────────────────────────────────────────
    def _pre_physics_step(self, actions: torch.Tensor):
        self._total_steps += self.num_envs
        a = actions.clone().clamp(-1.0, 1.0)

        # Curriculum: enforce minimum forward velocity
        if self._total_steps < CURRICULUM_STEPS * self.num_envs:
            a[:, 0] = a[:, 0].clamp(VX_MIN_CURRICULUM, 1.0)

        # Wall-ahead gate: suppress forward velocity when wall is close
        centre_strip = self._last_depth_m[
            :, self._mid_row, self._cx_lo:self._cx_hi]
        min_ahead = centre_strip.min(axis=-1)
        wall_near = torch.tensor(
            min_ahead < WALL_NEAR_M, dtype=torch.bool, device=self.device)
        a[:, 0] = torch.where(wall_near, torch.zeros_like(a[:, 0]), a[:, 0])

        self._actions   = a
        lin_b           = torch.zeros(self.num_envs, 3, device=self.device)
        lin_b[:, 0]     = a[:, 0] * self.cfg.max_forward_vel

        vz = (self.cfg.altitude_kp *
              (self.cfg.hover_height + self._terrain.env_origins[:, 2]
               - self._robot.data.root_pos_w[:, 2])
              ).clamp(-self.cfg.max_altitude_vel, self.cfg.max_altitude_vel)

        lin_w      = _quat_rotate(self._robot.data.root_state_w[:, 3:7], lin_b)
        lin_w[:,2] = vz
        ang_w      = torch.zeros(self.num_envs, 3, device=self.device)
        ang_w[:,2] = a[:, 1] * self.cfg.max_yaw_rate
        self._robot.write_root_velocity_to_sim(torch.cat([lin_w, ang_w], dim=-1))

        jv = torch.zeros_like(self._robot.data.joint_vel)
        jv[:,0], jv[:,1] =  200.0, -200.0
        jv[:,2], jv[:,3] =  200.0, -200.0
        self._robot.set_joint_velocity_target(jv)

    def _apply_action(self): pass

    # ── Observations ──────────────────────────────────────────────────────────
    def _get_observations(self) -> dict:
        depth_m = self._fetch_depth()
        self._last_depth_m = depth_m
        self._update_global_grid(depth_m)
        self._step_count += 1

        depth_norm = self._normalise_depth(depth_m)

        tc           = GROWS * GCOLS
        free_frac    = (self._global_grid == FREE).sum(axis=(1,2)).astype(np.float32) / tc
        unexplored_t = torch.tensor(1.0 - free_frac, dtype=torch.float32,
                                    device=self.device)
        N, H, W = depth_norm.shape
        cov_ch  = unexplored_t.view(N, 1, 1).expand(N, H, W)
        frame   = torch.stack([depth_norm, cov_ch], dim=-1)

        if self._depth_hist is None:
            self._depth_hist = frame.unsqueeze(1).repeat(
                1, self.cfg.history_len, 1, 1, 1).contiguous()
        else:
            self._depth_hist = torch.cat(
                [self._depth_hist[:, 1:], frame.unsqueeze(1)], dim=1)

        return {"policy": self._depth_hist}

    # ── Rewards ───────────────────────────────────────────────────────────────
    def _get_rewards(self) -> torch.Tensor:

        # 1. New cells — log-scaled
        free_c    = (self._global_grid == FREE).sum(axis=(1,2)).astype(np.int32)
        new_cells = np.maximum(0, free_c - self._prev_free_count).astype(np.float32)
        self._prev_free_count = free_c.copy()
        new_r = torch.tensor(np.log1p(new_cells), dtype=torch.float32,
                              device=self.device) * self.cfg.new_cell_scale

        # 2. Velocity reward
        vel_r = self._actions[:, 0].clamp(0.0, 1.0) * self.cfg.velocity_reward_scale

        # 3. Curiosity — mean depth of central strip
        strip     = self._last_depth_m[:, self._mid_row, self._cx_lo:self._cx_hi]
        d_norm    = (strip.mean(axis=-1) - self.cfg.cam_min_depth) / (
                     self.cfg.cam_max_depth - self.cfg.cam_min_depth)
        curious_r = torch.tensor(
            np.clip(d_norm, 0.0, 1.0).astype(np.float32),
            dtype=torch.float32, device=self.device) * self.cfg.curiosity_scale

        # 4. Anti-revisit — fully vectorised
        pos_np  = self._robot.data.root_pos_w.cpu().numpy()
        origins = self._terrain.env_origins.cpu().numpy()
        ghx = (GCOLS * GCS) / 2.0
        ghy = (GROWS * GCS) / 2.0
        ox  = pos_np[:, 0] - origins[:, 0]
        oy  = pos_np[:, 1] - origins[:, 1]
        dc  = np.clip(np.floor((ox + ghx) / GCS).astype(int), 0, GCOLS-1)
        dr  = np.clip(np.floor((ghy - oy) / GCS).astype(int), 0, GROWS-1)
        env_idx = np.arange(self.num_envs)
        visits  = self._visit_count[env_idx, dr, dc].astype(np.float32)
        over    = np.maximum(0.0, visits - self.cfg.revisit_limit)
        rev_pen = np.clip(over, 0.0, 5.0) * self.cfg.revisit_penalty
        revisit_r = torch.tensor(rev_pen, dtype=torch.float32, device=self.device)

        # 5. Time penalty
        time_r = torch.full((self.num_envs,), self.cfg.time_penalty, device=self.device)

        # 6. Collision
        min_d = self._last_depth_m.reshape(self.num_envs, -1).min(axis=-1)
        self._collided = torch.tensor(
            min_d < self.cfg.collision_thresh_m, dtype=torch.bool, device=self.device)
        col_r = self._collided.float() * self.cfg.collision_penalty

        # 7. Success
        tc        = GROWS * GCOLS
        free_frac = (self._global_grid == FREE).sum(axis=(1,2)).astype(np.float32) / tc
        succeeded = torch.tensor(
            free_frac >= self.cfg.success_threshold, dtype=torch.bool, device=self.device)
        new_success      = succeeded & ~self._succeeded
        self._succeeded |= succeeded
        success_r        = new_success.float() * self.cfg.success_bonus

        total = new_r + vel_r + curious_r + revisit_r + time_r + col_r + success_r

        if self._step_count % 200 == 0:
            cur_on = self._total_steps < CURRICULUM_STEPS * self.num_envs
            wall_blocked_pct = float(
                (self._actions[:, 0] == 0.0).float().mean() * 100)
            print(f"\n{'='*55}  step={self._step_count}")
            print(f"  curriculum={'ON ' if cur_on else 'off'}  "
                  f"vx_mean={float(self._actions[:,0].mean()):+.3f}  "
                  f"wall_gate={wall_blocked_pct:.0f}%")
            print(f"  coverage={float(free_frac.mean()*100):5.1f}%  "
                  f"(target {self.cfg.success_threshold*100:.0f}%)")
            for name, r in [
                ("new_cells",  new_r),    ("velocity",  vel_r),
                ("curiosity",  curious_r),("revisit",   revisit_r),
                ("time",       time_r),   ("collision", col_r),
                ("success",    success_r),
            ]:
                print(f"  {name:<12} {float(r.mean()):+.4f}")
            print(f"  {'total':<12} {float(total.mean()):+.4f}")

        self.extras["log"] = {
            "coverage_pct":   float(free_frac.mean() * 100),
            "new_cells_mean": float(new_cells.mean()),
            "vx_mean":        float(self._actions[:, 0].mean()),
            "collision_rate": float(self._collided.float().mean()),
            "success_rate":   float(self._succeeded.float().mean()),
        }
        return total

    # ── Termination ───────────────────────────────────────────────────────────
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        alt_fail = ((self._robot.data.root_pos_w[:, 2] < 0.2) |
                    (self._robot.data.root_pos_w[:, 2] > 4.0))
        return alt_fail | self._collided | self._succeeded, time_out

    # ── Reset ─────────────────────────────────────────────────────────────────
    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        ids_np = env_ids.cpu().numpy()
        pick   = np.random.randint(0, len(SPAWN_TABLE), size=len(ids_np))

        for i, eid in enumerate(ids_np):
            t = torch.tensor([eid], device=self.device)

            sx, sy, sz, _ = SPAWN_TABLE[pick[i]]
            qw, qx, qy, qz = SPAWN_QUATS[pick[i]]

            s = self._robot.data.default_root_state[t].clone()
            # write_root_pose_to_sim expects WORLD-frame positions directly.
            # No env-origin offset needed — that is the whole fix.
            s[0, 0] = float(sx)
            s[0, 1] = float(sy)
            s[0, 2] = float(sz)
            s[0, 3] = float(qw)
            s[0, 4] = float(qx)
            s[0, 5] = float(qy)
            s[0, 6] = float(qz)
            s[0, 7:] = 0.0

            self._robot.write_root_pose_to_sim(s[:, :7], t)
            self._robot.write_root_velocity_to_sim(s[:, 7:], t)
            self._robot.write_joint_state_to_sim(
                self._robot.data.default_joint_pos[t],
                self._robot.data.default_joint_vel[t], None, t)

        self._global_grid[ids_np]     = UNKNOWN
        self._visit_count[ids_np]     = 0
        self._prev_free_count[ids_np] = 0
        self._last_depth_m[ids_np]    = self.cfg.cam_max_depth
        self._succeeded[env_ids]      = False
        self._collided[env_ids]       = False
        if self._depth_hist is not None:
            self._depth_hist[env_ids] = 0.0

    def _set_debug_vis_impl(self, debug_vis: bool): pass
    def _debug_vis_callback(self, event): pass


# =============================================================================
# UTILITY
# =============================================================================

def _quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    w, xyz = q[:, 0:1], q[:, 1:]
    t = 2.0 * torch.linalg.cross(xyz, v)
    return v + w * t + torch.linalg.cross(xyz, t)