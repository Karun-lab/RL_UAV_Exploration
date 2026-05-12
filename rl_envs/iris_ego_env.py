"""
iris_ego_env.py
===============
Iris drone exploration using an egocentric occupancy map.

The drone carries a MAP_SIZE×MAP_SIZE grid centred on itself.
"Up" in the grid is always the drone's forward direction.
Depth rays update the grid each step.
Reward = new cells discovered. Simple. Unambiguous.

Observation: (N, T=1, MAP_SIZE, MAP_SIZE, 2)
    channel 0: occupancy map   — unknown=0.5, free=1.0, occupied=0.0
    channel 1: state channel   — forward velocity, normalised to [0,1]
               expanded spatially, same pattern as ball tracking's
               search_active channel. Tells policy how fast it's moving.

Why T=1 (no history stack)?
    The egocentric map IS the memory. It accumulates across steps.
    No need to stack frames — the map already encodes history.

Actions: [vx, yaw_rate] in [-1, 1]
Altitude: P-controller

Rewards (proven simple approach):
    + new_cells * scale   primary exploration signal
    - time                forces efficiency
    - proximity           graded wall penalty from depth min
    + success_bonus       at coverage threshold (one-time)

Key improvements over old code:
    - Random spawn (was fixed)
    - Egocentric map resets correctly on reset
    - Proximity penalty uses float32 consistently (no dtype bugs)
    - Map update inner loop vectorised per ray (no Python j/s loop)
    - Observation uses CNN-compatible (T, H, W, C) format
    - Compatible with IrisEgoModel in iris_ego_agent.py
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCamera, TiledCameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
import gymnasium as gym
from isaaclab_assets.robots.iris import IRIS_CFG


# =============================================================================
# MAZE GEOMETRY
# =============================================================================

WALL_DEFS: List[Tuple[float, float, float, float]] = [
    # (centre_x, centre_y, size_x, size_y) metres
    ( 0.0,  4.0, 12.4,  0.4),   # north wall
    ( 0.0, -4.0, 12.4,  0.4),   # south wall
    (-6.0,  0.0,  0.4,  8.4),   # west wall
    ( 6.0,  0.0,  0.4,  8.4),   # east wall
    (-5.5,  0.0,  0.8,  0.4),   # room 1 west stub
    (-1.8,  0.0,  3.5,  0.4),   # room 1 east divider
    (-1.5,  2.0,  0.4,  4.0),   # room 1 north vertical
    ( 2.5,  0.0,  1.5,  0.4),   # room 2 west divider
    ( 5.5,  0.0,  1.0,  0.4),   # room 2 east stub
    ( 1.95,-2.0,  0.4,  4.0),   # room 2 south vertical
]

WALL_AABBS: np.ndarray = np.array(
    [[cx - sx/2, cx + sx/2, cy - sy/2, cy + sy/2]
     for cx, cy, sx, sy in WALL_DEFS],
    dtype=np.float32,
)

# Global grid for coverage tracking (not in observation)
GCOLS = 120   # 12 m / 0.1 m
GROWS = 80    #  8 m / 0.1 m
GCS   = 0.1   # metres per cell

FREE     = 1
OCCUPIED = 2
UNKNOWN  = 0

# Spawn candidates — built once at import
_MARGIN    = 0.6
_SPAWN_Z   = 0.9
_GRID_STEP = 0.7


def _build_spawn_candidates() -> np.ndarray:
    xs = np.arange(-6.0 + 0.2 + _MARGIN, 6.0 - 0.2 - _MARGIN,
                   _GRID_STEP, dtype=np.float32)
    ys = np.arange(-4.0 + 0.2 + _MARGIN, 4.0 - 0.2 - _MARGIN,
                   _GRID_STEP, dtype=np.float32)
    gx, gy     = np.meshgrid(xs, ys)
    candidates = np.stack([gx.ravel(), gy.ravel()], axis=1)
    valid      = np.ones(len(candidates), dtype=bool)
    for xmin, xmax, ymin, ymax in WALL_AABBS:
        in_wall = (
            (candidates[:, 0] > xmin - _MARGIN) &
            (candidates[:, 0] < xmax + _MARGIN) &
            (candidates[:, 1] > ymin - _MARGIN) &
            (candidates[:, 1] < ymax + _MARGIN)
        )
        valid &= ~in_wall
    result = candidates[valid]
    assert len(result) > 0, "No valid spawn candidates"
    return result


SPAWN_CANDIDATES: np.ndarray = _build_spawn_candidates()
print(f"[IrisEgoEnv] {len(SPAWN_CANDIDATES)} valid spawn positions")


# =============================================================================
# CONFIG
# =============================================================================

@configclass
class IrisEgoEnvCfg(DirectRLEnvCfg):

    episode_length_s = 90.0
    decimation       = 2          # 100Hz sim → 50Hz control

    # Egocentric map: 40×40 cells at 0.25m = 10m radius around drone
    map_size:      int   = 40
    map_cell_size: float = 0.25

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
        num_envs=32,
        env_spacing=20.0,
        replicate_physics=True,
    )

    robot: ArticulationCfg = IRIS_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )

    # Depth camera — 80×64, 90° FOV
    cam_width:     int   = 80
    cam_height:    int   = 64
    cam_fov_deg:   float = 90.0
    cam_min_depth: float = 0.15
    cam_max_depth: float = 5.0
    wall_height:   float = 2.0

    camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/quadrotor/body/EgoCam",
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
            clipping_range=(0.15, 8.0),
        ),
        width=80,
        height=64,
    )

    # Motion
    max_forward_vel:  float = 1.0
    max_yaw_rate:     float = 1.5
    hover_height:     float = 0.9
    altitude_kp:      float = 2.0
    max_altitude_vel: float = 1.0

    # Observation: (T=1, MAP_SIZE, MAP_SIZE, C=2)
    # channel 0: map, channel 1: velocity state
    observation_space = gym.spaces.Box(
        low=0.0, high=1.0,
        shape=(1, 40, 40, 2),
        dtype=float,
    )
    action_space  = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,))
    state_space   = gym.spaces.Box(
        low=-float("inf"), high=float("inf"), shape=(0,))

    # Rewards — tuned for fast learning
    new_cell_scale:    float =  5.0    # per newly discovered global cell
    time_penalty:      float = -0.01   # per step
    prox_warn_m:       float =  0.6    # start penalising below this
    prox_max_penalty:  float = -1.0    # at contact
    success_threshold: float =  0.70   # fraction of global grid = free
    success_bonus:     float =  200.0


# =============================================================================
# ENVIRONMENT
# =============================================================================

class IrisEgoEnv(DirectRLEnv):

    cfg: IrisEgoEnvCfg

    def __init__(self, cfg: IrisEgoEnvCfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        N = self.num_envs
        M = self.cfg.map_size

        # Egocentric map: drone always at (M//2, M//2), "up" = forward
        self._ego_map         = np.zeros((N, M, M), dtype=np.uint8)

        # Global grid for coverage reward (world frame, not in observation)
        self._global_grid     = np.zeros((N, GROWS, GCOLS), dtype=np.uint8)
        self._prev_free_count = np.zeros(N, dtype=np.int32)

        self._actions      = torch.zeros(N, 2, device=self.device)
        self._last_depth_m = np.full((N, cfg.cam_height, cfg.cam_width),
                                     cfg.cam_max_depth, dtype=np.float32)
        self._succeeded    = torch.zeros(N, dtype=torch.bool, device=self.device)
        self._step_count   = 0

        # Precompute per-column ray angles (body frame, horizontal scan)
        W    = cfg.cam_width
        hfov = math.radians(cfg.cam_fov_deg)
        fx   = (W / 2.0) / math.tan(hfov / 2.0)
        cols = np.arange(W, dtype=np.float32)
        self._ray_angles = np.arctan2(cols - W / 2.0, fx)   # (W,) body frame

    # ── Scene ─────────────────────────────────────────────────────────────────
    def _setup_scene(self):
        self._robot  = Articulation(self.cfg.robot)
        self._camera = TiledCamera(self.cfg.camera)

        self.scene.articulations["robot"] = self._robot
        self.scene.sensors["ego_cam"]     = self._camera

        self.cfg.terrain.num_envs    = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self._spawn_walls()
        self.scene.clone_environments(copy_from_source=False)

        sim_utils.DomeLightCfg(
            intensity=2000.0, color=(0.75, 0.75, 0.75)
        ).func("/World/Light",
               sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)))

    def _spawn_walls(self):
        import omni.usd
        from pxr import UsdGeom
        stage = omni.usd.get_context().get_stage()
        UsdGeom.Xform.Define(stage, "/World/envs/env_0/Maze")
        for i, (cx, cy, sx, sy) in enumerate(WALL_DEFS):
            wall_cfg = sim_utils.CuboidCfg(
                size=(sx, sy, self.cfg.wall_height),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=True, disable_gravity=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=1000.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.6, 0.6, 0.65), roughness=0.9),
            )
            wall_cfg.func(
                f"/World/envs/env_0/Maze/Wall_{i:03d}", wall_cfg,
                translation=(cx, cy, self.cfg.wall_height / 2.0),
            )

    # ── Depth fetch ───────────────────────────────────────────────────────────
    def _fetch_depth(self) -> np.ndarray:
        """Returns (N, H, W) depth in metres. Invalid → cam_max_depth."""
        raw = self._camera.data.output.get("distance_to_image_plane")
        N, H, W = self.num_envs, self.cfg.cam_height, self.cfg.cam_width

        if raw is None:
            return np.full((N, H, W), self.cfg.cam_max_depth, dtype=np.float32)

        d = raw.float().cpu().numpy() if isinstance(raw, torch.Tensor) \
            else np.asarray(raw, dtype=np.float32)

        if d.ndim == 2:   d = d[np.newaxis].repeat(N, axis=0)
        elif d.ndim == 4: d = d[:, :, :, 0]

        d = np.clip(d, self.cfg.cam_min_depth, self.cfg.cam_max_depth)
        return np.where(np.isfinite(d), d, self.cfg.cam_max_depth)

    # ── Map update ────────────────────────────────────────────────────────────
    def _update_maps(self, depth_m: np.ndarray):
        """
        Update egocentric map (body frame) and global grid (world frame).
        Uses the middle row of depth for a horizontal 1D scan.
        Vectorised per ray — no inner Python loop over steps.
        """
        M   = self.cfg.map_size
        mcs = self.cfg.map_cell_size
        mc  = M // 2    # drone centre col in ego map
        mr  = M // 2    # drone centre row in ego map

        ghx = (GCOLS * GCS) / 2.0
        ghy = (GROWS * GCS) / 2.0

        pos_np  = self._robot.data.root_pos_w.cpu().numpy()
        origins = self._terrain.env_origins.cpu().numpy()
        quat    = self._robot.data.root_state_w[:, 3:7].cpu().numpy()
        yaw     = np.arctan2(
            2.0*(quat[:,0]*quat[:,3] + quat[:,1]*quat[:,2]),
            1.0 - 2.0*(quat[:,2]**2 + quat[:,3]**2)
        )   # (N,)

        # Use middle row for horizontal scan
        mid = self.cfg.cam_height // 2
        scan = depth_m[:, mid, :]   # (N, W)

        for i in range(self.num_envs):
            ox = pos_np[i, 0] - origins[i, 0]
            oy = pos_np[i, 1] - origins[i, 1]

            depths = scan[i]   # (W,)
            valid  = (depths > self.cfg.cam_min_depth) & \
                     (depths < self.cfg.cam_max_depth) & \
                     np.isfinite(depths)
            if not valid.any():
                continue

            dv      = depths[valid]
            rays_b  = self._ray_angles[valid]   # body frame angles
            rays_w  = yaw[i] + rays_b           # world frame angles

            cos_b = np.cos(rays_b)
            sin_b = np.sin(rays_b)
            cos_w = np.cos(rays_w)
            sin_w = np.sin(rays_w)

            # ── Egocentric map (vectorised) ───────────────────────────────
            # Sample along each ray at every cell step simultaneously
            max_steps = int(self.cfg.cam_max_depth / mcs) + 1
            step_dists = np.arange(1, max_steps + 1, dtype=np.float32) * mcs  # (S,)

            # For each ray: free cells along it, endpoint = occupied
            # dv: (R,), step_dists: (S,) → broadcast → (R, S)
            along = step_dists[np.newaxis, :] <= dv[:, np.newaxis]  # (R, S)

            # Cell indices in ego map for each (ray, step)
            step_arr = step_dists[np.newaxis, :]           # (1, S)
            r_idx = (mr - np.round(step_arr * cos_b[:, np.newaxis] / mcs)
                     ).astype(int)   # (R, S)
            c_idx = (mc + np.round(step_arr * sin_b[:, np.newaxis] / mcs)
                     ).astype(int)   # (R, S)

            in_bounds = (r_idx >= 0) & (r_idx < M) & (c_idx >= 0) & (c_idx < M)
            free_mask = along & in_bounds

            # Mark free cells
            fr_flat = r_idx[free_mask]
            fc_flat = c_idx[free_mask]
            not_occ = self._ego_map[i, fr_flat, fc_flat] != OCCUPIED
            self._ego_map[i, fr_flat[not_occ], fc_flat[not_occ]] = FREE

            # Mark endpoints as occupied
            ep_r = (mr - np.round(dv * cos_b / mcs)).astype(int).clip(0, M-1)
            ep_c = (mc + np.round(dv * sin_b / mcs)).astype(int).clip(0, M-1)
            hit  = dv < self.cfg.cam_max_depth * 0.95
            self._ego_map[i, ep_r[hit], ep_c[hit]] = OCCUPIED

            # ── Global grid (world frame, for coverage reward) ────────────
            # Midpoints → FREE
            fx_ = ox + dv * 0.5 * cos_w
            fy_ = oy + dv * 0.5 * sin_w
            gc  = np.clip(np.floor((fx_ + ghx) / GCS).astype(int), 0, GCOLS-1)
            gr  = np.clip(np.floor((ghy - fy_) / GCS).astype(int), 0, GROWS-1)
            nm  = self._global_grid[i, gr, gc] != OCCUPIED
            self._global_grid[i, gr[nm], gc[nm]] = FREE

            # Endpoints → OCCUPIED
            hx_ = ox + dv * cos_w
            hy_ = oy + dv * sin_w
            hc  = np.clip(np.floor((hx_ + ghx) / GCS).astype(int), 0, GCOLS-1)
            hr  = np.clip(np.floor((ghy - hy_) / GCS).astype(int), 0, GROWS-1)
            self._global_grid[i, hr, hc] = OCCUPIED

            # Drone cell → FREE
            dc = np.clip(int((ox + ghx) / GCS), 0, GCOLS-1)
            dr = np.clip(int((ghy - oy) / GCS), 0, GROWS-1)
            self._global_grid[i, dr, dc] = FREE

    # ── Actions ───────────────────────────────────────────────────────────────
    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions    = actions.clone().clamp(-1.0, 1.0)
        lin_b            = torch.zeros(self.num_envs, 3, device=self.device)
        lin_b[:, 0]      = self._actions[:, 0] * self.cfg.max_forward_vel

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

    # ── Observations ──────────────────────────────────────────────────────────
    def _get_observations(self) -> dict:
        """
        Returns {"policy": (N, 1, M, M, 2)}

        channel 0: occupancy map
            unknown  → 0.5  (neutral, not yet seen)
            free     → 1.0  (bright, go here)
            occupied → 0.0  (dark, wall)

        channel 1: forward velocity normalised to [0,1]
            tells CNN whether drone is moving or stationary
            same role as search_active in ball tracking

        Why this encoding?
            Unknown=0.5 sits between free and occupied.
            The CNN learns: bright=go, dark=wall, grey=unknown/explore.
            This gives a natural visual gradient pointing toward unexplored space.
        """
        depth_m = self._fetch_depth()
        self._last_depth_m = depth_m
        self._update_maps(depth_m)
        self._step_count += 1

        N = self.num_envs
        M = self.cfg.map_size

        # Map channel: unknown→0.5, free→1.0, occupied→0.0
        ego   = self._ego_map.astype(np.float32)
        map_ch = np.where(ego == UNKNOWN, 0.5,
                 np.where(ego == FREE,    1.0, 0.0))   # (N, M, M)
        map_t  = torch.tensor(map_ch, dtype=torch.float32, device=self.device)

        # Velocity channel: forward velocity normalised to [0,1]
        fwd_vel  = self._robot.data.root_lin_vel_b[:, 0]   # (N,)
        vel_norm = ((fwd_vel / self.cfg.max_forward_vel).clamp(-1.0, 1.0) + 1.0) / 2.0
        vel_ch   = vel_norm.view(N, 1, 1).expand(N, M, M)   # (N, M, M)

        # Stack channels: (N, M, M, 2)
        frame = torch.stack([map_t, vel_ch], dim=-1)

        # Add T=1 dimension: (N, 1, M, M, 2)
        obs = frame.unsqueeze(1)

        return {"policy": obs}

    # ── Rewards ───────────────────────────────────────────────────────────────
    def _get_rewards(self) -> torch.Tensor:
        # 1. New cells — primary signal
        free_c    = (self._global_grid == FREE).sum(axis=(1, 2)).astype(np.int32)
        new_cells = np.maximum(0, free_c - self._prev_free_count).astype(np.float32)
        self._prev_free_count = free_c.copy()
        new_r = torch.tensor(new_cells, dtype=torch.float32,
                              device=self.device) * self.cfg.new_cell_scale

        # 2. Proximity penalty from minimum depth
        min_d = torch.tensor(
            self._last_depth_m.reshape(self.num_envs, -1).min(axis=-1).astype(np.float32),
            device=self.device,
        )
        prox_fac = ((self.cfg.prox_warn_m - min_d) / self.cfg.prox_warn_m
                    ).clamp(0.0, 1.0)
        prox_r = prox_fac * self.cfg.prox_max_penalty

        # 3. Time penalty
        time_r = torch.full(
            (self.num_envs,), self.cfg.time_penalty, device=self.device)

        # 4. Success bonus
        tc        = GROWS * GCOLS
        free_frac = (self._global_grid == FREE).sum(axis=(1, 2)).astype(np.float32) / tc
        succeeded = torch.tensor(
            free_frac >= self.cfg.success_threshold,
            dtype=torch.bool, device=self.device,
        )

        # Stagnation: penalise not moving forward
        fwd_vel   = self._robot.data.root_lin_vel_b[:, 0]
        stag_r    = (fwd_vel < 0.1).float() * -0.5   # penalise near-zero forward velocity

        new_success   = succeeded & ~self._succeeded
        self._succeeded |= succeeded
        success_r = new_success.float() * self.cfg.success_bonus

        total = new_r + prox_r + time_r + success_r + stag_r

        # ── Training health printout every 200 steps ──────────────────────
        if self._step_count % 200 == 0:
            cov_pct     = float(free_frac.mean() * 100)
            nc_mean     = float(new_cells.mean())
            prox_mean   = float(prox_r.mean())
            total_mean  = float(total.mean())
            succ_rate   = float(self._succeeded.float().mean() * 100)

            print(f"\n{'='*55}  step={self._step_count}")
            print(f"  coverage   = {cov_pct:5.1f}%   "
                  f"(target: {self.cfg.success_threshold*100:.0f}%)")
            print(f"  new_cells  = {nc_mean:6.2f}   "
                  f"{'▲ exploring' if nc_mean > 0.5 else '▼ stagnating'}")
            print(f"  prox_r     = {prox_mean:+.4f}  "
                  f"{'⚠ near walls' if prox_mean < -0.1 else 'ok'}")
            print(f"  total_r    = {total_mean:+.4f}")
            print(f"  success    = {succ_rate:.0f}% of envs done")

            # Map health: what fraction of ego map cells are known?
            known = float(((self._ego_map[0] != UNKNOWN).sum()) /
                          (self.cfg.map_size ** 2) * 100)
            print(f"  ego_known  = {known:.0f}%  "
                  f"(env_0 local map fill)")

        self.extras["log"] = {
            "coverage_pct":   float(free_frac.mean() * 100),
            "new_cells_mean": float(new_cells.mean()),
            "prox_penalty":   float(prox_r.mean()),
            "success_rate":   float(self._succeeded.float().mean()),
        }
        return total

    # ── Termination ───────────────────────────────────────────────────────────
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        alt_fail = (
            (self._robot.data.root_pos_w[:, 2] < 0.1) |
            (self._robot.data.root_pos_w[:, 2] > 3.5)
        )

        # Collision: min depth below danger threshold
        min_d_np = self._last_depth_m.reshape(self.num_envs, -1).min(axis=-1)
        col = torch.tensor(

            min_d_np < 0.15,
            dtype=torch.bool, device=self.device,
        )

        died = alt_fail | col
        return died | self._succeeded, time_out

    # ── Reset ─────────────────────────────────────────────────────────────────
    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        ids_np = env_ids.cpu().numpy()

        # Clear maps
        self._ego_map[ids_np]         = UNKNOWN
        self._global_grid[ids_np]     = UNKNOWN
        self._prev_free_count[ids_np] = 0
        self._last_depth_m[ids_np]    = self.cfg.cam_max_depth
        self._succeeded[env_ids]      = False

        # Random spawn — different position AND yaw every episode
        chosen = SPAWN_CANDIDATES[
            np.random.choice(len(SPAWN_CANDIDATES),
                             size=len(ids_np), replace=True)]
        yaws = np.random.uniform(-math.pi, math.pi, size=len(ids_np))

        for i, eid in enumerate(ids_np):
            t    = torch.tensor([eid], device=self.device)
            orig = self._terrain.env_origins[eid].cpu().numpy()
            s    = self._robot.data.default_root_state[t].clone()

            s[0, 0] = float(orig[0] + chosen[i, 0])
            s[0, 1] = float(orig[1] + chosen[i, 1])
            s[0, 2] = float(_SPAWN_Z)

            h       = float(yaws[i]) * 0.5
            s[0, 3] = float(np.cos(h))
            s[0, 4] = 0.0
            s[0, 5] = 0.0
            s[0, 6] = float(np.sin(h))
            s[0, 7:] = 0.0

            self._robot.write_root_pose_to_sim(s[:, :7], t)
            self._robot.write_root_velocity_to_sim(s[:, 7:], t)
            self._robot.write_joint_state_to_sim(
                self._robot.data.default_joint_pos[t],
                self._robot.data.default_joint_vel[t],
                None, t,
            )

    def _set_debug_vis_impl(self, debug_vis: bool): pass
    def _debug_vis_callback(self, event):           pass


# =============================================================================
# UTILITY
# =============================================================================

def _quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    w, xyz = q[:, 0:1], q[:, 1:]
    t = 2.0 * torch.linalg.cross(xyz, v)
    return v + w * t + torch.linalg.cross(xyz, t)