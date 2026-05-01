"""
iris_explore_ego_env.py
=======================
Drone exploration with egocentric occupancy map.

Core idea:
  The drone carries a small 2D grid centred on itself (egocentric).
  The grid rotates with the drone so "up" is always forward.
  Depth rays update the grid each step.
  Reward = number of newly discovered cells this step.

  This makes the reward unambiguous: move into new space → get reward.

Observation:
  {
    "map":   (1, MAP_SIZE, MAP_SIZE)  egocentric occupancy map image
    "state": (6,)                     velocity + yaw_sin + yaw_cos + coverage
  }
  Flattened to one vector for SKRL/RSL-RL compatibility.

Network (configured in runner):
  CNN on map image → features
  MLP on state     → features
  Concatenate      → policy/value heads

Actions: [vx, yaw_rate] in [-1, 1]
Altitude: P-controller (not RL-controlled)

Rewards:
  + new_cells   : cells newly marked free this step  (main signal)
  - time        : small per-step penalty
  - proximity   : graded penalty from depth min
  + success     : large bonus at coverage threshold
  Episode ends on success, collision, OOB, or timeout.

Register in __init__.py:
  gym.register(
      id="Isaac-Iris-Explore-Ego-v0",
      entry_point="rl_WorkSpace.rl_envs.iris_explore_ego_env:IrisExploreEgoEnv",
      kwargs={
          "env_cfg_entry_point":
              "rl_WorkSpace.rl_envs.iris_explore_ego_env:IrisExploreEgoEnvCfg",
          "skrl_cfg_entry_point":
              "rl_WorkSpace.agents:skrl_ppo_ego_cfg.yaml",
      },
  )

Train:
  CUDA_VISIBLE_DEVICES=0 /isaac-sim/python.sh rl_WorkSpace/scripts/train_skrl.py --task Isaac-Iris-Explore-Ego-v0 --num_envs 32 --headless --enable_cameras
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

from isaaclab_assets.robots.iris import IRIS_CFG


# ---------------------------------------------------------------------------
# Maze
# ---------------------------------------------------------------------------
WALL_DEFS: List[Tuple[float, float, float, float]] = [
        # ---- Outer perimeter ----
    ( 0.0,  4.0, 12.4,  0.4),   # north wall
    ( 0.0, -4.0, 12.4,  0.4),   # south wall
    (-6.0,  0.0,  0.4,  8.4),   # west wall
    ( 6.0,  0.0,  0.4,  8.4),   # east wall


    # ---- Room 1 ----
    (-5.5,  0.0,  0.8,  0.4),
    (-1.8,  0.0,  3.5,  0.4),
    (-1.5,  2.0,  0.4,  4.0),
    
    # ---- Room 2 ----
    ( 2.5, 0.0,  1.5,  0.4),
    ( 5.5, 0.0,  1.0,  0.4),
    ( 1.95,-2.0,  0.4,  4.0),
]

WALL_AABBS = np.array(
    [[cx - sx/2, cx + sx/2, cy - sy/2, cy + sy/2]
     for cx, cy, sx, sy in WALL_DEFS],
    dtype=np.float32,
)

# SPAWN_XYZ: Tuple[float, float, float] = (-4.0, -2.0, 0.9)
# ---------------------------------------------------------------------------
# Spawn randomisation — add this near the top of the file, after WALL_AABBS
# ---------------------------------------------------------------------------

# Safety margin around walls (metres). The drone body is ~0.2m radius,
# plus some buffer so the camera doesn't immediately see a wall on spawn.
SPAWN_MARGIN: float = 0.5

# Hover height is fixed — only X/Y is randomised
SPAWN_Z: float = 0.9

def _build_spawn_candidates(
    margin:    float = SPAWN_MARGIN,
    grid_step: float = 0.5,          # sample every 0.5 m — dense enough
) -> np.ndarray:
    """
    Pre-compute all valid spawn (x, y) positions inside the maze.

    Strategy: enumerate a grid across the maze interior, reject any
    point that falls inside a wall AABB (expanded by margin) or outside
    the inner boundary of the outer perimeter walls.

    Returns: (K, 2) float32 array of valid (x, y) local positions.
    """
    # Inner bounds: outer walls are at ±6m (x) and ±4m (y).
    # Subtract wall thickness (0.2m) and margin.
    x_min = -6.0 + 0.2 + margin
    x_max =  6.0 - 0.2 - margin
    y_min = -4.0 + 0.2 + margin
    y_max =  4.0 - 0.2 - margin

    xs = np.arange(x_min, x_max, grid_step, dtype=np.float32)
    ys = np.arange(y_min, y_max, grid_step, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    candidates = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)  # (M, 2)

    # Expand each wall AABB by margin and reject overlapping candidates
    valid_mask = np.ones(len(candidates), dtype=bool)
    for xmin, xmax, ymin, ymax in WALL_AABBS:
        in_wall = (
            (candidates[:, 0] > xmin - margin) &
            (candidates[:, 0] < xmax + margin) &
            (candidates[:, 1] > ymin - margin) &
            (candidates[:, 1] < ymax + margin)
        )
        valid_mask &= ~in_wall

    result = candidates[valid_mask]
    assert len(result) > 0, "No valid spawn points found — check WALL_DEFS and SPAWN_MARGIN"
    return result


# Build once at import time — free at runtime
SPAWN_CANDIDATES: np.ndarray = _build_spawn_candidates()
# Global grid extent (for coverage tracking only)
GLOBAL_COLS = 120   # 12 m / 0.1 m
GLOBAL_ROWS = 80    #  8 m / 0.1 m
GLOBAL_CS   = 0.1


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@configclass
class IrisExploreEgoEnvCfg(DirectRLEnvCfg):

    episode_length_s = 60.0
    decimation       = 2

    # Egocentric map: MAP_SIZE × MAP_SIZE cells, cell_size metres each
    # Drone is always at centre. Map rotates with drone heading.
    # 40 cells at 0.25 m = 10 m radius around drone
    map_size:      int   = 40
    map_cell_size: float = 0.25

    # Camera resolution — keep small, map is the real observation
    cam_height: int = 64
    cam_width:  int = 80

    # Observation = flat(map) + state
    # map: map_size^2 = 1600
    # state: 6
    action_space      = 2
    observation_space = 40 * 40 + 6   # 1606
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
        num_envs=32,
        env_spacing=20.0,
        replicate_physics=True,
    )

    robot: ArticulationCfg = IRIS_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )

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
        width=80,
        height=64,
    )

    cam_fov_deg:   float = 90.0
    cam_min_depth: float = 0.15
    cam_max_depth: float = 4.0

    wall_height: float = 2.0

    # Motion
    max_forward_vel:  float = 1.5
    max_yaw_rate:     float = 2.0
    hover_height:     float = 0.9
    altitude_kp:      float = 2.0
    max_altitude_vel: float = 1.0

    # Rewards
    new_cell_reward:    float =  2.5    # per newly discovered map cell
    time_penalty:       float = -0.02   # per step
    prox_warn_m:        float =  0.4    # proximity warning distance
    prox_max_penalty:   float = -0.2    # at zero distance
    success_threshold:  float =  0.70   # global coverage for success
    success_bonus:      float =  200.0

    # Logging
    grid_save_path: str = "/tmp/iris_ego_grid.npy"
    grid_save_every: int = 500


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
FREE     = 1
OCCUPIED = 2
UNKNOWN  = 0


class IrisExploreEgoEnv(DirectRLEnv):
    cfg: IrisExploreEgoEnvCfg

    def __init__(self, cfg: IrisExploreEgoEnvCfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        N  = self.num_envs
        M  = self.cfg.map_size

        # Egocentric map per env: values 0=unknown, 1=free, 2=occupied
        # Shape: (N, M, M)  — drone always at centre (M//2, M//2)
        self._ego_map = np.zeros((N, M, M), dtype=np.uint8)

        # Global grid for coverage tracking only (not in observation)
        self._global_grid     = np.zeros((N, GLOBAL_ROWS, GLOBAL_COLS), dtype=np.uint8)
        self._prev_free_count = np.zeros(N, dtype=np.int32)

        self._actions = torch.zeros(N, 2, device=self.device)
        self._step_counter = 0

        # Success flag
        self._succeeded = torch.zeros(N, dtype=torch.bool, device=self.device)

        # Precompute camera ray angles
        self._precompute_rays()

        # Precompute rotation matrices for egocentric map update
        # We rotate depth ray endpoints into drone frame each step

    def _precompute_rays(self):
        W    = self.cfg.cam_width
        hfov = math.radians(self.cfg.cam_fov_deg)
        fx   = (W / 2.0) / math.tan(hfov / 2.0)
        cols = np.arange(W, dtype=np.float32)
        # Angle of each pixel column relative to camera forward axis
        self._ray_angles_local = np.arctan2(cols - W / 2.0, fx)   # (W,)

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
            cfg = sim_utils.CuboidCfg(
                size=(sx, sy, self.cfg.wall_height),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=True, disable_gravity=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=1000.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.6, 0.6, 0.65), roughness=0.9),
            )
            cfg.func(
                f"/World/envs/env_0/Maze/Wall_{i:03d}", cfg,
                translation=(cx, cy, self.cfg.wall_height / 2.0),
            )

    # ------------------------------------------------------------------
    # Depth fetch
    # ------------------------------------------------------------------
    def _fetch_depth_metres(self) -> np.ndarray:
        """
        Returns (N, H, W) depth in metres.
        Invalid pixels set to cam_max_depth (treat as open).
        """
        raw = self._camera.data.output.get("distance_to_image_plane")
        N, H, W = self.num_envs, self.cfg.cam_height, self.cfg.cam_width

        if raw is None:
            return np.full((N, H, W), self.cfg.cam_max_depth, dtype=np.float32)

        if isinstance(raw, torch.Tensor):
            d = raw.float().cpu().numpy()
        else:
            d = np.asarray(raw, dtype=np.float32)

        # Normalise to (N, H, W)
        if d.ndim == 2:
            d = np.broadcast_to(d[np.newaxis], (N, H, W)).copy()
        elif d.ndim == 3 and d.shape[-1] == 1:
            d = d[:, :, 0]
        elif d.ndim == 4:
            d = d[:, :, :, 0]

        d = np.clip(d, self.cfg.cam_min_depth, self.cfg.cam_max_depth)
        d = np.where(np.isfinite(d), d, self.cfg.cam_max_depth)
        return d   # (N, H, W) metres

    # ------------------------------------------------------------------
    # Egocentric map update
    # ------------------------------------------------------------------
    def _update_ego_map(self, depth_m: np.ndarray):
        """
        Update the egocentric occupancy map from depth rays.

        The egocentric map is always centred on the drone.
        Drone heading is always "up" in the map (row=0 is forward).
        So we do NOT need to rotate the map — rays are already in
        the drone's local frame (heading = +x body = up in map).

        For each ray:
          - Cast from drone centre (map centre)
          - Mark cells along ray as FREE
          - Mark endpoint as OCCUPIED if depth < max_depth
          - Update global grid simultaneously (in world frame)

        Map coordinates:
          row = map_size//2 - forward_cells  (up = forward)
          col = map_size//2 + right_cells    (right = +y in body frame)
        """
        M   = self.cfg.map_size
        mcs = self.cfg.map_cell_size
        cx  = M // 2   # drone is always here in egocentric map
        cy  = M // 2

        # Global grid params
        GCS  = GLOBAL_CS
        GNC  = GLOBAL_COLS
        GNR  = GLOBAL_ROWS
        ghx  = (GNC * GCS) / 2.0
        ghy  = (GNR * GCS) / 2.0

        pos_np  = self._robot.data.root_pos_w.cpu().numpy()
        origins = self._terrain.env_origins.cpu().numpy()
        quat    = self._robot.data.root_state_w[:, 3:7].cpu().numpy()
        w_q, x_q, y_q, z_q = quat[:,0], quat[:,1], quat[:,2], quat[:,3]
        yaw = np.arctan2(2.0*(w_q*z_q + x_q*y_q),
                         1.0 - 2.0*(y_q*y_q + z_q*z_q))   # (N,)

        # Use middle row of depth for horizontal scan
        mid_row = self.cfg.cam_height // 2
        scan    = depth_m[:, mid_row, :]   # (N, W)

        for i in range(self.num_envs):
            yaw_i  = yaw[i]
            ox_w   = pos_np[i, 0] - origins[i, 0]  # local frame
            oy_w   = pos_np[i, 1] - origins[i, 1]

            # World-frame ray directions
            ray_world = yaw_i + self._ray_angles_local   # (W,) world angles
            cos_w = np.cos(ray_world)
            sin_w = np.sin(ray_world)

            # Body-frame ray directions (for egocentric map)
            # In body frame: forward=+x, right=+y
            # Rotate world direction by -yaw to get body frame
            cos_b = np.cos(self._ray_angles_local)   # forward component
            sin_b = np.sin(self._ray_angles_local)   # right component

            depths = scan[i]   # (W,)
            valid  = (depths > self.cfg.cam_min_depth) & \
                     (depths < self.cfg.cam_max_depth) & \
                     np.isfinite(depths)

            if not valid.any():
                continue

            dv     = depths[valid]
            cos_wv = cos_w[valid]
            sin_wv = sin_w[valid]
            cos_bv = cos_b[valid]
            sin_bv = sin_b[valid]

            # --- Egocentric map update ---
            # Number of steps along each ray in map cells
            n_steps = np.floor(dv / mcs).astype(int).clip(1, M)

            for j, (d_j, cw_j, sw_j, cb_j, sb_j, ns) in enumerate(
                zip(dv, cos_wv, sin_wv, cos_bv, sin_bv, n_steps)
            ):
                for s in range(1, ns + 1):
                    dist   = s * mcs
                    # Body frame displacement: forward=cos_bv, right=sin_bv
                    mr = cx - int(round(dist * cb_j / mcs))  # row: up = forward
                    mc = cy + int(round(dist * sb_j / mcs))  # col: right = +
                    if 0 <= mr < M and 0 <= mc < M:
                        if self._ego_map[i, mr, mc] != OCCUPIED:
                            self._ego_map[i, mr, mc] = FREE

                # Mark endpoint as occupied (if hit a wall)
                if d_j < self.cfg.cam_max_depth * 0.95:
                    mr = cx - int(round(d_j * cb_j / mcs))
                    mc = cy + int(round(d_j * sb_j / mcs))
                    if 0 <= mr < M and 0 <= mc < M:
                        self._ego_map[i, mr, mc] = OCCUPIED

            # --- Global grid update (world frame, for coverage tracking) ---
            # Midpoint → FREE
            fx_  = ox_w + dv * 0.5 * cos_wv
            fy_  = oy_w + dv * 0.5 * sin_wv
            fc   = np.clip(np.floor((fx_ + ghx) / GCS).astype(int), 0, GNC-1)
            fr   = np.clip(np.floor((ghy - fy_) / GCS).astype(int), 0, GNR-1)
            mask = self._global_grid[i, fr, fc] != OCCUPIED
            self._global_grid[i, fr[mask], fc[mask]] = FREE

            # Endpoint → OCCUPIED
            hx_  = ox_w + dv * cos_wv
            hy_  = oy_w + dv * sin_wv
            hc   = np.clip(np.floor((hx_ + ghx) / GCS).astype(int), 0, GNC-1)
            hr   = np.clip(np.floor((ghy - hy_) / GCS).astype(int), 0, GNR-1)
            self._global_grid[i, hr, hc] = OCCUPIED

            # Drone cell → FREE
            dc = np.clip(int((ox_w + ghx) / GCS), 0, GNC-1)
            dr = np.clip(int((ghy - oy_w) / GCS), 0, GNR-1)
            self._global_grid[i, dr, dc] = FREE

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone().clamp(-1.0, 1.0)

        lin_b       = torch.zeros(self.num_envs, 3, device=self.device)
        lin_b[:, 0] = self._actions[:, 0] * self.cfg.max_forward_vel

        target_z = self.cfg.hover_height + self._terrain.env_origins[:, 2]
        vz = (self.cfg.altitude_kp * (
            target_z - self._robot.data.root_pos_w[:, 2]
        )).clamp(-self.cfg.max_altitude_vel, self.cfg.max_altitude_vel)

        q         = self._robot.data.root_state_w[:, 3:7]
        lin_w     = _quat_rotate(q, lin_b)
        lin_w[:, 2] = vz

        ang_w       = torch.zeros(self.num_envs, 3, device=self.device)
        ang_w[:, 2] = self._actions[:, 1] * self.cfg.max_yaw_rate

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
        depth_m = self._fetch_depth_metres()         # (N, H, W)
        self._last_depth_m = depth_m                 # store for reward/done

        self._update_ego_map(depth_m)
        self._step_counter += 1

        if self._step_counter % self.cfg.grid_save_every == 0:
            np.save(self.cfg.grid_save_path, self._ego_map[0])
            # Also save global grid
            np.save(
                self.cfg.grid_save_path.replace(".npy", "_global.npy"),
                self._global_grid[0],
            )

        # Yaw
        q = self._robot.data.root_state_w[:, 3:7]
        w_q, x_q, y_q, z_q = q[:,0], q[:,1], q[:,2], q[:,3]
        yaw     = torch.atan2(2.0*(w_q*z_q + x_q*y_q),
                              1.0 - 2.0*(y_q*y_q + z_q*z_q))
        yaw_sin = torch.sin(yaw).unsqueeze(1)
        yaw_cos = torch.cos(yaw).unsqueeze(1)

        # Global coverage
        tc  = GLOBAL_ROWS * GLOBAL_COLS
        cov = torch.tensor(
            (self._global_grid == FREE).sum(axis=(1,2)).astype(np.float32) / tc,
            device=self.device,
        ).unsqueeze(1)

        # State: (N, 6)
        state = torch.cat([
            self._robot.data.root_lin_vel_b,           # 3
            self._robot.data.root_ang_vel_b[:, 2:3],   # 1
            yaw_sin,                                   # 1
            yaw_cos,                                   # 1
        ], dim=-1)   # 6 total (removed coverage from state, it's implicit in map)

        # Egocentric map: (N, M, M) → (N, M*M) normalised to [-1, 0, 1]
        # unknown=0 → 0.0,  free=1 → 1.0,  occupied=2 → -1.0
        ego_np    = self._ego_map.astype(np.float32)
        ego_norm  = np.where(ego_np == 0, 0.0,
                    np.where(ego_np == 1, 1.0, -1.0))   # (N, M, M)
        ego_flat = torch.tensor(
            ego_norm.reshape(self.num_envs, -1),
            dtype=torch.float32,        # ← add this
            device=self.device,
        )

        obs = torch.cat([ego_flat, state], dim=-1)   # (N, M*M + 6)
        return {"policy": obs}

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:

        # ── New cells in GLOBAL grid ────────────────────────────────────
        # This is the primary exploration signal.
        # Each newly marked free cell = reward.
        # The drone MUST move to uncover new cells — no equilibrium exploit.
        free_c    = (self._global_grid == FREE).sum(axis=(1,2)).astype(np.int32)
        new_cells = np.maximum(0, free_c - self._prev_free_count)
        self._prev_free_count = free_c.copy()
        new_cells_t = torch.tensor(
            new_cells.astype(np.float32), device=self.device
        )

        # ── Proximity from depth ────────────────────────────────────────
        depth_m = self._last_depth_m   # (N, H, W)
        min_d = torch.tensor(
            depth_m.reshape(self.num_envs, -1).min(axis=-1).astype(np.float32),  # ← .astype()
            device=self.device,
        )
        warn       = self.cfg.prox_warn_m
        prox_fac   = ((warn - min_d) / warn).clamp(0.0, 1.0)
        prox_r     = prox_fac * self.cfg.prox_max_penalty

        # ── Time penalty ────────────────────────────────────────────────
        time_r = torch.full(
            (self.num_envs,), self.cfg.time_penalty, device=self.device
        )

        # ── Success ─────────────────────────────────────────────────────
        tc         = GLOBAL_ROWS * GLOBAL_COLS
        free_frac  = torch.tensor(
            (self._global_grid == FREE).sum(axis=(1,2)).astype(np.float32) / tc,
            device=self.device,
        )
        self._succeeded = free_frac >= self.cfg.success_threshold
        success_r = self._succeeded.float() * self.cfg.success_bonus

        # ── Total ────────────────────────────────────────────────────────
        reward = (
            new_cells_t * self.cfg.new_cell_reward
            + prox_r
            + time_r
            + success_r
        )

        if self._step_counter % 200 == 0:
            cov = float((self._global_grid == FREE).mean() * 100)
            print(
                f"[step {self._step_counter:6d}]"
                f"  cov={cov:5.1f}%"
                f"  new_cells={new_cells.mean():.2f}"
                f"  prox={prox_r.mean():.3f}"
                f"  total={reward.mean():.3f}"
            )

        self.extras["log"] = {
            "coverage_pct":   float((self._global_grid == FREE).mean() * 100),
            "new_cells_mean": float(new_cells.mean()),
            "prox_penalty":   float(prox_r.mean()),
            "success_rate":   float(self._succeeded.float().mean()),
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

        # Out of bounds
        hx   = (GLOBAL_COLS * GLOBAL_CS) / 2.0
        hy   = (GLOBAL_ROWS * GLOBAL_CS) / 2.0
        p    = self._robot.data.root_pos_w.cpu().numpy()
        orig = self._terrain.env_origins[:, :2].cpu().numpy()
        loc  = p[:, :2] - orig
        oob  = torch.tensor(
            (np.abs(loc[:,0]) > hx) | (np.abs(loc[:,1]) > hy),
            device=self.device,
        )

        # Collision: depth below danger threshold
        depth_m = self._last_depth_m
        min_d_np = depth_m.reshape(self.num_envs, -1).min(axis=-1)
        col = torch.tensor(
            min_d_np < (self.cfg.prox_warn_m * 0.35),
            device=self.device,
        )

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
        self._ego_map[ids_np]         = UNKNOWN
        self._global_grid[ids_np]     = UNKNOWN
        self._prev_free_count[ids_np] = 0
        self._succeeded[env_ids]      = False

        if hasattr(self, "_last_depth_m"):
            self._last_depth_m[ids_np] = self.cfg.cam_max_depth

        # ── Random spawn ────────────────────────────────────────────────────
        # Draw one candidate per resetting env — with replacement so we never
        # run out of candidates even with many envs.
        chosen_idx = np.random.choice(len(SPAWN_CANDIDATES), size=len(ids_np), replace=True)
        chosen_xy  = SPAWN_CANDIDATES[chosen_idx]   # (len(ids_np), 2)

        # Random yaw: full 360° so policy learns from all headings, not just
        # the fixed heading it would have from a fixed spawn.
        random_yaws = np.random.uniform(-np.pi, np.pi, size=len(ids_np))

        for i, eid in enumerate(ids_np):
            t    = torch.tensor([eid], device=self.device)
            orig = self._terrain.env_origins[eid].cpu().numpy()

            state    = self._robot.data.default_root_state[t].clone()
            state[0, 0] = float(orig[0] + chosen_xy[i, 0])
            state[0, 1] = float(orig[1] + chosen_xy[i, 1])
            state[0, 2] = float(SPAWN_Z)

            yaw_half    = float(random_yaws[i]) * 0.5
            state[0, 3] = float(np.cos(yaw_half))   # w
            state[0, 4] = 0.0
            state[0, 5] = 0.0
            state[0, 6] = float(np.sin(yaw_half))   # z

            self._robot.write_root_pose_to_sim(state[:, :7], t)
            self._robot.write_root_velocity_to_sim(state[:, 7:], t)
            jp = self._robot.data.default_joint_pos[t]
            jv = self._robot.data.default_joint_vel[t]
            self._robot.write_joint_state_to_sim(jp, jv, None, t)

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