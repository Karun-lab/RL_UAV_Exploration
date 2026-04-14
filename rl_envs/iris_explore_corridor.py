"""
iris_maze_env.py  —  Hardcoded maze exploration with synthetic depth camera.

Maze layout (12 m wide × 8 m tall, origin at centre):
  - Outer perimeter walls
  - Room 1 (top-left)
  - Room 2 (bottom-right)
  - Long horizontal corridor connecting left to right
  - Long vertical corridor connecting top to bottom
  - T-junction where the two corridors meet

All walls are defined as a plain Python list at the bottom of this file.
Edit WALL_DEFS to change the layout without touching any RL logic.

Design:
  - 1 drone per env, 16 or 32 envs during training
  - Synthetic PinholeCameraaCfg — no hardware camera, no projection warnings
  - Depth image sliced to thin horizontal strip (11 rows) — 24x cheaper than full frame processing
  - Proximity-based collision — no PhysX contact sensor needed
  - vx + yaw only actions, altitude held by P-controller
  - Frontier exploration reward

Register in __init__.py:
    gym.register(
        id="Isaac-Iris-Maze-v0",
        entry_point="rl_WorkSpace.rl_envs.iris_maze_env:IrisMazeEnv",
        kwargs={
            "env_cfg_entry_point": "rl_WorkSpace.rl_envs.iris_maze_env:IrisMazeEnvCfg",
            "skrl_cfg_entry_point": "rl_WorkSpace.agents:skrl_ppo_maze_cfg.yaml",
        },
    )

Train:
    CUDA_VISIBLE_DEVICES=0 /isaac-sim/python.sh rl_WorkSpace/scripts/train_skrl.py \
        --task Isaac-Iris-Maze-v0 --num_envs 32 --headless --enable_cameras

Play:
    CUDA_VISIBLE_DEVICES=0 /isaac-sim/python.sh rl_WorkSpace/scripts/play_skrl.py \
        --task Isaac-Iris-Maze-v0 --num_envs 1 --livestream 2 --enable_cameras
"""

from __future__ import annotations

from typing import List, Tuple

import gymnasium as gym
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCamera, TiledCameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

from isaaclab_assets.robots.iris import IRIS_CFG
from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip


# ---------------------------------------------------------------------------
# Grid states
# ---------------------------------------------------------------------------
UNKNOWN  = 0
FREE     = 1
OCCUPIED = 2
# ---------------------------------------------------------------------------
# Hardcoded maze wall definitions
# ---------------------------------------------------------------------------
# Each entry: (centre_x, centre_y, size_x, size_y)  — metres, local to env
# All walls share the same height defined in cfg.wall_height
# Origin is at the centre of the maze.
# Positive X = right, Positive Y = up (in the top-down view)
#
# Maze footprint: X in [-6, +6],  Y in [-4, +4]  (12 m × 8 m)
#
# Edit this list to change the maze layout.
# The RL logic, reward, and observations do NOT need to change.

WALL_DEFS: List[Tuple[float, float, float, float]] = [

    # ---- Outer perimeter ----
    ( 0.0,  4.0, 20.4,  0.4),
    ( 0.0, -4.0, 20.4,  0.4),
    (-10.0,  0.0,  0.4,  8.4),
    ( 10.0,  0.0,  0.4,  8.4),

    ( 0.0,  1.5, 14.4,  0.4),
    ( 0.0, -1.5, 14.4,  0.4),
    (-6.0,  0.0,  0.4,  4.5),
    ( 6.0,  0.0,  0.4,  4.5),
]

# Drone spawn position (local frame, relative to env origin)
# Place it inside Room 1, away from walls
SPAWN_LOCAL = (-2.5, -2.5, 0.9)   # (x, y, z)

# Pre-build AABBs from wall definitions for fast collision detection
# Each AABB: [x_min, x_max, y_min, y_max]
WALL_AABBS: np.ndarray = np.array(
    [[cx - sx/2, cx + sx/2, cy - sy/2, cy + sy/2]
     for cx, cy, sx, sy in WALL_DEFS],
    dtype=np.float32,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@configclass
class IrisMazeEnvCfg(DirectRLEnvCfg):

    # Episode
    episode_length_s  = 60.0
    decimation        = 2
    action_space      = 2    # [v_forward, yaw_rate]
    observation_space = 13   # lin_vel(3)+ang_vel(3)+gravity(3)+frontier_b(3)+coverage(1)
    state_space       = 0
    debug_vis         = False

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
        env_spacing=20.0,   # must be > maze width (12 m) + buffer
        replicate_physics=True,
    )

    robot: ArticulationCfg = IRIS_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )

    # -----------------------------------------------------------------------
    # Synthetic depth camera — edit these to match your real camera later
    # -----------------------------------------------------------------------
    cam_width:       int   = 320
    cam_height:      int   = 240
    cam_fov_deg:     float = 90.0
    cam_min_depth:   float = 0.15
    cam_max_depth:   float = 5.0
    cam_slice_half:  int   = 5     # rows above+below centre to average

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
            clipping_range=(0.15, 8.0),
        ),
        width=320,
        height=240,
    )

    # -----------------------------------------------------------------------
    # Maze geometry (used for wall spawning — matches WALL_DEFS above)
    # -----------------------------------------------------------------------
    wall_height: float = 2.0   # metres

    # -----------------------------------------------------------------------
    # Occupancy grid
    # Covers 12 × 8 m at 0.1 m/cell → 120 × 80 cells
    # -----------------------------------------------------------------------
    cell_size:       float = 0.1
    grid_cols:       int   = 120   # X axis (12 m / 0.1 m)
    grid_rows:       int   = 80    # Y axis ( 8 m / 0.1 m)

    frontier_recompute_interval: int = 10

    # -----------------------------------------------------------------------
    # Motion
    # -----------------------------------------------------------------------
    max_forward_vel:  float = 1.0
    max_yaw_rate:     float = 1.5
    hover_height:     float = 0.9
    altitude_kp:      float = 2.0
    max_altitude_vel: float = 1.0

    # -----------------------------------------------------------------------
    # Collision
    # -----------------------------------------------------------------------
    collision_radius: float = 0.25

    # -----------------------------------------------------------------------
    # Rewards
    # -----------------------------------------------------------------------
    new_cell_reward_scale: float =  5.0
    frontier_reward_scale: float =  3.0
    ang_vel_penalty_scale: float = -0.01
    revisit_penalty_scale: float = -0.1
    out_of_bounds_penalty: float = -5.0
    collision_penalty:     float = -10.0

    # -----------------------------------------------------------------------
    # Logging
    # -----------------------------------------------------------------------
    grid_save_path:  str = "/tmp/iris_maze_grid.npy"
    grid_save_every: int = 300


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
class IrisMazeEnv(DirectRLEnv):
    cfg: IrisMazeEnvCfg

    def __init__(self, cfg: IrisMazeEnvCfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        N  = self.num_envs
        NR = self.cfg.grid_rows
        NC = self.cfg.grid_cols

        self._grid            = np.zeros((N, NR, NC), dtype=np.uint8)
        self._prev_free_count = np.zeros(N, dtype=np.int32)
        self._frontier_pos_w  = torch.zeros(N, 3, device=self.device)
        self._actions         = torch.zeros(N, 2, device=self.device)
        self._prev_frontier_dist = torch.zeros(N, device=self.device)

        self._episode_sums = {
            key: torch.zeros(N, dtype=torch.float, device=self.device)
            for key in ["new_cells", "frontier","progress", "ang_vel",
                        "revisit", "out_of_bounds","frontier_prox","directed", "idle","collision"]
        }

        self._step_counter = 0
        self._precompute_ray_angles()
        self.set_debug_vis(self.cfg.debug_vis)

    # ------------------------------------------------------------------
    # Camera ray precomputation
    # ------------------------------------------------------------------
    def _precompute_ray_angles(self):
        W    = self.cfg.cam_width
        H    = self.cfg.cam_height
        hfov = np.deg2rad(self.cfg.cam_fov_deg)
        fx   = (W / 2.0) / np.tan(hfov / 2.0)
        cols = np.arange(W, dtype=np.float32)
        self._ray_offsets = np.arctan2(cols - W / 2.0, fx)   # (W,)
        cy   = H // 2
        sh   = self.cfg.cam_slice_half
        self._slice_rows = slice(max(0, cy - sh), min(H, cy + sh))

    # ------------------------------------------------------------------
    # Scene setup — walls spawned in env_0 BEFORE clone_environments
    # ------------------------------------------------------------------
    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self._camera = TiledCamera(self.cfg.camera)
        self.scene.sensors["depth_cam"] = self._camera

        self.cfg.terrain.num_envs    = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # Spawn walls in env_0 only — cloner replicates them to all other envs
        self._spawn_walls_env0()

        self.scene.clone_environments(copy_from_source=False)

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _spawn_walls_env0(self):
        """
        Spawn all maze walls as static cuboid prims in env_0.
        Uses WALL_DEFS defined at the top of this file.
        The scene cloner will copy these into all other envs automatically.
        """
        import omni.usd
        from pxr import UsdGeom

        stage = omni.usd.get_context().get_stage()
        UsdGeom.Xform.Define(stage, "/World/envs/env_0/Maze")

        for idx, (cx, cy, sx, sy) in enumerate(WALL_DEFS):
            wall_cfg = sim_utils.CuboidCfg(
                size=(sx, sy, self.cfg.wall_height),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=True,
                    disable_gravity=True,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=1000.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.55, 0.55, 0.65),
                    roughness=0.9,
                ),
            )
            # Origin of env_0 is (0,0,0) at setup time — walls are in local frame
            wall_cfg.func(
                f"/World/envs/env_0/Maze/Wall_{idx:03d}",
                wall_cfg,
                translation=(cx, cy, self.cfg.wall_height / 2.0),
            )

    # ------------------------------------------------------------------
    # Action pipeline
    # ------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone().clamp(-1.0, 1.0)

        lin_vel_b = torch.zeros(self.num_envs, 3, device=self.device)
        lin_vel_b[:, 0] = self._actions[:, 0] * self.cfg.max_forward_vel

        target_z  = self.cfg.hover_height + self._terrain.env_origins[:, 2]
        current_z = self._robot.data.root_pos_w[:, 2]
        vz        = (self.cfg.altitude_kp * (target_z - current_z)).clamp(
            -self.cfg.max_altitude_vel, self.cfg.max_altitude_vel
        )

        quat_w    = self._robot.data.root_state_w[:, 3:7]
        lin_vel_w = quat_rotate(quat_w, lin_vel_b)
        lin_vel_w[:, 2] = vz

        ang_vel_w = torch.zeros(self.num_envs, 3, device=self.device)
        ang_vel_w[:, 2] = self._actions[:, 1] * self.cfg.max_yaw_rate

        self._cmd_lin_w = lin_vel_w
        self._cmd_ang_w = ang_vel_w

    def _apply_action(self):
        self._robot.write_root_velocity_to_sim(
            torch.cat([self._cmd_lin_w, self._cmd_ang_w], dim=-1)
        )
        jv = torch.zeros_like(self._robot.data.joint_vel)
        jv[:, 0], jv[:, 1] =  200.0, -200.0
        jv[:, 2], jv[:, 3] =  200.0, -200.0
        self._robot.set_joint_velocity_target(jv)

    # ------------------------------------------------------------------
    # Coordinate conversion helpers
    # ------------------------------------------------------------------
    def _world_to_grid(self, world_x: np.ndarray, world_y: np.ndarray,
                       env_origins: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert world-frame XY positions to grid (row, col) indices.
        Grid origin is at the SW corner of the maze area.
        Row 0 = north edge, Col 0 = west edge.
        """
        cs  = self.cfg.cell_size
        NR  = self.cfg.grid_rows
        NC  = self.cfg.grid_cols
        # Maze half-extents
        half_x = (NC * cs) / 2.0   # 6.0 m
        half_y = (NR * cs) / 2.0   # 4.0 m

        local_x = world_x - env_origins[:, 0]
        local_y = world_y - env_origins[:, 1]

        col = np.clip(np.floor((local_x + half_x) / cs).astype(int), 0, NC - 1)
        row = np.clip(np.floor((half_y - local_y) / cs).astype(int), 0, NR - 1)
        return row, col

    def _grid_to_world(self, row: int, col: int, env_id: int) -> Tuple[float, float]:
        cs     = self.cfg.cell_size
        half_x = (self.cfg.grid_cols * cs) / 2.0
        half_y = (self.cfg.grid_rows * cs) / 2.0
        orig   = self._terrain.env_origins[env_id].cpu().numpy()
        wx     = (col + 0.5) * cs - half_x + orig[0]
        wy     = half_y - (row + 0.5) * cs  + orig[1]
        return wx, wy

    # ------------------------------------------------------------------
    # Depth → occupancy grid
    # ------------------------------------------------------------------
    def _update_grid_from_depth(self):
        depth_data = self._camera.data.output.get("distance_to_image_plane")
        if depth_data is None:
            print("[DEPTH] No data — camera not ready")
            return

        # Ensure shape is (N, H, W, 1) regardless of Isaac Lab version
        if isinstance(depth_data, torch.Tensor):
            d = depth_data.cpu().numpy()
        else:
            d = depth_data

        # Normalise to 4D
        if d.ndim == 2:
            d = d[np.newaxis, :, :, np.newaxis]
        elif d.ndim == 3:
            d = d[np.newaxis] if d.shape[-1]==1 else d[:,:,:,np.newaxis]

        # DEBUG: print min/max of depth slice for env 0
        slice_data = d[0, self._slice_rows, :, 0]
        valid = slice_data[(slice_data > 0.1) & np.isfinite(slice_data)]
        # if len(valid) > 0:
        #     print(f"[DEPTH] env0 slice: min={valid.min():.2f}m  "
        #         f"max={valid.max():.2f}m  "
        #         f"valid_pct={100*len(valid)/slice_data.size:.0f}%")
        # else:
        #     print("[DEPTH] env0 slice: all invalid readings")
            # if d.shape[-1] == 1:
            #     d = d[np.newaxis]          # add batch dim
            # else:
            #     d = d[:, :, :, np.newaxis] # add channel dim
        # d is now (N, H, W, 1)

        cs    = self.cfg.cell_size
        NR    = self.cfg.grid_rows
        NC    = self.cfg.grid_cols
        half_x = (NC * cs) / 2.0
        half_y = (NR * cs) / 2.0

        # Average centre strip → (N, W) scan
        depth_1d = np.nanmean(d[:, self._slice_rows, :, 0], axis=1)

        pos_np  = self._robot.data.root_pos_w.cpu().numpy()       # (N, 3)
        origins = self._terrain.env_origins.cpu().numpy()         # (N, 3)

        quat = self._robot.data.root_state_w[:, 3:7].cpu().numpy()
        if quat.ndim == 1:
            quat = quat[np.newaxis, :]
        w_q, x_q, y_q, z_q = quat[:,0], quat[:,1], quat[:,2], quat[:,3]
        yaw = np.arctan2(2.0*(w_q*z_q + x_q*y_q),
                         1.0 - 2.0*(y_q*y_q + z_q*z_q))          # (N,)

        for i in range(self.num_envs):
            ox = pos_np[i, 0] - origins[i, 0]
            oy = pos_np[i, 1] - origins[i, 1]

            angles = yaw[i] + self._ray_offsets   # (W,)
            depths = depth_1d[i]                  # (W,)
            valid  = (
                (depths > self.cfg.cam_min_depth) &
                (depths < self.cfg.cam_max_depth) &
                np.isfinite(depths)
            )
            if not valid.any():
                continue

            cos_a = np.cos(angles[valid])
            sin_a = np.sin(angles[valid])
            dv    = depths[valid]

            # --- Midpoints → FREE ---
            fx   = ox + (dv * 0.5) * cos_a
            fy   = oy + (dv * 0.5) * sin_a
            fcol = np.clip(np.floor((fx + half_x) / cs).astype(int), 0, NC-1)
            frow = np.clip(np.floor((half_y - fy) / cs).astype(int), 0, NR-1)
            not_wall = self._grid[i, frow, fcol] != OCCUPIED
            self._grid[i, frow[not_wall], fcol[not_wall]] = FREE

            # --- Endpoints → OCCUPIED ---
            hx   = ox + dv * cos_a
            hy   = oy + dv * sin_a
            hcol = np.clip(np.floor((hx + half_x) / cs).astype(int), 0, NC-1)
            hrow = np.clip(np.floor((half_y - hy) / cs).astype(int), 0, NR-1)
            self._grid[i, hrow, hcol] = OCCUPIED

            # --- Drone cell → FREE ---
            dc = np.clip(int((ox + half_x) / cs), 0, NC-1)
            dr = np.clip(int((half_y - oy) / cs), 0, NR-1)
            self._grid[i, dr, dc] = FREE

    # ------------------------------------------------------------------
    # Frontier detection
    # ------------------------------------------------------------------
    def _compute_frontiers(self) -> np.ndarray:
        NR    = self.cfg.grid_rows
        NC    = self.cfg.grid_cols
        cs    = self.cfg.cell_size
        half_x = (NC * cs) / 2.0
        half_y = (NR * cs) / 2.0

        pos_np  = self._robot.data.root_pos_w.cpu().numpy()
        origins = self._terrain.env_origins.cpu().numpy()

        drone_rows, drone_cols = self._world_to_grid(
            pos_np[:, 0], pos_np[:, 1], origins
        )

        result = np.zeros((self.num_envs, 2), dtype=np.float32)

        for i in range(self.num_envs):
            g    = self._grid[i]
            free = g == FREE
            unk  = g == UNKNOWN

            has_unk = (
                np.roll(unk,  1, axis=0) | np.roll(unk, -1, axis=0) |
                np.roll(unk,  1, axis=1) | np.roll(unk, -1, axis=1)
            )
            has_unk[0,:] = has_unk[-1,:] = has_unk[:,0] = has_unk[:,-1] = False
            fronts = np.argwhere(free & has_unk)

            if len(fronts) == 0:
                # No frontier yet — point 2 m ahead in current heading
                quat = self._robot.data.root_state_w[i, 3:7].cpu().numpy()
                yaw  = np.arctan2(
                    2.0*(quat[0]*quat[3] + quat[1]*quat[2]),
                    1.0 - 2.0*(quat[2]**2 + quat[3]**2)
                )
                result[i] = [
                    pos_np[i, 0] + 2.0 * np.cos(yaw),
                    pos_np[i, 1] + 2.0 * np.sin(yaw),
                ]
                continue

            dists = (np.abs(fronts[:,0] - drone_rows[i]) +
                     np.abs(fronts[:,1] - drone_cols[i]))
            best  = fronts[np.argmin(dists)]
            wx, wy = self._grid_to_world(best[0], best[1], i)
            result[i] = [wx, wy]

        return result

    # ------------------------------------------------------------------
    # Collision and bounds
    # ------------------------------------------------------------------
    def _get_collisions(self) -> np.ndarray:
        pos_np  = self._robot.data.root_pos_w.cpu().numpy()
        origins = self._terrain.env_origins.cpu().numpy()
        col     = np.zeros(self.num_envs, dtype=bool)
        for i in range(self.num_envs):
            lx = pos_np[i, 0] - origins[i, 0]
            ly = pos_np[i, 1] - origins[i, 1]
            dx = np.maximum(0.0, np.maximum(
                WALL_AABBS[:, 0] - lx, lx - WALL_AABBS[:, 1]
            ))
            dy = np.maximum(0.0, np.maximum(
                WALL_AABBS[:, 2] - ly, ly - WALL_AABBS[:, 3]
            ))
            col[i] = np.sqrt(dx**2 + dy**2).min() < self.cfg.collision_radius
        return col

    def _is_oob(self) -> np.ndarray:
        half_x  = (self.cfg.grid_cols * self.cfg.cell_size) / 2.0
        half_y  = (self.cfg.grid_rows * self.cfg.cell_size) / 2.0
        pos_np  = self._robot.data.root_pos_w.cpu().numpy()
        origins = self._terrain.env_origins[:, :2].cpu().numpy()
        local   = pos_np[:, :2] - origins
        return (np.abs(local[:,0]) > half_x) | (np.abs(local[:,1]) > half_y)

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------
    def _get_observations(self) -> dict:
        self._update_grid_from_depth()

        if self._step_counter % self.cfg.frontier_recompute_interval == 0:
            fxy = self._compute_frontiers()
            fp  = np.zeros((self.num_envs, 3), dtype=np.float32)
            fp[:, :2] = fxy
            fp[:, 2]  = (self.cfg.hover_height +
                         self._terrain.env_origins[:, 2].cpu().numpy())
            self._frontier_pos_w = torch.tensor(fp, device=self.device)

        self._step_counter += 1

        if self._step_counter % self.cfg.grid_save_every == 0:
            np.save(self.cfg.grid_save_path, self._grid[0])
            self._save_grid_png(
                self._grid[0],
                self.cfg.grid_save_path.replace(".npy", ".png"),
            )

        frontier_b, _ = subtract_frame_transforms(
            self._robot.data.root_state_w[:, :3],
            self._robot.data.root_state_w[:, 3:7],
            self._frontier_pos_w,
        )

        total_cells = self.cfg.grid_rows * self.cfg.grid_cols
        cov = torch.tensor(
            (self._grid == FREE).sum(axis=(1,2)).astype(np.float32) / total_cells,
            device=self.device,
        ).unsqueeze(1)

        obs = torch.cat([
            self._robot.data.root_lin_vel_b,
            self._robot.data.root_ang_vel_b,
            self._robot.data.projected_gravity_b,
            frontier_b,
            cov,
        ], dim=-1)

        # Temporary: print grid fill rate every 200 steps
        # if self._step_counter % 200 == 0:
        #     free_pct = (self._grid[0] == FREE).mean() * 100
        #     occ_pct  = (self._grid[0] == OCCUPIED).mean() * 100
        #     print(f"[GRID] step={self._step_counter}  "
        #         f"free={free_pct:.1f}%  occupied={occ_pct:.1f}%  "
        #         f"unknown={100-free_pct-occ_pct:.1f}%")

        return {"policy": obs}

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        # --- New cells: the primary reward ---
        free_c    = (self._grid == FREE).sum(axis=(1,2)).astype(np.int32)
        new_cells = np.maximum(0, free_c - self._prev_free_count)
        self._prev_free_count = free_c.copy()
        nc_t = torch.tensor(new_cells, dtype=torch.float, device=self.device)

        # --- Frontier: reward for facing and moving toward frontier ---
        dist = torch.linalg.norm(
            self._frontier_pos_w - self._robot.data.root_pos_w, dim=1
        )

        # Heading alignment: cosine between drone forward and frontier direction
        quat_w     = self._robot.data.root_state_w[:, 3:7]
        forward_b  = torch.zeros(self.num_envs, 3, device=self.device)
        forward_b[:, 0] = 1.0
        forward_w  = quat_rotate(quat_w, forward_b)
        to_frontier = self._frontier_pos_w - self._robot.data.root_pos_w
        to_frontier[:, 2] = 0.0   # project to horizontal
        to_frontier_norm = torch.linalg.norm(to_frontier, dim=1, keepdim=True).clamp(min=1e-6)
        to_frontier_unit = to_frontier / to_frontier_norm
        heading_align = (forward_w * to_frontier_unit).sum(dim=1)  # cosine [-1, 1]

        # Forward velocity (positive = moving forward)
        forward_vel = self._robot.data.root_lin_vel_b[:, 0]

        # Combined: reward for moving forward WHILE facing frontier
        # This specifically rewards the yaw-then-advance behaviour
        directed_motion = (heading_align.clamp(min=0.0) * forward_vel.clamp(min=0.0))

        # --- Penalties ---
        ang_vel  = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)
        oob      = torch.tensor(self._is_oob().astype(np.float32),         device=self.device)
        collided = torch.tensor(self._get_collisions().astype(np.float32), device=self.device)

        # Strong idle penalty: punish doing nothing harder than small rotation
        is_idle = (forward_vel.abs() < 0.1) & (self._robot.data.root_ang_vel_b[:, 2].abs() < 0.2)
        idle_penalty = is_idle.float() * -0.5

        rewards = {
            # Discovering new cells is the main objective
            "new_cells":     nc_t           *  8.0,
            # Moving toward frontier while facing it
            "directed":      directed_motion *  2.0,
            # Small proximity signal — keep this weak
            "frontier_prox": (1.0 - torch.tanh(dist / 3.0)) * 0.5 * self.step_dt,
            # Penalties
            "idle":          idle_penalty,
            "ang_vel":       ang_vel  * -0.005 * self.step_dt,
            "out_of_bounds": oob      * -5.0,
            "collision":     collided * -10.0,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        for key, value in rewards.items():
            self._episode_sums[key] += value

        total_cells = self.cfg.grid_rows * self.cfg.grid_cols
        self.extras["log"] = {
            "coverage_pct":   float((self._grid==FREE).mean() * 100.0),
            "collision_rate": float(collided.mean().item()),
            "mean_new_cells": float(nc_t.mean().item()),
            "heading_align":  float(heading_align.mean().item()),
            "idle_rate":      float(is_idle.float().mean().item()),
        }

        return reward
    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        alt_fail = (
            (self._robot.data.root_pos_w[:, 2] < 0.1) |
            (self._robot.data.root_pos_w[:, 2] > 3.0)
        )
        oob      = torch.tensor(self._is_oob(),         device=self.device)
        collided = torch.tensor(self._get_collisions(), device=self.device)
        return alt_fail | oob | collided, time_out

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
                self.episode_length_buf, high=int(self.max_episode_length)
            )

        env_ids_np = env_ids.cpu().numpy()
        self._grid[env_ids_np]            = UNKNOWN
        self._prev_free_count[env_ids_np] = 0
        self._frontier_pos_w[env_ids]     = 0.0
        self._prev_frontier_dist[env_ids] = 0.0

        for env_id in env_ids_np:
            eid_t  = torch.tensor([env_id], device=self.device)
            origin = self._terrain.env_origins[env_id].cpu().numpy()
            state  = self._robot.data.default_root_state[eid_t].clone()

            spawn_choices = [
                (-7.5,  0.0),   # Room 1
                (0.0, 3.0),   # Room 2
                ( 0.0,  -3.0),   # center corridor
                (7.5,  0.0),   # upper corridor
            ]

            choice = spawn_choices[np.random.randint(len(spawn_choices))]
            state[0, 0] = origin[0] + choice[0]
            state[0, 1] = origin[1] + choice[1]

            # Spawn at the hardcoded position inside Room 1
            # state[0, 0] = origin[0] + SPAWN_LOCAL[0]
            # state[0, 1] = origin[1] + SPAWN_LOCAL[1]
            state[0, 2] = SPAWN_LOCAL[2]

            self._robot.write_root_pose_to_sim(state[:, :7], eid_t)
            self._robot.write_root_velocity_to_sim(state[:, 7:], eid_t)
            jp = self._robot.data.default_joint_pos[eid_t]
            jv = self._robot.data.default_joint_vel[eid_t]
            self._robot.write_joint_state_to_sim(jp, jv, None, eid_t)

    # ------------------------------------------------------------------
    # Grid PNG export
    # ------------------------------------------------------------------
    def _save_grid_png(self, grid: np.ndarray, path: str):
        import struct, zlib
        NR, NC = grid.shape
        rgb    = np.zeros((NR, NC, 3), dtype=np.uint8)
        rgb[grid == UNKNOWN]  = (180, 178, 169)
        rgb[grid == FREE]     = (125, 200, 122)
        rgb[grid == OCCUPIED] = ( 44,  44,  42)
        free, unk = grid == FREE, grid == UNKNOWN
        frontier  = free & (
            np.roll(unk,1,0)|np.roll(unk,-1,0)|
            np.roll(unk,1,1)|np.roll(unk,-1,1)
        )
        rgb[frontier] = (226, 75, 74)
        raw = b''.join(b'\x00' + rgb[r].tobytes() for r in range(NR))
        def chunk(n, d):
            c = n + d
            return struct.pack('>I',len(d))+c+struct.pack('>I',zlib.crc32(c)&0xffffffff)
        ihdr = struct.pack('>IIBBBBB', NC, NR, 8, 2, 0, 0, 0)
        with open(path, 'wb') as f:
            f.write(b'\x89PNG\r\n\x1a\n'
                    + chunk(b'IHDR', ihdr)
                    + chunk(b'IDAT', zlib.compress(raw))
                    + chunk(b'IEND', b''))

    # ------------------------------------------------------------------
    # Debug visualisation
    # ------------------------------------------------------------------
    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "_frontier_vis"):
                mc = CUBOID_MARKER_CFG.copy()
                mc.markers["cuboid"].size = (0.2, 0.2, 0.2)
                mc.prim_path = "/Visuals/Frontier/target"
                self._frontier_vis = VisualizationMarkers(mc)
            self._frontier_vis.set_visibility(True)
        else:
            if hasattr(self, "_frontier_vis"):
                self._frontier_vis.set_visibility(False)

    def _debug_vis_callback(self, event):
        if hasattr(self, "_frontier_vis"):
            self._frontier_vis.visualize(self._frontier_pos_w)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    w   = q[:, 0:1]
    xyz = q[:, 1:]
    t   = 2.0 * torch.linalg.cross(xyz, v)
    return v + w * t + torch.linalg.cross(xyz, t)