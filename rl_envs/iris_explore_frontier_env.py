"""
iris_explore_frontier_env.py
============================
Frontier-based exploration environment for the IRIS drone.
Designed for RSL-RL's OnPolicyRunner.

Key design decisions vs the original draft:
  - Flat observation vector {"policy": (N, obs_dim)} — RSL-RL requires this
  - Proper Isaac Lab scene setup with terrain, walls, lighting
  - Correct API: write_root_velocity_to_sim / write_root_pose_to_sim
  - 100 Hz sim / 50 Hz control (decimation=2)
  - Full reward system inherited from the best-performing maze env:
      new cells, coverage rate, frontier direction, motion toward openings,
      stagnation penalty, time penalty, milestone bonuses, success bonus
  - Randomised spawn from precomputed valid candidate grid
  - Depth strip (11 rows) for proximity + grid update — same as iris_maze_env
  - Altitude P-controller — same as iris_maze_env

Observation vector layout (obs_dim = 19):
    [0:3]   root_lin_vel_b       (forward, lateral, up)
    [3:6]   root_ang_vel_b       (roll, pitch, yaw rates)
    [6:9]   projected_gravity_b  (tilt sensing)
    [9:12]  frontier_b           (vector to nearest frontier, body frame)
    [12]    coverage             (fraction of maze explored)
    [13:15] depth_opening        (gap direction, gap distance)
    [15:17] depth_min_lr         (min depth left half, min depth right half)
    [17:19] prev_action          (last vx, yaw — helps with latency)

Register in __init__.py:
    gym.register(
        id="Isaac-Iris-Frontier-v0",
        entry_point="rl_WorkSpace.rl_envs.iris_explore_frontier_env:IrisExploreFrontierEnv",
        kwargs={
            "env_cfg_entry_point":
                "rl_WorkSpace.rl_envs.iris_explore_frontier_env:IrisExploreFrontierEnvCfg",
            "rsl_rl_cfg_entry_point":
                "rl_WorkSpace.agents.rsl_ppo_frontier_cfg:FrontierRunnerCfg",
        },
    )

Train:
    CUDA_VISIBLE_DEVICES=1 /isaac-sim/python.sh \\
        scripts/reinforcement_learning/rsl_rl/train.py \\
        --task Isaac-Iris-Frontier-v0 --num_envs 32 --headless --enable_cameras

Play:
    CUDA_VISIBLE_DEVICES=1 /isaac-sim/python.sh \\
        scripts/reinforcement_learning/rsl_rl/play.py \\
        --task Isaac-Iris-Frontier-v0 --num_envs 1 --livestream 2 --enable_cameras
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.markers import VisualizationMarkers, CUBOID_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCamera, TiledCameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

from rl_WorkSpace.models.drone.iris import IRIS_CFG


# =============================================================================
# MAZE DEFINITION  — edit WALL_DEFS only, nothing else needs to change
# =============================================================================
# Each wall: (centre_x, centre_y, size_x, size_y) metres, local to env origin.
# Maze footprint: X in [-6, +6],  Y in [-4, +4]  (12 m × 8 m)

WALL_DEFS: List[Tuple[float, float, float, float]] = [
    # ── Outer perimeter ──────────────────────────────────────────────────────
    ( 0.0,  4.0, 12.4,  0.4),   # north wall
    ( 0.0, -4.0, 12.4,  0.4),   # south wall
    (-6.0,  0.0,  0.4,  8.4),   # west wall
    ( 6.0,  0.0,  0.4,  8.4),   # east wall

    # ── Room 1 (west) ────────────────────────────────────────────────────────
    (-5.5,  0.0,  0.8,  0.4),   # short west stub of room divider
    (-1.8,  0.0,  3.5,  0.4),   # long east section of room divider
    (-1.5,  2.0,  0.4,  4.0),   # north vertical divider

    # ── Room 2 (east) ────────────────────────────────────────────────────────
    ( 2.5,  0.0,  1.5,  0.4),   # west section of east divider
    ( 5.5,  0.0,  1.0,  0.4),   # east stub of east divider
    ( 1.95,-2.0,  0.4,  4.0),   # south vertical divider
]

# Pre-build AABBs: [x_min, x_max, y_min, y_max] per wall — used for collision
WALL_AABBS: np.ndarray = np.array(
    [[cx - sx/2, cx + sx/2, cy - sy/2, cy + sy/2]
     for cx, cy, sx, sy in WALL_DEFS],
    dtype=np.float32,
)

# =============================================================================
# SPAWN CANDIDATES — precomputed at import, used every reset
# =============================================================================
_SPAWN_MARGIN  = 0.55   # metres clearance from walls
_SPAWN_Z       = 0.9    # hover height
_SPAWN_STEP    = 0.5    # grid spacing for candidate generation


def _build_spawn_candidates() -> np.ndarray:
    """
    Enumerate a grid over the maze interior, reject points inside any
    wall AABB (expanded by margin). Returns (K, 2) float32 (x, y).
    """
    m = _SPAWN_MARGIN
    xs = np.arange(-6.0 + 0.2 + m,  6.0 - 0.2 - m, _SPAWN_STEP, dtype=np.float32)
    ys = np.arange(-4.0 + 0.2 + m,  4.0 - 0.2 - m, _SPAWN_STEP, dtype=np.float32)
    gx, gy    = np.meshgrid(xs, ys)
    candidates = np.stack([gx.ravel(), gy.ravel()], axis=1)

    valid = np.ones(len(candidates), dtype=bool)
    for xmin, xmax, ymin, ymax in WALL_AABBS:
        in_wall = (
            (candidates[:, 0] > xmin - m) & (candidates[:, 0] < xmax + m) &
            (candidates[:, 1] > ymin - m) & (candidates[:, 1] < ymax + m)
        )
        valid &= ~in_wall

    result = candidates[valid]
    assert len(result) > 0, "No valid spawn candidates — check WALL_DEFS"
    return result


SPAWN_CANDIDATES: np.ndarray = _build_spawn_candidates()

# Grid constants
UNKNOWN  = 0
FREE     = 1
OCCUPIED = 2


# =============================================================================
# CONFIG
# =============================================================================

@configclass
class IrisExploreFrontierEnvCfg(DirectRLEnvCfg):

    # ── Episode ──────────────────────────────────────────────────────────────
    episode_length_s = 90.0
    decimation       = 2        # 100 Hz sim / 2 = 50 Hz control

    # ── Spaces — RSL-RL reads these directly ─────────────────────────────────
    # obs layout: lin_vel(3) + ang_vel(3) + gravity(3) + frontier_b(3)
    #           + coverage(1) + depth_opening(2) + depth_min_lr(2) + prev_action(2) = 19
    action_space      = 2       # [vx, yaw_rate]
    observation_space = 19
    state_space       = 0
    debug_vis         = False

    # ── Simulation ───────────────────────────────────────────────────────────
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
        env_spacing=20.0,   # > maze width (12 m) + buffer
        replicate_physics=True,
    )

    # ── Robot ────────────────────────────────────────────────────────────────
    robot: ArticulationCfg = IRIS_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )

    # ── Camera ───────────────────────────────────────────────────────────────
    cam_width:      int   = 320
    cam_height:     int   = 240
    cam_fov_deg:    float = 90.0
    cam_min_depth:  float = 0.15
    cam_max_depth:  float = 5.0
    cam_slice_half: int   = 5     # rows above+below centre → 11 rows total

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
            clipping_range=(0.15, 8.0),
        ),
        width=320,
        height=240,
    )

    # ── Maze ─────────────────────────────────────────────────────────────────
    wall_height: float = 2.0

    # ── Occupancy grid — 12×8 m at 0.1 m/cell ────────────────────────────────
    cell_size:  float = 0.1
    grid_cols:  int   = 120   # 12 m
    grid_rows:  int   = 80    #  8 m

    frontier_recompute_interval: int = 10

    # ── Motion ───────────────────────────────────────────────────────────────
    max_forward_vel:  float = 1.0
    max_yaw_rate:     float = 1.5
    hover_height:     float = 0.9
    altitude_kp:      float = 2.0
    max_altitude_vel: float = 1.0

    # ── Collision ────────────────────────────────────────────────────────────
    collision_radius: float = 0.25

    # ── Rewards ──────────────────────────────────────────────────────────────
    new_cell_reward_scale: float =  10.0
    cov_rate_scale:        float =   6.0
    motion_reward_scale:   float =   3.0
    stagnation_scale:      float =   5.0
    ang_vel_penalty_scale: float =  -0.005
    time_penalty_per_step: float =  -0.15
    collision_penalty:     float =  -5.0
    out_of_bounds_penalty: float =  -5.0

    # Milestone bonuses (one-time per episode)
    milestone_25_bonus:  float =  15.0
    milestone_50_bonus:  float =  30.0
    milestone_75_bonus:  float =  60.0

    # Success
    success_coverage_threshold: float = 0.85
    success_bonus:              float = 100.0

    # ── Logging ──────────────────────────────────────────────────────────────
    grid_save_path:  str = "/tmp/iris_frontier_grid.npy"
    grid_save_every: int = 300


# =============================================================================
# ENVIRONMENT
# =============================================================================

class IrisExploreFrontierEnv(DirectRLEnv):
    """
    Frontier-based exploration with RSL-RL compatible flat observations.

    What 'frontier' means here:
        A frontier cell is a FREE cell adjacent to at least one UNKNOWN cell.
        It marks the boundary of explored space. The nearest frontier to the
        drone is computed every N steps and fed into the observation as a
        body-frame direction vector. The policy uses this to navigate toward
        unknown territory — unlike ICM (which rewards surprise) or egocentric
        maps (which let the policy infer frontiers itself), this explicitly
        tells the policy where to go next.
    """

    cfg: IrisExploreFrontierEnvCfg

    # ─────────────────────────────────────────────────────────────────────────
    # Init
    # ─────────────────────────────────────────────────────────────────────────
    def __init__(self, cfg: IrisExploreFrontierEnvCfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        N  = self.num_envs
        NR = self.cfg.grid_rows
        NC = self.cfg.grid_cols

        # Occupancy grid: 0=unknown, 1=free, 2=occupied
        self._grid            = np.zeros((N, NR, NC), dtype=np.uint8)
        self._prev_free_count = np.zeros(N, dtype=np.int32)

        # Frontier direction (world frame, updated every N steps)
        self._frontier_pos_w  = torch.zeros(N, 3, device=self.device)

        # Action buffer — needed by observation on first step
        self._actions = torch.zeros(N, 2, device=self.device)

        # Stagnation: ring buffer of positions over last 2 seconds
        # At 50 Hz, 2 s = 100 steps
        N_HISTORY = 100
        self._pos_history       = torch.zeros(N, N_HISTORY, 2, device=self.device)
        self._history_idx       = 0
        self._pos_history_filled = False

        # Coverage rate tracking
        self._coverage_check_interval = 50
        self._coverage_at_last_check  = np.zeros(N, dtype=np.float32)

        # Milestone tracking: columns = [25%, 50%, 75%]
        self._milestone_claimed = np.zeros((N, 3), dtype=np.uint8)

        # Success flag (set by _get_dones, read by _get_rewards)
        self._succeeded = torch.zeros(N, dtype=torch.bool, device=self.device)

        # Episode sums for logging
        self._episode_sums = {
            key: torch.zeros(N, dtype=torch.float, device=self.device)
            for key in [
                "new_cells", "cov_rate", "motion", "stagnation",
                "ang_vel", "time", "collision", "out_of_bounds",
                "milestone", "success",
            ]
        }

        self._step_counter = 0
        self._precompute_ray_angles()
        self.set_debug_vis(self.cfg.debug_vis)

    # ─────────────────────────────────────────────────────────────────────────
    # Scene setup
    # ─────────────────────────────────────────────────────────────────────────
    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self._camera = TiledCamera(self.cfg.camera)
        self.scene.sensors["depth_cam"] = self._camera

        self.cfg.terrain.num_envs    = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # Spawn walls in env_0 then clone to all envs
        self._spawn_walls_env0()
        self.scene.clone_environments(copy_from_source=False)

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _spawn_walls_env0(self):
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
            wall_cfg.func(
                f"/World/envs/env_0/Maze/Wall_{idx:03d}",
                wall_cfg,
                translation=(cx, cy, self.cfg.wall_height / 2.0),
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Camera intrinsics — precompute ray angles once
    # ─────────────────────────────────────────────────────────────────────────
    def _precompute_ray_angles(self):
        W    = self.cfg.cam_width
        H    = self.cfg.cam_height
        hfov = np.deg2rad(self.cfg.cam_fov_deg)
        fx   = (W / 2.0) / np.tan(hfov / 2.0)
        cols = np.arange(W, dtype=np.float32)
        self._ray_offsets = np.arctan2(cols - W / 2.0, fx)   # (W,)
        cy   = H // 2
        sh   = self.cfg.cam_slice_half
        # +1 to include centre row: slice(115, 126) = 11 rows
        self._slice_rows = slice(max(0, cy - sh), min(H, cy + sh + 1))

    # ─────────────────────────────────────────────────────────────────────────
    # Depth processing
    # ─────────────────────────────────────────────────────────────────────────
    def _get_depth_strip(self) -> torch.Tensor:
        """
        Returns (N, strip_h, W) normalised depth strip, values in [0, 1].
        0 = min_depth (closest), 1 = max_depth or invalid (far/unknown).
        """
        depth_data = self._camera.data.output.get("distance_to_image_plane")
        N      = self.num_envs
        strip_h = 2 * self.cfg.cam_slice_half + 1

        if depth_data is None:
            return torch.ones(N, strip_h, self.cfg.cam_width, device=self.device)

        d = depth_data.float() if isinstance(depth_data, torch.Tensor) else \
            torch.tensor(depth_data, dtype=torch.float32, device=self.device)

        # Normalise shape to (N, H, W)
        if d.ndim == 2:
            d = d.unsqueeze(0)
        elif d.ndim == 4:
            d = d.squeeze(-1)

        strip = d[:, self._slice_rows, :]  # (N, strip_h, W)
        strip = strip.clamp(self.cfg.cam_min_depth, self.cfg.cam_max_depth)
        strip = (strip - self.cfg.cam_min_depth) / (
            self.cfg.cam_max_depth - self.cfg.cam_min_depth
        )
        return torch.nan_to_num(strip, nan=1.0, posinf=1.0, neginf=0.0)

    def _get_depth_opening(self) -> torch.Tensor:
        """
        Returns (N, 2): [gap_direction_sin, normalised_gap_distance].
        gap_direction: sin of the angle toward the widest opening (0=straight ahead).
        gap_distance:  normalised depth at that opening [0,1].
        This gives the policy a cheap 'best escape direction' signal.
        """
        strip = self._get_depth_strip()    # (N, strip_h, W)
        scan  = strip.mean(dim=1)          # (N, W) — average across strip height

        # Find column with maximum depth (widest opening)
        best_col = scan.argmax(dim=1)      # (N,)
        best_val = scan.max(dim=1).values  # (N,)

        W = self.cfg.cam_width
        # Normalise column to [-1, 1]: -1=left, 0=centre, +1=right
        col_norm = (best_col.float() / (W - 1)) * 2.0 - 1.0   # (N,)

        return torch.stack([col_norm, best_val], dim=1)   # (N, 2)

    def _get_depth_min_lr(self) -> torch.Tensor:
        """
        Returns (N, 2): [min_depth_left_half, min_depth_right_half] normalised.
        Gives the policy explicit left/right proximity awareness beyond what
        the frontier direction provides — useful for corridor navigation.
        """
        strip = self._get_depth_strip()
        scan  = strip.mean(dim=1)   # (N, W)
        W     = self.cfg.cam_width
        left  = scan[:, :W//2].min(dim=1).values   # (N,)
        right = scan[:, W//2:].min(dim=1).values   # (N,)
        return torch.stack([left, right], dim=1)   # (N, 2)

    # ─────────────────────────────────────────────────────────────────────────
    # Grid update from depth
    # ─────────────────────────────────────────────────────────────────────────
    def _update_grid_from_depth(self):
        """
        Project depth strip rays into the global occupancy grid.
        Mid-ray cells → FREE, endpoint cells → OCCUPIED.
        Uses only the horizontal scan (average of strip height).
        """
        strip  = self._get_depth_strip()
        depth_m = (
            strip.cpu().numpy() *
            (self.cfg.cam_max_depth - self.cfg.cam_min_depth) +
            self.cfg.cam_min_depth
        )                           # (N, strip_h, W) in metres
        scan = np.nanmean(depth_m, axis=1)   # (N, W)

        cs   = self.cfg.cell_size
        NR   = self.cfg.grid_rows
        NC   = self.cfg.grid_cols
        hx   = (NC * cs) / 2.0
        hy   = (NR * cs) / 2.0

        pos_np  = self._robot.data.root_pos_w.cpu().numpy()
        origins = self._terrain.env_origins.cpu().numpy()
        quat    = self._robot.data.root_state_w[:, 3:7].cpu().numpy()
        yaw     = np.arctan2(
            2.0*(quat[:,0]*quat[:,3] + quat[:,1]*quat[:,2]),
            1.0 - 2.0*(quat[:,2]**2 + quat[:,3]**2)
        )

        for i in range(self.num_envs):
            ox  = pos_np[i, 0] - origins[i, 0]
            oy  = pos_np[i, 1] - origins[i, 1]
            ray_angles = yaw[i] + self._ray_offsets
            depths     = scan[i]
            valid = (
                (depths > self.cfg.cam_min_depth) &
                (depths < self.cfg.cam_max_depth) &
                np.isfinite(depths)
            )
            if not valid.any():
                continue

            ca = np.cos(ray_angles[valid])
            sa = np.sin(ray_angles[valid])
            dv = depths[valid]

            # Midpoints → FREE (if not already OCCUPIED)
            fx = ox + dv * 0.5 * ca
            fy = oy + dv * 0.5 * sa
            fc = np.clip(np.floor((fx + hx) / cs).astype(int), 0, NC-1)
            fr = np.clip(np.floor((hy - fy) / cs).astype(int), 0, NR-1)
            nm = self._grid[i, fr, fc] != OCCUPIED
            self._grid[i, fr[nm], fc[nm]] = FREE

            # Endpoints → OCCUPIED
            ex = ox + dv * ca
            ey = oy + dv * sa
            ec = np.clip(np.floor((ex + hx) / cs).astype(int), 0, NC-1)
            er = np.clip(np.floor((hy - ey) / cs).astype(int), 0, NR-1)
            self._grid[i, er, ec] = OCCUPIED

            # Drone cell → FREE
            dc = np.clip(int((ox + hx) / cs), 0, NC-1)
            dr = np.clip(int((hy - oy) / cs), 0, NR-1)
            self._grid[i, dr, dc] = FREE

    # ─────────────────────────────────────────────────────────────────────────
    # Frontier computation
    # ─────────────────────────────────────────────────────────────────────────
    def _compute_frontiers(self) -> np.ndarray:
        """
        For each env, find the nearest frontier cell (FREE adjacent to UNKNOWN)
        and return its world-frame (x, y) position.
        Falls back to 2m ahead in current heading if no frontier exists yet.
        Returns (N, 2) float32.
        """
        cs     = self.cfg.cell_size
        NR     = self.cfg.grid_rows
        NC     = self.cfg.grid_cols
        hx     = (NC * cs) / 2.0
        hy     = (NR * cs) / 2.0
        pos_np = self._robot.data.root_pos_w.cpu().numpy()
        origins = self._terrain.env_origins.cpu().numpy()
        result = np.zeros((self.num_envs, 2), dtype=np.float32)

        for i in range(self.num_envs):
            ox = pos_np[i, 0] - origins[i, 0]
            oy = pos_np[i, 1] - origins[i, 1]

            # Drone cell
            dc = int(np.clip((ox + hx) / cs, 0, NC-1))
            dr = int(np.clip((hy - oy) / cs, 0, NR-1))

            g    = self._grid[i]
            free = g == FREE
            unk  = g == UNKNOWN

            # Frontier: free cell with at least one unknown neighbour
            has_unk = (
                np.roll(unk,  1, axis=0) | np.roll(unk, -1, axis=0) |
                np.roll(unk,  1, axis=1) | np.roll(unk, -1, axis=1)
            )
            has_unk[0,:] = has_unk[-1,:] = has_unk[:,0] = has_unk[:,-1] = False
            fronts = np.argwhere(free & has_unk)

            if len(fronts) == 0:
                # No frontier — point 2m ahead
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

            # Nearest frontier by Manhattan distance
            dists = np.abs(fronts[:, 0] - dr) + np.abs(fronts[:, 1] - dc)
            best  = fronts[np.argmin(dists)]

            # Convert grid (row, col) → world (x, y)
            world_x = (best[1] + 0.5) * cs - hx + origins[i, 0]
            world_y = hy - (best[0] + 0.5) * cs + origins[i, 1]
            result[i] = [world_x, world_y]

        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Stagnation penalty
    # ─────────────────────────────────────────────────────────────────────────
    def _get_stagnation_penalty(self) -> torch.Tensor:
        """
        Returns -1.0 for envs that have not moved >0.5m in the last 2 seconds,
        else 0.0. Penalises spinning in place and hovering.
        """
        cur = self._robot.data.root_pos_w[:, :2]
        self._pos_history[:, self._history_idx, :] = cur.detach()
        self._history_idx = (self._history_idx + 1) % self._pos_history.shape[1]

        if not self._pos_history_filled:
            if self._history_idx == 0:
                self._pos_history_filled = True
            return torch.zeros(self.num_envs, device=self.device)

        oldest = self._pos_history[:, self._history_idx, :]
        disp   = torch.linalg.norm(cur - oldest, dim=1)
        return (disp < 0.5).float() * -1.0

    # ─────────────────────────────────────────────────────────────────────────
    # Collision and bounds helpers
    # ─────────────────────────────────────────────────────────────────────────
    def _get_collisions(self) -> np.ndarray:
        pos_np  = self._robot.data.root_pos_w.cpu().numpy()
        origins = self._terrain.env_origins.cpu().numpy()
        col     = np.zeros(self.num_envs, dtype=bool)
        for i in range(self.num_envs):
            lx = pos_np[i, 0] - origins[i, 0]
            ly = pos_np[i, 1] - origins[i, 1]
            dx = np.maximum(0.0, np.maximum(WALL_AABBS[:,0] - lx, lx - WALL_AABBS[:,1]))
            dy = np.maximum(0.0, np.maximum(WALL_AABBS[:,2] - ly, ly - WALL_AABBS[:,3]))
            col[i] = np.sqrt(dx**2 + dy**2).min() < self.cfg.collision_radius
        return col

    def _is_oob(self) -> np.ndarray:
        hx      = (self.cfg.grid_cols * self.cfg.cell_size) / 2.0
        hy      = (self.cfg.grid_rows * self.cfg.cell_size) / 2.0
        pos_np  = self._robot.data.root_pos_w.cpu().numpy()
        origins = self._terrain.env_origins[:, :2].cpu().numpy()
        local   = pos_np[:, :2] - origins
        return (np.abs(local[:,0]) > hx) | (np.abs(local[:,1]) > hy)

    # ─────────────────────────────────────────────────────────────────────────
    # Action pipeline
    # ─────────────────────────────────────────────────────────────────────────
    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone().clamp(-1.0, 1.0)

        lin_vel_b = torch.zeros(self.num_envs, 3, device=self.device)
        lin_vel_b[:, 0] = self._actions[:, 0] * self.cfg.max_forward_vel

        # Altitude P-controller
        target_z  = self.cfg.hover_height + self._terrain.env_origins[:, 2]
        current_z = self._robot.data.root_pos_w[:, 2]
        vz        = (self.cfg.altitude_kp * (target_z - current_z)).clamp(
            -self.cfg.max_altitude_vel, self.cfg.max_altitude_vel
        )

        # Body → world frame rotation
        quat_w    = self._robot.data.root_state_w[:, 3:7]
        lin_vel_w = _quat_rotate(quat_w, lin_vel_b)
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

    # ─────────────────────────────────────────────────────────────────────────
    # Observations — flat vector for RSL-RL
    # ─────────────────────────────────────────────────────────────────────────
    def _get_observations(self) -> dict:
        """
        Returns {"policy": (N, 19)} flat tensor.

        RSL-RL's OnPolicyRunner expects exactly this format.
        The frontier direction and depth opening give the policy
        both long-range (where to go) and short-range (avoid walls) signals.
        """
        self._update_grid_from_depth()

        # Recompute frontier every N steps (expensive — don't do every step)
        if self._step_counter % self.cfg.frontier_recompute_interval == 0:
            fxy = self._compute_frontiers()   # (N, 2) world frame
            fp  = np.zeros((self.num_envs, 3), dtype=np.float32)
            fp[:, :2] = fxy
            fp[:, 2]  = (self.cfg.hover_height +
                         self._terrain.env_origins[:, 2].cpu().numpy())
            self._frontier_pos_w = torch.tensor(fp, dtype=torch.float32,
                                                 device=self.device)

        self._step_counter += 1

        # Periodic grid save
        if self._step_counter % self.cfg.grid_save_every == 0:
            np.save(self.cfg.grid_save_path, self._grid[0])
            self._save_grid_png(
                self._grid[0],
                self.cfg.grid_save_path.replace(".npy", ".png"),
            )

        # Frontier in body frame
        frontier_b, _ = subtract_frame_transforms(
            self._robot.data.root_state_w[:, :3],
            self._robot.data.root_state_w[:, 3:7],
            self._frontier_pos_w,
        )

        # Coverage fraction
        total_cells = self.cfg.grid_rows * self.cfg.grid_cols
        cov = torch.tensor(
            (self._grid == FREE).sum(axis=(1,2)).astype(np.float32) / total_cells,
            device=self.device,
        ).unsqueeze(1)   # (N, 1)

        depth_opening = self._get_depth_opening()   # (N, 2)
        depth_min_lr  = self._get_depth_min_lr()    # (N, 2)

        obs = torch.cat([
            self._robot.data.root_lin_vel_b,         # 3
            self._robot.data.root_ang_vel_b,          # 3
            self._robot.data.projected_gravity_b,    # 3
            frontier_b,                              # 3
            cov,                                     # 1
            depth_opening,                           # 2
            depth_min_lr,                            # 2
            self._actions,                           # 2  (prev action)
        ], dim=-1)   # total: 19

        return {"policy": obs}

    # ─────────────────────────────────────────────────────────────────────────
    # Rewards
    # ─────────────────────────────────────────────────────────────────────────
    def _get_rewards(self) -> torch.Tensor:
        """
        Reward components:
            new_cells    — cells discovered this step (primary driver)
            cov_rate     — coverage rate over last N steps (secondary driver)
            motion       — moving toward visible openings (behaviour shaping)
            stagnation   — penalty for not moving (anti-hovering)
            ang_vel      — penalty for spinning (smooth flight)
            time         — per-step penalty (forces efficiency)
            collision    — wall contact penalty
            out_of_bounds— leaving grid penalty
            milestone    — one-time bonuses at 25/50/75% coverage
            success      — one-time large bonus at 85% coverage
        """
        # ── New cells ────────────────────────────────────────────────────────
        free_c    = (self._grid == FREE).sum(axis=(1,2)).astype(np.int32)
        new_cells = np.maximum(0, free_c - self._prev_free_count)
        self._prev_free_count = free_c.copy()
        nc_t = torch.tensor(new_cells, dtype=torch.float32, device=self.device)

        # ── Coverage rate ────────────────────────────────────────────────────
        cov_rate = torch.zeros(self.num_envs, device=self.device)
        if self._step_counter % self._coverage_check_interval == 0:
            cur_cov = (self._grid == FREE).mean(axis=(1,2)).astype(np.float32)
            delta   = np.maximum(0.0, cur_cov - self._coverage_at_last_check)
            self._coverage_at_last_check = cur_cov.copy()
            cov_rate = torch.tensor(delta * 100.0, dtype=torch.float32,
                                    device=self.device)

        # ── Motion toward opening ────────────────────────────────────────────
        depth_opening  = self._get_depth_opening()
        gap_dir        = depth_opening[:, 0]          # sin of best gap angle
        gap_dist       = depth_opening[:, 1]          # normalised depth
        facing_opening = (1.0 - gap_dir.abs()) * gap_dist
        forward_vel    = self._robot.data.root_lin_vel_b[:, 0].clamp(min=0.0)
        motion_r       = facing_opening * forward_vel

        # ── Stagnation ───────────────────────────────────────────────────────
        stagnation = self._get_stagnation_penalty()

        # ── Time penalty ─────────────────────────────────────────────────────
        time_r = torch.full(
            (self.num_envs,), self.cfg.time_penalty_per_step, device=self.device
        )

        # ── Collision + OOB ──────────────────────────────────────────────────
        collided = torch.tensor(self._get_collisions().astype(np.float32),
                                dtype=torch.float32, device=self.device)
        oob      = torch.tensor(self._is_oob().astype(np.float32),
                                dtype=torch.float32, device=self.device)

        # ── Angular velocity penalty ─────────────────────────────────────────
        ang_vel_sq = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)

        # ── Milestone bonuses ────────────────────────────────────────────────
        total_cells   = self.cfg.grid_rows * self.cfg.grid_cols
        coverage_frac = (self._grid == FREE).sum(axis=(1,2)) / total_cells
        milestone_np  = np.zeros(self.num_envs, dtype=np.float32)
        for col, (thresh, bonus) in enumerate(zip(
            [0.25, 0.50, 0.75],
            [self.cfg.milestone_25_bonus,
             self.cfg.milestone_50_bonus,
             self.cfg.milestone_75_bonus],
        )):
            just_reached = (
                (coverage_frac >= thresh) &
                (self._milestone_claimed[:, col] == 0)
            )
            milestone_np[just_reached] += bonus
            self._milestone_claimed[just_reached, col] = 1
        milestone_t = torch.tensor(milestone_np, dtype=torch.float32,
                                   device=self.device)

        # ── Success bonus ────────────────────────────────────────────────────
        success_r = self._succeeded.float() * self.cfg.success_bonus

        # ── Compose ──────────────────────────────────────────────────────────
        rewards = {
            "new_cells":     nc_t       * self.cfg.new_cell_reward_scale,
            "cov_rate":      cov_rate   * self.cfg.cov_rate_scale,
            "motion":        motion_r   * self.cfg.motion_reward_scale,
            "stagnation":    stagnation * self.cfg.stagnation_scale,
            "ang_vel":       ang_vel_sq * self.cfg.ang_vel_penalty_scale * self.step_dt,
            "time":          time_r,
            "collision":     collided   * self.cfg.collision_penalty,
            "out_of_bounds": oob        * self.cfg.out_of_bounds_penalty,
            "milestone":     milestone_t,
            "success":       success_r,
        }
        total = torch.stack(list(rewards.values())).sum(dim=0)

        for k, v in rewards.items():
            self._episode_sums[k] += v

        # ── Logging ──────────────────────────────────────────────────────────
        self.extras["log"] = {
            "coverage_pct":   float((self._grid == FREE).mean() * 100.0),
            "new_cells_mean": float(nc_t.mean().item()),
            "collision_rate": float(collided.mean().item()),
            "stagnation_rate":float((stagnation < 0).float().mean().item()),
            "success_rate":   float(self._succeeded.float().mean().item()),
        }

        if self._step_counter % 100 == 0:
            self._print_reward_breakdown(rewards)

        return total

    def _print_reward_breakdown(self, rewards: dict):
        print(f"\n{'='*55}")
        print(f"[REWARD]  step={self._step_counter}  "
              f"cov={float((self._grid==FREE).mean()*100):.1f}%")
        total = 0.0
        for name, t in rewards.items():
            v = float(t.mean().item())
            total += v
            bar = ("+" if v >= 0 else "─") * min(int(abs(v) * 10), 30)
            print(f"  {name:<16} {v:+7.4f}  {bar}")
        print(f"  {'TOTAL':<16} {total:+7.4f}")
        print(f"{'='*55}")

    # ─────────────────────────────────────────────────────────────────────────
    # Termination
    # ─────────────────────────────────────────────────────────────────────────
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        alt_fail = (
            (self._robot.data.root_pos_w[:, 2] < 0.1) |
            (self._robot.data.root_pos_w[:, 2] > 3.0)
        )
        oob      = torch.tensor(self._is_oob(),         device=self.device)
        collided = torch.tensor(self._get_collisions(), device=self.device)

        total_cells    = self.cfg.grid_rows * self.cfg.grid_cols
        coverage_ratio = (self._grid == FREE).sum(axis=(1,2)) / total_cells
        self._succeeded = torch.tensor(
            coverage_ratio >= self.cfg.success_coverage_threshold,
            dtype=torch.bool, device=self.device,
        )

        died = alt_fail | oob | collided
        return died | self._succeeded, time_out

    # ─────────────────────────────────────────────────────────────────────────
    # Reset
    # ─────────────────────────────────────────────────────────────────────────
    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if len(env_ids) == self.num_envs:
            self.episode_length_buf = torch.randint_like(
                self.episode_length_buf, high=int(self.max_episode_length)
            )

        ids_np = env_ids.cpu().numpy()
        self._grid[ids_np]               = UNKNOWN
        self._prev_free_count[ids_np]    = 0
        self._frontier_pos_w[env_ids]    = 0.0
        self._milestone_claimed[ids_np]  = 0
        self._pos_history[env_ids]       = 0.0
        self._succeeded[env_ids]         = False
        self._coverage_at_last_check[ids_np] = 0.0

        # Random spawn from precomputed valid candidates
        chosen_idx = np.random.choice(len(SPAWN_CANDIDATES),
                                      size=len(ids_np), replace=True)
        chosen_xy  = SPAWN_CANDIDATES[chosen_idx]
        random_yaws = np.random.uniform(-np.pi, np.pi, size=len(ids_np))

        for i, eid in enumerate(ids_np):
            t    = torch.tensor([eid], device=self.device)
            orig = self._terrain.env_origins[eid].cpu().numpy()
            state = self._robot.data.default_root_state[t].clone()

            state[0, 0] = float(orig[0] + chosen_xy[i, 0])
            state[0, 1] = float(orig[1] + chosen_xy[i, 1])
            state[0, 2] = float(_SPAWN_Z)

            yh = float(random_yaws[i]) * 0.5
            state[0, 3] = float(np.cos(yh))   # w
            state[0, 4] = 0.0                  # x
            state[0, 5] = 0.0                  # y
            state[0, 6] = float(np.sin(yh))    # z

            self._robot.write_root_pose_to_sim(state[:, :7], t)
            self._robot.write_root_velocity_to_sim(state[:, 7:], t)
            jp = self._robot.data.default_joint_pos[t]
            jv = self._robot.data.default_joint_vel[t]
            self._robot.write_joint_state_to_sim(jp, jv, None, t)

    # ─────────────────────────────────────────────────────────────────────────
    # Grid PNG export
    # ─────────────────────────────────────────────────────────────────────────
    def _save_grid_png(self, grid: np.ndarray, path: str):
        import struct, zlib
        NR, NC = grid.shape
        rgb    = np.zeros((NR, NC, 3), dtype=np.uint8)
        rgb[grid == UNKNOWN]  = (180, 178, 169)
        rgb[grid == FREE]     = (125, 200, 122)
        rgb[grid == OCCUPIED] = ( 44,  44,  42)
        raw = b''.join(b'\x00' + rgb[r].tobytes() for r in range(NR))
        def chunk(n, d):
            c = n + d
            return (struct.pack('>I', len(d)) + c +
                    struct.pack('>I', zlib.crc32(c) & 0xffffffff))
        ihdr = struct.pack('>IIBBBBB', NC, NR, 8, 2, 0, 0, 0)
        with open(path, 'wb') as f:
            f.write(b'\x89PNG\r\n\x1a\n'
                    + chunk(b'IHDR', ihdr)
                    + chunk(b'IDAT', zlib.compress(raw))
                    + chunk(b'IEND', b''))

    # ─────────────────────────────────────────────────────────────────────────
    # Debug visualisation
    # ─────────────────────────────────────────────────────────────────────────
    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "_drone_vis"):
                mc = CUBOID_MARKER_CFG.copy()
                mc.markers["cuboid"].size = (0.2, 0.2, 0.2)
                mc.prim_path = "/Visuals/DroneMarker"
                self._drone_vis = VisualizationMarkers(mc)
            self._drone_vis.set_visibility(True)
        elif hasattr(self, "_drone_vis"):
            self._drone_vis.set_visibility(False)

    def _debug_vis_callback(self, event):
        if hasattr(self, "_drone_vis"):
            self._drone_vis.visualize(self._robot.data.root_pos_w)


# =============================================================================
# UTILITY
# =============================================================================

def _quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors v by quaternions q. Isaac Lab convention: (w, x, y, z)."""
    w   = q[:, 0:1]
    xyz = q[:, 1:]
    t   = 2.0 * torch.linalg.cross(xyz, v)
    return v + w * t + torch.linalg.cross(xyz, t)