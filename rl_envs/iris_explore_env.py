"""
iris_explore_env.py  –  Stage 2: Exploration with occupancy grid

What is new compared to iris_target_env.py:
  - No fixed target.  The drone is rewarded for uncovering new cells.
  - An occupancy grid (2D numpy array) is maintained per environment.
  - Frontier cells (free cells adjacent to unknown cells) are detected
    and the closest one is used as a soft navigation hint in the reward.
  - The grid is saved to disk periodically so viewer.py can display it.

Action space  (unchanged):  [v_forward, yaw_rate]
Observation space  (13):
    root_lin_vel_b      (3)
    root_ang_vel_b      (3)
    projected_gravity_b (3)
    closest_frontier_b  (3)  ← replaces "desired_pos_b"
    coverage_ratio      (1)  ← fraction of grid that is explored so far
"""

from __future__ import annotations

import os
import gymnasium as gym
import numpy as np 
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

from rl_WorkSpace.models.drone import IRIS_CFG
from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@configclass
class IrisExploreEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 30.0       # longer episodes so the drone has time to explore
    decimation      = 2

    # 2 actions: [v_forward, yaw_rate]
    action_space = 2

    # 13 observations: lin_vel(3) + ang_vel(3) + gravity(3) + frontier_dir(3) + coverage(1)
    observation_space = 13
    state_space = 0
    debug_vis = True

    # simulation
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
        num_envs=512, env_spacing=12.0, replicate_physics=True
    )
    # Note: env_spacing is larger because the drone will wander further

    robot: ArticulationCfg = IRIS_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # --- velocity limits (same as iris_target_env) ---
    max_forward_vel  = 1.5   # m/s
    max_yaw_rate     = 1.5   # rad/s

    # --- altitude hold ---
    hover_height     = 1.0   # m above terrain
    altitude_kp      = 2.0
    max_altitude_vel = 1.0   # m/s

    # --- occupancy grid ---
    # The grid covers a square centred on each environment's origin.
    # cell_size: how many metres each grid cell represents.
    # grid_size: number of cells per side.  Total area = (grid_size * cell_size)^2
    cell_size  = 0.25   # metres per cell  →  25 cm resolution
    grid_size  = 80     # cells per side   →  20 m × 20 m arena per env

    # How many steps between full frontier re-computation (cheap but not free)
    frontier_recompute_interval = 10

    # --- reward scales ---
    new_cell_reward_scale     = 5.0    # reward per newly discovered cell
    frontier_reward_scale     = 1.0    # reward for moving toward nearest frontier
    ang_vel_penalty_scale     = -0.01  # discourage spinning without moving
    revisit_penalty_scale     = -0.02  # small penalty for staying in already-seen cells
    out_of_bounds_penalty     = -5.0   # large penalty for leaving the grid

    # --- logging ---
    # Grid of env index 0 is saved here so viewer.py can display it.
    # Make sure this path is accessible from inside your Docker container.
    #grid_save_path  = "/tmp/iris_explore_grid.npy"
    grid_save_path  = "/workspace/storageMountDoc/iris_explore_grid.npy"
    grid_save_every = 200   # steps


# ---------------------------------------------------------------------------
# Grid cell states
# ---------------------------------------------------------------------------
UNKNOWN  = 0
FREE     = 1
OCCUPIED = 2   # reserved for Stage 3 when walls are present


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class IrisExploreEnv(DirectRLEnv):
    cfg: IrisExploreEnvCfg

    def __init__(self, cfg: IrisExploreEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # --- occupancy grids: one per env, stored on CPU as uint8 ---
        # Shape: (num_envs, grid_size, grid_size)
        G = self.cfg.grid_size
        self._grids = np.zeros((self.num_envs, G, G), dtype=np.uint8)

        # Track which cells were free last step to compute "new cells" reward
        self._prev_free_count = np.zeros(self.num_envs, dtype=np.int32)

        # Closest frontier position (world frame) per env, as a torch tensor
        self._frontier_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

        # Actions buffer
        self._actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )

        # Episode sums for logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in ["new_cells", "frontier", "ang_vel", "revisit", "out_of_bounds"]
        }

        # Step counter for periodic frontier recompute
        self._step_counter = 0

        self.set_debug_vis(self.cfg.debug_vis)

    # ------------------------------------------------------------------
    # Scene
    # ------------------------------------------------------------------

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self.cfg.terrain.num_envs  = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ------------------------------------------------------------------
    # Action pipeline  (identical to iris_target_env)
    # ------------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone().clamp(-1.0, 1.0)

        # Forward velocity in body frame
        lin_vel_b = torch.zeros(self.num_envs, 3, device=self.device)
        lin_vel_b[:, 0] = self._actions[:, 0] * self.cfg.max_forward_vel

        # Altitude hold P-controller
        target_z  = self.cfg.hover_height + self._terrain.env_origins[:, 2]
        current_z = self._robot.data.root_pos_w[:, 2]
        vz_world  = (self.cfg.altitude_kp * (target_z - current_z)).clamp(
            -self.cfg.max_altitude_vel, self.cfg.max_altitude_vel
        )

        # Rotate body → world
        quat_w    = self._robot.data.root_state_w[:, 3:7]
        lin_vel_w = quat_rotate(quat_w, lin_vel_b)
        lin_vel_w[:, 2] = vz_world

        # Yaw rate
        ang_vel_w = torch.zeros(self.num_envs, 3, device=self.device)
        ang_vel_w[:, 2] = self._actions[:, 1] * self.cfg.max_yaw_rate

        self._cmd_lin_vel_w = lin_vel_w
        self._cmd_ang_vel_w = ang_vel_w

    def _apply_action(self):
        velocity_cmd = torch.cat([self._cmd_lin_vel_w, self._cmd_ang_vel_w], dim=-1)
        self._robot.write_root_velocity_to_sim(velocity_cmd)

        # Spin props visually (cosmetic only)
        joint_vel = torch.zeros_like(self._robot.data.joint_vel)
        joint_vel[:, 0] =  200.0
        joint_vel[:, 1] = -200.0
        joint_vel[:, 2] =  200.0
        joint_vel[:, 3] = -200.0
        self._robot.set_joint_velocity_target(joint_vel)

    # ------------------------------------------------------------------
    # Grid helpers
    # ------------------------------------------------------------------

    def _world_to_grid(self, pos_w: torch.Tensor, env_ids=None) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert world-frame XY positions to integer grid indices.

        The grid is centred on each environment's terrain origin.
        Returns (row, col) arrays of shape (N,), clipped to grid bounds.
        Both row and col run 0 → grid_size-1.
        Row 0 is the +Y side (top of the grid image), col 0 is the -X side.
        """
        G  = self.cfg.grid_size
        cs = self.cfg.cell_size
        half = (G * cs) / 2.0

        # Terrain origin for each env
        if env_ids is None:
            origins = self._terrain.env_origins[:, :2].cpu().numpy()   # (N, 2)
            pos_xy  = pos_w[:, :2].cpu().numpy()
        else:
            origins = self._terrain.env_origins[env_ids, :2].cpu().numpy()
            pos_xy  = pos_w[:, :2].cpu().numpy()

        local_xy = pos_xy - origins                     # position relative to env centre
        col = np.floor((local_xy[:, 0] + half) / cs).astype(int)
        row = np.floor((half - local_xy[:, 1]) / cs).astype(int)   # Y flipped for image coords

        col = np.clip(col, 0, G - 1)
        row = np.clip(row, 0, G - 1)
        return row, col

    def _grid_to_world(self, row: int, col: int, env_id: int) -> np.ndarray:
        """
        Convert a single grid cell (row, col) back to a world-frame XY position.
        Returns a (2,) array [x, y].
        """
        G    = self.cfg.grid_size
        cs   = self.cfg.cell_size
        half = (G * cs) / 2.0
        origin = self._terrain.env_origins[env_id, :2].cpu().numpy()

        x = (col + 0.5) * cs - half + origin[0]
        y = half - (row + 0.5) * cs + origin[1]
        return np.array([x, y])

    # def _update_grids(self):
    #     """
    #     Mark cells as FREE for every drone's current position.
    #     In Stage 3 this is where ray-casting from the camera slice will go.
    #     For now: just mark the cell the drone occupies and a small radius around it
    #     (simulating a sensor footprint).
    #     """
    #     pos_w = self._robot.data.root_pos_w          # (N, 3)
    #     rows, cols = self._world_to_grid(pos_w)

    #     G  = self.cfg.grid_size
    #     cs = self.cfg.cell_size
    #     # Sensor radius: mark cells within ~1 m of the drone as free
    #     r  = max(1, int(1.0 / cs))

    #     for i in range(self.num_envs):
    #         r0, c0 = rows[i], cols[i]
    #         r_lo = max(0, r0 - r)
    #         r_hi = min(G - 1, r0 + r)
    #         c_lo = max(0, c0 - r)
    #         c_hi = min(G - 1, c0 + r)
    #         self._grids[i, r_lo:r_hi+1, c_lo:c_hi+1] = FREE

    def _update_grids(self):
        """
        Conical sensor update — mimics a front-facing monocular camera FOV.

        For each env the drone casts a fan of rays from its current position
        outward in the direction it is facing.  Any grid cell whose centre
        falls inside the cone (within max_depth AND within half_fov_rad of
        the forward heading) is marked FREE.

        Parameters (tune these to match your real camera):
            fov_deg   : total horizontal field of view  (pinhole ~90-110°)
            max_depth : how far the 'camera' can see     (metres)
            n_rays    : angular resolution of the fan    (more = smoother, slower)
        """
        fov_deg   = 100.0                        # total FOV in degrees
        max_depth = 3.5                          # metres
        n_rays    = 36                           # number of rays in the fan

        half_fov  = np.deg2rad(fov_deg / 2.0)
        G         = self.cfg.grid_size
        cs        = self.cfg.cell_size

        # --- Get drone yaw angle for every env ---
        # Quaternion (w,x,y,z) → yaw around world Z
        quat = self._robot.data.root_state_w[:, 3:7].cpu().numpy()   # (N,4)
        # yaw = atan2(2(wz + xy),  1 - 2(yy + zz))
        w, x, y, z = quat[:,0], quat[:,1], quat[:,2], quat[:,3]
        yaw = np.arctan2(2.0*(w*z + x*y), 1.0 - 2.0*(y*y + z*z))    # (N,)

        # --- Current drone grid position ---
        pos_w = self._robot.data.root_pos_w   # (N,3)
        drone_rows, drone_cols = self._world_to_grid(pos_w)           # both (N,)

        # --- Ray angles: spread evenly across the FOV ---
        ray_offsets = np.linspace(-half_fov, half_fov, n_rays)        # (n_rays,)

        for i in range(self.num_envs):
            dr, dc   = drone_rows[i], drone_cols[i]
            heading  = yaw[i]   # drone's forward direction in world frame

            for offset in ray_offsets:
                ray_angle = heading + offset

                # Step along this ray, cell by cell
                # Each step is one cell_size in world units
                n_steps = int(max_depth / cs)
                for step in range(1, n_steps + 1):
                    dist = step * cs

                    # World-frame displacement along this ray
                    dx =  dist * np.cos(ray_angle)   # +x = East
                    dy =  dist * np.sin(ray_angle)   # +y = North

                    # World position of this cell centre
                    wx = self._robot.data.root_pos_w[i, 0].item() + dx
                    wy = self._robot.data.root_pos_w[i, 1].item() + dy

                    # Convert to grid indices
                    origin = self._terrain.env_origins[i, :2].cpu().numpy()
                    half   = (G * cs) / 2.0
                    col = int((wx - origin[0] + half) / cs)
                    row = int((half - (wy - origin[1])) / cs)

                    if 0 <= row < G and 0 <= col < G:
                        self._grids[i, row, col] = FREE
    def _compute_frontiers(self) -> np.ndarray:
        """
        Find frontier cells for every environment.
        A frontier is a FREE cell that has at least one UNKNOWN neighbour.

        Returns:
            closest_frontier_xy: (num_envs, 2) world-frame XY of the nearest
            frontier to each drone.  If no frontier exists, returns the drone's
            own position (zero relative vector → zero reward, but no crash).
        """
        G   = self.cfg.grid_size
        pos_w = self._robot.data.root_pos_w.cpu().numpy()   # (N, 3)
        drone_rows, drone_cols = self._world_to_grid(self._robot.data.root_pos_w)

        result_xy = np.zeros((self.num_envs, 2), dtype=np.float32)

        for i in range(self.num_envs):
            grid = self._grids[i]   # (G, G)

            # Vectorised frontier detection using array slicing:
            # A cell is a frontier if it is FREE and any of its 4 neighbours is UNKNOWN
            free_mask = grid == FREE   # (G, G) bool

            # Shift the grid in 4 directions and check for UNKNOWN neighbours
            unknown_above = np.zeros_like(free_mask)
            unknown_below = np.zeros_like(free_mask)
            unknown_left  = np.zeros_like(free_mask)
            unknown_right = np.zeros_like(free_mask)

            unknown_above[1:,  :]  = grid[:-1, :] == UNKNOWN
            unknown_below[:-1, :]  = grid[1:,  :] == UNKNOWN
            unknown_left[ :,  1:]  = grid[:, :-1] == UNKNOWN
            unknown_right[:, :-1]  = grid[:, 1:]  == UNKNOWN

            frontier_mask = free_mask & (
                unknown_above | unknown_below | unknown_left | unknown_right
            )

            frontier_positions = np.argwhere(frontier_mask)   # (K, 2) → [row, col]

            if len(frontier_positions) == 0:
                # No frontier: use drone's own position (zero relative vector)
                result_xy[i] = pos_w[i, :2]
                continue

            # Find the closest frontier to the current drone position
            dr = drone_rows[i]
            dc = drone_cols[i]
            dists = np.abs(frontier_positions[:, 0] - dr) + np.abs(frontier_positions[:, 1] - dc)
            best = frontier_positions[np.argmin(dists)]
            result_xy[i] = self._grid_to_world(best[0], best[1], i)

        return result_xy

    def _is_out_of_bounds(self) -> np.ndarray:
        """
        Returns a (num_envs,) bool array: True if the drone is outside the grid.
        """
        G    = self.cfg.grid_size
        cs   = self.cfg.cell_size
        half = (G * cs) / 2.0

        pos_w   = self._robot.data.root_pos_w.cpu().numpy()
        origins = self._terrain.env_origins[:, :2].cpu().numpy()
        local   = pos_w[:, :2] - origins
        return (np.abs(local[:, 0]) > half) | (np.abs(local[:, 1]) > half)

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def _get_observations(self) -> dict:
        # Update grid with current drone positions
        self._update_grids()

        # Recompute frontiers periodically (not every step for performance)
        if self._step_counter % self.cfg.frontier_recompute_interval == 0:
            frontier_xy = self._compute_frontiers()   # (N, 2) world XY
            # Build world-frame 3D frontier position at hover height
            frontier_pos = np.zeros((self.num_envs, 3), dtype=np.float32)
            frontier_pos[:, :2] = frontier_xy
            frontier_pos[:, 2]  = self.cfg.hover_height + \
                                   self._terrain.env_origins[:, 2].cpu().numpy()
            self._frontier_pos_w = torch.tensor(frontier_pos, device=self.device)

        self._step_counter += 1

        # Save grid of env 0 to disk periodically for viewer.py
        if self._step_counter % self.cfg.grid_save_every == 0:
            np.save(self.cfg.grid_save_path, self._grids[0])

        # Transform frontier position into drone body frame
        frontier_b, _ = subtract_frame_transforms(
            self._robot.data.root_state_w[:, :3],
            self._robot.data.root_state_w[:, 3:7],
            self._frontier_pos_w,
        )

        # Coverage ratio: fraction of cells that are FREE
        G = self.cfg.grid_size
        free_counts = (self._grids == FREE).sum(axis=(1, 2)).astype(np.float32)
        coverage = torch.tensor(
            free_counts / (G * G), device=self.device
        ).unsqueeze(1)   # (N, 1)

        obs = torch.cat(
            [
                self._robot.data.root_lin_vel_b,       # 3
                self._robot.data.root_ang_vel_b,       # 3
                self._robot.data.projected_gravity_b,  # 3
                frontier_b,                            # 3
                coverage,                              # 1  → total: 13
            ],
            dim=-1,
        )
        return {"policy": obs}

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------

    def _get_rewards(self) -> torch.Tensor:
        # --- New cells discovered this step ---
        free_counts = (self._grids == FREE).sum(axis=(1, 2)).astype(np.int32)
        new_cells   = np.maximum(0, free_counts - self._prev_free_count)
        self._prev_free_count = free_counts.copy()
        new_cells_t = torch.tensor(new_cells, dtype=torch.float, device=self.device)

        # --- Frontier approach reward ---
        # Positive when drone is moving toward the frontier
        dist_to_frontier = torch.linalg.norm(
            self._frontier_pos_w - self._robot.data.root_pos_w, dim=1
        )
        # Smooth mapping: reward peaks when very close to frontier
        frontier_proximity = 1 - torch.tanh(dist_to_frontier / 2.0)

        # --- Penalties ---
        ang_vel = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)

        # Revisit penalty: drone is in an already-explored region and not discovering new cells
        revisit = torch.tensor(
            (new_cells == 0).astype(np.float32), device=self.device
        )

        # Out of bounds penalty
        oob = torch.tensor(
            self._is_out_of_bounds().astype(np.float32), device=self.device
        )

        rewards = {
            "new_cells":    new_cells_t       * self.cfg.new_cell_reward_scale,
            "frontier":     frontier_proximity * self.cfg.frontier_reward_scale     * self.step_dt,
            "ang_vel":      ang_vel            * self.cfg.ang_vel_penalty_scale     * self.step_dt,
            "revisit":      revisit            * self.cfg.revisit_penalty_scale,
            "out_of_bounds": oob              * self.cfg.out_of_bounds_penalty,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        for key, value in rewards.items():
            self._episode_sums[key] += value

        return reward

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # Terminate if drone crashes (too low or too high) or leaves the grid
        altitude_fail = torch.logical_or(
            self._robot.data.root_pos_w[:, 2] < 0.1,
            self._robot.data.root_pos_w[:, 2] > 3.0,
        )
        oob = torch.tensor(self._is_out_of_bounds(), device=self.device)
        died = torch.logical_or(altitude_fail, oob)

        return died, time_out

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

        self._actions[env_ids] = 0.0

        # Reset grids for the resetting environments
        env_ids_np = env_ids.cpu().numpy()
        self._grids[env_ids_np] = UNKNOWN
        self._prev_free_count[env_ids_np] = 0
        self._frontier_pos_w[env_ids] = 0.0

        # Reset robot state
        joint_pos          = self._robot.data.default_joint_pos[env_ids]
        joint_vel          = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    # ------------------------------------------------------------------
    # Debug visualisation  (frontier marker)
    # ------------------------------------------------------------------

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "frontier_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.1, 0.1, 0.1)
                marker_cfg.prim_path = "/Visuals/Frontier/position"
                self.frontier_visualizer = VisualizationMarkers(marker_cfg)
            self.frontier_visualizer.set_visibility(True)
        else:
            if hasattr(self, "frontier_visualizer"):
                self.frontier_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        self.frontier_visualizer.visualize(self._frontier_pos_w)


# ---------------------------------------------------------------------------
# Utility: rotate vectors by quaternions  [w, x, y, z] convention
# ---------------------------------------------------------------------------

def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    w   = q[:, 0:1]
    xyz = q[:, 1:]
    t   = 2.0 * torch.linalg.cross(xyz, v)
    return v + w * t + torch.linalg.cross(xyz, t)