"""
iris_explore_walls_env.py  —  Stage A/B: Exploration with occupancy grid + walls

Changes from the previous version:
  - Removed ContactSensor entirely (caused 'activate_contact_sensors' error)
  - Collision detection is now proximity-based: if the drone centre comes
    within cfg.collision_radius metres of any wall AABB, the episode ends.
    This is more robust, requires no PhysX API changes, and is fast.
  - Wall spawning uses sim_utils.CuboidCfg correctly for Isaac Lab 4.x

Switch layouts by changing wall_layout in the config:
    wall_layout = field(default_factory=lambda: PILLAR_LAYOUT)    # Stage A
    wall_layout = field(default_factory=lambda: CORRIDOR_LAYOUT)  # Stage B
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

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
# Wall layout definitions
# ---------------------------------------------------------------------------
# Each tuple: (centre_x, centre_y, size_x, size_y, height)  — metres, local frame

PILLAR_LAYOUT: List[Tuple[float, float, float, float, float]] = [
    ( 3.0,  3.0,  0.4,  0.4,  2.0),
    (-3.0,  3.0,  0.4,  0.4,  2.0),
    ( 3.0, -3.0,  0.4,  0.4,  2.0),
    (-3.0, -3.0,  0.4,  0.4,  2.0),
]

CORRIDOR_LAYOUT: List[Tuple[float, float, float, float, float]] = [
    ( 0.0,  5.0, 10.0,  0.4,  2.0),   # north wall
    ( 0.0, -5.0, 10.0,  0.4,  2.0),   # south wall
    (-5.0,  0.0,  0.4, 10.0,  2.0),   # west wall
    ( 5.0,  0.0,  0.4, 10.0,  2.0),   # east wall
    (-0.5,  3.0,  0.4,  4.0,  2.0),   # interior divider north
    (-0.5, -3.0,  0.4,  4.0,  2.0),   # interior divider south
]


# ---------------------------------------------------------------------------
# Grid cell states
# ---------------------------------------------------------------------------
UNKNOWN  = 0
FREE     = 1
OCCUPIED = 2


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@configclass
class IrisExploreWallsEnvCfg(DirectRLEnvCfg):
    episode_length_s  = 40.0
    decimation        = 2
    action_space      = 2
    observation_space = 13
    state_space       = 0
    debug_vis         = True

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
        num_envs=256, env_spacing=16.0, replicate_physics=True
    )

    robot: ArticulationCfg = IRIS_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )

    # --- wall layout ---
    wall_layout: List[Tuple[float, float, float, float, float]] = field(
        default_factory=lambda: PILLAR_LAYOUT
    )

    # --- proximity collision ---
    # Episode ends if drone centre is within this distance of any wall surface.
    # 0.3 m is roughly the drone's arm radius — tune if you see false positives.
    collision_radius: float = 0.3

    # --- velocity limits ---
    max_forward_vel  = 1.5
    max_yaw_rate     = 1.5

    # --- altitude hold ---
    hover_height     = 1.0
    altitude_kp      = 2.0
    max_altitude_vel = 1.0

    # --- occupancy grid ---
    cell_size  = 0.25
    grid_size  = 80     # 20 m × 20 m

    frontier_recompute_interval = 10

    # --- reward scales ---
    new_cell_reward_scale  =  5.0
    frontier_reward_scale  =  1.0
    ang_vel_penalty_scale  = -0.01
    revisit_penalty_scale  = -0.02
    out_of_bounds_penalty  = -5.0
    collision_penalty      = -10.0

    # --- logging ---
    grid_save_path  = "/tmp/iris_walls_grid.npy"
    grid_save_every = 200


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class IrisExploreWallsEnv(DirectRLEnv):
    cfg: IrisExploreWallsEnvCfg

    def __init__(self, cfg: IrisExploreWallsEnvCfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        G = self.cfg.grid_size
        self._grids           = np.zeros((self.num_envs, G, G), dtype=np.uint8)
        self._prev_free_count = np.zeros(self.num_envs, dtype=np.int32)
        self._frontier_pos_w  = torch.zeros(self.num_envs, 3, device=self.device)

        self._actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space),
            device=self.device
        )

        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in ["new_cells", "frontier", "ang_vel",
                        "revisit", "out_of_bounds", "collision"]
        }

        self._step_counter = 0

        # Pre-compute wall AABBs in local (env-relative) frame: (W, 4)
        # Format: [x_min, x_max, y_min, y_max]
        self._wall_aabbs = self._build_wall_aabbs()

        self.set_debug_vis(self.cfg.debug_vis)

    # ------------------------------------------------------------------
    # Wall geometry helpers
    # ------------------------------------------------------------------

    def _build_wall_aabbs(self) -> np.ndarray:
        aabbs = []
        for (cx, cy, sx, sy, _) in self.cfg.wall_layout:
            aabbs.append([cx - sx/2, cx + sx/2, cy - sy/2, cy + sy/2])
        return np.array(aabbs, dtype=np.float32)   # (W, 4)

    def _point_to_aabb_distance(self, px: float, py: float) -> float:
        """
        Returns the minimum distance from point (px, py) to the nearest
        wall surface across all wall AABBs.  Uses the standard clamped-point
        formula so interior points return 0 (inside wall = collision).
        """
        if len(self._wall_aabbs) == 0:
            return np.inf

        # Vectorised over all walls at once
        # dx: distance outside the box in X (0 if inside)
        dx = np.maximum(0.0, np.maximum(
            self._wall_aabbs[:, 0] - px,   # left face
            px - self._wall_aabbs[:, 1]    # right face
        ))
        dy = np.maximum(0.0, np.maximum(
            self._wall_aabbs[:, 2] - py,
            py - self._wall_aabbs[:, 3]
        ))
        distances = np.sqrt(dx**2 + dy**2)
        return float(distances.min())

    def _get_collisions(self) -> np.ndarray:
        """
        Returns (num_envs,) bool array.
        True if drone is within collision_radius of any wall.
        """
        pos_np  = self._robot.data.root_pos_w.cpu().numpy()   # (N, 3)
        origins = self._terrain.env_origins.cpu().numpy()     # (N, 3)

        collided = np.zeros(self.num_envs, dtype=bool)
        for i in range(self.num_envs):
            # Local position relative to env origin
            lx = pos_np[i, 0] - origins[i, 0]
            ly = pos_np[i, 1] - origins[i, 1]
            dist = self._point_to_aabb_distance(lx, ly)
            collided[i] = dist < self.cfg.collision_radius

        return collided

    def _ray_hits_wall(self, ox, oy, dx, dy, max_dist) -> Tuple[bool, float]:
        """
        Slab-method 2-D ray vs AABB intersection.
        Returns (hit, t) where t is distance to first hit along the ray.
        """
        t_hit = max_dist
        for aabb in self._wall_aabbs:
            x_min, x_max, y_min, y_max = aabb

            if abs(dx) > 1e-9:
                tx1, tx2 = (x_min - ox) / dx, (x_max - ox) / dx
                te_x, tx_x = min(tx1, tx2), max(tx1, tx2)
            else:
                if ox < x_min or ox > x_max:
                    continue
                te_x, tx_x = -np.inf, np.inf

            if abs(dy) > 1e-9:
                ty1, ty2 = (y_min - oy) / dy, (y_max - oy) / dy
                te_y, tx_y = min(ty1, ty2), max(ty1, ty2)
            else:
                if oy < y_min or oy > y_max:
                    continue
                te_y, tx_y = -np.inf, np.inf

            t_enter = max(te_x, te_y)
            t_exit  = min(tx_x, tx_y)

            if t_enter <= t_exit and 0.0 < t_enter < t_hit:
                t_hit = t_enter

        return t_hit < max_dist, t_hit

    # ------------------------------------------------------------------
    # Scene setup
    # ------------------------------------------------------------------

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self.cfg.terrain.num_envs    = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)

        # Spawn wall boxes into every environment
        for idx, (cx, cy, sx, sy, sz) in enumerate(self.cfg.wall_layout):
            wall_cfg = sim_utils.CuboidCfg(
                size=(sx, sy, sz),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=True,
                    disable_gravity=True,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=1000.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.45, 0.45, 0.5),
                    roughness=0.8,
                ),
            )
            wall_cfg.func(
                f"/World/envs/env_.*/Wall_{idx:02d}",
                wall_cfg,
                translation=(cx, cy, sz / 2.0),
            )

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ------------------------------------------------------------------
    # Action pipeline
    # ------------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone().clamp(-1.0, 1.0)

        lin_vel_b = torch.zeros(self.num_envs, 3, device=self.device)
        lin_vel_b[:, 0] = self._actions[:, 0] * self.cfg.max_forward_vel

        target_z  = self.cfg.hover_height + self._terrain.env_origins[:, 2]
        current_z = self._robot.data.root_pos_w[:, 2]
        vz_world  = (self.cfg.altitude_kp * (target_z - current_z)).clamp(
            -self.cfg.max_altitude_vel, self.cfg.max_altitude_vel
        )

        quat_w    = self._robot.data.root_state_w[:, 3:7]
        lin_vel_w = quat_rotate(quat_w, lin_vel_b)
        lin_vel_w[:, 2] = vz_world

        ang_vel_w = torch.zeros(self.num_envs, 3, device=self.device)
        ang_vel_w[:, 2] = self._actions[:, 1] * self.cfg.max_yaw_rate

        self._cmd_lin_vel_w = lin_vel_w
        self._cmd_ang_vel_w = ang_vel_w

    def _apply_action(self):
        velocity_cmd = torch.cat([self._cmd_lin_vel_w, self._cmd_ang_vel_w], dim=-1)
        self._robot.write_root_velocity_to_sim(velocity_cmd)

        joint_vel = torch.zeros_like(self._robot.data.joint_vel)
        joint_vel[:, 0] =  200.0
        joint_vel[:, 1] = -200.0
        joint_vel[:, 2] =  200.0
        joint_vel[:, 3] = -200.0
        self._robot.set_joint_velocity_target(joint_vel)

    # ------------------------------------------------------------------
    # Grid coordinate helpers
    # ------------------------------------------------------------------

    def _world_to_grid(self, pos_w):
        G, cs = self.cfg.grid_size, self.cfg.cell_size
        half  = (G * cs) / 2.0
        origins = self._terrain.env_origins[:, :2].cpu().numpy()
        pos_xy  = pos_w[:, :2].cpu().numpy()
        local   = pos_xy - origins
        col = np.clip(np.floor((local[:, 0] + half) / cs).astype(int), 0, G-1)
        row = np.clip(np.floor((half - local[:, 1]) / cs).astype(int), 0, G-1)
        return row, col

    def _grid_to_world(self, row, col, env_id):
        G, cs = self.cfg.grid_size, self.cfg.cell_size
        half  = (G * cs) / 2.0
        origin = self._terrain.env_origins[env_id, :2].cpu().numpy()
        x = (col + 0.5) * cs - half + origin[0]
        y = half - (row + 0.5) * cs  + origin[1]
        return np.array([x, y])

    # ------------------------------------------------------------------
    # Conical sensor with wall occlusion
    # ------------------------------------------------------------------

    def _update_grids(self):
        """
        Vectorised conical sensor — runs in ~2 ms for 64 envs instead of ~200 ms.
        Safe to call every step even with the renderer running.
        """
        fov_deg   = 100.0
        max_depth = 3.5
        n_rays    = 24          # reduced from 36 — still good coverage
        half_fov  = np.deg2rad(fov_deg / 2.0)
        G, cs     = self.cfg.grid_size, self.cfg.cell_size
        half      = (G * cs) / 2.0

        # --- batch-compute yaw for all envs at once ---
        quat = self._robot.data.root_state_w[:, 3:7].cpu().numpy()
        w, x, y, z = quat[:,0], quat[:,1], quat[:,2], quat[:,3]
        yaw  = np.arctan2(2.0*(w*z + x*y), 1.0 - 2.0*(y*y + z*z))  # (N,)

        pos_np  = self._robot.data.root_pos_w.cpu().numpy()           # (N, 3)
        origins = self._terrain.env_origins.cpu().numpy()             # (N, 3)

        ray_offsets = np.linspace(-half_fov, half_fov, n_rays)        # (R,)
        step_dists  = np.arange(1, int(max_depth / cs) + 1) * cs     # (S,)

        for i in range(self.num_envs):
            ox = pos_np[i, 0] - origins[i, 0]
            oy = pos_np[i, 1] - origins[i, 1]
            angles = yaw[i] + ray_offsets                             # (R,)

            # vectorised over rays AND steps simultaneously
            cos_a = np.cos(angles)   # (R,)
            sin_a = np.sin(angles)   # (R,)

            # All sample points: (R, S)
            dx = np.outer(cos_a, step_dists)   # (R, S)
            dy = np.outer(sin_a, step_dists)   # (R, S)

            # World positions of all sample points
            wx = ox + dx                       # (R, S)  local frame
            wy = oy + dy

            # Grid indices for all points at once
            cols = np.floor((wx + half) / cs).astype(int)            # (R, S)
            rows = np.floor((half - wy)  / cs).astype(int)           # (R, S)

            # Valid mask
            valid = (rows >= 0) & (rows < G) & (cols >= 0) & (cols < G)

            if len(self._wall_aabbs) > 0:
                # For each ray, find the step index where it first hits a wall
                # Shape: (R, S) — True if this sample point is inside or past a wall
                blocked = np.zeros((n_rays, len(step_dists)), dtype=bool)
                for aabb in self._wall_aabbs:
                    x_min, x_max, y_min, y_max = aabb
                    inside = ((wx >= x_min) & (wx <= x_max) &
                            (wy >= y_min) & (wy <= y_max))
                    blocked |= inside

                # For each ray, find first blocked step
                # Everything after first block on that ray should be masked out
                for r in range(n_rays):
                    hit_steps = np.where(blocked[r])[0]
                    if len(hit_steps) > 0:
                        first_hit = hit_steps[0]
                        # Mark the wall cell OCCUPIED
                        rc, cc = rows[r, first_hit], cols[r, first_hit]
                        if 0 <= rc < G and 0 <= cc < G:
                            self._grids[i, rc, cc] = OCCUPIED
                        # Mask out steps beyond the wall
                        valid[r, first_hit:] = False

            # Mark all remaining valid cells FREE (don't overwrite OCCUPIED)
            r_idx = rows[valid]
            c_idx = cols[valid]
            # Only update UNKNOWN cells — preserve OCCUPIED walls
            current = self._grids[i, r_idx, c_idx]
            free_mask = current != OCCUPIED
            self._grids[i, r_idx[free_mask], c_idx[free_mask]] = FREE

    # ------------------------------------------------------------------
    # Frontier detection
    # ------------------------------------------------------------------

    def _compute_frontiers(self):
        G     = self.cfg.grid_size
        pos_w = self._robot.data.root_pos_w
        d_rows, d_cols = self._world_to_grid(pos_w)
        result_xy = np.zeros((self.num_envs, 2), dtype=np.float32)

        for i in range(self.num_envs):
            grid = self._grids[i]
            free = grid == FREE
            unk  = grid == UNKNOWN
            has_unk = (
                np.roll(unk,  1, axis=0) | np.roll(unk, -1, axis=0) |
                np.roll(unk,  1, axis=1) | np.roll(unk, -1, axis=1)
            )
            has_unk[0,:]=has_unk[-1,:]=has_unk[:,0]=has_unk[:,-1]=False
            frontiers = np.argwhere(free & has_unk)

            if len(frontiers) == 0:
                result_xy[i] = pos_w[i, :2].cpu().numpy()
                continue

            dists = (np.abs(frontiers[:,0] - d_rows[i]) +
                     np.abs(frontiers[:,1] - d_cols[i]))
            best = frontiers[np.argmin(dists)]
            result_xy[i] = self._grid_to_world(best[0], best[1], i)

        return result_xy

    def _is_out_of_bounds(self) -> np.ndarray:
        G, cs = self.cfg.grid_size, self.cfg.cell_size
        half  = (G * cs) / 2.0
        pos_np  = self._robot.data.root_pos_w.cpu().numpy()
        origins = self._terrain.env_origins[:, :2].cpu().numpy()
        local   = pos_np[:, :2] - origins
        return (np.abs(local[:,0]) > half) | (np.abs(local[:,1]) > half)

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def _get_observations(self) -> dict:
        self._update_grids()

        if self._step_counter % self.cfg.frontier_recompute_interval == 0:
            frontier_xy = self._compute_frontiers()
            fp = np.zeros((self.num_envs, 3), dtype=np.float32)
            fp[:, :2] = frontier_xy
            fp[:, 2]  = (self.cfg.hover_height +
                         self._terrain.env_origins[:, 2].cpu().numpy())
            self._frontier_pos_w = torch.tensor(fp, device=self.device)

        self._step_counter += 1

        if self._step_counter % self.cfg.grid_save_every == 0:
            np.save(self.cfg.grid_save_path, self._grids[0])
            self._save_grid_png(
                self._grids[0],
                self.cfg.grid_save_path.replace('.npy', '.png')
            )

        frontier_b, _ = subtract_frame_transforms(
            self._robot.data.root_state_w[:, :3],
            self._robot.data.root_state_w[:, 3:7],
            self._frontier_pos_w,
        )

        G = self.cfg.grid_size
        free_counts = (self._grids == FREE).sum(axis=(1,2)).astype(np.float32)
        coverage = torch.tensor(
            free_counts / (G * G), device=self.device
        ).unsqueeze(1)

        obs = torch.cat([
            self._robot.data.root_lin_vel_b,
            self._robot.data.root_ang_vel_b,
            self._robot.data.projected_gravity_b,
            frontier_b,
            coverage,
        ], dim=-1)
        return {"policy": obs}

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------

    def _get_rewards(self) -> torch.Tensor:
        free_counts = (self._grids == FREE).sum(axis=(1,2)).astype(np.int32)
        new_cells   = np.maximum(0, free_counts - self._prev_free_count)
        self._prev_free_count = free_counts.copy()
        new_cells_t = torch.tensor(new_cells, dtype=torch.float, device=self.device)

        dist_to_frontier = torch.linalg.norm(
            self._frontier_pos_w - self._robot.data.root_pos_w, dim=1
        )
        frontier_prox = 1 - torch.tanh(dist_to_frontier / 2.0)

        ang_vel  = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)
        revisit  = torch.tensor((new_cells == 0).astype(np.float32), device=self.device)
        oob      = torch.tensor(self._is_out_of_bounds().astype(np.float32), device=self.device)
        collided = torch.tensor(self._get_collisions().astype(np.float32), device=self.device)

        rewards = {
            "new_cells":     new_cells_t   * self.cfg.new_cell_reward_scale,
            "frontier":      frontier_prox  * self.cfg.frontier_reward_scale  * self.step_dt,
            "ang_vel":       ang_vel        * self.cfg.ang_vel_penalty_scale  * self.step_dt,
            "revisit":       revisit        * self.cfg.revisit_penalty_scale,
            "out_of_bounds": oob            * self.cfg.out_of_bounds_penalty,
            "collision":     collided       * self.cfg.collision_penalty,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        for key, value in rewards.items():
            self._episode_sums[key] += value

        # TensorBoard extras — visible under Episode/... in TensorBoard
        self.extras["log"] = {
            "coverage_pct":   float((self._grids == FREE).mean() * 100.0),
            "collision_rate": float(collided.mean().item()),
            "mean_new_cells": float(new_cells_t.mean().item()),
            "frontier_dist":  float(dist_to_frontier.mean().item()),
        }

        return reward

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        altitude_fail = (self._robot.data.root_pos_w[:, 2] < 0.1) | \
                        (self._robot.data.root_pos_w[:, 2] > 3.0)
        oob      = torch.tensor(self._is_out_of_bounds(), device=self.device)
        collided = torch.tensor(self._get_collisions(),   device=self.device)

        died = altitude_fail | oob | collided
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

        env_ids_np = env_ids.cpu().numpy()
        self._actions[env_ids]        = 0.0
        self._grids[env_ids_np]       = UNKNOWN
        self._prev_free_count[env_ids_np] = 0
        self._frontier_pos_w[env_ids] = 0.0

        # Log final coverage before reset — useful for thesis episode-end stats
        final_cov = float((self._grids[env_ids_np] == FREE).mean() * 100.0)
        self.extras.setdefault("log", {})["episode_final_coverage"] = final_cov

        joint_pos          = self._robot.data.default_joint_pos[env_ids]
        joint_vel          = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    # ------------------------------------------------------------------
    # Grid PNG export (no matplotlib — pure Python stdlib only)
    # ------------------------------------------------------------------

    def _save_grid_png(self, grid: np.ndarray, path: str):
        import struct, zlib
        G   = grid.shape[0]
        rgb = np.zeros((G, G, 3), dtype=np.uint8)
        rgb[grid == UNKNOWN]  = (180, 178, 169)
        rgb[grid == FREE]     = (125, 200, 122)
        rgb[grid == OCCUPIED] = ( 44,  44,  42)
        free, unk = grid == FREE, grid == UNKNOWN
        frontier  = free & (
            np.roll(unk,  1,0)|np.roll(unk,-1,0)|
            np.roll(unk,  1,1)|np.roll(unk,-1,1)
        )
        rgb[frontier] = (226, 75, 74)

        raw = b''.join(b'\x00' + rgb[r].tobytes() for r in range(G))
        def chunk(n, d):
            c = n + d
            return struct.pack('>I',len(d)) + c + struct.pack('>I',zlib.crc32(c)&0xffffffff)
        ihdr = struct.pack('>IIBBBBB', G, G, 8, 2, 0, 0, 0)
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
# Utility
# ---------------------------------------------------------------------------

def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    w   = q[:, 0:1]
    xyz = q[:, 1:]
    t   = 2.0 * torch.linalg.cross(xyz, v)
    return v + w * t + torch.linalg.cross(xyz, t)