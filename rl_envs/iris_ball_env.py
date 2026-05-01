"""
iris_ball_env.py
================
Iris drone learns to:
  1. Search by yawing when the yellow ball is not visible
  2. Yaw to centre the ball in frame when visible
  3. Fly toward the ball
  4. Stop at stop_distance and hover

The entire task is solved from RGB vision only — no GPS, no pose of the ball.
This is a faithful drone equivalent of the Jetbot red-object tracking task.

Observation: (N, T, H, W, 4) stacked frames
    channels 0-2: RGB normalised [0,1]
    channel    3: search_active [0 or 1] — tells policy if it should be searching

Why channel 3 (search_active)?
    When ball is not visible, the policy needs to know to rotate rather than
    fly forward. Encoding this as a spatial channel (same as jetbot's goal vec)
    keeps the input format identical to what the CNN expects.

Action: [vx, yaw_rate] in [-1, 1]
    vx:       forward velocity command
    yaw_rate: rotation command

Rewards are vision-based — the policy never receives the ball's world position.
All spatial awareness comes from the camera.
"""
 
from __future__ import annotations

import math
import torch
import numpy as np

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sensors import TiledCamera
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from .iris_ball_env_cfg import IrisBallEnvCfg


# =============================================================================
# YELLOW DETECTION HELPER
# =============================================================================

def _detect_yellow(rgb: torch.Tensor, threshold: float = 0.005):
    """
    Detect yellow pixels in an RGB image tensor.

    Yellow: high R, high G, low B.
    Thresholds tuned for the emissive yellow material defined in cfg.

    Args:
        rgb:       (N, H, W, 3) float32 in [0, 1]
        threshold: fraction of pixels that must be yellow for "ball visible"

    Returns:
        visible:    (N,) bool — True if ball detected
        cx_norm:    (N,) float — horizontal centroid of yellow pixels, [-1,1]
                    0 = centre, -1 = left edge, +1 = right edge
        area_norm:  (N,) float — fraction of pixels that are yellow [0,1]
    """
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    # Yellow: R > 0.6, G > 0.5, B < 0.35
    yellow_mask = (r > 0.6) & (g > 0.5) & (b < 0.35)   # (N, H, W)

    N, H, W    = rgb.shape[:3]
    pixel_count = H * W

    area = yellow_mask.float().sum(dim=(1, 2)) / pixel_count   # (N,)
    visible = area > threshold                                  # (N,)

    # Horizontal centroid — where in the frame is the ball?
    col_idx = torch.arange(W, device=rgb.device, dtype=torch.float32)
    col_idx = col_idx.view(1, 1, W).expand(N, H, W)            # (N, H, W)

    yellow_float = yellow_mask.float()
    sum_cols = (yellow_float * col_idx).sum(dim=(1, 2))
    sum_area = yellow_float.sum(dim=(1, 2)).clamp(min=1.0)

    cx_pixel = sum_cols / sum_area                             # (N,) in [0, W)
    cx_norm  = (cx_pixel / (W - 1)) * 2.0 - 1.0              # (N,) in [-1, 1]

    # For non-visible envs, centroid is meaningless — zero it
    cx_norm = cx_norm * visible.float()

    return visible, cx_norm, area


# =============================================================================
# MAIN ENVIRONMENT
# =============================================================================

class IrisBallEnv(DirectRLEnv):
    """
    Iris drone yellow ball tracking.
    Mirrors JetbotNavEnv structure exactly — same scene setup pattern,
    same observation construction, same reward components adapted for a drone.
    """

    cfg: IrisBallEnvCfg

    def __init__(self, cfg: IrisBallEnvCfg, render_mode: str | None = None, **kwargs):
        self._camera_hist: torch.Tensor | None = None
        super().__init__(cfg, render_mode, **kwargs)

        N = self.num_envs

        # Ball world positions (set at reset)
        self._ball_pos_w = torch.zeros(N, 3, device=self.device)

        # Distance to ball (for delta-distance reward, same as jetbot)
        self._prev_dist = torch.zeros(N, device=self.device)

        # Search state: True when ball not visible
        self._searching = torch.ones(N, dtype=torch.bool, device=self.device)

        # Success flag — drone has reached stop distance at least once
        self._succeeded = torch.zeros(N, dtype=torch.bool, device=self.device)

        # Rotor joint indices
        self._rotor_ids = None   # set after articulation init

        # Visualisation markers
        self._ball_marker  = self._make_ball_vis_marker()
        self._drone_marker = self._make_drone_vis_marker()

        self._ball_marker.set_visibility(True)
        self._drone_marker.set_visibility(True)

    # ─────────────────────────────────────────────────────────────────────────
    # Markers
    # ─────────────────────────────────────────────────────────────────────────
    def _make_ball_vis_marker(self) -> VisualizationMarkers:
        cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/BallMarker",
            markers={
                "sphere": sim_utils.SphereCfg(
                    radius=0.18,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(1.0, 1.0, 0.0)
                    ),
                )
            },
        )
        return VisualizationMarkers(cfg)

    def _make_drone_vis_marker(self) -> VisualizationMarkers:
        cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/DroneMarker",
            markers={
                "arrow": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                    scale=(0.3, 0.3, 0.6),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.0, 0.8, 1.0)
                    ),
                )
            },
        )
        return VisualizationMarkers(cfg)

    # ─────────────────────────────────────────────────────────────────────────
    # Scene setup — mirrors jetbot pattern
    # ─────────────────────────────────────────────────────────────────────────
    def _setup_scene(self):
        # Terrain
        self.cfg.terrain.num_envs    = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # Lighting
        if hasattr(self.cfg, "sky_light") and self.cfg.sky_light is not None:
            self.cfg.sky_light.spawn.func(
                self.cfg.sky_light.prim_path,
                self.cfg.sky_light.spawn,
            )

        # Robot
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        # Ball
        self._ball = RigidObject(self.cfg.ball)
        self.scene.rigid_objects["ball"] = self._ball

        # Camera
        self._camera = TiledCamera(self.cfg.tiled_camera)
        self.scene.sensors["tiled_camera"] = self._camera

        self.scene.clone_environments(copy_from_source=False)

    # ─────────────────────────────────────────────────────────────────────────
    # Action pipeline — same P-controller altitude hold as all drone envs
    # ─────────────────────────────────────────────────────────────────────────
    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone().clamp(-1.0, 1.0)

        # Forward velocity in body frame
        lin_vel_b = torch.zeros(self.num_envs, 3, device=self.device)
        lin_vel_b[:, 0] = self._actions[:, 0] * self.cfg.max_forward_vel

        # Altitude P-controller
        target_z  = self.cfg.hover_height + self._terrain.env_origins[:, 2]
        current_z = self._robot.data.root_pos_w[:, 2]
        vz = (self.cfg.altitude_kp * (target_z - current_z)).clamp(
            -self.cfg.max_altitude_vel, self.cfg.max_altitude_vel
        )

        # Body → world
        quat_w    = self._robot.data.root_state_w[:, 3:7]
        lin_vel_w = _quat_rotate(quat_w, lin_vel_b)
        lin_vel_w[:, 2] = vz

        ang_vel_w = torch.zeros(self.num_envs, 3, device=self.device)
        ang_vel_w[:, 2] = self._actions[:, 1] * self.cfg.max_yaw_rate

        self._robot.write_root_velocity_to_sim(
            torch.cat([lin_vel_w, ang_vel_w], dim=-1)
        )

        # Spin rotors (cosmetic)
        jv = torch.zeros_like(self._robot.data.joint_vel)
        jv[:, 0], jv[:, 1] =  200.0, -200.0
        jv[:, 2], jv[:, 3] =  200.0, -200.0
        self._robot.set_joint_velocity_target(jv)

        # Update ball marker
        self._ball_marker.visualize(self._ball_pos_w)
        self._drone_marker.visualize(
            self._robot.data.root_pos_w,
            self._robot.data.root_quat_w,
        )

    def _apply_action(self):
        pass   # velocity written in _pre_physics_step

    # ─────────────────────────────────────────────────────────────────────────
    # Observations — mirrors jetbot exactly
    # ─────────────────────────────────────────────────────────────────────────
    def _get_observations(self) -> dict:
        """
        Returns {"policy": (N, T, H, W, 4)}.

        Channel layout:
            0-2: RGB normalised [0,1]
            3:   search_active (1.0 if ball not visible, 0.0 if visible)
                 expanded spatially so CNN can process it per-pixel.

        The search_active channel plays the same role as the jetbot's
        goal vector — it gives the policy non-visual state information
        embedded into the spatial structure the CNN expects.
        """
        # Raw RGB from camera: (N, H, W, 3) uint8 or float
        raw_rgb = self._camera.data.output["rgb"]
        if raw_rgb.dtype == torch.uint8:
            rgb_f = raw_rgb.float() / 255.0
        else:
            rgb_f = raw_rgb.float().clamp(0.0, 1.0)
        # rgb_f: (N, H, W, 3)

        # Detect yellow ball in current frame
        visible, cx_norm, area = _detect_yellow(rgb_f, self.cfg.yellow_threshold)
        self._ball_visible = visible
        self._ball_cx      = cx_norm    # (N,) horizontal centroid, [-1, 1]
        self._searching    = ~visible   # (N,) True = searching

        # Build search_active channel: (N, H, W, 1)
        N, H, W = rgb_f.shape[:3]
        search_ch = self._searching.float().view(N, 1, 1, 1).expand(N, H, W, 1)

        # Combine: (N, H, W, 4)
        frame = torch.cat([rgb_f, search_ch], dim=-1)

        # Build history stack — same as jetbot
        if self._camera_hist is None:
            self._camera_hist = (
                frame.unsqueeze(1)
                .repeat(1, self.cfg.history_len, 1, 1, 1)
                .contiguous()
            )
        else:
            self._camera_hist = torch.cat(
                [self._camera_hist[:, 1:], frame.unsqueeze(1)], dim=1
            )

        return {"policy": self._camera_hist}

    # ─────────────────────────────────────────────────────────────────────────
    # Rewards — vision-based, ball world position never used for reward
    # ─────────────────────────────────────────────────────────────────────────
    def _get_rewards(self) -> torch.Tensor:
        """
        Five reward components, each serving a different phase of the task:

        1. approach_reward:   reward for getting closer to ball
                              (only when visible AND farther than stop_distance)
        2. alignment_reward:  reward for keeping ball centred in frame
                              (yaw correction signal)
        3. hover_reward:      reward for hovering at exactly stop_distance
                              (the actual goal behaviour)
        4. search_reward:     small reward for yawing while searching
                              (prevents the drone from just hovering mid-air)
        5. success_bonus:     one-time bonus for first reaching stop_distance

        Note: we compute actual distance to ball here (the reward function
        can see world state). The POLICY can not — it only sees RGB.
        """
        pos  = self._robot.data.root_pos_w
        dist = torch.linalg.norm(pos - self._ball_pos_w, dim=-1)   # (N,)

        # ── 1. Approach reward ───────────────────────────────────────────────
        # Reward for closing distance, but only when outside stop_distance.
        # When inside stop_distance, approach is NOT rewarded (prevents overshoot).
        dist_delta     = self._prev_dist - dist
        self._prev_dist = dist.clone()
        beyond_stop    = (dist > self.cfg.stop_distance).float()
        approach_r     = dist_delta * beyond_stop * self.cfg.approach_scale

        # ── 2. Alignment reward ──────────────────────────────────────────────
        # Ball centred in frame (cx_norm ≈ 0) = high reward.
        # Not visible = zero reward (don't reward for accidental centring).
        centred    = (1.0 - self._ball_cx.abs()).clamp(0.0, 1.0)
        alignment_r = centred * self._ball_visible.float() * self.cfg.alignment_scale

        # ── 3. Hover reward ──────────────────────────────────────────────────
        # Reward for being at stop_distance (±0.2m tolerance).
        # This is the hardest reward to earn — requires both approach AND stop.
        at_stop    = (dist - self.cfg.stop_distance).abs() < 0.2
        hover_r    = at_stop.float() * self._ball_visible.float() * self.cfg.hover_scale

        # ── 4. Search reward ─────────────────────────────────────────────────
        # Small reward for rotating (yaw action) when searching.
        # Prevents degenerate "hover and do nothing" during search.
        yaw_action  = self._actions[:, 1].abs()
        search_r    = yaw_action * self._searching.float() * self.cfg.search_scale

        # ── 5. Success bonus ─────────────────────────────────────────────────
        just_succeeded = at_stop & self._ball_visible & ~self._succeeded
        self._succeeded |= just_succeeded
        success_r = just_succeeded.float() * self.cfg.success_bonus

        # ── Time penalty ─────────────────────────────────────────────────────
        time_r = torch.full(
            (self.num_envs,), self.cfg.time_penalty, device=self.device
        )

        total = approach_r + alignment_r + hover_r + search_r + success_r + time_r

        self.extras["log"] = {
            "visible_pct":  float(self._ball_visible.float().mean() * 100),
            "dist_mean":    float(dist.mean()),
            "at_stop_pct":  float(at_stop.float().mean() * 100),
            "success_rate": float(self._succeeded.float().mean()),
            "searching_pct":float(self._searching.float().mean() * 100),
        }

        return total

    # ─────────────────────────────────────────────────────────────────────────
    # Termination
    # ─────────────────────────────────────────────────────────────────────────
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # Altitude failure
        alt_fail = (
            (self._robot.data.root_pos_w[:, 2] < 0.1) |
            (self._robot.data.root_pos_w[:, 2] > 3.0)
        )

        # Drone too far from ball (flew the wrong way for too long)
        dist = torch.linalg.norm(
            self._robot.data.root_pos_w - self._ball_pos_w, dim=-1
        )
        too_far = dist > 8.0

        died = alt_fail | too_far
        return died, time_out

    # ─────────────────────────────────────────────────────────────────────────
    # Reset — mirrors jetbot _reset_idx
    # ─────────────────────────────────────────────────────────────────────────
    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        N_reset = len(env_ids)
        origins = self._terrain.env_origins[env_ids]

        # ── Reset drone ──────────────────────────────────────────────────────
        state = self._robot.data.default_root_state[env_ids].clone()
        state[:, :3] = origins + torch.tensor(
            [0.0, 0.0, self.cfg.hover_height], device=self.device
        )

        # Random spawn yaw — drone doesn't start facing the ball
        rand_yaw = torch.rand(N_reset, device=self.device) * 2 * math.pi
        half     = rand_yaw * 0.5
        state[:, 3] = torch.cos(half)   # w
        state[:, 4] = 0.0               # x
        state[:, 5] = 0.0               # y
        state[:, 6] = torch.sin(half)   # z
        state[:, 7:] = 0.0

        self._robot.write_root_pose_to_sim(state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(state[:, 7:], env_ids)
        jp = self._robot.data.default_joint_pos[env_ids]
        jv = self._robot.data.default_joint_vel[env_ids]
        self._robot.write_joint_state_to_sim(jp, jv, None, env_ids)

        # ── Reset ball ───────────────────────────────────────────────────────
        # Random position in a ring around the drone
        radii  = torch.empty(N_reset, device=self.device).uniform_(
            self.cfg.ball_min_dist, self.cfg.ball_max_dist
        )
        thetas = torch.empty(N_reset, device=self.device).uniform_(
            -math.pi, math.pi
        )

        ball_state = self._ball.data.default_root_state[env_ids].clone()
        ball_state[:, 0] = origins[:, 0] + radii * torch.cos(thetas)
        ball_state[:, 1] = origins[:, 1] + radii * torch.sin(thetas)
        ball_state[:, 2] = 0.15   # ball radius — sits on ground
        ball_state[:, 3] = 1.0    # quaternion w
        ball_state[:, 4:7] = 0.0

        self._ball.write_root_state_to_sim(ball_state, env_ids)

        # Cache ball world positions for reward computation
        self._ball_pos_w[env_ids, 0] = ball_state[:, 0]
        self._ball_pos_w[env_ids, 1] = ball_state[:, 1]
        self._ball_pos_w[env_ids, 2] = ball_state[:, 2]

        # ── Reset bookkeeping ────────────────────────────────────────────────
        if self._camera_hist is not None:
            self._camera_hist[env_ids] = 0.0

        self._succeeded[env_ids] = False
        self._searching[env_ids] = True

        # prev_dist from reset position
        dist0 = torch.linalg.norm(
            self._robot.data.root_pos_w[env_ids] - self._ball_pos_w[env_ids],
            dim=-1,
        )
        self._prev_dist[env_ids] = dist0


# =============================================================================
# UTILITY
# =============================================================================

def _quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors v by quaternions q. Isaac Lab convention: (w, x, y, z)."""
    w   = q[:, 0:1]
    xyz = q[:, 1:]
    t   = 2.0 * torch.linalg.cross(xyz, v)
    return v + w * t + torch.linalg.cross(xyz, t)