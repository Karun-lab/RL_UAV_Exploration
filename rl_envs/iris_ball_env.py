"""
iris_ball_env.py
================
Iris drone learns to find and hover near a yellow ball using RGB vision.

Behaviours learned (in order of emergence):
    1. Search  — yaw when ball not visible
    2. Align   — yaw to centre ball in frame
    3. Approach — fly toward ball
    4. Stop    — hold hover at stop_distance

Actions : [vx, yaw_rate]  in [-1, 1]
Obs     : (T=3, 64, 64, 4) stacked RGB + search_active channel
"""

from __future__ import annotations
import math
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCamera, TiledCameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import gymnasium as gym
from rl_WorkSpace.models.drone.iris import IRIS_CFG


# =============================================================================
# CONFIG
# =============================================================================

@configclass
class IrisBallEnvCfg(DirectRLEnvCfg):

    episode_length_s = 10.0
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
        num_envs=64,
        env_spacing=12.0,
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

    sky_light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=1500.0, color=(1.0, 0.98, 0.95)),
    )

    robot: ArticulationCfg = IRIS_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )

    ball: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Ball",
        spawn=sim_utils.SphereCfg(
            radius=0.15,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.95, 0.0),
                emissive_color=(0.3, 0.28, 0.0),
                roughness=0.5,
                metallic=0.0,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(3.0, 0.0, 0.9)),
    )

    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/quadrotor/body/TrackCam",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.1, 0.0, 0.0),
            rot=(-0.5, -0.5, 0.5, 0.5),
            convention="opengl",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 20.0),
        ),
        width=64,
        height=64,
    )

    # Motion
    max_forward_vel:  float = 1.5
    max_yaw_rate:     float = 2.0
    hover_height:     float = 0.9
    altitude_kp:      float = 3.0
    max_altitude_vel: float = 1.5

    # Observation: (T=3, H=64, W=64, C=4)
    history_len:  int = 3
    num_channels: int = 4

    observation_space = gym.spaces.Box(
        low=0.0, high=1.0, shape=(3, 64, 64, 4), dtype=float
    )
    action_space  = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,))
    state_space   = gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(0,))

    # Ball spawn
    ball_min_dist: float = 1.5
    ball_max_dist: float = 5.5 

    # Rewards — optimised for fast learning
    # Higher approach + alignment scales surface the task signal early.
    # Higher hover scale makes stopping clearly better than overshooting.
    # Low search scale just prevents idle hovering — not a training objective.
    # Tighter time penalty forces efficiency without crushing exploration.
    # stop_distance:   float = 0.5
    # approach_scale:  float = 8.0    # was 3.0
    alignment_scale: float = 3.0    # was 2.0
    # hover_scale:     float = 3.0    # was 1.5
    search_scale:    float = 0.05    # was 0.5 then 0.2 
    success_bonus:   float = 100.0  # was 50.0
    # time_penalty:    float = -0.05  # was -0.02
    yellow_threshold: float = 0.002

    approach_scale: float = 5.0
    hover_scale:    float = 10.0   # large — makes stopping at ball very attractive
    time_penalty:   float = -0.1
    stop_distance:  float = 0.6
 

# =============================================================================
# YELLOW DETECTION
# =============================================================================

def _detect_yellow(rgb: torch.Tensor, threshold: float):
    """
    Detect yellow pixels in (N, H, W, 3) float image in [0, 1].
    Yellow: high R+G, low B.
    Returns: visible (N,) bool, cx_norm (N,) in [-1,1], area (N,) float
    """
    r, g, b  = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mask     = (r > 0.6) & (g > 0.5) & (b < 0.35)
    N, H, W  = rgb.shape[:3]
    area     = mask.float().sum(dim=(1, 2)) / (H * W)
    visible  = area > threshold

    col_idx  = torch.arange(W, device=rgb.device, dtype=torch.float32).view(1, 1, W).expand(N, H, W)
    mask_f   = mask.float()
    cx_pixel = (mask_f * col_idx).sum(dim=(1, 2)) / mask_f.sum(dim=(1, 2)).clamp(min=1.0)
    cx_norm  = ((cx_pixel / (W - 1)) * 2.0 - 1.0) * visible.float()

    return visible, cx_norm, area


# =============================================================================
# ENVIRONMENT
# =============================================================================

class IrisBallEnv(DirectRLEnv):

    cfg: IrisBallEnvCfg

    def __init__(self, cfg: IrisBallEnvCfg, render_mode: str | None = None, **kwargs):
        self._camera_hist: torch.Tensor | None = None
        super().__init__(cfg, render_mode, **kwargs)

        N = self.num_envs
        self._ball_pos_w   = torch.zeros(N, 3, device=self.device)
        self._prev_dist    = torch.zeros(N, device=self.device)
        self._ball_visible = torch.zeros(N, dtype=torch.bool, device=self.device)
        self._ball_cx      = torch.zeros(N, device=self.device)
        self._searching    = torch.ones(N, dtype=torch.bool, device=self.device)
        self._succeeded    = torch.zeros(N, dtype=torch.bool, device=self.device)
        self._actions      = torch.zeros(N, 2, device=self.device)
        self._step_count   = 0

        self._ball_marker  = self._make_marker(
            "/Visuals/BallMarker", "sphere",
            sim_utils.SphereCfg(radius=0.18,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)))
        )
        self._drone_marker = self._make_marker(
            "/Visuals/DroneMarker", "arrow",
            sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                scale=(0.3, 0.3, 0.6),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.8, 1.0)),
            )
        )

    def _make_marker(self, path: str, key: str, shape_cfg) -> VisualizationMarkers:
        m = VisualizationMarkers(VisualizationMarkersCfg(
            prim_path=path, markers={key: shape_cfg}))
        m.set_visibility(True)
        return m

    # ── Scene ─────────────────────────────────────────────────────────────────
    def _setup_scene(self):
        self.cfg.terrain.num_envs    = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.cfg.sky_light.spawn.func(self.cfg.sky_light.prim_path,
                                      self.cfg.sky_light.spawn)
        self._robot  = Articulation(self.cfg.robot)
        self._ball   = RigidObject(self.cfg.ball)
        self._camera = TiledCamera(self.cfg.tiled_camera)
        self.scene.articulations["robot"]  = self._robot
        self.scene.rigid_objects["ball"]   = self._ball
        self.scene.sensors["tiled_camera"] = self._camera
        self.scene.clone_environments(copy_from_source=False)

    # ── Actions ───────────────────────────────────────────────────────────────
    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions    = actions.clone().clamp(-1.0, 1.0)
        lin_vel_b        = torch.zeros(self.num_envs, 3, device=self.device)
        lin_vel_b[:, 0]  = self._actions[:, 0] * self.cfg.max_forward_vel

        vz = (self.cfg.altitude_kp *
              (self.cfg.hover_height + self._terrain.env_origins[:, 2]
               - self._robot.data.root_pos_w[:, 2])
              ).clamp(-self.cfg.max_altitude_vel, self.cfg.max_altitude_vel)

        lin_vel_w       = _quat_rotate(self._robot.data.root_state_w[:, 3:7], lin_vel_b)
        lin_vel_w[:, 2] = vz
        ang_vel_w       = torch.zeros(self.num_envs, 3, device=self.device)
        ang_vel_w[:, 2] = self._actions[:, 1] * self.cfg.max_yaw_rate

        self._robot.write_root_velocity_to_sim(torch.cat([lin_vel_w, ang_vel_w], dim=-1))

        jv = torch.zeros_like(self._robot.data.joint_vel)
        jv[:, 0], jv[:, 1] =  200.0, -200.0
        jv[:, 2], jv[:, 3] =  200.0, -200.0
        self._robot.set_joint_velocity_target(jv)

        self._ball_marker.visualize(self._ball_pos_w)
        drone_pos       = self._robot.data.root_pos_w.clone()
        drone_pos[:, 2] += 0.3
        self._drone_marker.visualize(drone_pos, self._robot.data.root_quat_w)

    def _apply_action(self):
        pass

    # ── Observations ──────────────────────────────────────────────────────────
    def _get_observations(self) -> dict:
        raw = self._camera.data.output["rgb"]
        rgb = (raw.float() / 255.0 if raw.dtype == torch.uint8
               else raw.float().clamp(0.0, 1.0))

        self._ball_visible, self._ball_cx, _ = _detect_yellow(
            rgb, self.cfg.yellow_threshold)
        self._searching = ~self._ball_visible

        N, H, W   = rgb.shape[:3]
        search_ch = self._searching.float().view(N, 1, 1, 1).expand(N, H, W, 1)
        frame     = torch.cat([rgb, search_ch], dim=-1)   # (N, H, W, 4)

        if self._camera_hist is None:
            self._camera_hist = frame.unsqueeze(1).repeat(
                1, self.cfg.history_len, 1, 1, 1).contiguous()
        else:
            self._camera_hist = torch.cat(
                [self._camera_hist[:, 1:], frame.unsqueeze(1)], dim=1)

        return {"policy": self._camera_hist}

    # ── Rewards ───────────────────────────────────────────────────────────────
    def _get_rewards(self) -> torch.Tensor:
        self._step_count += 1
        dist = torch.linalg.norm(
            self._robot.data.root_pos_w - self._ball_pos_w, dim=-1)

        # 1. Approach — reward closing distance
        delta           = self._prev_dist - dist
        self._prev_dist = dist.clone()
        approach_r      = delta * self.cfg.approach_scale

        # 2. Hover — continuous reward for being within stop distance
        at_stop   = dist < self.cfg.stop_distance
        success_r = at_stop.float() * self.cfg.hover_scale

        # 3. Time penalty
        time_r = torch.full(
            (self.num_envs,), self.cfg.time_penalty, device=self.device)

        total = approach_r + success_r + time_r

        if self._step_count % 200 == 0:
            print(f"\n{'='*50}  step={self._step_count}")
            print(f"  dist={float(dist.mean()):.2f}m  "
                f"visible={float(self._ball_visible.float().mean()*100):.0f}%  "
                f"at_stop={float(at_stop.float().mean()*100):.0f}%")
            for n, r in zip(
                ["approach", "success", "time"],
                [approach_r, success_r, time_r],
            ):
                print(f"  {n:<10} {float(r.mean()):+.4f}")
            print(f"  {'total':<10} {float(total.mean()):+.4f}")

        self.extras["log"] = {
            "visible_pct": float(self._ball_visible.float().mean() * 100),
            "dist_mean":   float(dist.mean()),
            "at_stop_pct": float(at_stop.float().mean() * 100),
        }
        return total

    # ── Termination ───────────────────────────────────────────────────────────
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        alt_fail = ((self._robot.data.root_pos_w[:, 2] < 0.1) |
                    (self._robot.data.root_pos_w[:, 2] > 3.0))
        too_far  = torch.linalg.norm(
            self._robot.data.root_pos_w - self._ball_pos_w, dim=-1) > 8.0
        return alt_fail | too_far, time_out

    # ── Reset ─────────────────────────────────────────────────────────────────
    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        N_r     = len(env_ids)
        origins = self._terrain.env_origins[env_ids]

        # Drone: random yaw, fixed hover height
        state        = self._robot.data.default_root_state[env_ids].clone()
        state[:, :3] = origins + torch.tensor([0., 0., self.cfg.hover_height], device=self.device)
        half         = torch.rand(N_r, device=self.device) * math.pi
        state[:, 3]  = torch.cos(half)
        state[:, 4]  = 0.0
        state[:, 5]  = 0.0
        state[:, 6]  = torch.sin(half)
        state[:, 7:] = 0.0
        self._robot.write_root_pose_to_sim(state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(
            self._robot.data.default_joint_pos[env_ids],
            self._robot.data.default_joint_vel[env_ids],
            None, env_ids,
        )

        # Ball: random ring
        r      = torch.empty(N_r, device=self.device).uniform_(
            self.cfg.ball_min_dist, self.cfg.ball_max_dist)
        theta  = torch.empty(N_r, device=self.device).uniform_(-math.pi, math.pi)
        bstate = self._ball.data.default_root_state[env_ids].clone()
        bstate[:, 0] = origins[:, 0] + r * torch.cos(theta)
        bstate[:, 1] = origins[:, 1] + r * torch.sin(theta)
        bstate[:, 2] = 0.9
        bstate[:, 3] = 1.0
        bstate[:, 4:7] = 0.0
        self._ball.write_root_state_to_sim(bstate, env_ids)
        self._ball_pos_w[env_ids] = bstate[:, :3]

        if self._camera_hist is not None:
            self._camera_hist[env_ids] = 0.0
        self._succeeded[env_ids] = False
        self._searching[env_ids] = True
        self._prev_dist[env_ids] = torch.linalg.norm(
            self._robot.data.root_pos_w[env_ids] - self._ball_pos_w[env_ids], dim=-1)


# =============================================================================
# UTILITY
# =============================================================================

def _quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    w, xyz = q[:, 0:1], q[:, 1:]
    t = 2.0 * torch.linalg.cross(xyz, v)
    return v + w * t + torch.linalg.cross(xyz, t)