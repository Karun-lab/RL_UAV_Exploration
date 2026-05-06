"""
iris_door_env.py
================
Iris drone learns to escape a walled room by finding and flying through
a gap in the walls using a forward-facing depth camera.

The task:
    The drone spawns inside a rectangular room defined by WALL_DEFS.
    One gap in the perimeter acts as the exit.  The drone must:
        1. Search  — yaw when no large depth opening is visible
        2. Align   — yaw to centre the opening in frame
        3. Approach — fly toward the opening
        4. Escape  — cross the exit threshold  →  episode success

Room layout (top-down, metres, local to each env origin):
    +-----12.4m------+
    |                |     ← N wall  y=+3.0
    |                |
    8.4m gap→        |     ← S wall  y=-3.0  (gap on left side)
    |                |
    +----------------+
    W wall x=-6.0    E wall x=+6.0

Actions : [vx, yaw_rate]  in [-1, 1]   (same as ball task)
Obs     : (T=3, 64, 64, 2)  depth (normalised 0-1) + search_active channel
"""

from __future__ import annotations
import math
from typing import List, Tuple

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCamera, TiledCameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import gymnasium as gym
from isaaclab_assets.robots.iris import IRIS_CFG


# =============================================================================
# ROOM DEFINITION
# =============================================================================

# Each tuple: (centre_x, centre_y, size_x, size_y)  in metres, env-local
WALL_DEFS: List[Tuple[float, float, float, float]] = [
    # ---- Outer perimeter ----
    ( 0.0,  3.0, 12.4, 0.4),   # North wall
    ( 1.7, -3.0,  8.4, 0.4),   # South wall  (gap on the west side ≈ 2.6 m wide)
    (-6.0,  0.0,  0.4, 6.4),   # West wall
    ( 6.0,  0.0,  0.4, 6.4),   # East wall
    (-5.5, -3.0,  0.8, 0.4),   # South-west stub (closes the west end of south wall)
]

# Gap centre (env-local): the opening is between x ≈ -5.1 and x ≈ -2.1
# on the south wall (y = -3.0).  Mid-gap ≈ (-3.6, -3.0).
# We reward the drone for crossing y < -3.0 with |x + 3.6| < 1.3.
EXIT_CENTRE_X:  float = -3.6
EXIT_CENTRE_Y:  float = -3.0
EXIT_HALF_W:    float = 1.3   # half-width of the opening


# =============================================================================
# CONFIG
# =============================================================================

@configclass
class IrisDoorEnvCfg(DirectRLEnvCfg):

    episode_length_s = 40.0   # slightly longer — room navigation is harder
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
        env_spacing=20.0,   # larger spacing — room is ~12 m wide
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

    # Depth camera — forward-facing, same resolution as ball task
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/quadrotor/body/TrackCam",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.1, 0.0, 0.0),
            rot=(-0.5, -0.5, 0.5, 0.5),
            convention="opengl",
        ),
        data_types=["distance_to_camera"],   # ← depth instead of RGB
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 12.0),      # clip at room diameter
        ),
        width=64,
        height=64,
    )

    # Motion — identical to ball task
    max_forward_vel:  float = 1.5
    max_yaw_rate:     float = 2.0
    hover_height:     float = 0.9
    altitude_kp:      float = 3.0
    max_altitude_vel: float = 1.5

    # Observation: (T=3, H=64, W=64, C=2)  [depth, search_active]
    history_len:  int = 3
    num_channels: int = 2   # depth + search_active

    observation_space = gym.spaces.Box(
        low=0.0, high=1.0, shape=(3, 64, 64, 2), dtype=float
    )
    action_space  = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,))
    state_space   = gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(0,))

    # Depth opening detection
    depth_clip:       float = 12.0   # metres — matches camera clipping_range
    opening_thresh:   float = 0.15   # fraction of pixels that must be "far" to count as opening
    far_frac:         float = 0.80   # depth > far_frac of clip → "open" pixel

    # Drone spawn region (env-local, kept well inside the room)
    spawn_x_range: Tuple[float, float] = (-3.0, 3.0)
    spawn_y_range: Tuple[float, float] = (-1.5, 2.0)

    # Rewards
    approach_scale: float = 6.0    # reward closing horizontal distance to exit
    hover_scale:    float = 12.0   # continuous reward for being near the exit
    escape_bonus:   float = 150.0  # one-shot bonus for clearing the exit
    search_scale:   float = 0.05   # tiny reward for yawing while lost
    time_penalty:   float = -0.1
    near_exit_dist: float = 1.5    # metres — "close to exit" threshold


# =============================================================================
# DEPTH OPENING DETECTION
# =============================================================================

def _detect_opening(depth: torch.Tensor, cfg: IrisDoorEnvCfg):
    """
    Detect a large depth opening (gap/door) in a forward-facing depth image.

    depth : (N, H, W, 1) float, raw metres from camera, NaN/inf where invalid.

    Strategy:
        - Normalise to [0, 1] with clip distance.
        - A pixel is "open" (far away) when its normalised depth > far_frac.
          For a wall gap, those pixels cluster in the centre of the opening.
        - opening_visible: (N,) bool  — enough open pixels exist
        - cx_norm        : (N,) float — horizontal centroid in [-1, 1]
        - open_frac      : (N,) float — fraction of open pixels
    """
    d = depth[..., 0].clone()                          # (N, H, W)
    # Replace NaN / inf with clip (treat as wall at max distance)
    d = torch.nan_to_num(d, nan=cfg.depth_clip, posinf=cfg.depth_clip)
    d = d.clamp(0.0, cfg.depth_clip) / cfg.depth_clip  # normalise [0, 1]

    open_mask = d > cfg.far_frac                       # True where far = open space
    N, H, W   = d.shape
    open_frac = open_mask.float().sum(dim=(1, 2)) / (H * W)
    opening_visible = open_frac > cfg.opening_thresh

    # Horizontal centroid of open pixels
    col_idx  = torch.arange(W, device=d.device, dtype=torch.float32).view(1, 1, W).expand(N, H, W)
    mask_f   = open_mask.float()
    cx_pixel = (mask_f * col_idx).sum(dim=(1, 2)) / mask_f.sum(dim=(1, 2)).clamp(min=1.0)
    cx_norm  = ((cx_pixel / (W - 1)) * 2.0 - 1.0) * opening_visible.float()

    return opening_visible, cx_norm, open_frac, d   # also return normalised depth for obs


# =============================================================================
# WALL SPAWNER
# =============================================================================

def _spawn_walls(env_origins: torch.Tensor, wall_height: float = 2.0):
    """
    Spawn cuboid walls at each env origin using WALL_DEFS.
    Called once during scene setup.  Uses IsaacLab's USD cuboid prim.
    """
    import omni.usd
    from pxr import UsdGeom, Gf, UsdPhysics, PhysxSchema

    stage = omni.usd.get_context().get_stage()
    N     = env_origins.shape[0]

    for env_id in range(N):
        ox, oy, oz = (env_origins[env_id, 0].item(),
                      env_origins[env_id, 1].item(),
                      env_origins[env_id, 2].item())
        for w_idx, (cx, cy, sx, sy) in enumerate(WALL_DEFS):
            path = f"/World/Walls/env_{env_id}/wall_{w_idx}"
            xform: UsdGeom.Xform = UsdGeom.Xform.Define(stage, path)
            xform.AddTranslateOp().Set(Gf.Vec3d(ox + cx, oy + cy, oz + wall_height / 2.0))
            cube  = UsdGeom.Cube.Define(stage, path + "/cube")
            half  = Gf.Vec3d(sx / 2.0, sy / 2.0, wall_height / 2.0)
            cube.GetSizeAttr().Set(1.0)
            xform_op = cube.AddScaleOp()
            xform_op.Set(Gf.Vec3d(sx, sy, wall_height))

            # Physics
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
            PhysxSchema.PhysxCollisionAPI.Apply(cube.GetPrim())


# =============================================================================
# ENVIRONMENT
# =============================================================================

class IrisDoorEnv(DirectRLEnv):

    cfg: IrisDoorEnvCfg

    def __init__(self, cfg: IrisDoorEnvCfg, render_mode: str | None = None, **kwargs):
        self._camera_hist: torch.Tensor | None = None
        super().__init__(cfg, render_mode, **kwargs)

        N = self.num_envs

        # Per-env exit position in world frame (filled after scene setup)
        self._exit_pos_w   = torch.zeros(N, 2, device=self.device)  # (x, y) only
        self._prev_dist    = torch.zeros(N, device=self.device)

        # Detection state
        self._opening_visible = torch.zeros(N, dtype=torch.bool,  device=self.device)
        self._opening_cx      = torch.zeros(N,                     device=self.device)
        self._searching       = torch.ones(N,  dtype=torch.bool,   device=self.device)
        self._escaped         = torch.zeros(N, dtype=torch.bool,   device=self.device)
        self._actions         = torch.zeros(N, 2,                  device=self.device)
        self._step_count      = 0

        # Markers
        self._exit_marker  = self._make_marker(
            "/Visuals/ExitMarker", "sphere",
            sim_utils.SphereCfg(
                radius=0.3,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.3)),
            )
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
        self._camera = TiledCamera(self.cfg.tiled_camera)
        self.scene.articulations["robot"]  = self._robot
        self.scene.sensors["tiled_camera"] = self._camera
        self.scene.clone_environments(copy_from_source=False)

        # Spawn walls after cloning so each env gets its own walls
        _spawn_walls(self._terrain.env_origins)

    # ── Actions ───────────────────────────────────────────────────────────────
    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions   = actions.clone().clamp(-1.0, 1.0)
        lin_vel_b       = torch.zeros(self.num_envs, 3, device=self.device)
        lin_vel_b[:, 0] = self._actions[:, 0] * self.cfg.max_forward_vel

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

        # Markers
        exit_vis = torch.cat(
            [self._exit_pos_w,
             torch.full((self.num_envs, 1), self.cfg.hover_height, device=self.device)],
            dim=-1)
        self._exit_marker.visualize(exit_vis)
        drone_pos = self._robot.data.root_pos_w.clone()
        drone_pos[:, 2] += 0.3
        self._drone_marker.visualize(drone_pos, self._robot.data.root_quat_w)

    def _apply_action(self):
        pass

    # ── Observations ──────────────────────────────────────────────────────────
    def _get_observations(self) -> dict:
        # Raw depth: (N, H, W, 1), metres
        raw = self._camera.data.output["distance_to_camera"]
        depth_raw = raw.float()

        self._opening_visible, self._opening_cx, _, depth_norm = _detect_opening(
            depth_raw, self.cfg)
        self._searching = ~self._opening_visible

        N, H, W = depth_norm.shape
        depth_ch  = depth_norm.unsqueeze(-1)                            # (N, H, W, 1)
        search_ch = self._searching.float().view(N, 1, 1, 1).expand(N, H, W, 1)
        frame     = torch.cat([depth_ch, search_ch], dim=-1)            # (N, H, W, 2)

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

        pos_xy  = self._robot.data.root_pos_w[:, :2]
        dist    = torch.linalg.norm(pos_xy - self._exit_pos_w, dim=-1)

        # 1. Approach — reward closing XY distance to exit gap
        delta           = self._prev_dist - dist
        self._prev_dist = dist.clone()
        approach_r      = delta * self.cfg.approach_scale

        # 2. Proximity — continuous reward for being near the exit
        near_r = (dist < self.cfg.near_exit_dist).float() * self.cfg.hover_scale

        # 3. Escape — one-shot bonus for flying through the gap
        #    Condition: drone crosses y < exit_y AND is within gap half-width
        origin_y  = self._terrain.env_origins[:, 1]
        exit_y_w  = origin_y + EXIT_CENTRE_Y
        exit_x_w  = self._terrain.env_origins[:, 0] + EXIT_CENTRE_X
        past_wall = self._robot.data.root_pos_w[:, 1] < exit_y_w - 0.1
        in_gap    = (self._robot.data.root_pos_w[:, 0] - exit_x_w).abs() < EXIT_HALF_W
        just_escaped = past_wall & in_gap & ~self._escaped
        self._escaped |= just_escaped
        escape_r  = just_escaped.float() * self.cfg.escape_bonus

        # 4. Search reward — small incentive to keep yawing while lost
        search_r = self._searching.float() * self.cfg.search_scale

        # 5. Time penalty
        time_r = torch.full((self.num_envs,), self.cfg.time_penalty, device=self.device)

        total = approach_r + near_r + escape_r + search_r + time_r

        if self._step_count % 200 == 0:
            print(f"\n{'='*50}  step={self._step_count}")
            print(f"  dist_to_exit={float(dist.mean()):.2f}m  "
                  f"opening={float(self._opening_visible.float().mean()*100):.0f}%  "
                  f"escaped={float(self._escaped.float().mean()*100):.0f}%")
            for name, r in zip(
                ["approach", "near", "escape", "search", "time"],
                [approach_r, near_r, escape_r, search_r, time_r],
            ):
                print(f"  {name:<10} {float(r.mean()):+.4f}")
            print(f"  {'total':<10} {float(total.mean()):+.4f}")

        self.extras["log"] = {
            "opening_pct":  float(self._opening_visible.float().mean() * 100),
            "dist_mean":    float(dist.mean()),
            "escaped_pct":  float(self._escaped.float().mean() * 100),
        }
        return total

    # ── Termination ───────────────────────────────────────────────────────────
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # Altitude failure
        alt_fail = ((self._robot.data.root_pos_w[:, 2] < 0.1) |
                    (self._robot.data.root_pos_w[:, 2] > 3.5))

        # Wandered far outside expected area (sanity kill)
        origin_xy = self._terrain.env_origins[:, :2]
        too_far   = torch.linalg.norm(
            self._robot.data.root_pos_w[:, :2] - origin_xy, dim=-1) > 12.0

        # Success terminal: escaped!
        escaped = self._escaped

        terminated = alt_fail | too_far | escaped
        return terminated, time_out

    # ── Reset ─────────────────────────────────────────────────────────────────
    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        N_r     = len(env_ids)
        origins = self._terrain.env_origins[env_ids]

        # Drone: random position inside room, random yaw
        state        = self._robot.data.default_root_state[env_ids].clone()
        spawn_x = (torch.rand(N_r, device=self.device) *
                   (self.cfg.spawn_x_range[1] - self.cfg.spawn_x_range[0]) +
                   self.cfg.spawn_x_range[0])
        spawn_y = (torch.rand(N_r, device=self.device) *
                   (self.cfg.spawn_y_range[1] - self.cfg.spawn_y_range[0]) +
                   self.cfg.spawn_y_range[0])
        state[:, 0] = origins[:, 0] + spawn_x
        state[:, 1] = origins[:, 1] + spawn_y
        state[:, 2] = origins[:, 2] + self.cfg.hover_height
        half        = torch.rand(N_r, device=self.device) * math.pi
        state[:, 3] = torch.cos(half)
        state[:, 4] = 0.0
        state[:, 5] = 0.0
        state[:, 6] = torch.sin(half)
        state[:, 7:] = 0.0
        self._robot.write_root_pose_to_sim(state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(
            self._robot.data.default_joint_pos[env_ids],
            self._robot.data.default_joint_vel[env_ids],
            None, env_ids,
        )

        # Exit position (world frame, per env)
        self._exit_pos_w[env_ids, 0] = origins[:, 0] + EXIT_CENTRE_X
        self._exit_pos_w[env_ids, 1] = origins[:, 1] + EXIT_CENTRE_Y

        # Distance tracking
        pos_xy = self._robot.data.root_pos_w[env_ids, :2]
        self._prev_dist[env_ids] = torch.linalg.norm(
            pos_xy - self._exit_pos_w[env_ids], dim=-1)

        # Clear state
        if self._camera_hist is not None:
            self._camera_hist[env_ids] = 0.0
        self._escaped[env_ids]  = False
        self._searching[env_ids] = True


# =============================================================================
# UTILITY
# =============================================================================

def _quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector v by quaternion q.  q: (N,4) wxyz, v: (N,3)."""
    w, xyz = q[:, 0:1], q[:, 1:]
    t = 2.0 * torch.linalg.cross(xyz, v)
    return v + w * t + torch.linalg.cross(xyz, t)