from __future__ import annotations

import gymnasium as gym
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

from isaaclab_assets.robots.iris import IRIS_CFG
from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip


@configclass
class IrisEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 10.0
    decimation = 2

    # --- ACTION SPACE ---
    # 2 actions: [v_forward, yaw_rate]
    #   v_forward : [-1, 1] scaled to max_forward_vel  (body +x axis only)
    #   yaw_rate  : [-1, 1] scaled to max_yaw_rate     (rotation around world z)
    #
    # The drone cannot strafe (vy = 0) and altitude is managed by a P-controller,
    # not the policy. This matches a unicycle-style motion model suited to a
    # front-facing monocular camera.
    action_space = 2

    # --- OBSERVATION SPACE ---
    # 12 values (same layout as before so the SKRL network config still works):
    #   root_lin_vel_b      (3)  – current linear velocity in body frame
    #   root_ang_vel_b      (3)  – current angular velocity in body frame
    #   projected_gravity_b (3)  – gravity vector in body frame (attitude proxy)
    #   desired_pos_b       (3)  – goal position in body frame
    observation_space = 12
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

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=True)

    robot: ArticulationCfg = IRIS_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # --- VELOCITY LIMITS ---
    max_forward_vel = 1.5   # m/s  – maximum forward speed (body +x)
    max_yaw_rate    = 1.5   # rad/s – maximum yaw rate

    # --- ALTITUDE HOLD (P-controller, independent of policy) ---
    # The drone always tries to hover at this height above the terrain origin.
    hover_height     = 1.0   # metres above terrain origin
    altitude_kp      = 2.0   # proportional gain  (increase if drone drifts up/down)
    max_altitude_vel = 1.0   # m/s  – clamp on the altitude correction

    # reward scales
    lin_vel_reward_scale          = -0.05
    ang_vel_reward_scale          = -0.01
    distance_to_goal_reward_scale = 15.0
    # Heading reward: cosine similarity between drone forward axis and goal direction.
    # Positive value encourages the drone to face the goal before (or while) advancing.
    heading_reward_scale          = 2.0


# -----------------------
# ENV
# -----------------------
class IrisEnv(DirectRLEnv):
    cfg: IrisEnvCfg

    def __init__(self, cfg: IrisEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in ["lin_vel", "ang_vel", "distance_to_goal", "heading"]
        }

        self.set_debug_vis(self.cfg.debug_vis)

    # ------------------------------------------------------------------
    # Scene
    # ------------------------------------------------------------------
    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ------------------------------------------------------------------
    # Action pipeline
    # ------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor):
        """
        Convert 2-D policy output → world-frame velocity command.

        actions[:, 0]  →  v_forward  (body +x, scaled by max_forward_vel)
        actions[:, 1]  →  yaw_rate   (world +z, scaled by max_yaw_rate)

        Altitude is maintained by a proportional controller that is
        completely independent of the policy.
        """
        self._actions = actions.clone().clamp(-1.0, 1.0)

        # --- Forward velocity in body frame (+x only, no strafing) ---
        lin_vel_b = torch.zeros(self.num_envs, 3, device=self.device)
        lin_vel_b[:, 0] = self._actions[:, 0] * self.cfg.max_forward_vel

        # --- Altitude hold: P-controller ---
        target_z  = self.cfg.hover_height + self._terrain.env_origins[:, 2]
        current_z = self._robot.data.root_pos_w[:, 2]
        vz_world  = self.cfg.altitude_kp * (target_z - current_z)
        vz_world  = vz_world.clamp(-self.cfg.max_altitude_vel, self.cfg.max_altitude_vel)

        # --- Rotate body-frame forward velocity → world frame ---
        quat_w    = self._robot.data.root_state_w[:, 3:7]
        lin_vel_w = quat_rotate(quat_w, lin_vel_b)   # (N, 3)
        lin_vel_w[:, 2] = vz_world                    # override z with altitude hold

        # --- Yaw rate (pure rotation around world z-axis) ---
        ang_vel_w = torch.zeros(self.num_envs, 3, device=self.device)
        ang_vel_w[:, 2] = self._actions[:, 1] * self.cfg.max_yaw_rate

        self._cmd_lin_vel_w = lin_vel_w
        self._cmd_ang_vel_w = ang_vel_w
    def _apply_action(self):
    # Physics: sets actual drone motion
        velocity_cmd = torch.cat([self._cmd_lin_vel_w, self._cmd_ang_vel_w], dim=-1)
        self._robot.write_root_velocity_to_sim(velocity_cmd)

        # Cosmetic: spins the rotor meshes visually 
        joint_vel = torch.zeros_like(self._robot.data.joint_vel)
        joint_vel[:, 0] =  200.0
        joint_vel[:, 1] = -200.0
        joint_vel[:, 2] =  200.0
        joint_vel[:, 3] = -200.0
        self._robot.set_joint_velocity_target(joint_vel)

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------
    def _get_observations(self) -> dict:
        desired_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_state_w[:, :3],
            self._robot.data.root_state_w[:, 3:7],
            self._desired_pos_w,
        )
        obs = torch.cat(
            [
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                self._robot.data.projected_gravity_b,
                desired_pos_b,
            ],
            dim=-1,
        )
        return {"policy": obs}

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        lin_vel = torch.sum(torch.square(self._robot.data.root_lin_vel_b), dim=1)
        ang_vel = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)

        distance_to_goal = torch.linalg.norm(
            self._desired_pos_w - self._robot.data.root_pos_w, dim=1
        )
        distance_to_goal_mapped = 1 - torch.tanh(distance_to_goal / 0.8)

        heading_reward = self._compute_heading_reward()

        rewards = {
            "lin_vel":          lin_vel          * self.cfg.lin_vel_reward_scale          * self.step_dt,
            "ang_vel":          ang_vel          * self.cfg.ang_vel_reward_scale          * self.step_dt,
            "distance_to_goal": distance_to_goal_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt,
            "heading":          heading_reward   * self.cfg.heading_reward_scale          * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        for key, value in rewards.items():
            self._episode_sums[key] += value

        return reward

    def _compute_heading_reward(self) -> torch.Tensor:
        """
        Cosine similarity between the drone's forward axis and the horizontal
        direction to the goal.  Range: [-1, 1].  1 = perfectly aligned.

        Only the horizontal plane is considered so altitude difference doesn't
        confuse the heading signal.
        """
        quat_w = self._robot.data.root_state_w[:, 3:7]

        # Drone's forward unit vector in world frame
        forward_b = torch.zeros(self.num_envs, 3, device=self.device)
        forward_b[:, 0] = 1.0
        forward_w = quat_rotate(quat_w, forward_b)   # (N, 3)

        # Horizontal unit vector from drone to goal
        to_goal = self._desired_pos_w - self._robot.data.root_pos_w   # (N, 3)
        to_goal[:, 2] = 0.0                                            # project to XY plane
        to_goal_norm = torch.linalg.norm(to_goal, dim=1, keepdim=True).clamp(min=1e-6)
        to_goal_unit = to_goal / to_goal_norm

        return torch.sum(forward_w * to_goal_unit, dim=1)   # dot product = cos(angle)

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = torch.logical_or(
            self._robot.data.root_pos_w[:, 2] < 0.1,
            self._robot.data.root_pos_w[:, 2] > 2.0,
        )
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

        # Sample new goal positions
        self._desired_pos_w[env_ids, :2] = (
            torch.zeros_like(self._desired_pos_w[env_ids, :2]).uniform_(-2.0, 2.0)
            + self._terrain.env_origins[env_ids, :2]
        )
        self._desired_pos_w[env_ids, 2] = torch.zeros_like(
            self._desired_pos_w[env_ids, 2]
        ).uniform_(0.5, 1.5)

        # Reset robot state
        joint_pos          = self._robot.data.default_joint_pos[env_ids]
        joint_vel          = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    # ------------------------------------------------------------------
    # Debug visualisation
    # ------------------------------------------------------------------
    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            self.goal_pos_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        self.goal_pos_visualizer.visualize(self._desired_pos_w)


# ---------------------------------------------------------------------------
# Utility: rotate a batch of 3-D vectors by a batch of unit quaternions
# Isaac Lab quaternion convention: (w, x, y, z)
# ---------------------------------------------------------------------------
def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Args:
        q: (N, 4)  [w, x, y, z]
        v: (N, 3)
    Returns:
        (N, 3) rotated vectors
    """
    w   = q[:, 0:1]
    xyz = q[:, 1:]
    t   = 2.0 * torch.linalg.cross(xyz, v)
    return v + w * t + torch.linalg.cross(xyz, t)
