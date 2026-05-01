"""
iris_ball_env_cfg.py
====================
Config for Iris drone tracking a yellow ball.

Observation space: (T, H, W, 4) stacked RGB frames + search_state channel
    - 3 RGB channels (normalised 0-1)
    - 1 state channel (search yaw rate, expanded spatially like jetbot goal vec)
    Shape: (history_len=3, 64, 64, 4)

Action space: [vx, yaw_rate] in [-1, 1]
    - vx:       forward velocity (body frame)
    - yaw_rate: yaw angular velocity

The drone hovers at fixed altitude via P-controller.
No depth, no obstacles. Clean visual tracking task.
"""
 
import gymnasium as gym
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots.iris import IRIS_CFG


@configclass
class IrisBallEnvCfg(DirectRLEnvCfg):

    # ── Episode ───────────────────────────────────────────────────────────────
    episode_length_s = 30.0
    decimation       = 2        # 100 Hz sim → 50 Hz control

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

    # ── Scene ────────────────────────────────────────────────────────────────
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=64,
        env_spacing=12.0,
        replicate_physics=True,
    )

    # ── Terrain ──────────────────────────────────────────────────────────────
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

    # ── Lighting ─────────────────────────────────────────────────────────────
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=1500.0,
            color=(1.0, 0.98, 0.95),
        ),
    )

    # ── Robot ────────────────────────────────────────────────────────────────
    robot: ArticulationCfg = IRIS_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )

    # ── Yellow ball ───────────────────────────────────────────────────────────
    # Bright yellow sphere — highly visible in RGB, easy to detect with
    # colour thresholding in the reward function.
    # Ball sits on the ground (radius=0.15 → z=0.15).
    ball: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Ball",
        spawn=sim_utils.SphereCfg(
            radius=0.15,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.95, 0.0),   # bright yellow
                emissive_color=(0.3, 0.28, 0.0),  # slight glow — helps detection
                roughness=0.5,
                metallic=0.0,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,   # ball doesn't move — drone tracks it
                disable_gravity=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=False,  # no physics collision — reward-based stop
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(3.0, 0.0, 0.15)),
    )

    # ── Camera ───────────────────────────────────────────────────────────────
    # Forward-facing camera on the drone body.
    # 64×64 RGB — same as jetbot, small enough to train fast.
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/quadrotor/body/TrackCam",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.1, 0.0, 0.0),
            rot=(-0.5, -0.5, 0.5, 0.5),   # forward-facing
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

    # ── Motion ───────────────────────────────────────────────────────────────
    max_forward_vel:  float = 1.5    # m/s  — slightly faster than exploration
    max_yaw_rate:     float = 2.0    # rad/s
    hover_height:     float = 0.6    # m    — low hover so ball is in frame
    altitude_kp:      float = 3.0
    max_altitude_vel: float = 1.5

    # ── Observation ──────────────────────────────────────────────────────────
    history_len:  int = 3    # frames to stack — gives velocity perception
    img_width:    int = 64
    img_height:   int = 64
    # 3 RGB + 1 state channel (search_active flag expanded spatially)
    num_channels: int = 4

    observation_space = gym.spaces.Box(
        low=0.0, high=1.0,
        shape=(history_len, img_height, img_width, num_channels),
        dtype=float,
    )
    action_space  = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,))
    state_space   = gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(0,))

    # ── Ball spawn ───────────────────────────────────────────────────────────
    ball_min_dist: float = 2.5    # minimum spawn distance from drone
    ball_max_dist: float = 5.0    # maximum spawn distance from drone

    # ── Rewards ──────────────────────────────────────────────────────────────
    # Stop distance: drone should hover this far from the ball
    # (ball radius 0.15 + drone ~0.2 + buffer 0.1 = ~0.45m)
    stop_distance:     float = 0.5    # metres — target hover distance

    # Reward scales
    approach_scale:    float = 3.0    # reward for closing distance (beyond stop_dist)
    alignment_scale:   float = 2.0    # reward for ball centred in frame
    hover_scale:       float = 1.5    # reward for holding stop_distance
    search_scale:      float = 0.5    # small reward for rotating during search
    success_bonus:     float = 50.0   # one-time bonus for reaching stop_distance
    time_penalty:      float = -0.02  # per-step cost

    # Detection threshold — pixel fraction of yellow in frame
    # that counts as "ball visible"
    yellow_threshold:  float = 0.005  # 0.5% of pixels = ball visible

    # Search behaviour: yaw rate applied when ball not visible
    search_yaw_rate:   float = 1.0    # rad/s — slow rotation to find ball