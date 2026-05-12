"""
test_office_spawn.py
====================
Standalone Isaac Sim script to verify:
    1. Drone spawns correctly inside the office at each SPAWN_TABLE location
    2. Depth camera produces valid output
    3. Depth values are in the expected range

Run:
    CUDA_VISIBLE_DEVICES=0 /isaac-sim/python.sh \\
        rl_WorkSpace/scripts/test_office_spawn.py --livestream 2

Controls (in livestream viewer):
    Press SPACE to cycle through spawn positions
    The depth image is printed to terminal as ASCII art

No RL, no training — pure verification.
"""

import argparse
import sys
import math
import numpy as np
sys.path.insert(0, "/workspace/isaaclab")

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test office spawn and depth camera")
parser.add_argument("--spawn_idx", type=int, default=0,
                    help="Which spawn point to use (0-4)")
parser.add_argument("--cycle_spawns", action="store_true", default=False,
                    help="Cycle through all spawn points every 100 steps")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── All imports AFTER AppLauncher ─────────────────────────────────────────────
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sensors import TiledCamera, TiledCameraCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab_assets.robots.iris import IRIS_CFG


# =============================================================================
# CONFIGURATION
# =============================================================================

OFFICE_USD_PATH = (
    "/workspace/isaaclab/rl_WorkSpace/models/environments/TestEnvOfficeC.usd"
)

# Five verified interior spawn positions
SPAWN_TABLE = [
    (-2.0, 56.0, 1.0,   0.0),   # spawn 0
    (-3.0, 39.5, 1.0,   0.0),   # spawn 1
    ( 4.0, 39.5, 1.0, 180.0),   # spawn 2
    (-3.0, 50.0, 1.0, -90.0),   # spawn 3
    ( 0.0, 58.0, 1.0, -90.0),   # spawn 4
]

CAM_H          = 64
CAM_W          = 80
CAM_MIN_DEPTH  = 0.2
CAM_MAX_DEPTH  = 6.0
HOVER_HEIGHT   = 1.0
STEPS_PER_SPAWN = 150   # steps to hold at each spawn before cycling


# =============================================================================
# SCENE SETUP
# =============================================================================

def setup_scene(sim: SimulationContext) -> tuple:
    """Build scene: office + drone + depth camera."""

    # Ground plane (needed for physics even with office)
    sim_utils.GroundPlaneCfg().func("/World/defaultGroundPlane", sim_utils.GroundPlaneCfg())

    # Dome light — needed for depth rendering in dark office
    sim_utils.DomeLightCfg(
        intensity=1500.0, color=(1.0, 1.0, 1.0)
    ).func("/World/Light", sim_utils.DomeLightCfg(intensity=1500.0, color=(1.0, 1.0, 1.0)))

    # Office USD
    print(f"[TEST] Loading office: {OFFICE_USD_PATH}")
    office_cfg = sim_utils.UsdFileCfg(usd_path=OFFICE_USD_PATH)
    office_cfg.func("/World/Office", office_cfg, translation=(0.0, 0.0, 0.0))
    print("[TEST] Office loaded")

    # Drone
    robot_cfg = IRIS_CFG.replace(prim_path="/World/Robot")
    robot = Articulation(robot_cfg)

    # Depth camera — same config as ICM env
    cam_cfg = TiledCameraCfg(
        prim_path="/World/Robot/quadrotor/body/TestCam",
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
            clipping_range=(0.2, 10.0),
        ),
        width=CAM_W,
        height=CAM_H,
    )
    camera = TiledCamera(cam_cfg)

    return robot, camera


def spawn_drone(robot: Articulation, spawn_idx: int):
    """Teleport drone to spawn position."""
    sx, sy, sz, syaw_deg = SPAWN_TABLE[spawn_idx]
    syaw_rad = math.radians(syaw_deg)
    half     = syaw_rad * 0.5

    # Build state tensor
    state    = robot.data.default_root_state.clone()
    state[0, 0] = sx
    state[0, 1] = sy
    state[0, 2] = sz
    state[0, 3] = math.cos(half)   # w
    state[0, 4] = 0.0              # x
    state[0, 5] = 0.0              # y
    state[0, 6] = math.sin(half)   # z
    state[0, 7:] = 0.0

    robot.write_root_pose_to_sim(state[:, :7])
    robot.write_root_velocity_to_sim(state[:, 7:])
    robot.write_joint_state_to_sim(
        robot.data.default_joint_pos,
        robot.data.default_joint_vel,
    )

    print(f"\n[TEST] ══ Spawn {spawn_idx} ══")
    print(f"  Position:  x={sx:.1f}  y={sy:.1f}  z={sz:.1f}")
    print(f"  Heading:   {syaw_deg:.0f}°")


# =============================================================================
# DEPTH ANALYSIS
# =============================================================================

def analyse_depth(camera: TiledCamera, spawn_idx: int, step: int):
    """Print depth statistics and ASCII visualisation."""
    raw = camera.data.output.get("distance_to_image_plane")

    if raw is None:
        print("[TEST] ⚠ No depth data received — camera may not be ready yet")
        return

    d = raw.float()
    if d.ndim == 4:
        d = d.squeeze(-1)   # (1, H, W)
    elif d.ndim == 2:
        d = d.unsqueeze(0)

    d_np = d[0].cpu().numpy()   # (H, W)

    # Replace invalid zeros with max range
    d_np = np.where(d_np == 0, CAM_MAX_DEPTH, d_np)
    d_np = np.clip(d_np, CAM_MIN_DEPTH, CAM_MAX_DEPTH)

    # Statistics
    valid = d_np[(d_np > CAM_MIN_DEPTH) & (d_np < CAM_MAX_DEPTH)]
    print(f"\n[TEST] Step {step} — Spawn {spawn_idx} depth stats:")
    print(f"  shape       : {d_np.shape}")
    print(f"  min         : {d_np.min():.3f}m")
    print(f"  max         : {d_np.max():.3f}m")
    print(f"  mean        : {d_np.mean():.3f}m")
    print(f"  valid pixels: {len(valid)} / {d_np.size} "
          f"({100*len(valid)/d_np.size:.0f}%)")

    if len(valid) == 0:
        print("  ⚠ ALL PIXELS INVALID — check camera prim path and office USD")
        return

    if d_np.min() < 0.3:
        print(f"  ⚠ Very close reading ({d_np.min():.2f}m) — drone may be inside geometry")
    elif d_np.min() > 4.0:
        print(f"  ⚠ All readings far ({d_np.min():.2f}m) — drone may be outside office")
    else:
        print(f"  ✓ Depth readings look valid for indoor environment")

    # ASCII depth map — middle rows, downsampled to 40 cols
    mid    = CAM_H // 2
    strip  = d_np[mid-2:mid+3, :]     # 5 middle rows
    avg    = strip.mean(axis=0)        # (W,) horizontal scan
    norm   = (avg - CAM_MIN_DEPTH) / (CAM_MAX_DEPTH - CAM_MIN_DEPTH)
    norm   = np.clip(norm, 0, 1)

    # Downsample to 40 chars wide
    step_  = max(1, len(norm) // 40)
    norm_d = norm[::step_][:40]

    chars  = " .:;+=xX$&#"
    bar    = "".join(chars[int(v * (len(chars)-1))] for v in norm_d)
    print(f"  depth strip : [{bar}]")
    print(f"               L=close(wall)                 R=far(open)")

    # Normalised depth frame for ICM compatibility check
    norm_full = (d_np - CAM_MIN_DEPTH) / (CAM_MAX_DEPTH - CAM_MIN_DEPTH)
    norm_full = np.clip(norm_full, 0, 1)
    print(f"  norm range  : [{norm_full.min():.3f}, {norm_full.max():.3f}]  "
          f"(should be 0–1 for CNN input)")

    # Save depth frame as npy for external visualisation
    save_path = f"/tmp/test_depth_spawn{spawn_idx}_step{step}.npy"
    np.save(save_path, norm_full)
    print(f"  saved to    : {save_path}  (load with np.load, view with plt.imshow)")


# =============================================================================
# MAIN LOOP
# =============================================================================

def main():
    # Simulation config — single env, no replication
    sim_cfg = SimulationCfg(
        dt=1/100,
        render_interval=2,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[10.0, 50.0, 5.0], target=[0.0, 55.0, 1.0])

    # Build scene
    robot, camera = setup_scene(sim)

    # Play (initialise physics)
    sim.reset()
    robot.reset()

    print("\n[TEST] ══════════════════════════════════════")
    print("[TEST] Office spawn + depth camera test")
    print("[TEST] ══════════════════════════════════════")
    print(f"[TEST] Spawn table has {len(SPAWN_TABLE)} positions")
    print(f"[TEST] Camera: {CAM_W}×{CAM_H} px, "
          f"depth range [{CAM_MIN_DEPTH}, {CAM_MAX_DEPTH}]m")
    print("[TEST] Depth saved to /tmp/test_depth_spawn*.npy each step")

    spawn_idx  = args_cli.spawn_idx
    step_count = 0

    # Initial spawn
    spawn_drone(robot, spawn_idx)
    robot.write_data_to_sim()
    sim.step()
    camera.update(sim.current_time)

    while simulation_app.is_running():
        # Cycle spawn positions
        if args_cli.cycle_spawns and step_count % STEPS_PER_SPAWN == 0 and step_count > 0:
            spawn_idx = (spawn_idx + 1) % len(SPAWN_TABLE)
            spawn_drone(robot, spawn_idx)

        # Hold altitude — simple P-controller
        current_z = robot.data.root_pos_w[0, 2].item()
        vz        = 3.0 * (HOVER_HEIGHT - current_z)
        vz        = max(-1.5, min(1.5, vz))

        vel = torch.zeros(1, 6, device=sim.device)
        vel[0, 2] = vz
        robot.write_root_velocity_to_sim(vel)
        robot.write_data_to_sim()

        # Step sim and update camera
        sim.step()
        robot.update(sim.cfg.dt)
        camera.update(sim.current_time)

        step_count += 1

        # Print depth analysis every 50 steps
        if step_count % 50 == 0:
            pos = robot.data.root_pos_w[0].cpu().numpy()
            print(f"\n[TEST] Drone position: "
                  f"x={pos[0]:.2f}  y={pos[1]:.2f}  z={pos[2]:.2f}")
            analyse_depth(camera, spawn_idx, step_count)

        # Exit after cycling all spawns once (if cycling)
        if args_cli.cycle_spawns and spawn_idx == len(SPAWN_TABLE) - 1 \
                and step_count % STEPS_PER_SPAWN == STEPS_PER_SPAWN - 1:
            print("\n[TEST] All spawn positions tested. Exiting.")
            break

    simulation_app.close()


if __name__ == "__main__":
    main()