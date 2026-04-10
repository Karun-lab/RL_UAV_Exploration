"""
iris_env.py  —  Spawn IRIS drone in a USD office environment.

To switch environment: change USD_PATH below.
To change spawn position: change SPAWN_POS below.

Run:
    from /workspace/isaaclab

    /isaac-sim/python.sh source/isaaclab_tasks/isaaclab_tasks/direct/iris/iris_env.py --livestream 2

"""

import argparse
from isaaclab.app import AppLauncher


# EDIT THESE to switch environment or spawn position
USD_PATH = (
    "/workspace/isaaclab/source/isaaclab_tasks/"
    "isaaclab_tasks/direct/iris/environments/TestEnvOfficeA.usd"
)
# Options:
#   TestEnvOfficeA.usd  /  TestEnvOfficeB.usd  /  TestEnvOfficeC.usd
#   TrainEnvOffice1.usd /  TrainEnvOffice2.usd  /  TrainEnvOffice3.usd  /  TrainEnvOffice4.usd

SPAWN_POS = (-10.0, 30.0, 1.2)   # (x, y, z) in metres — adjust if drone spawns inside geometry

NUM_ENVS = 1   # increase to spawn multiple drones

# App launch — must happen before any Isaac Lab imports
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Isaac Lab imports
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from rl_WorkSpace.models.drone import IRIS_CFG

# Scene
@configclass
class IrisOfficeCfg(InteractiveSceneCfg):

    office: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/Office",
        spawn=sim_utils.UsdFileCfg(usd_path=USD_PATH),
    )

    dome_light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=1500.0, color=(0.8, 0.8, 0.8)),
    )

    robot: ArticulationCfg = IRIS_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=IRIS_CFG.InitialStateCfg(pos=SPAWN_POS),
    )


# Simulation loop
def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    sim_dt = sim.get_physics_dt()
    step   = 0
    print("[INFO]: Running. Connect via WebRTC to view. Ctrl+C to exit.")

    while simulation_app.is_running():
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        step += 1

        # Print drone position every 5 seconds
        if step % int(5.0 / sim_dt) == 0:
            pos = scene["robot"].data.root_pos_w[0]
            print(f"[t={step*sim_dt:.0f}s]  drone: x={pos[0]:.2f}  y={pos[1]:.2f}  z={pos[2]:.2f}")

# Main
def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim     = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[5.0, 5.0, 4.0], target=SPAWN_POS)

    scene_cfg = IrisOfficeCfg(num_envs=NUM_ENVS, env_spacing=20.0)
    scene     = InteractiveScene(scene_cfg)
    sim.reset()

    print(f"[INFO]: Loaded: {USD_PATH}")
    print(f"[INFO]: Drone spawn: {SPAWN_POS}")
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()