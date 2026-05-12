import argparse
import sys
sys.path.insert(0, "/workspace/isaaclab")
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train Iris with ICM.")
parser.add_argument("--num_envs", type=int, default=64)
# ← removed --headless, AppLauncher adds it automatically

AppLauncher.add_app_launcher_args(parser)   # adds --headless, --enable_cameras, etc.
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app 

import torch
import gymnasium as gym
from skrl.envs.wrappers.torch import wrap_env
from skrl.trainers.torch import SequentialTrainer
import rl_WorkSpace

device = torch.device("cuda:0")

from rl_WorkSpace.rl_envs.iris_icm_exploration import IrisICMOfficeEnvCfg
from rl_WorkSpace.rl_envs.iris_icm_office_agent import get_agent

cfg = IrisICMOfficeEnvCfg()
cfg.scene.num_envs = args_cli.num_envs

env = gym.make("Isaac-Iris-ICM-v0", cfg=cfg)
env = wrap_env(env)

agent = get_agent(env, device=device)

#trainer_cfg = {"timesteps": 1_000_000, "headless": True}
trainer_cfg = {"timesteps": 600_000, "headless": True}
#trainer_cfg = {"timesteps": 500_000, "headless": True}
trainer = SequentialTrainer(cfg=trainer_cfg, env=env, agents=agent)

try:
    trainer.train()
finally:
    env.close()
    simulation_app.close()