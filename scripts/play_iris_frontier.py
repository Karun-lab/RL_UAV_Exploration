"""
play_iris_frontier.py
=================
Custom play script for iris frontier exploration.
Uses Python agent/model directly — no YAML needed.

Run:
    CUDA_VISIBLE_DEVICES=0 /isaac-sim/python.sh \
        rl_WorkSpace/scripts/play_iris_frontier.py \
        --num_envs 4 \
        --livestream 2 \
        --enable_cameras
"""

import argparse
import os
import sys

sys.path.insert(0, "/workspace/isaaclab")

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs",   type=int,  default=4)
parser.add_argument("--checkpoint", type=str,  default=None)
parser.add_argument("--log_dir",    type=str,  default="logs/skrl/iris_frontier")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher   = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- imports after app launch ---
import torch
import gymnasium as gym
from skrl.envs.wrappers.torch import wrap_env
from skrl.trainers.torch import SequentialTrainer

import rl_WorkSpace  # triggers gym.register()

from rl_WorkSpace.rl_envs.iris_frontier_env import IrisFrontierEnvCfg
from rl_WorkSpace.rl_envs.iris_frontier_agent import get_agent


def find_checkpoint(log_dir: str, requested: str | None) -> str:
    if requested:
        if not os.path.isfile(requested):
            raise FileNotFoundError(f"Checkpoint not found: {requested}")
        return requested

    if not os.path.isdir(log_dir):
        raise FileNotFoundError(f"Log dir not found: {log_dir}")

    runs = sorted([
        d for d in os.listdir(log_dir)
        if os.path.isdir(os.path.join(log_dir, d))
    ])
    if not runs:
        raise FileNotFoundError(f"No runs in {log_dir}")

    ckpt_dir = os.path.join(log_dir, runs[-1], "checkpoints")
    best = os.path.join(ckpt_dir, "best_agent.pt")
    if os.path.isfile(best):
        return best

    ckpts = sorted([f for f in os.listdir(ckpt_dir) if f.endswith(".pt")])
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints in {ckpt_dir}")
    return os.path.join(ckpt_dir, ckpts[-1])


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Build env
    cfg = IrisFrontierEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    env_raw = gym.make("Isaac-Iris-Frontier-v0", cfg=cfg)
    env     = wrap_env(env_raw)

    # Build agent using your Python agent file
    agent = get_agent(env, device=device)

    # Find and load checkpoint
    ckpt_path = find_checkpoint(args_cli.log_dir, args_cli.checkpoint)
    print(f"[Play] Loading: {ckpt_path}")
    agent.load(ckpt_path)   # SKRL's built-in load handles the key structure
    agent.set_running_mode("eval")

    # Play loop
    states, _ = env.reset()
    print("[Play] Running. Connect via WebRTC. Ctrl+C to stop.")

    try:
        while simulation_app.is_running():
            with torch.no_grad():
                actions, _, _ = agent.policy.act(
                    {"states": agent._state_preprocessor(states)},
                    role="policy",
                )
            states, _, terminated, truncated, _ = env.step(actions)
            if (terminated | truncated).any():
                states, _ = env.reset()
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()