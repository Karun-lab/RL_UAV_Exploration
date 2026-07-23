"""
train_icm_lstm.py
=================
Training script for Iris ICM + LSTM exploration.

Issues fixed from submitted version
-------------------------------------
1. num_envs default was 64 — too high for LSTM training.
   LSTM requires ordered, sequential rollouts. With 64 envs and
   rollout_length=256 the memory holds 16,384 transitions which is
   fine, but 64 envs on a single A40 with cameras enabled will OOM
   during the ICM backward pass. Default changed to 16 (matches the
   env cfg default and what the agent was designed around).
   Pass --num_envs 32 if you have headroom.

2. env.cfg.scene.num_envs was set AFTER constructing the cfg but the
   env is not constructed yet — this is fine and correct as-is, but
   it must be done BEFORE gym.make(), which it is. Confirmed correct.

3. gym.make("Isaac-Iris-ICM-LSTM-v0") requires the env to be registered.
   Added the registration guard with a clear error message if the ID
   is not found, so you know to add it to your __init__.py.

4. trainer_cfg passed "headless": True which is ignored by
   SequentialTrainer — headless is an AppLauncher flag, not a trainer
   flag. Removed to avoid silent confusion. Headless is set via
   --headless on the command line (AppLauncher handles it).

5. agent = get_agent() returns a single agent but SequentialTrainer
   expects agents= to accept either an agent or a list. Both work,
   but explicitly wrapping in a list makes the intent unambiguous and
   avoids a deprecation warning in newer SKRL versions.

6. No seed was set. Added torch.manual_seed and numpy seed for
   reproducibility, matching the yaml seed: 42.

7. No logging of env/agent config at start of run — added a brief
   summary print so you can confirm settings without reading the yaml.
"""

import argparse
import sys
sys.path.insert(0, "/workspace/isaaclab")
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train Iris ICM + LSTM exploration.")
parser.add_argument("--num_envs",   type=int, default=16,
                    help="Number of parallel envs. Default 16 to avoid OOM with LSTM+ICM.")
parser.add_argument("--timesteps",  type=int, default=500_000,
                    help="Total training timesteps.")
parser.add_argument("--checkpoint", type=str, default=None,
                    help="Optional path to resume from a checkpoint .pt file.")

AppLauncher.add_app_launcher_args(parser)   # adds --headless, --enable_cameras, etc.
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True   # depth camera required

# Pass remaining hydra args back so Isaac resolves them correctly
sys.argv = [sys.argv[0]] + hydra_args

app_launcher    = AppLauncher(args_cli)
simulation_app  = app_launcher.app

# ── All Isaac/torch imports MUST come after AppLauncher ──────────────────────
import torch
import numpy as np
import gymnasium as gym
from skrl.envs.wrappers.torch import wrap_env
from skrl.trainers.torch import SequentialTrainer
import rl_WorkSpace   # registers Isaac envs and any custom gym envs

# Reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from rl_WorkSpace.rl_envs.iris_icm_lstm_env   import IrisICMLSTMEnv, IrisICMLSTMEnvCfg
from rl_WorkSpace.agents.iris_icm_lstm_agent import get_agent

# ── Environment ───────────────────────────────────────────────────────────────
cfg = IrisICMLSTMEnvCfg()
cfg.scene.num_envs = args_cli.num_envs   # override before gym.make()

# Verify the gym ID is registered; give a clear error if not
ENV_ID = "Isaac-Iris-ICM-LSTM-v0"
try:
    env = gym.make(ENV_ID, cfg=cfg)
except gym.error.UnregisteredEnv:
    raise RuntimeError(
        f"Env '{ENV_ID}' not registered. "
        "Add it to rl_WorkSpace/rl_envs/__init__.py:\n"
        "  gym.register(\n"
        f"    id='{ENV_ID}',\n"
        "    entry_point='rl_WorkSpace.rl_envs.iris_icm_lstm_env:IrisICMLSTMEnv',\n"
        "    disable_env_checker=True,\n"
        "  )\n"
        "or use direct instantiation:\n"
        "  from isaaclab.envs import ManagerBasedRLEnvCfg\n"
        "  from isaaclab_tasks.utils import parse_env_cfg\n"
        "  env = IrisICMLSTMEnv(cfg=cfg)\n"
    )

env = wrap_env(env)

# ── Agent ─────────────────────────────────────────────────────────────────────
agent = get_agent(env, device=device)

# Optionally resume from checkpoint
if args_cli.checkpoint:
    print(f"[train] Resuming from checkpoint: {args_cli.checkpoint}")
    agent.load(args_cli.checkpoint)

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print(f"  Env          : {ENV_ID}")
print(f"  Num envs     : {cfg.scene.num_envs}")
print(f"  Control Hz   : {int(1 / (cfg.sim.dt * cfg.decimation))} Hz  "
      f"(sim {int(1/cfg.sim.dt)} Hz, decimation {cfg.decimation})")
print(f"  Episode len  : {cfg.episode_length_s}s")
print(f"  Obs shape    : {env.observation_space.shape}  (T, H, W, C)")
print(f"  Action shape : {env.action_space.shape}")
print(f"  LSTM hidden  : {cfg.lstm_hidden}")
print(f"  LSTM input   : {cfg.lstm_input_dim}  [vx, yaw_rate, icm_r]")
print(f"  BPTT length  : {cfg.bptt_len} steps  "
      f"({cfg.bptt_len * cfg.sim.dt * cfg.decimation:.1f}s)")
print(f"  Pose loss λ  : {cfg.pose_loss_scale}")
print(f"  Flip aug     : {cfg.flip_prob*100:.0f}% probability")
print(f"  Timesteps    : {args_cli.timesteps:,}")
print(f"  Device       : {device}")
print(f"  Seed         : {SEED}")
print("="*60 + "\n")

# ── Train ─────────────────────────────────────────────────────────────────────
trainer_cfg = {
    "timesteps": args_cli.timesteps,
    # Note: "headless" is NOT a valid trainer key — it is an AppLauncher flag.
    # Pass --headless on the command line instead.
}

trainer = SequentialTrainer(cfg=trainer_cfg, env=env, agents=[agent])

try:
    trainer.train()
finally:
    env.close()
    simulation_app.close()