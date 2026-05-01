"""
iris_ball_agent.py
==================
SKRL PPO agent for the Iris drone yellow ball tracking task.
Mirrors jetbot agent.py exactly — same structure, same hyperparameters,
adapted for the drone task.

Usage:
    from rl_WorkSpace.rl_envs.iris_ball_agent import get_agent
    agent = get_agent(env, device)
"""
 
import torch
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.resources.preprocessors.torch import RunningStandardScaler

from .iris_ball_models import IrisBallModel


def get_agent(env, device, experiment_cfg: dict | None = None):
    """
    Constructs the PPO agent for Iris ball tracking.
    Call this from your train script after creating the env.

    Example train script usage:
        env = gym.make("Isaac-Iris-Ball-v0", ...)
        env = wrap_env(env)
        agent = get_agent(env, device=env.device)
        trainer = SequentialTrainer(cfg=trainer_cfg, env=env, agents=agent)
        trainer.train()
    """

    rollout_length = 256   # steps per env before PPO update

    # ── Memory ───────────────────────────────────────────────────────────────
    memory = RandomMemory(
        memory_size=rollout_length,
        num_envs=env.num_envs,
        device=device,
    )

    # ── Model — shared backbone (same as jetbot) ──────────────────────────────
    model = IrisBallModel(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )
    models = {
        "policy": model,
        "value":  model,   # shared backbone
    }

    # ── PPO config ────────────────────────────────────────────────────────────
    cfg = PPO_DEFAULT_CONFIG.copy()

    cfg["rollouts"]        = rollout_length
    cfg["learning_epochs"] = 8
    cfg["mini_batches"]    = 8

    cfg["discount_factor"] = 0.99    # 30s episode at 50Hz = 1500 steps
                                     # γ=0.99 → horizon ≈ 100 steps (5s)
    cfg["lambda"]          = 0.95

    # Learning rate
    cfg["learning_rate"]                   = 3e-4
    cfg["learning_rate_scheduler"]         = KLAdaptiveRL
    cfg["learning_rate_scheduler_kwargs"]  = {"kl_threshold": 0.01}

    # Gradient and ratio clipping
    cfg["grad_norm_clip"]        = 1.0
    cfg["ratio_clip"]            = 0.2
    cfg["value_clip"]            = 0.2
    cfg["clip_predicted_values"] = True

    # Loss scaling
    # entropy 0.01: enough to prevent early collapse without being too noisy
    # The search behaviour needs some entropy to explore yaw directions
    cfg["entropy_loss_scale"] = 0.01
    cfg["value_loss_scale"]   = 1.0

    # Preprocessors
    # State is image-based — no scaler on input (normalised to [0,1] in env)
    cfg["state_preprocessor"] = None
    # Value scaler: same as jetbot — helps PPO with varying reward scales
    cfg["value_preprocessor"]        = RunningStandardScaler
    cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}

    # Logging
    cfg["experiment"]["write_interval"]      = 100
    cfg["experiment"]["checkpoint_interval"] = 500
    cfg["experiment"]["directory"]           = "logs/skrl/iris_ball"
    cfg["experiment"]["wandb"]               = False   # set True if you use wandb

    if experiment_cfg:
        cfg["experiment"].update(experiment_cfg)

    # ── Agent ─────────────────────────────────────────────────────────────────
    agent = PPO(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )

    return agent