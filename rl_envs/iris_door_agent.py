"""
iris_door_agent.py
==================
SKRL PPO agent for the Iris drone door-finding (escape room) task.
Mirrors iris_ball_agent.py exactly — same structure, same hyperparameters,
adapted for depth input and the door task.

Key difference from ball agent:
    - Observation: (N, T, H, W, 2)  [depth_norm, search_active]
      instead of   (N, T, H, W, 4)  [R, G, B, search_active]
    - CNN input channels: 2 instead of 4
    - Everything else (architecture, PPO config, mixins) is identical.

Usage:
    from rl_WorkSpace.rl_envs.iris_door_agent import get_agent
    agent = get_agent(env, device)
"""

import torch
import torch.nn as nn

from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin
from skrl.utils.spaces.torch import unflatten_tensorized_space


# =============================================================================
# MODEL
# =============================================================================

class IrisDoorModel(GaussianMixin, DeterministicMixin, Model):
    """
    Shared backbone actor-critic for the door-finding task.

    Input: (N, T, H, W, 2) observation stack
        - 2 channels: depth_normalised (1) + search_active (1)
        - T=3 stacked frames for motion/flow perception
        - H=W=64 image

    Architecture (identical to IrisBallModel except n_ch=2):
        For each of T frames:
            frame (N, H, W, 2) → CNN → feature vector (N, cnn_out)
        Concatenate T features: (N, T * cnn_out)
        → Fusion MLP → shared embedding (N, 256)
        → policy head: mean action (N, 2)
        → value head:  scalar value (N, 1)
    """

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        clip_actions=False,
        clip_log_std=True,
        min_log_std=-20.0,
        max_log_std=2.0,
        reduction="sum",
        **kwargs,
    ):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std,
                               min_log_std, max_log_std, reduction)
        DeterministicMixin.__init__(self, clip_actions)

        # ── Input shape ───────────────────────────────────────────────────────
        # observation_space.shape = (T, H, W, C) = (3, 64, 64, 2)
        try:
            self.t_steps = observation_space.shape[0]   # 3
            self.h       = observation_space.shape[1]   # 64
            self.w       = observation_space.shape[2]   # 64
            self.n_ch    = observation_space.shape[3]   # 2  ← depth + search_active
        except Exception:
            self.t_steps, self.h, self.w, self.n_ch = 3, 64, 64, 2

        # ── CNN (shared across time steps) ────────────────────────────────────
        # Identical to IrisBallModel's CNN except first conv: n_ch=2 channels.
        # 64 → 30 → 13 → 5 → 2  (spatial resolution with stride-2 convs)
        self.cnn = nn.Sequential(
            nn.Conv2d(self.n_ch, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute CNN output size with a dummy forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, self.n_ch, self.h, self.w)
            self.cnn_out = self.cnn(dummy).shape[1]

        # ── Fusion MLP ────────────────────────────────────────────────────────
        input_dim = self.t_steps * self.cnn_out

        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )

        # ── Heads ─────────────────────────────────────────────────────────────
        self.policy_mean = nn.Linear(256, action_space.shape[0])
        self.log_std     = nn.Parameter(torch.zeros(action_space.shape[0]))
        self.value_head  = nn.Linear(256, 1)

    # ── SKRL interface ────────────────────────────────────────────────────────

    def act(self, inputs, role):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)

    def compute(self, inputs, role=""):
        """
        Forward pass.

        inputs["states"]: flattened observation, unflattened back to
                          (N, T, H, W, 2) by SKRL's unflatten utility.
        """
        # ── 1. Unpack ─────────────────────────────────────────────────────────
        obs = unflatten_tensorized_space(
            self.observation_space, inputs.get("states")
        )
        # obs: (N, T, H, W, 2)

        # ── 2. CNN per time step ──────────────────────────────────────────────
        cnn_feats = []
        for t in range(self.t_steps):
            frame = obs[:, t, ...]            # (N, H, W, 2)
            frame = frame.permute(0, 3, 1, 2) # (N, 2, H, W) for Conv2d
            cnn_feats.append(self.cnn(frame)) # (N, cnn_out)

        # ── 3. Concatenate time steps ─────────────────────────────────────────
        visual_emb = torch.cat(cnn_feats, dim=1)  # (N, T * cnn_out)

        # ── 4. Fusion ─────────────────────────────────────────────────────────
        shared = self.net(visual_emb)             # (N, 256)

        # ── 5. Output ─────────────────────────────────────────────────────────
        if role == "policy":
            return self.policy_mean(shared), self.log_std, {}
        elif role == "value":
            return self.value_head(shared), {}
        else:
            return self.policy_mean(shared), self.log_std, {}


# =============================================================================
# AGENT FACTORY
# =============================================================================

def get_agent(env, device, experiment_cfg: dict | None = None):
    """
    Constructs the PPO agent for Iris door-finding (escape room).
    Call this from your train script after creating the env.

    Example train script usage:
        env = gym.make("Isaac-Iris-Door-v0", ...)
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

    # ── Model — shared backbone ───────────────────────────────────────────────
    model = IrisDoorModel(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )
    models = {
        "policy": model,
        "value":  model,   # shared backbone
    }

    # ── PPO config (identical to ball agent) ──────────────────────────────────
    cfg = PPO_DEFAULT_CONFIG.copy()

    cfg["rollouts"]        = rollout_length
    cfg["learning_epochs"] = 8
    cfg["mini_batches"]    = 8

    cfg["discount_factor"] = 0.99    # 40s episode at 50Hz = 2000 steps
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
    # entropy 0.01: encourages exploration of yaw directions to find the gap
    cfg["entropy_loss_scale"] = 0.01
    cfg["value_loss_scale"]   = 1.0

    # Preprocessors
    # Depth input is already normalised [0,1] in the env — no input scaler.
    cfg["state_preprocessor"] = None
    # Value scaler: helps PPO handle the escape bonus spike in returns.
    cfg["value_preprocessor"]        = RunningStandardScaler
    cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}

    # Logging
    cfg["experiment"]["write_interval"]      = 100
    cfg["experiment"]["checkpoint_interval"] = 500
    cfg["experiment"]["directory"]           = "logs/skrl/iris_door"
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