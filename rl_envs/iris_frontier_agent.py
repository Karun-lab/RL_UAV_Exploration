"""
iris_frontier_agent.py
======================
SKRL PPO agent + CNN+MLP model for frontier exploration.

Identical architecture to iris_ball_models/agent — CNN per frame,
concatenate T=3 frames, MLP fusion, shared policy/value heads.

Input: (N, T=3, H=64, W=80, C=2) — depth + coverage_bias
"""

import torch
import torch.nn as nn
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.utils.spaces.torch import unflatten_tensorized_space


# =============================================================================
# MODEL
# =============================================================================

class IrisFrontierModel(GaussianMixin, DeterministicMixin, Model):
    """
    CNN per depth frame → concatenate T=3 → MLP → heads.
    Identical structure to IrisBallModel, different input shape.

    Input:  (N, T=3, H=64, W=80, C=2)
    CNN:    processes each (H=64, W=80) frame with 2 channels
    Output: [vx, yaw_rate] in [-1, 1]
    """

    def __init__(self, observation_space, action_space, device,
                 clip_actions=False, clip_log_std=True,
                 min_log_std=-20.0, max_log_std=2.0,
                 reduction="sum", **kwargs):

        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std,
                               min_log_std, max_log_std, reduction)
        DeterministicMixin.__init__(self, clip_actions)

        # obs shape: (T=3, H=64, W=80, C=2)
        self.t_steps = 3
        self.h       = 64
        self.w       = 80
        self.n_ch    = 2

        # CNN — same structure as ball tracking
        # 64×80 → 30×38 → 13×17 → 5×7 → 2×3 → flatten
        self.cnn = nn.Sequential(
            nn.Conv2d(self.n_ch, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy        = torch.zeros(1, self.n_ch, self.h, self.w)
            self.cnn_out = self.cnn(dummy).shape[1]

        # MLP fusion — T frames concatenated
        self.net = nn.Sequential(
            nn.Linear(self.t_steps * self.cnn_out, 512),
            nn.LayerNorm(512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
        )

        self.policy_mean = nn.Linear(256, action_space.shape[0])
        self.log_std     = nn.Parameter(torch.zeros(action_space.shape[0]))
        self.value_head  = nn.Linear(256, 1)

        # Orthogonal init
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.policy_mean.weight, gain=0.01)

    def act(self, inputs, role):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)

    def compute(self, inputs, role=""):
        obs = unflatten_tensorized_space(
            self.observation_space, inputs.get("states")
        )
        # obs: (N, T=3, H=64, W=80, C=2)

        feats = []
        for t in range(self.t_steps):
            frame = obs[:, t].permute(0, 3, 1, 2)   # (N, C, H, W)
            feats.append(self.cnn(frame))

        shared = self.net(torch.cat(feats, dim=1))

        if role == "policy":
            return self.policy_mean(shared), self.log_std, {}
        elif role == "value":
            return self.value_head(shared), {}
        return self.policy_mean(shared), self.log_std, {}


# =============================================================================
# AGENT FACTORY
# =============================================================================

def get_agent(env, device, experiment_cfg: dict | None = None):
    """
    Build SKRL PPO agent for frontier exploration.

    Differences from ball tracking agent:
        rollouts=512  — longer episodes, need more data per update
        discount=0.995 — 90s episode, need high gamma for success signal
        entropy=0.03  — exploration needs more policy randomness
        lr=1e-4       — conservative for stable CNN training
    """
    rollout_length = 512

    memory = RandomMemory(
        memory_size=rollout_length,
        num_envs=env.num_envs,
        device=device,
    )

    model = IrisFrontierModel(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )
    models = {"policy": model, "value": model}

    cfg = PPO_DEFAULT_CONFIG.copy()

    cfg["rollouts"]        = rollout_length
    cfg["learning_epochs"] = 8
    cfg["mini_batches"]    = 8
    cfg["discount_factor"] = 0.995
    cfg["lambda"]          = 0.95

    cfg["learning_rate"]                  = 1e-4
    cfg["learning_rate_scheduler"]        = KLAdaptiveRL
    cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.01}

    cfg["grad_norm_clip"]        = 0.5
    cfg["ratio_clip"]            = 0.2
    cfg["value_clip"]            = 0.2
    cfg["clip_predicted_values"] = True

    cfg["entropy_loss_scale"] = 0.03
    cfg["value_loss_scale"]   = 1.0

    cfg["state_preprocessor"]        = None
    cfg["value_preprocessor"]        = RunningStandardScaler
    cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}

    cfg["experiment"]["write_interval"]      = 200
    cfg["experiment"]["checkpoint_interval"] = 1000
    cfg["experiment"]["directory"]           = "logs/skrl/iris_frontier"

    if experiment_cfg:
        cfg["experiment"].update(experiment_cfg)
 
    return PPO(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )