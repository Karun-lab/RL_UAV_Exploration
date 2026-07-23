"""
iris_icm_map_agent.py
========================
SKRL PPO agent + CNN model for ICM-driven office exploration
with pose-based visited-area memory.

Observation change vs previous version:
    (T=3, H=64, W=80, C=1)  depth only
    →
    (T=3, H=64, W=80, C=2)  depth + local novelty map

The CNN now has 2 input channels per frame instead of 1.
Everything else — architecture depth, PPO hyperparameters,
init scheme — is unchanged.

The novelty map channel (C=1) feeds directly into the same CNN as depth,
so the policy can spatially correlate "this region looks open" (depth)
with "this region has not been visited yet" (novelty) and learn to
prefer novel open space over revisited open space.
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

class IrisICMOfficeModel(GaussianMixin, DeterministicMixin, Model):
    """
    Depth + novelty CNN actor-critic for ICM exploration.

    Input:  (N, T=3, H=64, W=80, C=2)
        C=0: normalised depth
        C=1: local novelty map  (1=unvisited, 0=heavily revisited)

    CNN processes each frame independently with 2 input channels,
    then concatenates T=3 feature vectors before the MLP.
    """

    def __init__(self, observation_space, action_space, device,
                 clip_actions=False, clip_log_std=True,
                 min_log_std=-20.0, max_log_std=2.0,
                 reduction="sum", **kwargs):

        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std,
                               min_log_std, max_log_std, reduction)
        DeterministicMixin.__init__(self, clip_actions)

        # obs: (T=3, H=64, W=80, C=2)
        self.t_steps = 3
        self.h       = 64
        self.w       = 80
        self.n_ch    = 2   # ← was 1, now 2 (depth + novelty map)

        self.cnn = nn.Sequential(
            nn.Conv2d(self.n_ch, 16, kernel_size=5, stride=2),
            nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy        = torch.zeros(1, self.n_ch, self.h, self.w)
            self.cnn_out = self.cnn(dummy).shape[1]

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
            self.observation_space, inputs.get("states"))
        # obs: (N, T=3, H=64, W=80, C=2)

        feats = []
        for t in range(self.t_steps):
            frame = obs[:, t].permute(0, 3, 1, 2)   # (N, 2, H, W)
            feats.append(self.cnn(frame))

        shared = self.net(torch.cat(feats, dim=1))   # (N, 256)

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
    PPO agent for ICM + pose-memory office exploration.
    Observation is now 2-channel, handled automatically by the model above.
    All PPO hyperparameters are unchanged from the working single-channel version.
    """
    rollout_length = 256

    memory = RandomMemory(
        memory_size=rollout_length,
        num_envs=env.num_envs,
        device=device,
    )

    model = IrisICMOfficeModel(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )
    models = {"policy": model, "value": model}

    cfg = PPO_DEFAULT_CONFIG.copy()

    cfg["rollouts"]        = rollout_length
    cfg["learning_epochs"] = 8
    cfg["mini_batches"]    = 4

    cfg["discount_factor"] = 0.99
    cfg["lambda"]          = 0.95

    cfg["learning_rate"]                  = 3e-4
    cfg["learning_rate_scheduler"]        = KLAdaptiveRL
    cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.015}

    cfg["grad_norm_clip"]        = 1.0
    cfg["ratio_clip"]            = 0.2
    cfg["value_clip"]            = 0.2
    cfg["clip_predicted_values"] = True

    cfg["entropy_loss_scale"] = 0.05
    cfg["value_loss_scale"]   = 1.0

    cfg["state_preprocessor"]        = None
    cfg["value_preprocessor"]        = RunningStandardScaler
    cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}

    cfg["experiment"]["write_interval"]      = 200
    cfg["experiment"]["checkpoint_interval"] = 500
    cfg["experiment"]["directory"]           = "logs/skrl/iris_icm_office"
    cfg["experiment"]["wandb"]               = False

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