"""
iris_icm_agent.py
========================
SKRL PPO agent + CNN model for ICM-driven office exploration.

The PPO policy is intentionally simple — just processes the depth
history and outputs actions. ICM does the heavy lifting for exploration
drive. The PPO agent only needs to learn:
    - Move forward when there is open space ahead (high depth)
    - Turn when walls are close (low depth)
    - Maintain forward velocity to earn velocity bonus

Architecture:
    (T=3, H=64, W=80, 1) depth stack
    → CNN per frame (single channel)
    → concatenate T frame features
    → MLP [512, 256]
    → policy / value heads

The model is intentionally shallower than the ball tracking model
because the ICM reward is much denser than the new_cells reward —
the policy update signal is strong, so the network doesn't need to
be as large to converge.
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
    Depth CNN + MLP for ICM exploration.
    Single channel input (depth only — no state channels needed).
    ICM encoder is separate and trained by the env, not part of this model.
    """

    def __init__(self, observation_space, action_space, device,
                 clip_actions=False, clip_log_std=True,
                 min_log_std=-20.0, max_log_std=2.0,
                 reduction="sum", **kwargs):

        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std,
                               min_log_std, max_log_std, reduction)
        DeterministicMixin.__init__(self, clip_actions)

        # obs: (T=3, H=64, W=80, C=1)
        self.t_steps = 3
        self.h       = 64
        self.w       = 80
        self.n_ch    = 2

        # CNN — lighter than ball model (single channel, ICM does feature work)
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
            self.observation_space, inputs.get("states")
        )
        # obs: (N, T=3, H=64, W=80, C=1)

        feats = []
        for t in range(self.t_steps):
            frame = obs[:, t].permute(0, 3, 1, 2)   # (N, 1, H, W)
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
    PPO agent for ICM office exploration.

    Key differences from ball tracking:
        entropy=0.05     — much higher, exploration needs randomness
        discount=0.99    — 60s episode, moderate gamma
        lr=3e-4          — slightly higher, ICM provides dense reward signal
        rollouts=256     — shorter, ICM reward is dense so less data needed

    Note: ICM is trained inside the env's _get_rewards(), not here.
    The agent only trains the PPO policy on the total reward
    (ICM intrinsic + velocity bonus + penalties).
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
    cfg["mini_batches"]    = 4   # fewer — smaller batch size with 16 envs

    cfg["discount_factor"] = 0.99
    cfg["lambda"]          = 0.95

    cfg["learning_rate"]                  = 3e-4
    cfg["learning_rate_scheduler"]        = KLAdaptiveRL
    cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.015}

    cfg["grad_norm_clip"]        = 1.0
    cfg["ratio_clip"]            = 0.2
    cfg["value_clip"]            = 0.2
    cfg["clip_predicted_values"] = True

    # High entropy — crucial for exploration
    # ICM provides exploration incentive but PPO still needs entropy
    # to avoid converging to a single yaw-and-drift behaviour
    cfg["entropy_loss_scale"] = 0.05
    cfg["value_loss_scale"]   = 1.0

    cfg["state_preprocessor"]        = None
    cfg["value_preprocessor"]        = RunningStandardScaler
    cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}

    cfg["experiment"]["write_interval"]      = 200
    cfg["experiment"]["checkpoint_interval"] = 500
    cfg["experiment"]["directory"]           = "logs/skrl/iris_icm_office"

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