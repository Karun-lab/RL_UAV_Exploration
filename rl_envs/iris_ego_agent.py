"""
iris_ego_agent.py
=================
SKRL PPO agent + CNN+MLP model for egocentric exploration.

Input: (N, T=1, MAP_SIZE, MAP_SIZE, 2)
    Same CNN+MLP architecture as iris_ball_models.py.
    T=1 because the map already encodes temporal history.

Network:
    map (40×40×2) → CNN → 256-d features
    → MLP [512, 256] → policy head / value head

The CNN sees the occupancy map as a spatial image:
    bright (1.0) = free space  → go here
    dark   (0.0) = wall        → avoid
    grey   (0.5) = unknown     → explore here

The policy naturally learns to steer toward grey (unknown) regions
because new cells discovered = reward.
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

class IrisEgoModel(GaussianMixin, DeterministicMixin, Model):
    """
    CNN processes the occupancy map image.
    Output goes directly to policy/value heads via a small MLP.

    Input shape from obs space: (T=1, M=40, W=40, C=2)
    We use T=1 so effectively: (M, W, C) = (40, 40, 2) per frame.

    CNN input: (N, C=2, H=40, W=40) — map + velocity channels
    CNN output: flat feature vector
    MLP: features → [512, 256] → heads
    """

    def __init__(self, observation_space, action_space, device,
                 clip_actions=False, clip_log_std=True,
                 min_log_std=-20.0, max_log_std=2.0,
                 reduction="sum", **kwargs):

        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std,
                               min_log_std, max_log_std, reduction)
        DeterministicMixin.__init__(self, clip_actions)

        # obs shape: (T=1, M=40, W=40, C=2)
        self.t_steps = 1
        self.h       = 40
        self.w       = 40
        self.n_ch    = 2

        # CNN — lighter than ball tracking since map is smaller and cleaner
        # Input: (N, 2, 40, 40)
        # 40 → 18 → 8 → 3
        self.cnn = nn.Sequential(
            nn.Conv2d(self.n_ch, 16, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm2d(16), nn.ELU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ELU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy        = torch.zeros(1, self.n_ch, self.h, self.w)
            self.cnn_out = self.cnn(dummy).shape[1]

        # MLP fusion
        self.net = nn.Sequential(
            nn.Linear(self.cnn_out, 512),
            nn.LayerNorm(512), nn.ELU(),
            nn.Linear(512, 256), nn.ELU(),
        )

        self.policy_mean = nn.Linear(256, action_space.shape[0])
        self.log_std     = nn.Parameter(torch.zeros(action_space.shape[0]))
        self.value_head  = nn.Linear(256, 1)

        # Orthogonal init — standard for PPO
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
        # obs: (N, T=1, H=40, W=40, C=2)

        feats = []
        for t in range(self.t_steps):
            frame = obs[:, t].permute(0, 3, 1, 2)   # (N, C, H, W)
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
    Build SKRL PPO agent for egocentric exploration.

    Key hyperparameter choices vs ball tracking:
        rollouts=512    — longer episodes need more transitions per update
        discount=0.995  — 90s episode at 50Hz = 4500 steps, need high gamma
        entropy=0.02    — exploration task needs higher entropy than tracking
        lr=1e-4         — conservative, CNN+map can be unstable early
    """
    rollout_length = 512

    memory = RandomMemory(
        memory_size=rollout_length,
        num_envs=env.num_envs,
        device=device,
    )

    model = IrisEgoModel(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )
    models = {"policy": model, "value": model}

    cfg = PPO_DEFAULT_CONFIG.copy()

    cfg["rollouts"]        = rollout_length
    cfg["learning_epochs"] = 8
    cfg["mini_batches"]    = 8

    # High gamma: success bonus at end of 90s episode must be visible
    # γ^4500 with γ=0.995 ≈ 1.2e-10 — episode reward still visible via GAE
    cfg["discount_factor"] = 0.995
    cfg["lambda"]          = 0.95

    cfg["learning_rate"]                  = 1e-4
    cfg["learning_rate_scheduler"]        = KLAdaptiveRL
    cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.01}

    cfg["grad_norm_clip"]        = 0.5
    cfg["ratio_clip"]            = 0.2
    cfg["value_clip"]            = 0.2
    cfg["clip_predicted_values"] = True

    # Higher entropy than ball tracking — exploration task must not collapse
    cfg["entropy_loss_scale"] = 0.02
    cfg["value_loss_scale"]   = 1.0

    # State is image-based — no scaler on input
    cfg["state_preprocessor"] = None
    cfg["value_preprocessor"]        = RunningStandardScaler
    cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}

    cfg["experiment"]["write_interval"]      = 200
    cfg["experiment"]["checkpoint_interval"] = 1000
    cfg["experiment"]["directory"]           = "logs/skrl/iris_ego"

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