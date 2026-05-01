"""
frontier_actor_critic.py
========================
RSL-RL compatible Actor-Critic for frontier-based exploration.

Observation vector (19 values — must match env's observation_space):
    [0:3]   lin_vel_b
    [3:6]   ang_vel_b
    [6:9]   projected_gravity_b
    [9:12]  frontier_b           ← the key signal: where to go next
    [12]    coverage
    [13:15] depth_opening        ← short-range: best gap direction + distance
    [15:17] depth_min_lr         ← short-range: proximity left/right
    [17:19] prev_action

All 19 values are fed into a single MLP — no CNN needed because the
observation is already a compact hand-crafted feature vector, not raw
pixels or a spatial map. The frontier direction and depth signals carry
all spatial information the policy needs.

RSL-RL interface contract:
    act(obs)              → (actions, log_prob, values)
    evaluate(obs, actions) → (values, log_prob, entropy)
    act_inference(obs)    → actions  (deterministic, mean)
    get_std()             → current std of the Gaussian policy
"""

import torch
import torch.nn as nn
from torch.distributions import Normal


class FrontierActorCritic(nn.Module):
    """
    Shared-trunk MLP actor-critic for the frontier exploration env.

    Why shared trunk (separate=False)?
        The actor and critic need the same features — "where is the frontier"
        and "how close are the walls" are relevant to both the policy and
        the value estimate. Sharing the trunk halves parameters and encourages
        the representation to be useful for both tasks.

    Architecture:
        obs (19) → trunk [256, 128] → actor head (2) / critic head (1)
        log_std: learned parameter, not input-dependent
    """

    def __init__(
        self,
        num_obs:            int,    # injected by RSL-RL runner = 19
        num_privileged_obs: int,    # injected by RSL-RL runner = 0 (unused)
        num_actions:        int,    # injected by RSL-RL runner = 2
        init_noise_std:     float = 1.0,
        **kwargs,                   # absorb any extra kwargs from runner cfg
    ):
        super().__init__()

        self.num_actions = num_actions

        # ── Shared trunk ─────────────────────────────────────────────────────
        # ELU throughout: smooth gradients, works well for continuous control
        self.trunk = nn.Sequential(
            nn.Linear(num_obs, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
        )

        # ── Actor head ───────────────────────────────────────────────────────
        self.actor_head = nn.Linear(128, num_actions)

        # Learned log_std — not input-dependent (standard for PPO locomotion)
        # init_noise_std=1.0 → initial std=1.0 → drone explores full action range
        self.log_std = nn.Parameter(
            torch.full((num_actions,), float(init_noise_std))
        )

        # ── Critic head ──────────────────────────────────────────────────────
        self.critic_head = nn.Linear(128, 1)

        # ── Weight init ──────────────────────────────────────────────────────
        # Orthogonal init: standard in RL, prevents vanishing/exploding grads
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)
        # Small gain on output layers: initial actions near zero, stable early training
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)

    # ── RSL-RL interface ──────────────────────────────────────────────────────

    def act(self, observations: torch.Tensor, **kwargs):
        """
        Sample actions from policy. Called during rollout collection.
        Returns: (actions, log_prob, values)
        """
        features = self.trunk(observations)
        mean     = self.actor_head(features)
        std      = self.log_std.exp().expand_as(mean)

        dist     = Normal(mean, std)
        actions  = dist.sample()
        log_prob = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        values   = self.critic_head(features)

        return actions, log_prob, values

    def evaluate(self, observations: torch.Tensor, actions: torch.Tensor, **kwargs):
        """
        Evaluate log_prob and entropy of given actions under current policy.
        Called during PPO update step.
        Returns: (values, log_prob, entropy)
        """
        features = self.trunk(observations)
        mean     = self.actor_head(features)
        std      = self.log_std.exp().expand_as(mean)

        dist     = Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        entropy  = dist.entropy().sum(dim=-1, keepdim=True)
        values   = self.critic_head(features)

        return values, log_prob, entropy

    def act_inference(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Deterministic action (policy mean). Used at play/deploy time.
        No sampling — takes the most likely action.
        """
        features = self.trunk(observations)
        return self.actor_head(features)

    def get_std(self) -> torch.Tensor:
        """RSL-RL calls this for TensorBoard std logging."""
        return self.log_std.exp()

    # ── Not used by RSL-RL but kept for debugging ─────────────────────────────
    def forward(self, observations: torch.Tensor):
        features = self.trunk(observations)
        return self.actor_head(features), self.critic_head(features)