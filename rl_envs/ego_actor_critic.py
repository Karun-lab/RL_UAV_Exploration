"""
ego_actor_critic.py
===================
Custom RSL-RL ActorCritic for the egocentric map environment.

Network architecture:
    Observation vector (1606,) split into:
        obs[:1600]  → reshape (1, 40, 40) → CNN  → 128-d features
        obs[1600:]  → MLP                         →  64-d features
    Concatenated (192-d) → shared trunk → policy head / value head

Why split input?
    The 40×40 map is a spatial image — a CNN respects its 2D structure
    and shares weights across spatial positions. Treating it as a flat
    1600-d vector forces the MLP to re-learn spatial relationships from
    scratch, which is much harder and needs far more parameters.
    The 6-d state (velocities, yaw) is already compact and structured,
    so a small MLP is appropriate there.

RSL-RL interface contract (what this file must satisfy):
    - Class inherits nn.Module
    - Exposes:  act(obs)              → actions (sampled)
                evaluate(obs, actions) → (value, log_prob, entropy)
                act_inference(obs)    → actions (deterministic, mean)
    - Actor outputs: mean of Gaussian (log_std is a learned parameter)
    - Critic outputs: scalar value per env
    - Both share the CNN+MLP feature extractor (separate=False)

Usage:
    Instantiated by IrisEgoRunnerCfg below — you do not call this directly.
"""

import torch
import torch.nn as nn
from torch.distributions import Normal


# ── CNN branch ──────────────────────────────────────────────────────────────
class MapCNN(nn.Module):
    """
    Small CNN for the 40×40 egocentric occupancy map.

    Input:  (N, 1, 40, 40)  — single-channel "image" with values {-1, 0, 1}
    Output: (N, 128)        — spatial feature vector

    Architecture notes:
    - 3 conv layers with stride-2 downsampling (no pooling — preserves gradient flow)
    - ELU activation: smooth around zero, good for {-1,0,1} inputs
    - Kernel sizes: 5→3→3 (large first kernel to capture multi-cell patterns)
    - Final adaptive pool to handle edge cases if map size ever changes
    """
    def __init__(self, map_size: int = 40, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            # (N,  1, 40, 40) → (N, 16, 18, 18)
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=1),
            nn.ELU(),
            # (N, 16, 18, 18) → (N, 32,  8,  8)
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            # (N, 32,  8,  8) → (N, 64,  4,  4)
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Flatten(),       # (N, 64*4*4) = (N, 1024)
        )

        # Compute actual flat size with a dummy pass (safe if map_size changes)
        with torch.no_grad():
            dummy    = torch.zeros(1, 1, map_size, map_size)
            flat_dim = self.net(dummy).shape[-1]

        self.proj = nn.Sequential(
            nn.Linear(flat_dim, out_dim),
            nn.ELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, 1600) flat map → returns (N, out_dim)"""
        img = x.reshape(x.shape[0], 1, 40, 40)   # unflatten to image
        return self.proj(self.net(img))


# ── State MLP branch ─────────────────────────────────────────────────────────
class StateMLP(nn.Module):
    """
    Small MLP for the 6-d state vector.
    [lin_vel(3), ang_vel_z(1), yaw_sin(1), yaw_cos(1)]

    Input:  (N, 6)   → Output: (N, 64)
    """
    def __init__(self, in_dim: int = 6, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ELU(),
            nn.Linear(64, out_dim),
            nn.ELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Shared trunk (after fusion) ───────────────────────────────────────────────
class FusionTrunk(nn.Module):
    """
    Takes concatenated CNN + MLP features and produces a shared embedding
    used by both actor and critic heads.

    Input:  (N, 128 + 64) = (N, 192)
    Output: (N, 256)
    """
    def __init__(self, in_dim: int = 192, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ELU(),
            nn.Linear(256, out_dim),
            nn.ELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Full Actor-Critic ─────────────────────────────────────────────────────────
class EgoActorCritic(nn.Module):
    """
    RSL-RL compatible Actor-Critic for egocentric map observations.

    Observation layout (must match env's _get_observations):
        obs[:1600]  — flat egocentric map  (40*40)
        obs[1600:]  — state vector         (6)

    The actor outputs a Gaussian policy:
        mean    = actor_head(features)
        log_std = learned parameter (not input-dependent)

    The critic outputs a scalar value estimate.

    Both share the CNN, StateMLP, and FusionTrunk — saves memory and
    encourages the feature extractor to learn representations useful
    for both policy and value estimation.
    """

    def __init__(
        self,
        num_actions:   int   = 2,
        map_size:      int   = 40,
        map_feat_dim:  int   = 128,
        state_in_dim:  int   = 6,
        state_feat_dim: int  = 64,
        trunk_out_dim: int   = 256,
        init_noise_std: float = 1.0,
    ):
        super().__init__()

        self.map_dim   = map_size * map_size   # 1600
        self.state_dim = state_in_dim          # 6

        # ── Feature extractors (shared) ──────────────────────────────────
        self.map_cnn   = MapCNN(map_size, map_feat_dim)
        self.state_mlp = StateMLP(state_in_dim, state_feat_dim)
        self.trunk     = FusionTrunk(map_feat_dim + state_feat_dim, trunk_out_dim)

        # ── Actor head ───────────────────────────────────────────────────
        self.actor = nn.Linear(trunk_out_dim, num_actions)

        # Log std as a learned parameter (not input-dependent)
        # init_noise_std=1.0 → initial std=1.0 → full action range exploration
        self.log_std = nn.Parameter(
            torch.ones(num_actions) * torch.tensor(init_noise_std).log()
        )

        # ── Critic head ──────────────────────────────────────────────────
        self.critic = nn.Linear(trunk_out_dim, 1)

        # ── Weight initialisation ────────────────────────────────────────
        self._init_weights()

    def _init_weights(self):
        """
        Orthogonal init for linear layers — standard in RL.
        Small gain on output layers to keep initial actions near zero.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)
        # Small gain on output layers
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def _extract_features(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Split obs → run CNN + MLP → fuse → trunk.
        obs: (N, 1606)
        returns: (N, trunk_out_dim)
        """
        map_flat  = obs[:, :self.map_dim]          # (N, 1600)
        state     = obs[:, self.map_dim:]           # (N, 6)

        map_feat   = self.map_cnn(map_flat)         # (N, 128)
        state_feat = self.state_mlp(state)          # (N, 64)

        fused = torch.cat([map_feat, state_feat], dim=-1)  # (N, 192)
        return self.trunk(fused)                            # (N, 256)

    # ── RSL-RL interface ─────────────────────────────────────────────────────

    def act(self, obs: torch.Tensor):
        """
        Sample actions from the policy. Called during rollout collection.
        Returns: (actions, log_prob, value)
        """
        features = self._extract_features(obs)

        mean     = self.actor(features)
        std      = self.log_std.exp().expand_as(mean)
        dist     = Normal(mean, std)
        actions  = dist.sample()
        log_prob = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        value    = self.critic(features)

        # Store for evaluate() — RSL-RL calls evaluate separately
        self._features = features.detach()

        return actions, log_prob, value

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor):
        """
        Evaluate log_prob and entropy of given actions under current policy.
        Called during PPO update.
        Returns: (value, log_prob, entropy)
        """
        features = self._extract_features(obs)

        mean     = self.actor(features)
        std      = self.log_std.exp().expand_as(mean)
        dist     = Normal(mean, std)

        log_prob = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        entropy  = dist.entropy().sum(dim=-1, keepdim=True)
        value    = self.critic(features)

        return value, log_prob, entropy

    def act_inference(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Deterministic action (mean of policy). Used at play/inference time.
        """
        features = self._extract_features(obs)
        return self.actor(features)

    def get_std(self) -> torch.Tensor:
        """RSL-RL calls this for logging."""
        return self.log_std.exp()