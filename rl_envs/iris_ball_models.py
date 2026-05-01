"""
iris_ball_models.py
===================
CNN + MLP model for the Iris drone yellow ball tracking task.
Mirrors JetbotSharedModel exactly — same architecture, same SKRL interface.

Input: (N, T, H, W, 4) observation stack
    - 4 channels: RGB (3) + search_active (1)
    - T=3 stacked frames for velocity perception
    - H=W=64 image

Architecture:
    For each of T frames:
        frame (N, H, W, 4) → CNN → feature vector (N, cnn_out)
    Concatenate T features: (N, T * cnn_out)
    → Fusion MLP → shared embedding (N, 256)
    → policy head: mean action (N, 2)
    → value head:  scalar value (N, 1)

Why CNN per frame then concatenate (vs 3D conv or LSTM)?
    Same reason as the jetbot: simple, fast, and the policy gets explicit
    per-frame features that it can compare across time to infer velocity.
    LSTM adds training complexity without much benefit for this task length.

Why channel 4 (search_active) in the CNN instead of a separate MLP?
    The search behaviour is spatially tied to what the drone sees.
    Processing it through the CNN lets the policy learn "when I see nothing
    AND search_active=1, rotate" as a spatial-visual rule rather than a
    separate logical branch. This mirrors how the jetbot goal vector works.
"""
 
import torch
import torch.nn as nn
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin
from skrl.utils.spaces.torch import unflatten_tensorized_space


class IrisBallModel(GaussianMixin, DeterministicMixin, Model):
    """
    Shared backbone actor-critic for the yellow ball tracking task.
    Identical interface to JetbotSharedModel — same act() dispatch,
    same compute() signature, same SKRL mixins.
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
        # observation_space.shape = (T, H, W, C) = (3, 64, 64, 4)
        try:
            self.t_steps  = observation_space.shape[0]   # 3
            self.h        = observation_space.shape[1]   # 64
            self.w        = observation_space.shape[2]   # 64
            self.n_ch     = observation_space.shape[3]   # 4
        except Exception:
            self.t_steps, self.h, self.w, self.n_ch = 3, 64, 64, 4

        # ── CNN (shared across time steps) ────────────────────────────────────
        # Identical to JetbotSharedModel's cnn.
        # Input: (N, C, H, W) where C=4 (RGB + search_active)
        # 64 → 30 → 13 → 5 → 2 (spatial resolution with stride-2 convs)
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
        # Input: T frames × CNN features (no separate state vector — all info
        # is encoded in the 4-channel image including search_active)
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
        """Dispatch to GaussianMixin (policy) or DeterministicMixin (value)."""
        if role == "policy":
            return GaussianMixin.act(self, inputs, role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)

    def compute(self, inputs, role=""):
        """
        Forward pass.

        inputs["states"]: flattened observation, unflattened back to
                          (N, T, H, W, C) by SKRL's unflatten utility.
        """
        # ── 1. Unpack ─────────────────────────────────────────────────────────
        obs = unflatten_tensorized_space(
            self.observation_space, inputs.get("states")
        )
        # obs: (N, T, H, W, 4)

        # ── 2. CNN per time step ──────────────────────────────────────────────
        cnn_feats = []
        for t in range(self.t_steps):
            frame = obs[:, t, ...]                # (N, H, W, 4)
            frame = frame.permute(0, 3, 1, 2)     # (N, 4, H, W) for Conv2d
            cnn_feats.append(self.cnn(frame))     # (N, cnn_out)

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
            # Fallback — return both (used in some SKRL versions)
            return self.policy_mean(shared), self.log_std, {}