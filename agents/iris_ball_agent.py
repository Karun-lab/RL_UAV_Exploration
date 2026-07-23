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
#And

"""CNN + MLP model for the Iris drone yellow ball tracking task.
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
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.resources.preprocessors.torch import RunningStandardScaler

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