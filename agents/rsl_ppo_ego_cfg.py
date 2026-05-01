"""
rsl_ppo_ego_cfg.py
==================
RSL-RL runner configuration for IrisExploreEgoEnv.

RSL-RL does not use YAML configs — everything is a Python dataclass.
This file defines:
  1. PPOAlgorithmCfg  — PPO hyperparameters
  2. EgoRunnerCfg     — ties together env, network, algorithm, and logging

Register in your task __init__.py by adding:
    "rsl_rl_cfg_entry_point": "rl_WorkSpace.agents.rsl_ppo_ego_cfg:EgoRunnerCfg"

Train:
    CUDA_VISIBLE_DEVICES=0 /isaac-sim/python.sh \\
        rl_WorkSpace/scripts/rsl_rl/train.py \\
        --task Isaac-Iris-Explore-Ego-v0 \\
        --num_envs 32 --headless --enable_cameras

Play:
    CUDA_VISIBLE_DEVICES=0 /isaac-sim/python.sh \\
        rl_WorkSpace/scripts/rsl_rl/play.py \\
        --task Isaac-Iris-Explore-Ego-v0 \\
        --num_envs 1 --livestream 2 --enable_cameras
"""

from rsl_rl.runners import OnPolicyRunner

# Import your custom network
from rl_WorkSpace.rl_envs.ego_actor_critic import EgoActorCritic


# ── PPO Algorithm Config ──────────────────────────────────────────────────────

PPO_CFG = dict(

    # ── Rollout ──────────────────────────────────────────────────────────────
    # 256 steps × 32 envs = 8192 transitions per PPO update.
    # At 50Hz, 256 steps = 5.12 seconds — long enough for the drone to
    # traverse a room and see the reward from new cells in adjacent areas.
    # Previously used 64 (too short for a 60s episode with sparse success).
    num_steps_per_env = 256,

    # ── PPO core ─────────────────────────────────────────────────────────────
    num_learning_epochs  = 8,    # passes over the rollout buffer per update
    num_mini_batches     = 8,    # 8192 / 8 = 1024 samples per mini-batch
    clip_param           = 0.2,  # PPO clip ratio — standard
    gamma                = 0.995, # high: lets success bonus at end of 60s episode
                                  #       have discounted value ≈ e^(-0.005*3000)≈7%
                                  #       still visible; with 0.99 it would be ~9e-14
    lam                  = 0.95, # GAE lambda — standard

    # ── Learning rate ────────────────────────────────────────────────────────
    # RSL-RL uses a fixed LR with optional schedule, not KL-adaptive.
    # 1e-4 is conservative — the CNN branch adds more parameters than a pure MLP.
    learning_rate        = 1e-4,
    schedule             = "adaptive",  # RSL-RL built-in: reduces LR on KL spike
    desired_kl           = 0.02,        # more tolerant than SKRL default 0.01
    max_grad_norm        = 0.5,         # gradient clip

    # ── Value function ───────────────────────────────────────────────────────
    value_loss_coef      = 0.5,
    use_clipped_value_loss = True,

    # ── Entropy ──────────────────────────────────────────────────────────────
    # 0.05 — higher than typical (0.01) because we need sustained exploration.
    # If the policy collapses to hovering early, increase to 0.08.
    # If training is noisy at 500k+ steps, reduce to 0.02.
    entropy_coef         = 0.05,
)


# ── Network Config ────────────────────────────────────────────────────────────

NETWORK_CFG = dict(
    # EgoActorCritic is instantiated by the runner with these kwargs.
    # num_actions is injected automatically by the runner from env.action_space.
    class_name    = EgoActorCritic,
    map_size       = 40,
    map_feat_dim   = 128,
    state_in_dim   = 6,
    state_feat_dim = 64,
    trunk_out_dim  = 256,
    init_noise_std = 1.0,   # initial action std — full range exploration
)


# ── Runner Config (top-level) ─────────────────────────────────────────────────

class EgoRunnerCfg:
    """
    Passed to OnPolicyRunner.__init__() by the Isaac Lab RSL-RL train script.

    RSL-RL reads these attributes directly — they are not dataclass fields,
    just plain class attributes (RSL-RL's convention).
    """

    seed = 42

    # ── Policy ───────────────────────────────────────────────────────────────
    policy = NETWORK_CFG

    # ── Algorithm ────────────────────────────────────────────────────────────
    algorithm = PPO_CFG

    # ── Runner ───────────────────────────────────────────────────────────────
    runner = dict(
        class_name          = "OnPolicyRunner",

        # Total environment steps (not PPO updates).
        # 1M steps / (256 steps × 32 envs) ≈ 122 PPO updates — increase if needed.
        max_iterations      = 1_000_000 // (256 * 32),   # ≈ 122

        # How often to log to TensorBoard (in PPO iterations)
        log_interval        = 1,

        # How often to save a checkpoint (in PPO iterations)
        save_interval       = 10,

        # Experiment directory — checkpoints saved under runs/<experiment_name>/
        experiment_name     = "iris_ego_explore",
        run_name            = "cnn_map_mlp_state",

        # Resume from checkpoint (set to path string to resume, None for fresh)
        resume              = False,
        load_run            = -1,    # -1 = latest run
        checkpoint          = -1,    # -1 = latest checkpoint
    )

    # ── Normalisation ─────────────────────────────────────────────────────────
    # RSL-RL has a built-in empirical normaliser for observations.
    # The map values are already in {-1, 0, 1} and state is small-scale,
    # so normalisation helps but isn't critical. Set to False if it causes issues.
    normalize_observation = True