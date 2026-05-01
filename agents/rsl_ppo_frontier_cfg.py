"""
rsl_ppo_frontier_cfg.py
=======================
RSL-RL OnPolicyRunner configuration for IrisExploreFrontierEnv.

RSL-RL does NOT use YAML or dataclasses — it reads plain class attributes
from a config class. The runner inspects cfg.policy, cfg.algorithm, and
cfg.runner as dicts and instantiates the right objects.

How RSL-RL instantiates the actor-critic:
    actor_critic = cfg.policy["class"](
        num_obs            = env.num_obs,
        num_privileged_obs = env.num_privileged_obs,
        num_actions        = env.num_actions,
        **{k:v for k,v in cfg.policy.items() if k != "class"},
    )
So num_obs=19 and num_actions=2 come from the env automatically.

Register in __init__.py under the task kwargs:
    "rsl_rl_cfg_entry_point":
        "rl_WorkSpace.agents.rsl_ppo_frontier_cfg:FrontierRunnerCfg"
"""

from rl_WorkSpace.rl_envs.frontier_actor_critic import FrontierActorCritic


class FrontierRunnerCfg:

    seed = 42

    # ── Policy network ────────────────────────────────────────────────────────
    # "class" is instantiated by OnPolicyRunner with (num_obs, num_privileged_obs,
    # num_actions) as the first three args, plus any extra kwargs below.
    policy = dict(
        class          = FrontierActorCritic,
        init_noise_std = 1.0,   # initial action std — full range exploration
    )

    # ── PPO algorithm ─────────────────────────────────────────────────────────
    algorithm = dict(
        class = "PPO",

        # Rollout length: 256 steps × 32 envs = 8192 transitions per update.
        # At 50 Hz, 256 steps = 5.12 seconds — long enough for the drone to
        # traverse a room and see the reward from discovering new cells ahead.
        num_steps_per_env = 256,

        # PPO core
        clip_param             = 0.2,
        gamma                  = 0.995,  # high: success bonus at end of 90s episode
        lam                    = 0.95,   # GAE lambda
        num_learning_epochs    = 8,
        num_mini_batches       = 8,      # 8192 / 8 = 1024 samples per mini-batch
        value_loss_coef        = 0.5,
        use_clipped_value_loss = True,
        max_grad_norm          = 0.5,

        # Learning rate — adaptive: reduces when KL spikes
        learning_rate = 1e-4,
        schedule      = "adaptive",
        desired_kl    = 0.02,

        # Entropy: keep high to prevent early collapse to hovering.
        # 0.05 = strong exploration pressure.
        # Reduce to 0.02 after 500k steps if policy is noisy.
        entropy_coef = 0.05,
    )

    # ── Runner ────────────────────────────────────────────────────────────────
    runner = dict(
        class      = "OnPolicyRunner",

        # Total training iterations (each = one PPO update over all envs).
        # 1M env steps / (256 steps × 32 envs) ≈ 122 iterations.
        # Increase max_iterations for longer training — faster than changing
        # timesteps because you don't have to recalculate.
        max_iterations = 122,   # ≈ 4M env steps with 256 steps × 32 envs

        # Logging and checkpointing (in iterations)
        log_interval        = 1,
        save_interval       = 25,

        # Experiment directory: checkpoints saved under runs/<experiment_name>/
        experiment_name = "iris_frontier_explore",
        run_name        = "frontier_mlp_v1",

        # Resume training from checkpoint
        # Set resume=True and load_run to the run folder name to continue.
        resume     = False,
        load_run   = -1,    # -1 = latest run in experiment_name directory
        checkpoint = -1,    # -1 = latest checkpoint in that run
    )