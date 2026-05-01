# /workspace/isaaclab/rl_WorkSpace/__init__.py

import gymnasium as gym
from . import agents
from .rl_envs import (
    iris_target_env,
    iris_explore_env,
    iris_explore_walls_env,
    iris_maze_env,
) 
print(">>> rl_WorkSpace __init__ LOADED")
gym.register(
    id="Isaac-Iris-Target-v0",
    entry_point=f"{__name__}.rl_envs.iris_target_env:IrisEnv", 
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_envs.iris_target_env:IrisEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Iris-Explore-v0",
    entry_point=f"{__name__}.rl_envs.iris_explore_env:IrisExploreEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_envs.iris_explore_env:IrisExploreEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_explore_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Iris-Walls-v0",
    entry_point=f"{__name__}.rl_envs.iris_explore_walls_env:IrisExploreWallsEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_envs.iris_explore_walls_env:IrisExploreWallsEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_explore_cfg.yaml",
    },
)
gym.register(
    id="Isaac-Iris-Maze-v0",
    entry_point=f"{__name__}.rl_envs.iris_maze_env:IrisMazeEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_envs.iris_maze_env:IrisMazeEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_maze_cfg.yaml",
    },
)
gym.register(
    id="Isaac-Iris-Maze-v1",
    entry_point=f"{__name__}.rl_envs.iris_explore_corridor:IrisMazeEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_envs.iris_explore_corridor:IrisMazeEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_maze_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Iris-Explore-ICM-v0",
    entry_point="rl_WorkSpace.rl_envs.iris_icm_exploration:IrisExploreEnv",
    kwargs={
        "env_cfg_entry_point":
            "rl_WorkSpace.rl_envs.iris_icm_exploration:IrisExploreEnvCfg",
        "skrl_cfg_entry_point":
            "rl_WorkSpace.agents:skrl_ppo_icm_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Iris-Explore-Ego-v0",
    entry_point="rl_WorkSpace.rl_envs.iris_explore_ego_env:IrisExploreEgoEnv",
    kwargs={
        "env_cfg_entry_point":
            "rl_WorkSpace.rl_envs.iris_explore_ego_env:IrisExploreEgoEnvCfg",
        "skrl_cfg_entry_point":
            "rl_WorkSpace.agents:skrl_ppo_ego_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Iris-Explore-Ego-v1",
    entry_point="rl_WorkSpace.rl_envs.iris_explore_ego_env:IrisExploreEgoEnv",
    kwargs={
        "env_cfg_entry_point":
            "rl_WorkSpace.rl_envs.iris_explore_ego_env:IrisExploreEgoEnvCfg",

        # ── ADD this for RSL-RL ───────────────────────────────────
        "rsl_rl_cfg_entry_point":
            "rl_WorkSpace.agents.rsl_ppo_ego_cfg:EgoRunnerCfg",
    },
)

gym.register(
    id="Isaac-Iris-Frontier-v0",
    entry_point="rl_WorkSpace.rl_envs.iris_explore_frontier_env:IrisExploreFrontierEnv",
    kwargs={
        "env_cfg_entry_point":
            "rl_WorkSpace.rl_envs.iris_explore_frontier_env:IrisExploreFrontierEnvCfg",
        "rsl_rl_cfg_entry_point":
            "rl_WorkSpace.agents.rsl_ppo_frontier_cfg:FrontierRunnerCfg",
    },
)

from .rl_envs import iris_ball_env   # ← add to existing import block

gym.register(
    id="Isaac-Iris-Ball-v0",
    entry_point="rl_WorkSpace.rl_envs.iris_ball_env:IrisBallEnv",
    kwargs={
        "env_cfg_entry_point":
            "rl_WorkSpace.rl_envs.iris_ball_env_cfg:IrisBallEnvCfg",
        "skrl_cfg_entry_point":
            "rl_WorkSpace.agents:skrl_ppo_ball_cfg.yaml",
    },
)
