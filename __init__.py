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