# /workspace/isaaclab/rl_WorkSpace/__init__.py

import gymnasium as gym
from . import agents
from .rl_envs import (
    iris_target_env,
    iris_explore_env,
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

from .rl_envs import iris_ball_env   
gym.register(
    id="Isaac-Iris-Ball-v0",
    entry_point="rl_WorkSpace.rl_envs.iris_ball_env:IrisBallEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "rl_WorkSpace.rl_envs.iris_ball_env:IrisBallEnvCfg",
        "skrl_cfg_entry_point": "rl_WorkSpace.agents:skrl_ppo_ball_cfg.yaml",
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

from .rl_envs import iris_icm_exploration   
gym.register(
    id="Isaac-Iris-ICM-v1",
    entry_point="rl_WorkSpace.rl_envs.iris_icm_exploration:IrisICMOfficeEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "rl_WorkSpace.rl_envs.iris_icm_exploration:IrisICMOfficeEnvCfg",
        "skrl_cfg_entry_point": "rl_WorkSpace.agents:skrl_ppo_icm_cfg.yaml",
    },
)