import gymnasium as gym
from . import agents
gym.register(
    id="Isaac-Iris-Direct-v0",
    entry_point=f"{__name__}.iris_target_env:IrisEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.iris_target_env:IrisEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Iris-Explore-v0",
    entry_point=f"{__name__}.iris_explore_env:IrisExploreEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.iris_explore_env:IrisExploreEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_explore_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Iris-Walls-v0",
    entry_point=f"{__name__}.iris_explore_walls_env:IrisExploreWallsEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.iris_explore_walls_env:IrisExploreWallsEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_explore_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Iris-Maze-v0",
    entry_point=f"{__name__}.iris_maze_env:IrisMazeEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.iris_maze_env:IrisMazeEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_maze_cfg.yaml",
    },
)