# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

##
# IRIS CONFIG
##

IRIS_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",

    spawn=sim_utils.UsdFileCfg(
        usd_path="/workspace/isaaclab/rl_WorkSpace/models/drone/iris_quadrotor.usd",

        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            enable_gyroscopic_forces=True,
            max_depenetration_velocity=10.0,
        ),
 
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=1,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),

        copy_from_source=False,
    ),

    # Initial state
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.3),
        rot=(1.0, 0.0, 0.0, 0.0),  # quaternion (w,x,y,z)
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),

    # Dummy actuator (IMPORTANT)
    actuators={
        "dummy": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=0.0,
            damping=0.0,
        ),
    },
)