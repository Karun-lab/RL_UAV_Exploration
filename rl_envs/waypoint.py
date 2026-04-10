# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Waypoints manager for Iris RL environment.
"""

import numpy as np

class Waypoints:
    def __init__(self):
        self.x_range = (-5.0, 5.0)
        self.y_range = (-5.0, 5.0)
        self.z_range = (1.0, 3.0)

    def sample(self):
        return np.array([
            np.random.uniform(*self.x_range),
            np.random.uniform(*self.y_range),
            np.random.uniform(*self.z_range),
        ], dtype=np.float32)