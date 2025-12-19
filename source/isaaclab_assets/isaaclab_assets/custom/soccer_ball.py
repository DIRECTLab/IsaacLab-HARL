# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for a Minitank robot with an arm joint."""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg

sim_path = sim_utils.__path__[0]
SOURCE_PATH = sim_path[: sim_path.index("source")] + "source"

SOCCERBALL_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{SOURCE_PATH}/isaaclab_assets/isaaclab_assets/custom/assets/soccer_ball.usda",
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
    ),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    actuators={},
)
