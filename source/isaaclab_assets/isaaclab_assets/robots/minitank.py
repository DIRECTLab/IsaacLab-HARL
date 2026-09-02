# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for a Minitank robot with an arm joint."""

from harl.utils.hf_assets import asset_path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

MINITANK_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=asset_path("minitank.usda"),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
    ),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    actuators={
        "arm_joint": ImplicitActuatorCfg(
            joint_names_expr=["arm_joint"],
            effort_limit=1000000.0,
            velocity_limit=5.0,
            stiffness=0.0,
            damping=100000000.0,
        ),
        "rotor_joint": ImplicitActuatorCfg(
            joint_names_expr=["rotor_joint"],
            effort_limit=1000000.0,
            velocity_limit=5.0,
            stiffness=0.0,
            damping=100000000.0,
        ),
    },
)
