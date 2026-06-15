# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for a Minitank robot with an arm joint."""

from pathlib import Path

import isaaclab
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg

isaaclab_asset_path = Path(
    Path(isaaclab.__path__[0]).parent.parent, "isaaclab_assets", "isaaclab_assets", "custom", "assets"
)
USD_PATH = str(Path(isaaclab_asset_path, "leatherback_simple_better.usd"))

SOCCERBALL_CFG = RigidObjectCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(Path(isaaclab_asset_path, "soccer_ball.usda")),
        # The authored USD is scaled down twice internally, so we compensate here
        # to bring the ball back to a visible, soccer-ball-sized object in meters.
        scale=(0.5, 0.5, 0.5),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
)
