# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for a Minitank robot with an arm joint."""

from pathlib import Path

import isaaclab
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg

isaaclab_asset_path = Path(
    Path(isaaclab.__path__[0]).parent.parent, "isaaclab_assets", "isaaclab_assets", "custom", "assets"
)
USD_PATH = str(Path(isaaclab_asset_path, "leatherback_simple_better.usd"))

SOCCERBALL_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(Path(isaaclab_asset_path, "soccer_ball.usda")),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
    ),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    actuators={},
)
