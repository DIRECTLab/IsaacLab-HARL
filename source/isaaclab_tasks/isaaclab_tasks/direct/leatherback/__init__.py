# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Leatherback Rover locomotion environment.
"""

import gymnasium as gym

from . import agents
from .leatherback import (
    LeatherbackEnvCfg
)

from .leatherback_soccar import (
    LeatherbackSoccerEnvCfg
)

##
# Register Gym environments.
##

gym.register(
    id="leatherback-Direct-v0",
    entry_point="isaaclab_tasks.direct.leatherback.leatherback:LeatherbackEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": LeatherbackEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_flat_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalCFlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
        "harl_happo_cfg_entry_point": f"{agents.__name__}:harl_happo_cfg.yaml",
        "harl_mappo_cfg_entry_point": f"{agents.__name__}:harl_mappo_cfg.yaml",
    },
)

gym.register(
    id="leatherback-soccer-Direct-v0",
    entry_point="isaaclab_tasks.direct.leatherback.leatherback_2:LeatherbackSoccerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": LeatherbackSoccerEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_flat_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalCFlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
        "harl_happo_cfg_entry_point": f"{agents.__name__}:harl_happo_cfg.yaml",
        "harl_mappo_cfg_entry_point": f"{agents.__name__}:harl_mappo_cfg.yaml",
    },
)
