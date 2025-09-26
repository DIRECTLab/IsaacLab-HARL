# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Leatherback Rover locomotion environment.
"""

import gymnasium as gym

from . import agents
from .yahboom_search_and_rescue import YahboomSearchAndRescueEnv, YahboomSearchAndRescueEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Yahboom-Search-And-Rescue-v0",
    entry_point=YahboomSearchAndRescueEnv, 
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": YahboomSearchAndRescueEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_flat_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalCSoccerFlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
        "harl_happo_cfg_entry_point": f"{agents.__name__}:harl_happo_cfg.yaml",
    },
)
