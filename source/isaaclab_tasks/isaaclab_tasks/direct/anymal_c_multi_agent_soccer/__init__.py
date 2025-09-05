# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Ant locomotion environment.
"""

import gymnasium as gym

from . import agents
from .anymal_c_soccer_find_ball_teacher import AnymalCFindBallHappoEnv, AnymalCHappoFindBallEnvCfg
from .leatherback_anymal_soccer_stage_2 import AnymalCPlaySoccer, AnymalCSoccerCfg
from .anymal_c_soccer_running import AnymalSoccerRunningEnv, AnymalSoccerRunningEnvCfg
from .anymal_c_soccer_stage_1 import AnymalStage1SoccerEnv, AnymalStage1SoccerEnvCfg
##
# Register Gym environments.
##

gym.register(
    id="Flat_AnymalC_Find_Ball_Teacher-v0",
    entry_point=AnymalCFindBallHappoEnv, 
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": AnymalCHappoFindBallEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_flat_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalCSoccerFlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
        "harl_happo_cfg_entry_point": f"{agents.__name__}:harl_happo_cfg.yaml",
    },
)

gym.register(
    id="Flat_AnymalC_Play_Soccer_Direct-v0",
    entry_point=AnymalCPlaySoccer, 
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": AnymalCSoccerCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_flat_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalCSoccerFlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
        "harl_happo_cfg_entry_point": f"{agents.__name__}:harl_happo_cfg.yaml",
    },
)

gym.register(
    id="AnymalC_Soccer_Soccer_Running-v0",
    entry_point=AnymalSoccerRunningEnv, 
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": AnymalSoccerRunningEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_flat_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalCSoccerFlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
        "harl_happo_cfg_entry_point": f"{agents.__name__}:harl_happo_cfg.yaml",
    },
)

gym.register(
    id="AnymalC_Soccer_Stage1-v0",
    entry_point=AnymalStage1SoccerEnv, 
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": AnymalStage1SoccerEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_flat_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalCSoccerFlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
        "harl_happo_cfg_entry_point": f"{agents.__name__}:harl_happo_cfg.yaml",
    },
)