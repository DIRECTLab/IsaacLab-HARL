# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Ant locomotion environment.
"""

import gymnasium as gym

from . import agents
from .anymal_c_soccer_running import AnymalSoccerRunningEnv, AnymalSoccerRunningEnvCfg
from .anymal_c_soccer_stage_1 import AnymalStage1SoccerEnv, AnymalStage1SoccerEnvCfg
from .anymal_c_soccer_stage_2 import AnymalStage2SoccerEnv, AnymalStage2SoccerEnvCfg
from .anymal_c_soccer_ma_homo import AnymalSoccerMAHomoEnv, AnymalSoccerMAHomoEnvCfg
from .anymals_vs_leatherbacks import AnymalSoccerHeteroByTeamEnv, AnymalSoccerHeteroByTeamEnvCfg
from .anymal_leatherback_on_each_team import AnymalSoccerHeteroInTeamEnv, AnymalSoccerHeteroInTeamEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="AnymalC_Soccer_Soccer_Running-v1",
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
    id="AnymalC_Soccer_Stage1-v1",
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

gym.register(
    id="AnymalC_Soccer_Stage2-v1",
    entry_point=AnymalStage2SoccerEnv, 
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": AnymalStage2SoccerEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_flat_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalCSoccerFlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
        "harl_happo_cfg_entry_point": f"{agents.__name__}:harl_happo_cfg.yaml",
    },
)

gym.register(
    id="AnymalC_Soccer_MA_Homo-v1",
    entry_point=AnymalSoccerMAHomoEnv, 
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": AnymalSoccerMAHomoEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_flat_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalCSoccerFlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
        "harl_happo_cfg_entry_point": f"{agents.__name__}:harl_happo_cfg.yaml",
        "harl_happo_adv_cfg_entry_point": f"{agents.__name__}:harl_happo_adv_cfg.yaml",
    },
)

gym.register(
    id="AnymalC_Soccer_Hetero_By_Team-v0",
    entry_point=AnymalSoccerHeteroByTeamEnv, 
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": AnymalSoccerHeteroByTeamEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_flat_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalCSoccerFlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
        "harl_happo_cfg_entry_point": f"{agents.__name__}:harl_happo_cfg.yaml",
        "harl_happo_adv_cfg_entry_point": f"{agents.__name__}:harl_happo_adv_cfg.yaml",
    },
)

gym.register(
    id="AnymalC_Soccer_Hetero_In_Team-v0",
    entry_point=AnymalSoccerHeteroInTeamEnv, 
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": AnymalSoccerHeteroInTeamEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_flat_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalCSoccerFlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
        "harl_happo_cfg_entry_point": f"{agents.__name__}:harl_happo_cfg.yaml",
        "harl_happo_adv_cfg_entry_point": f"{agents.__name__}:harl_happo_adv_cfg.yaml",
    },
)