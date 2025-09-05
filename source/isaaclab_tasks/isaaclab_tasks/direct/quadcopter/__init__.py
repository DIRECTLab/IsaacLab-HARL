# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Quacopter environment.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Quadcopter-Direct-v0",
    entry_point=f"{__name__}.quadcopter_env:QuadcopterEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.quadcopter_env:QuadcopterEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadcopterPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Quadcopter-Direct-HARL-v0",
    entry_point=f"{__name__}.quadcopter_env_happo:QuadcopterMARLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.quadcopter_env_happo:QuadcopterMARLEnvCfg",
        "harl_happo_cfg_entry_point": f"{agents.__name__}:harl_happo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadcopterPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Quadcopter-Direct-Team-HAPPO-v0",
    entry_point=f"{__name__}.quadcopter_env_happo_stage_1:QuadcopterMARLEnvTeam",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.quadcopter_env_happo_stage_1:QuadcopterMARLEnvTeamCfg",
        "harl_happo_adv_cfg_entry_point": f"{agents.__name__}:harl_happo_adv_cfg.yaml",
        # "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        # "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadcopterPPORunnerCfg",
        # "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
gym.register(
    id="Isaac-Quadcopter-Direct-Stage1-SingleAgent-v0",
    entry_point=f"{__name__}.quadcopter_env_happo_v2_state_1:DroneStage1EnvSingleAgentMARLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.quadcopter_env_happo_v2_state_1:DroneStage1EnvSingleAgentCfg",
        # "harl_happo_cfg_entry_point": f"{agents.__name__}:harl_happo_cfg.yaml",
        "harl_happo_adv_cfg_entry_point": f"{agents.__name__}:harl_happo_adv_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Quadcopter-Direct-Stage2-MultiAgent-v0",
    entry_point=f"{__name__}.quadcopter_env_happo_v2_state_2:DroneStage2EnvMultiAgentMARLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.quadcopter_env_happo_v2_state_2:DroneStage2EnvMultiAgentCfg",
        # "harl_happo_cfg_entry_point": f"{agents.__name__}:harl_happo_cfg.yaml",
        "harl_happo_adv_cfg_entry_point": f"{agents.__name__}:harl_happo_adv_cfg.yaml",
    },
)