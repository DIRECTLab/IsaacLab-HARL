# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Leatherback Rover locomotion environment.
"""

import gymnasium as gym

from . import agents
from .leatherback import LeatherbackEnvCfg
from ..adversarial.sumo.leatherback.leatherback_sumo_ma_stage1 import LeatherbackSumoMAStage1Env, LeatherbackSumoMAStage1EnvCfg
from .leatherback_sumo_ma_stage1_same_team import (
    LeatherbackSumoMAStage1SameTeamEnv,
    LeatherbackSumoMAStage1SameTeamEnvCfg,
)
from .leatherback_sumo_ma_stage2 import LeatherbackSumoMAStage2Env, LeatherbackSumoMAStage2EnvCfg
from .leatherback_vs_leatherback_sumo_stage_1 import LeatherbackVSLeatherbackSumoStage1Env, LeatherbackVSLeatherbackSumoStage1EnvCfg
from .leatherback_vs_leatherback_sumo_stage_2 import LeatherbackVSLeatherbackSumoStage2Env, LeatherbackVSLeatherbackSumoStage2EnvCfg

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
    id="leatherback-vs-leatherback-Sumo-Direct-Stage1-v0",
    entry_point=LeatherbackVSLeatherbackSumoStage1Env,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": LeatherbackVSLeatherbackSumoStage1EnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_flat_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalCFlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
        "harl_happo_adv_cfg_entry_point": f"{agents.__name__}:harl_happo_adv_cfg.yaml",
        "harl_happo_cfg_entry_point": f"{agents.__name__}:harl_happo_cfg.yaml",
        "harl_mappo_cfg_entry_point": f"{agents.__name__}:harl_mappo_cfg.yaml",
    },
)

gym.register(
    id="leatherback-vs-leatherback-Sumo-Direct-Stage2-v0",
    entry_point=LeatherbackVSLeatherbackSumoStage2Env,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": LeatherbackVSLeatherbackSumoStage2EnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_flat_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalCFlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
        "harl_happo_adv_cfg_entry_point": f"{agents.__name__}:harl_happo_adv_cfg.yaml",
        "harl_happo_cfg_entry_point": f"{agents.__name__}:harl_happo_cfg.yaml",
        "harl_mappo_cfg_entry_point": f"{agents.__name__}:harl_mappo_cfg.yaml",
    },
)



gym.register(
    id="leatherback-Sumo-Direct-MA-Stage2-v0",
    entry_point=LeatherbackSumoMAStage2Env,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": LeatherbackSumoMAStage2EnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_flat_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalCFlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
        "harl_happo_adv_cfg_entry_point": f"{agents.__name__}:harl_happo_adv_cfg.yaml",
        "harl_happo_cfg_entry_point": f"{agents.__name__}:harl_happo_cfg.yaml",
        "harl_mappo_cfg_entry_point": f"{agents.__name__}:harl_mappo_cfg.yaml",
    },
)

# gym.register(
#     id="Leatherback-Stage1-Soccer-v0",
#     entry_point=LeatherbackStage1SoccerEnv,
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": LeatherbackStage1SoccerEnvCfg,
#         "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_flat_ppo_cfg.yaml",
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalCSoccerFlatPPORunnerCfg",
#         "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
#         "harl_happo_cfg_entry_point": f"{agents.__name__}:harl_happo_cfg.yaml",
#     },
# )

gym.register(
    id="Leatherback-Stage1-Soccer-v0-Same-Team",
    entry_point=LeatherbackSumoMAStage1SameTeamEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": LeatherbackSumoMAStage1SameTeamEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_flat_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalCSoccerFlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
        "harl_happo_cfg_entry_point": f"{agents.__name__}:harl_happo_cfg.yaml",
        "harl_happo_adv_cfg_entry_point": f"{agents.__name__}:harl_happo_adv_cfg.yaml",
    },
)

