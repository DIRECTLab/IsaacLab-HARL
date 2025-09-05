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

from .leatherback_sumo_stage1 import LeatherbackSumoStage1Env, LeatherbackSumoStage1EnvCfg
from .leatherback_sumo_stage2 import LeatherbackSumoStage2Env, LeatherbackSumoStage2EnvCfg
from .leatherback_sumo_ma_stage1 import LeatherbackSumoMAStage1Env, LeatherbackSumoMAStage1EnvCfg
from .leatherback_sumo_ma_stage2 import LeatherbackSumoMAStage2Env, LeatherbackSumoMAStage2EnvCfg
from .leatherback_soccer_stage_1 import LeatherbackStage1SoccerEnv, LeatherbackStage1SoccerEnvCfg
from .leatherback_soccer_stage_2 import LeatherbackStage2AdversarialSoccerEnv, LeatherbackStage2AdversarialSoccerEnvCfg

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
    id="leatherback-Sumo-Direct-Stage1-v0",
    entry_point=LeatherbackSumoStage1Env,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": LeatherbackSumoStage1EnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_flat_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalCFlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
        "harl_happo_adv_cfg_entry_point": f"{agents.__name__}:harl_happo_adv_cfg.yaml",
        "harl_happo_cfg_entry_point": f"{agents.__name__}:harl_happo_cfg.yaml",
        "harl_mappo_cfg_entry_point": f"{agents.__name__}:harl_mappo_cfg.yaml",
    },
)

gym.register(
    id="leatherback-Sumo-Direct-Stage2-v0",
    entry_point=LeatherbackSumoStage2Env,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": LeatherbackSumoStage2EnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_flat_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalCFlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
        "harl_happo_adv_cfg_entry_point": f"{agents.__name__}:harl_happo_adv_cfg.yaml",
        "harl_happo_cfg_entry_point": f"{agents.__name__}:harl_happo_cfg.yaml",
        "harl_mappo_cfg_entry_point": f"{agents.__name__}:harl_mappo_cfg.yaml",
    },
)


gym.register(
    id="leatherback-Sumo-Direct-MA-Stage1-v0",
    entry_point=LeatherbackSumoMAStage1Env,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": LeatherbackSumoMAStage1EnvCfg,
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

gym.register(
    id="Leatherback-Stage1-Soccer-v0",
    entry_point=LeatherbackStage1SoccerEnv, 
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": LeatherbackStage1SoccerEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_flat_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalCSoccerFlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
        "harl_happo_cfg_entry_point": f"{agents.__name__}:harl_happo_cfg.yaml",
    },
)

gym.register(
    id="Leatherback-Stage2-Soccer-v0",
    entry_point=LeatherbackStage2AdversarialSoccerEnv, 
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": LeatherbackStage2AdversarialSoccerEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_flat_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalCSoccerFlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
        "harl_happo_cfg_entry_point": f"{agents.__name__}:harl_happo_cfg.yaml",
        "harl_happo_adv_cfg_entry_point": f"{agents.__name__}:harl_happo_adv_cfg.yaml",
    },
)