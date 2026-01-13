# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Ant locomotion environment.
"""

import gymnasium as gym

from . import agents
from ..adversarial.sumo.anymal_c.anymal_c_go_to_point_sumo import AnymalCGoToPointSumo, AnymalCGoToPointSumoCfg
from ..adversarial.sumo.anymal_c.sumo_stage_1_blocks_push import AnymalCBlocksPushEnv, AnymalCBlocksPushEnvCfg
from .sumo_stage_2 import SumoStage2Env, SumoStage2EnvCfg
from ..adversarial.sumo.heterogeneous.sumo_stage_2_hetero_by_team import SumoStage2HeteroByTeamEnv, SumoStage2HeteroByTeamEnvCfg
from ..adversarial.sumo.heterogeneous.sumo_stage_2_hetero_within_team import SumoStage2HeteroEnv, SumoStage2HeteroEnvCfg

##
# Register Gym environments.
##





gym.register(
    id="Anymal-C-Sumo-Stage2-v0",
    entry_point=SumoStage2Env,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": SumoStage2EnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_flat_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalCFlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
        "harl_happo_cfg_entry_point": f"{agents.__name__}:harl_happo_cfg.yaml",
        "harl_happo_adv_cfg_entry_point": f"{agents.__name__}:harl_happo_adv_cfg.yaml",
    },
)




