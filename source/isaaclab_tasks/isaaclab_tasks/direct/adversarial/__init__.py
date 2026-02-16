# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Leatherback Rover locomotion environment.
"""

import gymnasium as gym

from . import agents
from .soccer.leatherback.leatherback_soccer_stage_1 import LeatherbackStage1SoccerEnv, LeatherbackStage1SoccerEnvCfg
from .soccer.leatherback.leatherbacks_vs_leatherbacks import LeatherbacksVSLeatherbacksSoccerEnv, LeatherbacksVSLeatherbacksSoccerEnvCfg
from .soccer.heterogeneous.anymals_vs_leatherbacks import AnymalSoccerHeteroByTeamEnv, AnymalSoccerHeteroByTeamEnvCfg
from .soccer.heterogeneous.go2_vs_leatherbacks import go2SoccerHeteroByTeamEnv, go2SoccerHeteroByTeamEnvCfg
from .soccer.anymal_c.anymal_c_go_to_point_soccer import AnymalCGoToPointSoccerEnv, AnymalCGoToPointSoccerEnvCfg
from .soccer.anymal_c.anymal_c_soccer_stage_2 import AnymalStage2SoccerEnv, AnymalStage2SoccerEnvCfg
from .soccer.anymal_c.anymal_c_soccer_stage_1 import AnymalStage1SoccerEnv, AnymalStage1SoccerEnvCfg
from .soccer.heterogeneous.anymal_vs_leatherback import AnymalVsLeatherbackSoccerEnv, AnymalVsLeatherbackSoccerEnvCfg
from .sumo.heterogeneous.sumo_stage_2_hetero_by_team import SumoStage2HeteroByTeamEnv, SumoStage2HeteroByTeamEnvCfg
from .sumo.heterogeneous.sumo_stage_2_hetero_within_team import SumoStage2HeteroEnv, SumoStage2HeteroEnvCfg
from .sumo.leatherback.leatherback_sumo_ma_stage1 import LeatherbackSumoMAStage1Env, LeatherbackSumoMAStage1EnvCfg
from .sumo.anymal_c.anymal_c_go_to_point_sumo import AnymalCGoToPointSumo, AnymalCGoToPointSumoCfg
from .minitank_drone.heterogeneous.minitank_adversarial import MinitankAdversarialEnv, MinitankAdversarialEnvCfg
from .sumo.anymal_c.sumo_stage_1_blocks_push import AnymalCBlocksPushEnv, AnymalCBlocksPushEnvCfg
##
# Register Gym environments.
##

gym.register(
    id="Leatherback-Stage1-Soccer-v0",
    entry_point=LeatherbackStage1SoccerEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": LeatherbackStage1SoccerEnvCfg,
        "harl_happo_cfg_entry_point": f"{agents.__name__}:harl_happo_cfg.yaml",
        "harl_happo_adv_cfg_entry_point": f"{agents.__name__}:harl_happo_adv_cfg.yaml",
    },
)

gym.register(
    id="Leatherback-Stage2-Soccer-v0",
    entry_point=LeatherbacksVSLeatherbacksSoccerEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": LeatherbacksVSLeatherbacksSoccerEnvCfg,
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
        "harl_happo_adv_cfg_entry_point": f"{agents.__name__}:harl_happo_adv_cfg.yaml",
    },
)

gym.register(
    id="go2_Soccer_Hetero_By_Team-v0",
    entry_point=go2SoccerHeteroByTeamEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": go2SoccerHeteroByTeamEnvCfg,
        "harl_happo_adv_cfg_entry_point": f"{agents.__name__}:harl_happo_adv_cfg.yaml",
    },
)

gym.register(
    id="Sumo-Stage2-Hetero-By-Team-v0",
    entry_point=SumoStage2HeteroByTeamEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": SumoStage2HeteroByTeamEnvCfg,
        "harl_happo_adv_cfg_entry_point": f"{agents.__name__}:harl_happo_adv_cfg.yaml",
    },
)

gym.register(
    id="Sumo-Stage2-Hetero-v0",
    entry_point=SumoStage2HeteroEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": SumoStage2HeteroEnvCfg,
        "harl_happo_adv_cfg_entry_point": f"{agents.__name__}:harl_happo_adv_cfg.yaml",
    },
)

gym.register(
    id="Minitank-Adversarial-Direct-v0",
    entry_point=MinitankAdversarialEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": MinitankAdversarialEnvCfg,
        "harl_happo_adv_cfg_entry_point": f"{agents.__name__}:harl_happo_adv_cfg.yaml",
    },
)

gym.register(
    id="Anymal-C-Sumo-Stage1-Blocks-Push-v0",
    entry_point=AnymalCBlocksPushEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": AnymalCBlocksPushEnvCfg,
        "harl_happo_adv_cfg_entry_point": f"{agents.__name__}:harl_happo_adv_cfg.yaml",
    },
)

gym.register(
    id="Anymal-C-Go-To-Point-Sumo",
    entry_point=AnymalCGoToPointSumo,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": AnymalCGoToPointSumoCfg,
        "harl_happo_adv_cfg_entry_point": f"{agents.__name__}:harl_happo_adv_cfg.yaml",
    },
)

gym.register(
    id="AnymalC_Soccer_Go_To_Point_Stage_0",
    entry_point=AnymalCGoToPointSoccerEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": AnymalCGoToPointSoccerEnvCfg,
        "harl_happo_adv_cfg_entry_point": f"{agents.__name__}:harl_happo_adv_cfg.yaml",
    },
)

gym.register(
    id="AnymalC_Soccer_Go_To_Ball_Stage_1",
    entry_point=AnymalStage1SoccerEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": AnymalStage1SoccerEnvCfg,
        "harl_happo_cfg_entry_point": f"{agents.__name__}:harl_happo_cfg.yaml",
    },
)

gym.register(
    id="AnymalC_Soccer_Score_Goals_Stage_2",
    entry_point=AnymalStage2SoccerEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": AnymalStage2SoccerEnvCfg,
        "harl_happo_cfg_entry_point": f"{agents.__name__}:harl_happo_cfg.yaml",
    },
)

gym.register(
    id="leatherback-Sumo-Direct-MA-Stage1-v0",
    entry_point=LeatherbackSumoMAStage1Env,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": LeatherbackSumoMAStage1EnvCfg,
        "harl_happo_adv_cfg_entry_point": f"{agents.__name__}:harl_happo_adv_cfg.yaml",
    },
)

gym.register(
    id="AnymalC-VS-Leatherback-Soccer-v0",
    entry_point=AnymalVsLeatherbackSoccerEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": AnymalVsLeatherbackSoccerEnvCfg,
        "harl_happo_adv_cfg_entry_point": f"{agents.__name__}:harl_happo_adv_cfg.yaml",
    },
)