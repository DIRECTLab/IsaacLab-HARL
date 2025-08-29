# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Ant locomotion environment.
"""

import gymnasium as gym

from . import agents
from .anymal_c_multi_agent import AnymalCMultiAgentBar, AnymalCMultiAgentFlatEnvCfg
from .anymal_c_multi_agent_adversarial import AnymalCAdversarialEnv, AnymalCAdversarialEnvCfg
from .anymal_c_multi_agent_adversarial_soccer import AnymalCAdversarialSoccerEnv, AnymalCAdversarialSoccerEnvCfg
from .anymal_c_multi_agent_adversarial_single_agent import AnymalCAdversarialSingleAgentEnv, AnymalCAdversarialSingleAgentEnvCfg
from .anymal_c_multi_agent_adversarial_sumo_stage1 import AnymalCAdversarialSumoStage1Env, AnymalCAdversarialSumoStage1EnvCfg
from .anymal_c_multi_agent_adversarial_sumo_stage2 import AnymalCAdversarialSumoStage2Env, AnymalCAdversarialSumoStage2EnvCfg
# from .sumo_stage_1 import SumoStage1Env, SumoStage1EnvCfg
from .sumo_stage_1_single_agent import SumoStage1EnvSingleAgent, SumoStage1EnvSingleAgentCfg
from .anymal_c_multi_agent_adversarial_same_team import AnymalCAdversarialSameTeamEnv, AnymalCAdversarialSameTeamEnvCfg
from .anymal_c_multi_agent_bar_same_team import AnymalCMultiAgentFlatSameTeamBarEnv, AnymalCMultiAgentFlatSameTeamBarEnvCfg
from .sumo_stage_1_blocks import SumoStage1BlocksEnv, SumoStage1BlocksEnvCfg
from .sumo_stage_1_blocks_full_obs import SumoStage1BlocksFullEnv, SumoStage1BlocksFullEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Multi-Agent-Flat-Anymal-C-Direct-v0",
    entry_point=AnymalCMultiAgentBar,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": AnymalCMultiAgentFlatEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_flat_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalCFlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
        "harl_happo_cfg_entry_point": f"{agents.__name__}:harl_happo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Multi-Agent-Flat-Anymal-C-Direct-Bar-Same-Team-v0",
    entry_point=AnymalCMultiAgentFlatSameTeamBarEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": AnymalCMultiAgentFlatSameTeamBarEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_flat_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalCFlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
        "harl_happo_cfg_entry_point": f"{agents.__name__}:harl_happo_cfg.yaml",
        "harl_happo_adv_cfg_entry_point": f"{agents.__name__}:harl_happo_adv_cfg.yaml"
    },
)

gym.register(
    id="Isaac-Multi-Agent-Flat-Anymal-C-Direct-Adversarial-v0",
    entry_point=AnymalCAdversarialEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": AnymalCAdversarialEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_flat_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalCFlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
        "harl_happo_cfg_entry_point": f"{agents.__name__}:harl_happo_cfg.yaml",
        "harl_happo_adv_cfg_entry_point": f"{agents.__name__}:harl_happo_adv_cfg.yaml"
    },
)


gym.register(
    id="Isaac-Multi-Agent-Flat-Anymal-C-Direct-Adversarial-Same-Team-v0",
    entry_point=AnymalCAdversarialSameTeamEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": AnymalCAdversarialSameTeamEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_flat_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalCFlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
        "harl_happo_cfg_entry_point": f"{agents.__name__}:harl_happo_cfg.yaml",
        "harl_happo_adv_cfg_entry_point": f"{agents.__name__}:harl_happo_adv_cfg.yaml"
    },
)

gym.register(
    id="Isaac-Multi-Agent-Flat-Anymal-C-Direct-Adversarial-Soccer-v0",
    entry_point=AnymalCAdversarialSoccerEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": AnymalCAdversarialSoccerEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_flat_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalCFlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
        "harl_happo_cfg_entry_point": f"{agents.__name__}:harl_happo_cfg.yaml",
        "harl_happo_adv_cfg_entry_point": f"{agents.__name__}:harl_happo_adv_cfg.yaml"
    },
)



gym.register(
    id="Isaac-Multi-Agent-Flat-Anymal-C-Direct-Adversarial-SA-v0",
    entry_point=AnymalCAdversarialSingleAgentEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": AnymalCAdversarialSingleAgentEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_flat_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalCFlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
        "harl_happo_cfg_entry_point": f"{agents.__name__}:harl_happo_cfg.yaml",
        "harl_happo_adv_cfg_entry_point": f"{agents.__name__}:harl_happo_adv_cfg.yaml"
    },
)

gym.register(
    id="Isaac-Multi-Agent-Flat-Anymal-C-Direct-Adversarial-Sumo-Stage1-v0",
    entry_point=AnymalCAdversarialSumoStage1Env,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": AnymalCAdversarialSumoStage1EnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_flat_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalCFlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
        "harl_happo_cfg_entry_point": f"{agents.__name__}:harl_happo_cfg.yaml",
        "harl_happo_adv_cfg_entry_point": f"{agents.__name__}:harl_happo_adv_cfg.yaml"
    },
)


gym.register(
    id="Isaac-Multi-Agent-Flat-Anymal-C-Direct-Adversarial-Sumo-Stage2-v0",
    entry_point=AnymalCAdversarialSumoStage2Env,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": AnymalCAdversarialSumoStage2EnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_flat_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalCFlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
        "harl_happo_cfg_entry_point": f"{agents.__name__}:harl_happo_cfg.yaml",
        "harl_happo_adv_cfg_entry_point": f"{agents.__name__}:harl_happo_adv_cfg.yaml"
    },
)


# gym.register(
#     id="Isaac-Multi-Agent-Flat-Sumo-Stage1-v0",
#     entry_point=SumoStage1Env,
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": SumoStage1EnvCfg,
#         "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_flat_ppo_cfg.yaml",
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalCFlatPPORunnerCfg",
#         "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
#         "harl_happo_cfg_entry_point": f"{agents.__name__}:harl_happo_cfg.yaml",
#         "harl_happo_adv_cfg_entry_point": f"{agents.__name__}:harl_happo_adv_cfg.yaml"
#     },
# )

gym.register(
    id="Isaac-Multi-Agent-Flat-Sumo-Stage1-Single-Agent-v0",
    entry_point=SumoStage1EnvSingleAgent,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": SumoStage1EnvSingleAgentCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_flat_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalCFlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
        "harl_happo_cfg_entry_point": f"{agents.__name__}:harl_happo_cfg.yaml",
        "harl_happo_adv_cfg_entry_point": f"{agents.__name__}:harl_happo_adv_cfg.yaml"
    },
)

gym.register(
    id="Isaac-Multi-Agent-Flat-Sumo-Stage1-Blocks-v0",
    entry_point=SumoStage1BlocksEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": SumoStage1BlocksEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_flat_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalCFlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
        "harl_happo_cfg_entry_point": f"{agents.__name__}:harl_happo_cfg.yaml",
        "harl_happo_adv_cfg_entry_point": f"{agents.__name__}:harl_happo_adv_cfg.yaml"
    },
)

gym.register(
    id="Isaac-Multi-Agent-Flat-Sumo-Stage1-Blocks-Full-v0",
    entry_point=SumoStage1BlocksFullEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": SumoStage1BlocksFullEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_flat_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalCFlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
        "harl_happo_cfg_entry_point": f"{agents.__name__}:harl_happo_cfg.yaml",
        "harl_happo_adv_cfg_entry_point": f"{agents.__name__}:harl_happo_adv_cfg.yaml"
    },
)