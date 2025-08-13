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
from .anymal_c_multi_agent_adversarial_single_agent import AnymalCAdversarialSingleAgentEnv, AnymalCAdversarialSingleAgentEnvCfg

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