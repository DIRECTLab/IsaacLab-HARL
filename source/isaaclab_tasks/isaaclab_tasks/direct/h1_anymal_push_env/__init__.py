# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Ant locomotion environment.
"""

import gymnasium as gym

from . import agents
from .h1_anymal_push_env import (
    HeterogeneousPushMultiAgent,
    HeterogeneousPushMultiAgentFlatEnvCfg,
    HeterogeneousPushMultiAgentRoughEnvCfg,
)

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Anymal-H1-Push-Rough-Direct-v0",
    entry_point="isaaclab_tasks.direct.h1_anymal_push_env:HeterogeneousPushMultiAgent",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": HeterogeneousPushMultiAgentRoughEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_flat_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalCFlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
        "harl_happo_cfg_entry_point": f"{agents.__name__}:harl_happo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Anymal-H1-Push-Direct-v0",
    entry_point="isaaclab_tasks.direct.h1_anymal_push_env:HeterogeneousPushMultiAgent",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": HeterogeneousPushMultiAgentFlatEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_flat_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalCFlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
        "harl_happo_cfg_entry_point": f"{agents.__name__}:harl_happo_cfg.yaml",
    },
)
