# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Ant locomotion environment.
"""

import gymnasium as gym

from . import agents
from .three_dim_gal import (
    ThreeDimGalEnvCfg,
    ThreeDimGalEnv
)

from .minitank import (
    MinitankEnvCfg,
    MinitankEnv
)

from .three_dim_gal_cameras import (
    ThreeDimGalCamerasEnvCfg,
    ThreeDimGalCamerasEnv
)

from .minitank_adversarial import (
    MinitankAdversarialEnvCfg,
    MinitankAdversarialEnv
)

# from .minitank_adversarial_teams_stage_1 import (
#     MinitankAdversarialTeamsEnv,
#     MinitankAdversarialTeamsEnvCfg
# )

from .minitank_stage_1 import (
    MinitankStage1Env,
    MinitankStage1EnvCfg
)
from .minitank_stage_2 import (
    MinitankStage2Env,
    MinitankStage2EnvCfg
)

##
# Register Gym environments.
##

gym.register(
    id="3dg-Direct-v0",
    entry_point=ThreeDimGalEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ThreeDimGalEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_flat_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalCFlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
        "harl_happo_cfg_entry_point": f"{agents.__name__}:harl_happo_cfg.yaml",
        "harl_mappo_cfg_entry_point": f"{agents.__name__}:harl_mappo_cfg.yaml",
    },
)


gym.register(
    id="Minitank-Direct-v0",
    entry_point=MinitankEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": MinitankEnvCfg,
        "harl_happo_cfg_entry_point": f"{agents.__name__}:harl_happo_cfg.yaml",
    },
)


gym.register(
    id="Minitank-Adversarial-Direct-v0",
    entry_point=MinitankAdversarialEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": MinitankAdversarialEnvCfg,
        "harl_happo_cfg_entry_point": f"{agents.__name__}:harl_happo_cfg.yaml",
        "harl_happo_adv_cfg_entry_point": f"{agents.__name__}:harl_happo_adv_cfg.yaml",
    },
)


gym.register(
    id="Minitank-Adversarial-Cameras-Direct-v0",
    entry_point=ThreeDimGalCamerasEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ThreeDimGalCamerasEnvCfg,
        "harl_happo_cfg_entry_point": f"{agents.__name__}:harl_happo_cfg.yaml",
        "harl_happo_adv_cfg_entry_point": f"{agents.__name__}:harl_happo_adv_cfg.yaml",
    },
)


# Register the new minitank adversarial teams stage 1 environment
# gym.register(
#     id="Isaac-Minitank-Adversarial-Teams-Stage-1-v0",
#     entry_point=MinitankAdversarialTeamsEnv,
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": MinitankAdversarialTeamsEnvCfg,
#         "harl_happo_adv_cfg_entry_point": f"{agents.__name__}:harl_happo_cfg.yaml",
#     },
# )

gym.register(
    id="Minitank-Stage-1-Direct-v0",
    entry_point=MinitankStage1Env,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": MinitankStage1EnvCfg,
        # "harl_happo_cfg_entry_point": f"{agents.__name__}:harl_happo_cfg.yaml",
        "harl_happo_adv_cfg_entry_point": f"{agents.__name__}:harl_happo_adv_cfg.yaml",
    },
)

gym.register(
    id="Minitank-Stage-2-Direct-v0",
    entry_point=MinitankStage2Env,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": MinitankStage2EnvCfg,
        # "harl_happo_cfg_entry_point": f"{agents.__name__}:harl_happo_cfg.yaml",
        "harl_happo_adv_cfg_entry_point": f"{agents.__name__}:harl_happo_adv_cfg.yaml",
    },
)