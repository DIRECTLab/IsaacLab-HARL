# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Ant locomotion environment.
"""

import gymnasium as gym

from . import agents
from .minitank import MinitankEnv, MinitankEnvCfg
from .three_dim_gal import ThreeDimGalEnv, ThreeDimGalEnvCfg
from .three_dim_gal_cameras import ThreeDimGalCamerasEnv, ThreeDimGalCamerasEnvCfg

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
    id="Minitank-Adversarial-Cameras-Direct-v0",
    entry_point=ThreeDimGalCamerasEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ThreeDimGalCamerasEnvCfg,
        "harl_happo_cfg_entry_point": f"{agents.__name__}:harl_happo_cfg.yaml",
        "harl_happo_adv_cfg_entry_point": f"{agents.__name__}:harl_happo_adv_cfg.yaml",
    },
)
