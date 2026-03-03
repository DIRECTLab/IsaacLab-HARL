# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch


def gait_reward(env, period: float, offset: list[float], threshold: float = 0.5, commands=None) -> torch.Tensor:

    contact_sensor = env.contact_sensors["robot_0"]
    is_contact = contact_sensor.data.current_contact_time[:, env.feet_ids] > 0

    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    phases = []
    for offset_ in offset:
        phase = (global_phase + offset_) % 1.0
        phases.append(phase)
    leg_phase = torch.cat(phases, dim=-1)

    reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    for i in range(len(env.feet_ids)):
        is_stance = leg_phase[:, i] < threshold
        reward += ~(is_stance ^ is_contact[:, i])

    if commands is not None:
        cmd_norm = torch.norm(commands, dim=1)
        reward *= cmd_norm > 0.1
    return reward


def feet_slide_reward(env) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor = env.contact_sensors["robot_0"]
    contacts = contact_sensor.data.net_forces_w_history[:, :, env.feet_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene["robot_0"]

    body_vel = asset.data.body_lin_vel_w[:, env.feet_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward
