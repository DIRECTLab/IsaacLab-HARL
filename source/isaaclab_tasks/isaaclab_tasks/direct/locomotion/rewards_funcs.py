import torch
from isaaclab.utils.math import quat_rotate_inverse, yaw_quat

def gait_reward(
    env,
    period: float,
    offset: list[float],
    threshold: float = 0.5,
    commands=None
) -> torch.Tensor:

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

def feet_slide_reward(
        env
    ) -> torch.Tensor:
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

def foot_clearance_reward(
    env, target_height: float, std: float = .05, tanh_mult: float = 2.0
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset = env.robots["robot_0"]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, env.feet_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, env.feet_ids, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)

def track_lin_vel_xy(
        robot,
        commands,
        std=0.25
):
    # linear velocity tracking
    vel_yaw = quat_rotate_inverse(yaw_quat(robot.data.root_quat_w), robot.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(commands[:, :2] - vel_yaw[:, :2]), dim=1
    )
    lin_vel_error_mapped = torch.exp(-lin_vel_error / std)
    return lin_vel_error_mapped

def track_yaw_rate(
        robot,
        commands,
        std=0.25
):
    yaw_rate_error = torch.square(commands[:, 2] - robot.data.root_ang_vel_w[:, 2])
    yaw_rate_error_mapped = torch.exp(-yaw_rate_error / std)
    return yaw_rate_error_mapped