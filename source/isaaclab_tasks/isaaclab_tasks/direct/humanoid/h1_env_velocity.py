# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab_assets import H1_CFG

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.sensors import ContactSensor, ContactSensorCfg, TiledCamera, TiledCameraCfg

from isaaclab_tasks.direct.locomotion.locomotion_velocity_control import LocomotionVelocityEnv


@configclass
class H1VelocityEnvCfg(DirectMARLEnvCfg):
    # env
    episode_length_s = 15.0
    decimation = 4
    action_scale = 1.0
    action_space = 19
    observation_space = 69
    state_space = 0


    # env
    action_spaces = {f"robot_{i}": 19 for i in range(1)}
    # observation_space = 48
    observation_spaces = {f"robot_{i}": 30 for i in range(1)}
    state_space = 0
    state_spaces = {f"robot_{i}": 0 for i in range(1)}
    possible_agents = ["robot_0"]

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # robot
    robot_0: ArticulationCfg = H1_CFG.replace(prim_path="/World/envs/env_.*/Robot_0")
    contact_sensor_0: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot_0/.*_ankle_link", history_length=3, update_period=(1/120), track_air_time=True,
    )
    joint_gears: list = [
        50.0,  # left_hip_yaw
        50.0,  # right_hip_yaw
        50.0,  # torso
        50.0,  # left_hip_roll
        50.0,  # right_hip_roll
        50.0,  # left_shoulder_pitch
        50.0,  # right_shoulder_pitch
        50.0,  # left_hip_pitch
        50.0,  # right_hip_pitch
        50.0,  # left_shoulder_roll
        50.0,  # right_shoulder_roll
        50.0,  # left_knee
        50.0,  # right_knee
        50.0,  # left_shoulder_yaw
        50.0,  # right_shoulder_yaw
        50.0,  # left_ankle
        50.0,  # right_ankle
        50.0,  # left_elbow
        50.0,  # right_elbow
    ]

    period = 2
    
    # Reward Scales
    gait_weight = 0.5
    feet_slide_weight = -0.2
    feet_clearance_weight = 20.0
    heading_weight: float = 0.5
    up_weight: float = 0.1

    energy_cost_scale: float = 0.05
    smoothness_cost_scale: float = 0.05
    actions_cost_scale: float = 0.01
    alive_reward_scale: float = 0.15
    dof_vel_scale: float = 0.1

    death_cost: float = -1.0
    termination_height: float = 0.8

    task_velocity_scale: float = 1.0
    angular_velocity_scale: float = 0.5
    contact_force_scale: float = 0.01


class H1VelocityEnv(LocomotionVelocityEnv):
    cfg: H1VelocityEnvCfg

    def __init__(self, cfg: H1VelocityEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
