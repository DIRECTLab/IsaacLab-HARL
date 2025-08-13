# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

import isaacsim.core.utils.torch as torch_utils
from isaacsim.core.utils.torch.rotations import compute_heading_and_up, compute_rot, quat_conjugate

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg
from isaaclab.utils.math import quat_from_angle_axis
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.utils.math import quat_rotate_inverse, yaw_quat
from . import rewards_funcs as rewards

from pathlib import Path

def normalize_angle(x):
    return torch.atan2(torch.sin(x), torch.cos(x))

def define_markers() -> VisualizationMarkers:
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/myMarkers",
        markers={
            "sphere1": sim_utils.SphereCfg(
                radius=0.1,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.0, 0.0)),
            ),
            "sphere2": sim_utils.SphereCfg(
                radius=0.1,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),
            ),
            "arrow1": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                scale=(0.1, 0.1, 1.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.0, 0.0)),
            ),
            "arrow2": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                scale=(0.1, 0.1, 1.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),
            ),
        },
    )
    return VisualizationMarkers(marker_cfg)

class LocomotionVelocityEnv(DirectMARLEnv):
    cfg: DirectMARLEnvCfg

    def __init__(self, cfg: DirectMARLEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.actions = {
            agent: torch.zeros(self.num_envs, action_space, device=self.device)
            for agent, action_space in self.cfg.action_spaces.items()
        }

        self.prev_actions = {
            agent: torch.zeros(self.num_envs, action_space, device=self.device)
            for agent, action_space in self.cfg.action_spaces.items()
        }

        # Initialize the gait phase observation
        self.gait_phase_obs = self.gait_phase(self.cfg.period)

        self.action_scale = self.cfg.action_scale
        self.joint_gears = torch.tensor(self.cfg.joint_gears, dtype=torch.float32, device=self.sim.device)
        self.motor_effort_ratio = torch.ones_like(self.joint_gears, device=self.sim.device)
        self._joint_dof_idx, _ = self.robots["robot_0"].find_joints(".*")

        self.potentials = torch.zeros(self.num_envs, dtype=torch.float32, device=self.sim.device)
        self.prev_potentials = torch.zeros_like(self.potentials)
        # X/Y linear velocity and yaw angular velocity commands
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)
        
        self.start_rotation = torch.tensor([1, 0, 0, 0], device=self.sim.device, dtype=torch.float32)
        self.up_vec = torch.tensor([0, 0, 1], dtype=torch.float32, device=self.sim.device).repeat((self.num_envs, 1))

        self.inv_start_rot = quat_conjugate(self.start_rotation).repeat((self.num_envs, 1))
        self.basis_vec1 = self.up_vec.clone()

        self._episode_sums = {
            "total_reward": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "gait_reward": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "slide_reward": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "clearance_reward": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "other_rewards": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
        }

        self.feet_ids, self.feet_names = self.contact_sensors[f"robot_0"].find_bodies(".*_ankle_link")

    def _draw_markers(self, command):
        xy_commands = command.clone()
        z_commands = xy_commands[:, 2].clone()
        xy_commands[:, 2] = 0

        marker_ids = torch.concat(
            [
                0 * torch.zeros(2 * self._commands.shape[0]),
                1 * torch.ones(self._commands.shape[0]),
                2 * torch.ones(self._commands.shape[0]),
                3 * torch.ones(self._commands.shape[0]),
            ],
            axis=0,
        )

        robot_pos = self.robots["robot_0"].data.root_pos_w
        robot_yaw = self.robots["robot_0"].data.root_com_ang_vel_b[:, 2]

        scale1 = torch.ones((self._commands.shape[0], 3), device=self.device)
        scale1[:, 0] = torch.abs(z_commands)

        scale2 = torch.ones((self._commands.shape[0], 3), device=self.device)
        scale2[:, 0] = torch.abs(robot_yaw)

        offset1 = torch.zeros((self._commands.shape[0], 3), device=self.device)
        offset1[:, 1] = 0

        offset2 = torch.zeros((self._commands.shape[0], 3), device=self.device)
        offset2[:, 1] = 0

        _90 = (-3.14 / 2) * torch.ones(self._commands.shape[0]).to(self.device)

        marker_orientations = quat_from_angle_axis(
            torch.concat(
                [
                    torch.zeros(3 * self._commands.shape[0]).to(self.device),
                    torch.sign(z_commands) * _90,
                    torch.sign(robot_yaw) * _90,
                ],
                axis=0,
            ),
            torch.tensor([0.0, 1.0, 0.0], device=self.device),
        )

        marker_scales = torch.concat(
            [torch.ones((3 * self._commands.shape[0], 3), device=self.device), scale1, scale2], axis=0
        )

        marker_locations = torch.concat(
            [
                robot_pos,
                robot_pos + xy_commands * 10,
                robot_pos + self.robots["robot_0"].data.root_com_lin_vel_b * 10,
                robot_pos + offset1,
                robot_pos + offset2,
            ],
            axis=0,
        )

        self.my_visualizer.visualize(
            marker_locations, marker_orientations, scales=marker_scales, marker_indices=marker_ids
        )

    def _setup_scene(self):
        self.robots = {"robot_0": Articulation(self.cfg.robot_0)}
        self.my_visualizer = define_markers()
        # add ground plane
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add articulation to scene
        self.scene.articulations["robot_0"] = self.robots["robot_0"]
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # contact sensors        
        self.contact_sensors = {}
        self.contact_sensors["robot_0"] = ContactSensor(self.cfg.__dict__["contact_sensor_0"])

    def _pre_physics_step(self, actions: torch.Tensor):
        self.prev_actions["robot_0"] = self.actions["robot_0"].clone()
        self.actions["robot_0"] = actions["robot_0"].clone()

    def _apply_action(self):
        self.contact_sensors["robot_0"].update(self.step_dt)
        forces = self.action_scale * self.joint_gears * self.actions["robot_0"]
        self.robots["robot_0"].set_joint_effort_target(forces, joint_ids=self._joint_dof_idx)

    def _compute_intermediate_values(self):
        self.torso_position, self.torso_rotation = self.robots["robot_0"].data.root_pos_w, self.robots["robot_0"].data.root_quat_w
        self.velocity, self.ang_velocity = self.robots["robot_0"].data.root_lin_vel_w, self.robots["robot_0"].data.root_ang_vel_w
        self.dof_pos, self.dof_vel = self.robots["robot_0"].data.joint_pos, self.robots["robot_0"].data.joint_vel
        self.dof_pos_scaled = torch_utils.maths.unscale(self.dof_pos, 
                                                        self.robots["robot_0"].data.soft_joint_pos_limits[0, :, 0], 
                                                        self.robots["robot_0"].data.soft_joint_pos_limits[0, :, 1])

    def _get_observations(self) -> dict:
        obs = torch.cat(
            (
                self._commands,
                self.velocity,
                self.ang_velocity,
                self.gait_phase_obs,
                self.actions["robot_0"],
            ),
            dim=-1,
        )
        observations = {"robot_0": obs}
        return observations
    
    # This functions computes the gait phase of the robot for observations.
    def gait_phase(
            self,
            period: float
    ) -> torch.Tensor:
        if not hasattr(self, "episode_length_buf"):
            self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        global_phase = ((self.episode_length_buf * self.step_dt) % period / period)

        phase = torch.zeros(self.num_envs, 2, device=self.device)
        phase[:, 0] = torch.sin(global_phase * 2 * torch.pi)
        phase[:, 1] = torch.cos(global_phase * 2 * torch.pi)

        return phase


    def _get_rewards(self) -> dict:
        self._draw_markers(self._commands)

        feet_slide = rewards.feet_slide_reward(
            self
        ) 

        gait_reward = rewards.gait_reward(
            self,
            self.cfg.period,
            [0.0, 0.5],
            0.55,
            self._commands
        )

        clearance_reward = rewards.foot_clearance_reward(
            self,
            target_height=0.15,
            std=0.05,
            tanh_mult=2.0
        )

        other_rewards = self.compute_rewards(
            self.robots["robot_0"],
            self._commands,
            self.actions["robot_0"],
            self.prev_actions["robot_0"],
            self.dof_vel,
            self.dof_pos_scaled,
            self.potentials,
            self.prev_potentials,
            self.cfg.smoothness_cost_scale,
            self.cfg.actions_cost_scale,
            self.cfg.energy_cost_scale,
            self.cfg.dof_vel_scale,
            self.motor_effort_ratio,
        )
        # new rewards from rewards_funcs.py get added here
        total_reward = other_rewards.clone()
        slide_rewards = feet_slide * self.cfg.feet_slide_weight
        gait_rewards = gait_reward * self.cfg.gait_weight
        clearance_rewards = clearance_reward * self.cfg.feet_clearance_weight
        total_reward += slide_rewards
        total_reward += gait_rewards
        total_reward += clearance_rewards

        self._episode_sums["total_reward"] += total_reward
        self._episode_sums["gait_reward"] += gait_rewards
        self._episode_sums["slide_reward"] += slide_rewards
        self._episode_sums["clearance_reward"] += clearance_rewards
        self._episode_sums["other_rewards"] += other_rewards

        return {"robot_0": total_reward}

    def _get_dones(self) -> tuple[dict, dict]:
        self._compute_intermediate_values()
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = self.torso_position[:, 2] < self.cfg.termination_height
        return {"robot_0": died}, {"robot_0": time_out}

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robots["robot_0"]._ALL_INDICES
        self.robots["robot_0"].reset(env_ids)
        super()._reset_idx(env_ids)

        joint_pos = self.robots["robot_0"].data.default_joint_pos[env_ids]
        joint_vel = self.robots["robot_0"].data.default_joint_vel[env_ids]
        default_root_state = self.robots["robot_0"].data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.robots["robot_0"].write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robots["robot_0"].write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robots["robot_0"].write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # p = torch.zeros(1).uniform_(0, 1.0)
        #TODO we are assigning angular and linear velocity at the same time here. Possibly causing an error
        lin_ang_velocity = torch.zeros((len(env_ids), 3)).uniform_(-0.1, 0.1).to(self.device) # type:ignore

        self._commands[env_ids] = lin_ang_velocity
        


        # if p < 0.5:
        #     self._commands[env_ids, 0] = v
        # else:
        #     self._commands[env_ids, 2] = v


        self._compute_intermediate_values()

        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)


    def compute_rewards(
        self,
        robot: Articulation,
        commands: torch.Tensor,
        actions: torch.Tensor,
        prev_actions: torch.Tensor,
        dof_vel: torch.Tensor,
        dof_pos_scaled: torch.Tensor,
        potentials: torch.Tensor,
        prev_potentials: torch.Tensor,
        smoothness_cost_scale: float,
        actions_cost_scale: float,
        energy_cost_scale: float,
        dof_vel_scale: float,
        motor_effort_ratio: torch.Tensor,
    ):
        # Requires storing prev_actions
        action_delta = actions - prev_actions
        smoothness_penalty = torch.sum(action_delta ** 2, dim=-1)

        # linear velocity tracking

        vel_yaw = quat_rotate_inverse(yaw_quat(robot.data.root_quat_w), robot.data.root_lin_vel_w[:, :3])
        lin_vel_error = torch.sum(
            torch.square(commands[:, :2] - vel_yaw[:, :2]), dim=1
        )
        lin_vel_error_mapped = torch.exp(-lin_vel_error / .25)
        
        # yaw rate tracking
        yaw_rate_error = torch.square(commands[:, 2] - robot.data.root_ang_vel_w[:, 2])
        yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.25)

        # energy penalty for movement
        actions_cost = torch.sum(actions**2, dim=-1)
        electricity_cost = torch.sum(
            torch.abs(actions * dof_vel * dof_vel_scale) * motor_effort_ratio.unsqueeze(0),
            dim=-1,
        )

        # dof at limit cost
        dof_at_limit_cost = torch.sum(dof_pos_scaled > 0.98, dim=-1)

        # reward for duration of staying alive
        alive_reward = torch.ones_like(potentials)
        progress_reward = potentials - prev_potentials

        total_reward = (
            # progress_reward
            + (lin_vel_error_mapped * self.cfg.task_velocity_scale)
            + (yaw_rate_error_mapped * self.cfg.angular_velocity_scale)
            + (alive_reward * self.cfg.alive_reward_scale)
            # - actions_cost_scale * actions_cost
            # - energy_cost_scale * electricity_cost
            # - dof_at_limit_cost
            # - smoothness_cost_scale * smoothness_penalty
        )
        return total_reward

