from __future__ import annotations

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.utils.math import quat_from_angle_axis, quat_from_euler_xyz, quat_rotate_inverse
import random

def get_quaternion_tuple_from_xyz(x, y, z):
    quat_tensor = quat_from_euler_xyz(torch.tensor([x]), torch.tensor([y]), torch.tensor([z])).flatten()
    return (quat_tensor[0].item(), quat_tensor[1].item(), quat_tensor[2].item(), quat_tensor[3].item())


from .maze_course_env_cfg import MazeCourseEnvCfg


class MazeCourseEnv(DirectRLEnv):
    cfg: MazeCourseEnvCfg

    def __init__(self, cfg: MazeCourseEnvCfg, render_mode: str | None = None, headless: bool | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.headless = headless

        # Get specific body indices
        self._throttle_dof_idx, _ = self._robot.find_joints(self.cfg.throttle_dof_name)
        self._steering_dof_idx, _ = self._robot.find_joints(self.cfg.steering_dof_name)

        self._throttle_state = torch.zeros((self.num_envs, 4), device=self.device, dtype=torch.float32)
        self._steering_state = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.float32)

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "dist_to_ball_reward",
            ]
        }

        marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/myMarkers",
            markers={
                "goal1": sim_utils.CuboidCfg(
                    size = (1, 3, 0.1),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
                ),
                "goal2":sim_utils.CuboidCfg(
                        size=(1, 3, 0.1),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                ),
            },
        )
        self.goal_area = VisualizationMarkers(marker_cfg)
        self.goal1_pos, self.goal2_pos, self.goal1_area, self.goal2_area = self._get_goal_areas()
        self.target_goal = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

    def _setup_scene(self):
        # Setting up the maze walls
        self.wall_0 = RigidObject(self.cfg.wall_0)
        self.wall_1 = RigidObject(self.cfg.wall_1)
        self.wall_2 = RigidObject(self.cfg.wall_2)
        self.wall_3 = RigidObject(self.cfg.wall_3)

        # Terrain setup
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # Robot Declaration        
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        # self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        # self.scene.sensors["contact_sensor"] = self._contact_sensor

        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)


    def _draw_goal_areas(self):
        marker_ids0 = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        marker_ids1 = torch.ones(self.num_envs, dtype=torch.int32, device=self.device)
        marker_ids2 = 2 * torch.ones(self.num_envs, dtype=torch.int32, device=self.device)

        marker_ids = torch.concat([marker_ids0, marker_ids1, marker_ids2], dim=0)

        goal_to_score_pos = torch.where(self.target_goal.unsqueeze(1) == 0, self.goal1_pos, self.goal2_pos)

        marker_locations = torch.concat([self.goal1_pos, self.goal2_pos, goal_to_score_pos], dim=0)

        self.goal_area.visualize(marker_locations, marker_indices=marker_ids)



    def _pre_physics_step(self, actions: torch.Tensor):
        raw_throttle = actions[:, 0]
        self._throttle_action = raw_throttle.repeat_interleave(4).reshape((-1,4)) * self.cfg.throttle_scale
        self._throttle_action = torch.clamp(self._throttle_action, -self.cfg.throttle_max, self.cfg.throttle_max)
        self._throttle_state = self._throttle_action

        raw_steering = actions[:, 1]
        self._steering_action = raw_steering.repeat_interleave(2).reshape((-1,2)) * self.cfg.steering_scale
        self._steering_action = torch.clamp(self._steering_action, -self.cfg.steering_max, self.cfg.steering_max)
        self._steering_state = self._steering_action

    def _apply_action(self):
        self._robot.set_joint_velocity_target(
            self._throttle_state,
            joint_ids=self._throttle_dof_idx
        )

        self._robot.set_joint_position_target(
            self._steering_state,
            joint_ids=self._steering_dof_idx
        )


    def relative_velocity(self, obj_vel_w, ref_rot_w):
        """Convert world-frame velocity into reference-frame coordinates.
        
        Args:
            obj_vel_w (torch.Tensor): (E, 3) linear velocity of object in world frame
            ref_rot_w (torch.Tensor): (E, 4) quaternion of reference orientation in world frame
        
        Returns:
            torch.Tensor: (E, 3) velocity in the reference's local frame
        """
        return quat_rotate_inverse(ref_rot_w, obj_vel_w)

    def _get_observations(self) -> dict:
        robot_vel = self._robot.data.root_lin_vel_b # robot linear velocity in body frame

        target_goal_pos, _ = subtract_frame_transforms(
            self._robot.data.root_state_w[:, :3],
            self._robot.data.root_state_w[:, 3:7],
            torch.where(self.target_goal.unsqueeze(1) == 0, self.goal1_pos, self.goal2_pos)
        )

        other_goal_pos, _ = subtract_frame_transforms(
            self._robot.data.root_state_w[:, :3],
            self._robot.data.root_state_w[:, 3:7],
            torch.where(self.target_goal.unsqueeze(1) == 0, self.goal2_pos, self.goal1_pos)
        )


        self._previous_actions = self._actions.clone()
        height_data = None
        obs = torch.cat((
            robot_vel, # Robot velocity in body frame
            target_goal_pos, # Target goal position in world frame 
            other_goal_pos,  # other goal position in world frame 
        ), dim=-1)

        # handle NaN and inf values
        obs = torch.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)

        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # linear velocity tracking
        lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - self._robot.data.root_lin_vel_b[:, :2]), dim=1)
        lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25)
        # yaw rate tracking
        yaw_rate_error = torch.square(self._commands[:, 2] - self._robot.data.root_ang_vel_b[:, 2])
        yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.25)
        # z velocity tracking
        z_vel_error = torch.square(self._robot.data.root_lin_vel_b[:, 2])
        # angular velocity x/y
        ang_vel_error = torch.sum(torch.square(self._robot.data.root_ang_vel_b[:, :2]), dim=1)
        # joint torques
        joint_torques = torch.sum(torch.square(self._robot.data.applied_torque), dim=1)
        # joint acceleration
        joint_accel = torch.sum(torch.square(self._robot.data.joint_acc), dim=1)
        # action rate
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        # feet air time
        first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[:, self._feet_ids]
        last_air_time = self._contact_sensor.data.last_air_time[:, self._feet_ids]
        air_time = torch.sum((last_air_time - 0.5) * first_contact, dim=1) * (
            torch.norm(self._commands[:, :2], dim=1) > 0.1
        )
        # undesired contacts
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        is_contact = (
            torch.max(torch.norm(net_contact_forces[:, :, self._undesired_contact_body_ids], dim=-1), dim=1)[0] > 1.0
        )
        contacts = torch.sum(is_contact, dim=1)
        # flat orientation
        flat_orientation = torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1)

        rewards = {
            "track_lin_vel_xy_exp": lin_vel_error_mapped * self.cfg.lin_vel_reward_scale * self.step_dt,
            "track_ang_vel_z_exp": yaw_rate_error_mapped * self.cfg.yaw_rate_reward_scale * self.step_dt,
            "lin_vel_z_l2": z_vel_error * self.cfg.z_vel_reward_scale * self.step_dt,
            "ang_vel_xy_l2": ang_vel_error * self.cfg.ang_vel_reward_scale * self.step_dt,
            "dof_torques_l2": joint_torques * self.cfg.joint_torque_reward_scale * self.step_dt,
            "dof_acc_l2": joint_accel * self.cfg.joint_accel_reward_scale * self.step_dt,
            "action_rate_l2": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "feet_air_time": air_time * self.cfg.feet_air_time_reward_scale * self.step_dt,
            "undesired_contacts": contacts * self.cfg.undesired_contact_reward_scale * self.step_dt,
            "flat_orientation_l2": flat_orientation * self.cfg.flat_orientation_reward_scale * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        died = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0] > 1.0, dim=1)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        # Sample new commands
        self._commands[env_ids] = torch.zeros_like(self._commands[env_ids]).uniform_(-1.0, 1.0)
        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/base_contact"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"].update(extras)
