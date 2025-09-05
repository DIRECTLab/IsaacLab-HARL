# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers import VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

##
# Pre-defined configs
##
from isaaclab_assets import CRAZYFLIE_CFG  # isort: skip
from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip




@configclass
class DroneStage2EnvMultiAgentCfg(DirectMARLEnvCfg):
    # env
    episode_length_s = 20.0
    decimation = 4
    anymal_action_scale = 0.5
    action_space = 4
    action_spaces = {f"robot_{i}": 4 for i in range(2)}

    # Padded observation: 12 (original) + 3 (teammate_pos) + 3 (other_pos) + 1 (dist_to_center) + 1 (arena_radius) = 20
    observation_space = 20
    observation_spaces = {f"robot_{i}": 20 for i in range(2)}
    state_space = 0
    state_spaces = {f"robot_{i}": 0 for i in range(2)}
    possible_agents = [f"robot_{i}" for i in range(2)]

    # Teams for two agents
    teams = {"team_0": ["robot_0"], "team_1": ["robot_1"]}

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=10.0, replicate_physics=True)

    # events
    # events: EventCfg = EventCfg()

    ### CRAZYFLIE CONFIGURATION ###
    robot_0: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot_0")
    robot_0.init_state.pos = (0.0, 0.5, 2.0)

    robot_1: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot_1")
    robot_1.init_state.pos = (0.0, -0.5, 2.0)

    thrust_to_weight = 1.9
    moment_scale = 0.01

    # reward scales
    lin_vel_reward_scale = -0.05
    ang_vel_reward_scale = -0.01
    distance_to_goal_reward_scale = 15.0
    debug_vis = True

    ### CRAZYFLIE CONFIGURATION ###

class DroneStage2EnvMultiAgentMARLEnv(DirectMARLEnv):
    cfg: DroneStage2EnvMultiAgentCfg

    def __init__(self, cfg: DroneStage2EnvMultiAgentCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Total thrust and moment applied to the base of the quadcopter
        self.actions = {
            agent: torch.zeros(self.num_envs, action_space, device=self.device)
            for agent, action_space in self.cfg.action_spaces.items()
        }
        self._thrust = {agent: torch.zeros(self.num_envs, 1, 3, device=self.device) for agent in self.cfg.action_spaces}
        self._moment = {agent: torch.zeros(self.num_envs, 1, 3, device=self.device) for agent in self.cfg.action_spaces}
        self._desired_pos_w = {agent: torch.zeros(self.num_envs, 3, device=self.device) for agent in self.cfg.action_spaces}

        self._body_id = {}
        self._robot_mass = {}
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = {}
        for agent in self.cfg.action_spaces:
            self._body_id[agent] = self.robots[agent].find_bodies("body")[0]
            self._robot_mass[agent] = self.robots[agent].root_physx_view.get_masses()[0].sum()
            self._robot_weight[agent] = (self._robot_mass[agent] * self._gravity_magnitude).item()

        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "lin_vel",
                "ang_vel",
                "distance_to_goal",
                "crazyflie_cosine_reward",
                "tank_angle_reward",
            ]
        }

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        self.num_robots = sum(1 for key in self.cfg.__dict__.keys() if "robot_" in key)
        self.robots = {}
        for i in range(self.num_robots):
            robot_id = f"robot_{i}"
            if robot_id in self.cfg.__dict__:
                self.robots[robot_id] = Articulation(self.cfg.__dict__[robot_id])
                self.scene.articulations[robot_id] = self.robots[robot_id]

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=False)
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: dict):
        for agent in self.cfg.action_spaces:
            self.actions[agent] = actions[agent].clone().clamp(-1.0, 1.0)
            self._thrust[agent][:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight[agent] * (self.actions[agent][:, 0] + 1.0) / 2.0
            self._moment[agent][:, 0, :] = self.cfg.moment_scale * self.actions[agent][:, 1:]

    def _apply_action(self):
        for agent in self.cfg.action_spaces:
            self.robots[agent].set_external_force_and_torque(self._thrust[agent], self._moment[agent], body_ids=self._body_id[agent])

    def _get_observations(self) -> dict:
        # Arena radius and center
        arena_radius = torch.full((self.num_envs, 1), 2.0, device=self.device)  # Example value
        env_center = self._terrain.env_origins if hasattr(self._terrain, "env_origins") else torch.zeros((self.num_envs, 3), device=self.device)

        obs_dict = {"team_0": {}, "team_1": {}}
        for i, agent in enumerate(self.cfg.action_spaces):
            # Desired position in body frame
            desired_pos_b, _ = subtract_frame_transforms(
                self.robots[agent].data.root_state_w[:, :3], self.robots[agent].data.root_state_w[:, 3:7], self._desired_pos_w[agent]
            )
            # Teammate and other agent positions (in body frame)
            teammate_id = f"robot_{1-i}"
            teammate_pos, _ = subtract_frame_transforms(
                self.robots[agent].data.root_state_w[:, :3], self.robots[agent].data.root_state_w[:, 3:7], self.robots[teammate_id].data.root_pos_w
            )
            # For two agents, other_pos is same as teammate_pos
            other_pos = teammate_pos
            # Distance to center
            dist_to_center = torch.norm(self.robots[agent].data.root_pos_w - env_center, dim=-1, keepdim=True)

            obs = torch.cat(
                [
                    self.robots[agent].data.root_lin_vel_b,
                    self.robots[agent].data.root_ang_vel_b,
                    self.robots[agent].data.projected_gravity_b,
                    desired_pos_b,
                    teammate_pos,
                    other_pos,
                    dist_to_center,
                    arena_radius,
                ],
                dim=-1,
            )
            obs = torch.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)
            team = f"team_{i}"
            obs_dict[team][agent] = obs
        return obs_dict

    def _get_rewards(self) -> dict:
        all_rewards = {}
        team_rewards = {team: 0.0 for team in self.cfg.teams}
        for i, agent in enumerate(self.cfg.action_spaces):
            lin_vel = torch.sum(torch.square(self.robots[agent].data.root_lin_vel_b), dim=1)
            ang_vel = torch.sum(torch.square(self.robots[agent].data.root_ang_vel_b), dim=1)
            distance_to_goal = torch.linalg.norm(self._desired_pos_w[agent] - self.robots[agent].data.root_pos_w, dim=1)
            distance_to_goal_mapped = 1 - torch.tanh(distance_to_goal / 0.8)
            rewards = {
                "lin_vel": lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
                "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
                "distance_to_goal": distance_to_goal_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt,
            }
            reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
            for key, value in rewards.items():
                self._episode_sums[key] += value
            all_rewards[agent] = reward
            # Assign to team
            for team, agents in self.cfg.teams.items():
                if agent in agents:
                    team_rewards[team] += reward
        # Return per-team rewards
        return team_rewards

    def _get_dones(self) -> tuple[dict, dict]:
        time_out_tensor = (self.episode_length_buf >= self.max_episode_length - 1).to(self.device)
        dones = {}
        time_out = {}
        for i, agent in enumerate(self.cfg.action_spaces):
            died_tensor = self.robots[agent].data.root_pos_w[:, 2] < 0.1
            team = f"team_{i}"
            dones[team] = died_tensor
            time_out[team] = time_out_tensor
        return dones, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        # Use the first robot to get ALL_INDICES, but reset all robots
        first_agent = list(self.cfg.action_spaces.keys())[0]
        if env_ids is None or (hasattr(env_ids, "__len__") and len(env_ids) == self.num_envs):
            env_ids = self.robots[first_agent]._ALL_INDICES

        # Logging: average final distance to goal across all robots
        final_distances = []
        for agent in self.cfg.action_spaces:
            final_distances.append(torch.linalg.norm(self._desired_pos_w[agent][env_ids] - self.robots[agent].data.root_pos_w[env_ids], dim=1))
        final_distance_to_goal = torch.stack(final_distances).mean()

        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Metrics/final_distance_to_goal"] = final_distance_to_goal.item()
        self.extras["log"].update(extras)

        # Reset all robots
        for agent in self.cfg.action_spaces:
            self.robots[agent].reset(env_ids)

        super()._reset_idx(env_ids)
        if hasattr(env_ids, "__len__") and len(env_ids) == self.num_envs:
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        # Reset actions and sample new commands for all robots
        for agent in self.cfg.action_spaces:
            self.actions[agent][env_ids] = 0.0
            self._desired_pos_w[agent][env_ids, :2] = torch.zeros_like(self._desired_pos_w[agent][env_ids, :2]).uniform_(-2.0, 2.0)
            self._desired_pos_w[agent][env_ids, :2] += self._terrain.env_origins[env_ids, :2]
            self._desired_pos_w[agent][env_ids, 2] = torch.zeros_like(self._desired_pos_w[agent][env_ids, 2]).uniform_(0.5, 1.5)
            # Reset robot state
            joint_pos = self.robots[agent].data.default_joint_pos[env_ids]
            joint_vel = self.robots[agent].data.default_joint_vel[env_ids]
            default_root_state = self.robots[agent].data.default_root_state[env_ids]
            default_root_state[:, :3] += self._terrain.env_origins[env_ids]
            self.robots[agent].write_root_pose_to_sim(default_root_state[:, :7], env_ids)
            self.robots[agent].write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
            self.robots[agent].write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
                # -- goal pose
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            self.goal_pos_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # update the markers
        self.goal_pos_visualizer.visualize(self._desired_pos_w)
