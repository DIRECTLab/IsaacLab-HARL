# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import torch

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import quat_from_angle_axis, subtract_frame_transforms, quat_from_euler_xyz

##
# Pre-defined configs
##
from isaaclab_assets.robots.anymal import ANYMAL_C_CFG  # isort: skip
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip

@configclass
class EventCfg:
    """Configuration for randomization."""

    physics_material_0 = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot_0", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.2),
            "num_buckets": 64,
        },
    )

    # mass_scale_0 = EventTerm(
    #     func=mdp.randomize_rigid_body_mass,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot_0", body_names=".*"),
    #         # scale each link's mass by a factor in this range
    #         "mass_distribution_params": (0.90, 1.30),  # widen if you want more asymmetry
    #         "operation": "scale",                      # use multiplicative scaling
    #     },
    # )

    add_base_mass_0 = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot_0", body_names="base"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )

    # physics_material_1 = EventTerm(
    #     func=mdp.randomize_rigid_body_material,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot_1", body_names=".*"),
    #         "static_friction_range": (0.8, 0.8),
    #         "dynamic_friction_range": (0.6, 0.6),
    #         "restitution_range": (0.0, 0.2),
    #         "num_buckets": 64,
    #     },
    # )

    # mass_scale_1 = EventTerm(
    #     func=mdp.randomize_rigid_body_mass,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot_1", body_names=".*"),
    #         # scale each link's mass by a factor in this range
    #         "mass_distribution_params": (0.90, 1.30),  # widen if you want more asymmetry
    #         "operation": "scale",                      # use multiplicative scaling
    #         # "scale_inertia": True,  # if your Isaac Lab version supports this flag
    #     },
    # )

    # add_base_mass_1 = EventTerm(
    #     func=mdp.randomize_rigid_body_mass,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot_1", body_names="base"),
    #         "mass_distribution_params": (-5.0, 5.0),
    #         "operation": "add",
    #     },
    # )

def get_distance_to_center(arena_center: torch.tensor, robot: Articulation):
    robot_pos_w = robot.data.root_state_w[:, :3]                # (num_envs, 3)
    robot_delta_xy = robot_pos_w[:, :2] - arena_center          # (num_envs, 2)
    return torch.norm(robot_delta_xy, dim=1, keepdim=True)

def get_quaternion_tuple_from_xyz(x, y, z):
    quat_tensor = quat_from_euler_xyz(torch.tensor([x]), torch.tensor([y]), torch.tensor([z])).flatten()
    return (quat_tensor[0].item(), quat_tensor[1].item(), quat_tensor[2].item(), quat_tensor[3].item())

@configclass
class SumoStage1EnvSingleAgentCfg(DirectMARLEnvCfg):
    # env
    episode_length_s = 20.0
    decimation = 4
    action_scale = 0.5
    action_space = 12
    action_spaces = {f"robot_{i}": 12 for i in range(1)}
    observation_space = 48
    observation_spaces = {f"robot_{i}": 48 for i in range(1)}
    state_space = 0
    state_spaces = {f"robot_{i}": 0 for i in range(1)}
    possible_agents = [f"robot_{i}" for i in range(1)]

    teams = {
        "team_0": ["robot_0"],
    }

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
    events: EventCfg = EventCfg()

    # robot
    robot_0: ArticulationCfg = ANYMAL_C_CFG.replace(prim_path="/World/envs/env_.*/Robot_0")
    contact_sensor_0: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot_0/.*", history_length=3, update_period=0.005, track_air_time=True,
    )
    
    robot_0.init_state.rot = get_quaternion_tuple_from_xyz(0,0,torch.pi/2)
    robot_0.init_state.pos = (0.0, 1.0, 0.3)

    # robot_1: ArticulationCfg = ANYMAL_C_CFG.replace(prim_path="/World/envs/env_.*/Robot_1")
    # contact_sensor_1: ContactSensorCfg = ContactSensorCfg(
    #     prim_path="/World/envs/env_.*/Robot_1/.*", history_length=3, update_period=0.005, track_air_time=True,
    # )
    # robot_1.init_state.rot = get_quaternion_tuple_from_xyz(0,0,torch.pi/2)
    # robot_1.init_state.pos = (0.0, -1.0, 0.3)

    # --- NEW: goal params (visual + sampling) ---
    goal_reach_radius: float = 0.35          # within this distance counts as "reached"
    goal_height_z: float = 0.05              # small lift so the marker is visible
    goal_marker_radius: float = 0.08
    goal_a_color: tuple[float, float, float] = (0.2, 0.6, 1.0)  # light blue
    goal_b_color: tuple[float, float, float] = (1.0, 0.3, 0.8)  # pink
    goal_spawn_radius_min: float = 3
    goal_spawn_radius_max: float = 5
    goal_min_separation: float = 1.0    

    # reward scales
    reached_goal_reward = 10.0
    dist_to_goal_reward_scale = 2.0
    time_penalty_per_second = -0.01
    lin_vel_reward_scale = 1.0
    yaw_rate_reward_scale = 0.5
    z_vel_reward_scale = -2.0
    ang_vel_reward_scale = -0.05
    joint_torque_reward_scale = -2.5e-5
    joint_accel_reward_scale = -2.5e-7
    action_rate_reward_scale = -0.01
    feet_air_time_reward_scale = 0.5
    undesired_contact_reward_scale = -1.0
    flat_orientation_reward_scale = -5.0

def define_markers() -> VisualizationMarkers:
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/myMarkers",
        markers={
            "sphere1": sim_utils.SphereCfg(
                radius=0.1,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 1.0, 1.0),
                ),
            ),
        },
    )
    return VisualizationMarkers(marker_cfg)


class SumoStage1EnvSingleAgent(DirectMARLEnv):
    cfg: SumoStage1EnvSingleAgentCfg

    def __init__(
        self, cfg: SumoStage1EnvSingleAgentCfg, render_mode: str | None = None, debug=False, **kwargs
    ):
        self.debug = debug
        super().__init__(cfg, render_mode, **kwargs)
        self.actions = {
            agent: torch.zeros(self.num_envs, action_space, device=self.device)
            for agent, action_space in self.cfg.action_spaces.items()
        }
        self.previous_actions = {
            agent: torch.zeros(self.num_envs, action_space, device=self.device)
            for agent, action_space in self.cfg.action_spaces.items()
        }
        self._prev_min_goal_dist = {
            robot_id: torch.zeros(self.num_envs, device=self.device)
            for robot_id in self.robots.keys()
        }
        self._min_goal_dist = {
            robot_id: torch.zeros(self.num_envs, device=self.device)
            for robot_id in self.robots.keys()
        }
        self.my_visualizer = define_markers()
        self._desired_pos = torch.zeros((self.num_envs, 3), device=self.device)

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "goal_reached",
                "distance_to_goal",
                # "time_penalty",
                "lin_vel_z_l2",
                "ang_vel_xy_l2",
                "dof_torques_l2",
                "dof_acc_l2",
                "action_rate_l2",
                "feet_air_time",
                "undesired_contacts",
                "flat_orientation_l2",
            ]
        }

        self.base_ids = {}
        self.feet_ids = {}
        self.undesired_body_contact_ids = {}

        for robot_id, contact_sensor in self.contact_sensors.items():
            _base_id, _ = contact_sensor.find_bodies("base")
            _feet_ids, _ = contact_sensor.find_bodies(".*FOOT")
            _undesired_contact_body_ids, _ = contact_sensor.find_bodies([".*THIGH"])
            self.base_ids[robot_id] = _base_id
            self.feet_ids[robot_id] = _feet_ids
            self.undesired_body_contact_ids[robot_id] = _undesired_contact_body_ids

    def _draw_markers(self):

        marker_ids = torch.zeros(self.num_envs, dtype=torch.int32).to(self.device)



        self.my_visualizer.visualize(self._desired_pos, None, marker_indices=marker_ids)

    def _setup_scene(self):

        self.num_robots = sum(1 for key in self.cfg.__dict__.keys() if "robot_" in key)
        self.robots = {}
        self.contact_sensors = {}
        self.height_scanners = {}

        for i in range(self.num_robots):
            self.robots[f"robot_{i}"] = Articulation(self.cfg.__dict__["robot_" + str(i)])
            self.scene.articulations[f"robot_{i}"] = self.robots[f"robot_{i}"]
            self.contact_sensors[f"robot_{i}"] = ContactSensor(self.cfg.__dict__["contact_sensor_" + str(i)])
            self.scene.sensors[f"robot_{i}"] = self.contact_sensors[f"robot_{i}"]

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        self.goal_pos_w = torch.zeros(self.num_envs, 2, 3, device=self.device)

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]):
        # We need to process the actions for each scene independently

        self.processed_actions = {}
        self.actions = copy.deepcopy(actions)
        for robot_id, robot in self.robots.items():
            self.processed_actions[robot_id] = (
                self.cfg.action_scale * actions[robot_id] + robot.data.default_joint_pos
            )

    def _apply_action(self):
        for robot_id, robot in self.robots.items():
            robot.set_joint_position_target(self.processed_actions[robot_id])

    def _get_observations(self) -> dict:
        self.previous_actions = copy.deepcopy(self.actions)

        # time_remaining = (self.max_episode_length - self.episode_length_buf).unsqueeze(-1)
        goals_xy = self.goal_pos_w[:, :, :2]
        for robot_id, robot in self.robots.items():
            self._prev_min_goal_dist[robot_id] = self._min_goal_dist[robot_id].clone()
            robot_xy = robot.data.root_pos_w[:, :2]
            dists = torch.norm(goals_xy - robot_xy.unsqueeze(1), dim=-1)

            self._min_goal_dist[robot_id] = torch.min(dists, dim=-1).values

        obs = {}
        for team_name, robots in self.cfg.teams.items():
            team_obs = {}
            for i, robot_id in enumerate(robots):
                    
                goal_pos, _ = subtract_frame_transforms(
                        self.robots[robot_id].data.root_state_w[:, :3],
                        self.robots[robot_id].data.root_state_w[:, 3:7],
                        self._desired_pos
                    )

                # dist_to_center = get_distance_to_center(self.scene.env_origins[:, :2].to(self.device), self.robots[robot_id])
                dist_to_center = torch.zeros((self.num_envs, 1), device=self.device)
                arena_radius = torch.zeros((self.num_envs, 1), device=self.device)
                time_remaining = torch.zeros((self.num_envs, 1), device=self.device)
                teammate_pos = torch.full((self.num_envs, 3), 50, device=self.device)
                other_pos = torch.full((self.num_envs, 3), 50, device=self.device)

                obs_vec = torch.cat(
                    [
                        self.robots[robot_id].data.root_lin_vel_b,
                        self.robots[robot_id].data.root_ang_vel_b,
                        self.robots[robot_id].data.projected_gravity_b,
                        self.robots[robot_id].data.joint_pos - self.robots[robot_id].data.default_joint_pos,
                        self.robots[robot_id].data.joint_vel,
                        self.actions[robot_id],
                        goal_pos,
                        # teammate_pos,
                        # other_pos,
                        # dist_to_center,
                        # arena_radius,
                        # time_remaining,
                    ],
                    dim=-1,
                )
                team_obs[robot_id] = obs_vec
            obs[team_name] = team_obs

        return obs

    def _get_rewards(self) -> dict:
        self._draw_markers()
        all_rewards = {}
        reach_r = self.cfg.goal_reach_radius
        # any_robot_reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        goals_xy = self._desired_pos[:, :2]
        
        # for robot_id, robot in self.robots.items():
        #     # robot base XY: (N, 2)
        #     robot_xy = robot.data.root_pos_w[:, :2]
        #     # pairwise dists to both goals: (N, 2)
        #     dists = torch.norm(goals_xy - robot_xy.unsqueeze(1), dim=-1)
        #     # did this robot hit any goal? (N,)
        #     hit = (dists <= reach_r).any(dim=1)
        #     any_robot_reached |= hit

        for robot_id in self.robots.keys():
            # goal reached reward
            robot_xy = self.robots[robot_id].data.root_pos_w[:, :2]
            dists = torch.norm(goals_xy - robot_xy, dim=-1)
            hit = dists <= reach_r
            # distance reward
            distance_reward = 1 - torch.tanh(dists / 5)
            # z velocity tracking
            z_vel_error = torch.square(self.robots[robot_id].data.root_lin_vel_b[:, 2])
            # angular velocity x/y
            ang_vel_error = torch.sum(torch.square(self.robots[robot_id].data.root_ang_vel_b[:, :2]), dim=1)
            # joint torques
            joint_torques = torch.sum(torch.square(self.robots[robot_id].data.applied_torque), dim=1)
            # joint acceleration
            joint_accel = torch.sum(torch.square(self.robots[robot_id].data.joint_acc), dim=1)
            # action rate
            action_rate = torch.sum(torch.square(self.actions[robot_id] - self.previous_actions[robot_id]), dim=1)
            # feet air time
            first_contact = self.contact_sensors[robot_id].compute_first_contact(self.step_dt)[:, self.feet_ids[robot_id]]
            last_air_time = self.contact_sensors[robot_id].data.last_air_time[:, self.feet_ids[robot_id]]
            air_time = torch.sum((last_air_time - 0.5) * first_contact, dim=1)
            # undesired contacts
            net_contact_forces = self.contact_sensors[robot_id].data.net_forces_w_history
            is_contact = (
                torch.max(torch.norm(net_contact_forces[:, :, self.undesired_body_contact_ids[robot_id]], dim=-1), dim=1)[0] > 1.0
            )
            contacts = torch.sum(is_contact, dim=1)
            # flat orientation
            flat_orientation = torch.sum(torch.square(self.robots[robot_id].data.projected_gravity_b[:, :2]), dim=1)

            # time_penalty_per_second = self.cfg.time_penalty_per_second * torch.ones(self.num_envs, device=self.device)

            rewards = {
                "goal_reached": hit.float() * self.cfg.reached_goal_reward,
                "distance_to_goal": distance_reward * self.cfg.dist_to_goal_reward_scale * self.step_dt,
                # "time_penalty": time_penalty_per_second * self.step_dt,
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
            all_rewards[robot_id] = reward
            # Logging
            for key, value in rewards.items():
                self._episode_sums[key] += value
            
        return {"team_0" : all_rewards["robot_0"]}

    def _get_dones(self) -> tuple[dict, dict]:
        reach_r = self.cfg.goal_reach_radius

        any_robot_reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # goals in XY: (N, 2, 2)
        goals_xy = self._desired_pos[:, :2]
        for robot_id, robot in self.robots.items():
            net_contact_forces = self.contact_sensors[robot_id].data.net_forces_w_history
            died = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self.base_ids[robot_id]], dim=-1), dim=1)[0] > 1.0, dim=1)
            # robot base XY: (N, 2)
            robot_xy = robot.data.root_pos_w[:, :2]
            # pairwise dists to both goals: (N, 2)
            dists = torch.norm(goals_xy - robot_xy, dim=-1)
            # did this robot hit any goal? (N,)
            hit = (dists <= reach_r)
            any_robot_reached |= hit


        dones = {team: torch.logical_or(died.clone(), any_robot_reached.clone()) for team in self.cfg.teams.keys()}

        to_mask = self.episode_length_buf >= (self.max_episode_length - 1)
        time_out = {team: to_mask for team in self.cfg.teams.keys()}

        return dones, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robots["robot_0"]._ALL_INDICES
        super()._reset_idx(env_ids)  # once

        # spread out resets
        if len(env_ids) == self.num_envs: # type:ignore
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        # --- Reset robots & states ---
        origins = self.scene.env_origins[env_ids]  # (N, 3)
        N = env_ids.shape[0]

        min_separation = 1            
        spawn_radius_max = 2
        max_tries = 10

        sampled_offsets = {}            # per robot (N, 3)
        robot_ids = list(self.robots.keys())

        for robot_id, robot in self.robots.items():
            if env_ids is None or len(env_ids) == self.num_envs:
                env_ids = robot._ALL_INDICES
            self.actions[robot_id][env_ids] = 0.0
            self.previous_actions[robot_id][env_ids] = 0.0

            robot.reset(env_ids)

            # Reset robot state
            joint_pos = robot.data.default_joint_pos[env_ids]
            joint_vel = robot.data.default_joint_vel[env_ids]
            default_root_state = robot.data.default_root_state[env_ids]

            default_root_state[:, :2] += self._terrain.env_origins[env_ids][:, :2]
            robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
            robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
            robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self._desired_pos[env_ids, :2] = self.robots["robot_0"].data.root_pos_w[env_ids, :2] + \
            torch.zeros_like(self._desired_pos[env_ids, :2]).uniform_(-10.0, 10.0)
        self._desired_pos[env_ids, 2] = 0.25

        final_distance_to_goal = torch.linalg.norm(
            self._desired_pos[env_ids] - self.robots["robot_0"].data.root_pos_w[env_ids], dim=1
        ).mean()

        # logging
        extras = {}
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        extras["Metrics/final_distance_to_goal"] = final_distance_to_goal.item()
        self.extras["log"] = extras