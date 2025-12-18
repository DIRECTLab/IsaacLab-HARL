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
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply, quat_from_angle_axis

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
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass_0 = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot_0", body_names="base"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )


def define_markers() -> VisualizationMarkers:
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/myMarkers",
        markers={
            "arrow1": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                scale=(0.1, 0.1, 2.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.0, 0.0)),
            ),
            "arrow2": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                scale=(0.1, 0.1, 2.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),
            ),
        },
    )
    return VisualizationMarkers(marker_cfg)


@configclass
class AnymalCHappoFindBallEnvCfg(DirectMARLEnvCfg):
    # env
    episode_length_s = 20.0
    decimation = 4
    action_scale = 0.5
    action_space = 12
    action_spaces = {f"robot_{i}": 12 for i in range(1)}
    # observation_space = 48
    observation_space = 48
    observation_spaces = {f"robot_{i}": 48 for i in range(1)}
    state_space = 0
    state_spaces = {f"robot_{i}": 0 for i in range(1)}
    possible_agents = ["robot_0"]

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
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=6.0, replicate_physics=True)

    # events
    events: EventCfg = EventCfg()

    # robot
    robot_0: ArticulationCfg = ANYMAL_C_CFG.replace(prim_path="/World/envs/env_.*/Robot_0")
    contact_sensor_0: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot_0/.*", history_length=3, update_period=0.005, track_air_time=True
    )
    robot_0.init_state.rot = (1.0, 0.0, 0.0, 0.0)
    robot_0.init_state.pos = (0.0, 0.0, 0.5)

    cfg_soccer_ball = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object2",
        spawn=sim_utils.SphereCfg(
            radius=0.5,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="average",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.3,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(2.0, 0.0, 1.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    # reward scales (override from flat config)
    flat_orientation_reward_scale = 0.0

    # reward scales
    dist_to_ball_reward_scale = 5.0
    direction_reward_scale = 5.0
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


class AnymalCFindBallHappoEnv(DirectMARLEnv):
    cfg: AnymalCHappoFindBallEnvCfg

    def __init__(self, cfg: AnymalCHappoFindBallEnvCfg, render_mode: str | None = None, debug=False, **kwargs):
        self.debug = debug
        super().__init__(cfg, render_mode, **kwargs)

        # Joint position command (deviation from default joint positions)

        self.actions = {
            agent: torch.zeros(self.num_envs, action_space, device=self.device)
            for agent, action_space in self.cfg.action_spaces.items()
        }
        self.previous_actions = {
            agent: torch.zeros(self.num_envs, action_space, device=self.device)
            for agent, action_space in self.cfg.action_spaces.items()
        }

        self._desired_pos_w = self.soccer_ball.data.root_pos_w.clone()

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "direction_reward",
                "distance_to_ball_reward",
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
            _undesired_contact_body_ids, _ = contact_sensor.find_bodies(".*THIGH")
            self.base_ids[robot_id] = _base_id
            self.feet_ids[robot_id] = _feet_ids
            self.undesired_body_contact_ids[robot_id] = _undesired_contact_body_ids

        if self.debug:
            self.my_visualizer = define_markers()

        self.previous_ball_pos = self.soccer_ball.data.root_pos_w.clone()
        self.curr_ball_pos = self.soccer_ball.data.root_pos_w.clone()

        self.previous_anymal_pos = self.robots["robot_0"].data.root_pos_w.clone()
        self.curr_anymal_pos = self.robots["robot_0"].data.root_pos_w.clone()

    def _setup_scene(self):
        self.num_robots = sum(1 for key in self.cfg.__dict__.keys() if "robot_" in key)
        self.robots = {}
        self.contact_sensors = {}
        self.height_scanners = {}
        self.soccer_ball = RigidObject(self.cfg.cfg_soccer_ball)  # Soccer Ball

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

    def _pre_physics_step(self, actions: torch.Tensor):
        # We need to process the actions for each scene independently
        self.processed_actions = {}
        for robot_id, robot in self.robots.items():
            self.actions[robot_id] = actions[robot_id].clone()
            self.processed_actions[robot_id] = (
                self.cfg.action_scale * self.actions[robot_id] + robot.data.default_joint_pos
            )

    def _apply_action(self):
        for robot_id, robot in self.robots.items():
            robot.set_joint_position_target(self.processed_actions[robot_id])

    def _get_observations(self) -> dict:
        self.previous_actions = copy.deepcopy(self.actions)
        obs = {}

        for robot_id, robot in self.robots.items():
            obs[robot_id] = torch.cat(
                [
                    tensor
                    for tensor in (
                        robot.data.root_com_lin_vel_b,
                        robot.data.root_com_ang_vel_b,
                        robot.data.projected_gravity_b,
                        self._desired_pos_w,
                        robot.data.joint_pos - robot.data.default_joint_pos,
                        robot.data.joint_vel,
                        self.actions[robot_id],
                    )
                    if tensor is not None
                ],
                dim=-1,
            )

        return obs

    def _draw_markers(self, robot_pos, orientation_desired, orientation_forward):
        """
        Visualizes two orientation markers at the robot's position:
        - One pointing in the direction of the desired target
        - One showing the current velocity direction of the robot
        """

        # Marker metadata
        marker_ids = torch.concat(
            [
                torch.zeros(self.num_envs, dtype=torch.int32),
                torch.ones(self.num_envs, dtype=torch.int32),
            ],
            dim=0,
        )

        marker_locations = torch.concat(
            [
                robot_pos,
                robot_pos,
            ],
            dim=0,
        )

        orientations = torch.concat(
            [
                orientation_desired,
                orientation_forward,
            ],
            dim=0,
        )

        self.my_visualizer.visualize(marker_locations, orientations, marker_indices=marker_ids)

    def _get_rewards(self) -> dict:
        robot_pos = self.robots["robot_0"].data.root_pos_w
        ball_pos = self.soccer_ball.data.root_pos_w

        # Normalize direction vectors
        eps = 1e-6
        desired_dir = ball_pos - robot_pos
        desired_dir[:, 2] = 0.0
        desired_dir[:, 2] = 0.0  # Ignore z component
        desired_dir = desired_dir / torch.norm(desired_dir, dim=1, keepdim=True)

        # Reference vector: +X axis
        x_vector = torch.zeros_like(desired_dir)
        x_vector[:, 0] = 1.0

        robot_quat_w = self.robots["robot_0"].data.root_quat_w
        forward_world = quat_apply(robot_quat_w, x_vector)
        forward_world = forward_world / torch.norm(forward_world, dim=1, keepdim=True)

        # Rotation axis: cross product with x_vector
        r_desired = torch.cross(x_vector, desired_dir)
        r_desired_norm = torch.norm(r_desired, dim=1, keepdim=True)
        r_desired = torch.where(r_desired_norm > eps, r_desired / r_desired_norm, torch.zeros_like(r_desired))

        angle_desired = torch.atan2(desired_dir[:, 1], desired_dir[:, 0])
        forward_angle = torch.atan2(forward_world[:, 1], forward_world[:, 0])

        z_axis = torch.tensor([0.0, 0.0, 1.0], device=angle_desired.device).expand(self.num_envs, 3)
        orientation_desired = quat_from_angle_axis(angle_desired, z_axis)
        orientation_forward = quat_from_angle_axis(forward_angle, z_axis)

        if self.debug:
            self._draw_markers(robot_pos, orientation_desired, orientation_forward)

        # reward for getting closer to the ball
        direction_vector = self._desired_pos_w - self.robots["robot_0"].data.root_com_pos_w
        distance_to_ball = torch.linalg.norm(direction_vector, dim=1)
        distance_to_ball_mapped = 1 - torch.tanh(distance_to_ball / 0.8)

        forward_vector = quat_apply(orientation_forward, x_vector)
        desired_vector = quat_apply(orientation_desired, x_vector)

        forward_vector = forward_vector / torch.norm(forward_vector, dim=1, keepdim=True)
        desired_vector = desired_vector / torch.norm(desired_vector, dim=1, keepdim=True)

        direction_reward = torch.sum(forward_vector * desired_vector, dim=1)

        # z velocity tracking
        z_vel_error = torch.square(self.robots["robot_0"].data.root_lin_vel_b[:, 2])
        # angular velocity x/y
        ang_vel_error = torch.sum(torch.square(self.robots["robot_0"].data.root_ang_vel_b[:, :2]), dim=1)
        # joint torques
        joint_torques = torch.sum(torch.square(self.robots["robot_0"].data.applied_torque), dim=1)
        # joint acceleration
        joint_accel = torch.sum(torch.square(self.robots["robot_0"].data.joint_acc), dim=1)
        # action rate
        action_rate = torch.sum(torch.square(self.actions["robot_0"] - self.previous_actions["robot_0"]), dim=1)
        # feet air time
        first_contact = self.contact_sensors["robot_0"].compute_first_contact(self.step_dt)[:, self.feet_ids["robot_0"]]
        last_air_time = self.contact_sensors["robot_0"].data.last_air_time[:, self.feet_ids["robot_0"]]
        air_time = torch.sum((last_air_time - 0.5) * first_contact, dim=1) * (
            torch.norm(direction_vector[:, :2], dim=1) > 0.1
        )
        # undesired contacts
        net_contact_forces = self.contact_sensors["robot_0"].data.net_forces_w_history
        is_contact = (
            torch.max(torch.norm(net_contact_forces[:, :, self.undesired_body_contact_ids["robot_0"]], dim=-1), dim=1)[
                0
            ]
            > 1.0
        )
        contacts = torch.sum(is_contact, dim=1)
        # flat orientation
        flat_orientation = torch.sum(torch.square(self.robots["robot_0"].data.projected_gravity_b[:, :2]), dim=1)

        rewards = {
            "direction_reward": direction_reward * self.cfg.direction_reward_scale * self.step_dt,
            "distance_to_ball_reward": distance_to_ball_mapped * self.cfg.dist_to_ball_reward_scale * self.step_dt,
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

        return {"robot_0": reward}

    def _get_dones(self) -> tuple[dict, dict]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        net_contact_forces = self.contact_sensors["robot_0"].data.net_forces_w_history
        died = torch.any(
            torch.max(torch.norm(net_contact_forces[:, :, self.base_ids["robot_0"]], dim=-1), dim=1)[0] > 1.0, dim=1
        )
        return {"robot_0": died}, {"robot_0": time_out}

    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)

        min_dist_to_ball = 3.0

        object_default_state2 = self.soccer_ball.data.default_root_state.clone()[env_ids]
        object_default_state2[:, 0:3] = object_default_state2[:, 0:3] + self.scene.env_origins[env_ids]
        object_default_state2[:, :2] += torch.zeros_like(object_default_state2[:, :2], device=self.device).uniform_(
            -10, 10
        )
        mask1 = (object_default_state2[:, :2] < min_dist_to_ball) & (object_default_state2[:, :2] > 0.0)
        mask2 = (object_default_state2[:, :2] > -min_dist_to_ball) & (object_default_state2[:, :2] < 0.0)

        object_default_state2[:, :2][mask1] = min_dist_to_ball
        object_default_state2[:, :2][mask2] = -min_dist_to_ball

        self.soccer_ball.write_root_state_to_sim(object_default_state2, env_ids)
        self.soccer_ball.reset(env_ids)

        # Joint position command (deviation from default joint positions)
        for agent, action_space in self.cfg.action_spaces.items():
            self.actions[agent][env_ids] = torch.zeros(env_ids.shape[0], action_space, device=self.device)

        for _, robot in self.robots.items():
            if env_ids is None or len(env_ids) == self.num_envs:
                env_ids = robot._ALL_INDICES
            robot.reset(env_ids)
            if len(env_ids) == self.num_envs:
                # Spread out the resets to avoid spikes in training when many environments reset at a similar time
                self.episode_length_buf[:] = torch.randint_like(
                    self.episode_length_buf, high=int(self.max_episode_length)
                )

            # Reset robot state
            joint_pos = robot.data.default_joint_pos[env_ids]
            joint_vel = robot.data.default_joint_vel[env_ids]
            default_root_state = robot.data.default_root_state[env_ids]
            default_root_state[:, :3] += self._terrain.env_origins[env_ids]
            robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
            robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
            robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self.previous_anymal_pos = self.robots["robot_0"].data.root_pos_w.clone()
        self.previous_ball_pos = self.soccer_ball.data.root_link_pos_w.clone()
        self.curr_anymal_pos = self.robots["robot_0"].data.root_pos_w.clone()
        self.curr_ball_pos = self.soccer_ball.data.root_link_pos_w.clone()

        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0

        extras["final_distance_to_ball"] = (
            torch.linalg.norm(
                self.robots["robot_0"].data.root_pos_w[env_ids] - self.soccer_ball.data.root_pos_w[env_ids], dim=-1
            )
            .mean()
            .item()
        )
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
