# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import torch
import torch.nn as nn
from torch import nn

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

##
# Pre-defined configs
##
from isaaclab_assets.robots.minitank import MINITANK_CFG  # isort: skip
from isaaclab_assets.robots.anymal import ANYMAL_C_CFG  # isort: skip
from isaaclab_assets.robots.unitree import H1_CFG  # isort: skip
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip
from isaaclab_assets import CRAZYFLIE_CFG  # isort: skip


class SimpleCNN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, input_height: int, input_width: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Perform dummy forward pass to get flattened size
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, input_height, input_width)
            dummy_output = self._extract_features(dummy_input)
            self.flattened_size = dummy_output.view(1, -1).size(1)

        self.fc = nn.Linear(self.flattened_size, out_channels)

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the CNN.

        Args:
            x (Tensor): shape (batch_size, height, width, channels)

        Returns:
            Tensor: output tensor of shape (batch_size, out_channels)
        """
        x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
        x = torch.nan_to_num(x, nan=0.0, posinf=100_000)
        x = self._extract_features(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x


def normalize_angle(x):
    return torch.atan2(torch.sin(x), torch.cos(x))


# @configclass
# class EventCfg:
#     """Configuration for randomization."""

#     physics_material_0 = EventTerm(
#         func=mdp.randomize_rigid_body_material,
#         mode="startup",
#         params={
#             "asset_cfg": SceneEntityCfg("robot_0", body_names=".*"),
#             "static_friction_range": (0.8, 0.8),
#             "dynamic_friction_range": (0.6, 0.6),
#             "restitution_range": (0.0, 0.0),
#             "num_buckets": 64,
#         },
#     )

#     add_base_mass_0 = EventTerm(
#         func=mdp.randomize_rigid_body_mass,
#         mode="startup",
#         params={
#             "asset_cfg": SceneEntityCfg("robot_0", body_names="base"),
#             "mass_distribution_params": (-5.0, 5.0),
#             "operation": "add",
#         },
#     )


@configclass
class DronesEnvCfg(DirectMARLEnvCfg):
    # env
    episode_length_s = 20.0
    decimation = 4
    anymal_action_scale = 0.5
    action_space = 3
    action_spaces = {"robot_0": 4}

    # with camera = 12 + output dim of cnn = 32 = 44
    # observation_spaces = {"robot_0": 44}
    observation_spaces = {"robot_0": 12}
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
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=10.0, replicate_physics=True)

    # events
    # events: EventCfg = EventCfg()

    # CRAZYFLIE CONFIGURATION #
    robot_0: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot_0")
    robot_0.init_state.pos = (0.0, 0.0, 2.0)

    # camera_0 = TiledCameraCfg(
    #     prim_path="/World/envs/env_.*/Robot_0/body/front_cam",
    #     update_period=0.1,
    #     height=256,
    #     width=256,
    #     data_types=["depth"],
    #     spawn=sim_utils.FisheyeCameraCfg(
    #         projection_type="fisheyePolynomial",
    #     ),
    #     # spawn=sim_utils.PinholeCameraCfg(
    #     #     focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
    #     # ),
    #     offset=CameraCfg.OffsetCfg(pos=(0, 0, 0), rot=(1,0,0,0), convention="opengl"),
    # )

    thrust_to_weight = 1.9
    moment_scale = 0.01

    # reward scales
    lin_vel_reward_scale = -0.05
    ang_vel_reward_scale = -0.01
    distance_to_goal_reward_scale = 15.0

    # CRAZYFLIE CONFIGURATION #


def define_markers() -> VisualizationMarkers:
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/myMarkers",
        markers={
            "sphere1": sim_utils.SphereCfg(
                radius=0.1,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),
            ),
        },
    )
    return VisualizationMarkers(marker_cfg)


class DronesEnv(DirectMARLEnv):
    cfg: DronesEnvCfg

    def __init__(
        self,
        cfg: DronesEnvCfg,
        render_mode: str | None = None,
        headless: bool | None = False,
        **kwargs,
    ):
        super().__init__(cfg, render_mode, **kwargs)
        # self.cnnModel = SimpleCNN(1, 32, self.cfg.camera_0.height, self.cfg.camera_0.width).to(self.device)
        self.headless = headless
        # self.headless = True
        self.actions = {
            agent: torch.zeros(self.num_envs, action_space, device=self.device)
            for agent, action_space in self.cfg.action_spaces.items()
        }
        self.previous_actions = {
            agent: torch.zeros(self.num_envs, action_space, device=self.device)
            for agent, action_space in self.cfg.action_spaces.items()
        }
        self.processed_actions = {
            agent: torch.zeros(self.num_envs, action_space, device=self.device)
            for agent, action_space in self.cfg.action_spaces.items()
        }

        # CRAZYFLIE INITIALIZATION #
        # with open(crazy_flie_model, "rb") as f:
        #     self.crazy_flie_model = torch.jit.load(f)

        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

        crazyflie_mass = self.robots["robot_0"].root_physx_view.get_masses()[0].sum()
        self._crazyflie_body_ids = self.robots["robot_0"].find_bodies("body")[0]
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._crazyflie_weight = (crazyflie_mass * self._gravity_magnitude).item()
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
        # CRAZYFLIE INITIALIZATION #
        if not self.headless:
            self.my_visualizer = define_markers()

    def _draw_markers(self):

        marker_ids = torch.concat(
            [
                torch.zeros(self.num_envs, dtype=torch.int32).to(self.device),
            ],
            dim=0,
        )

        orientations = torch.zeros(self.num_envs, 4, device=self.device)

        self.my_visualizer.visualize(self._desired_pos_w, orientations, marker_indices=marker_ids)

    def _setup_scene(self):
        self.num_robots = sum(1 for key in self.cfg.__dict__.keys() if "robot_" in key)
        self.robots = {}
        # self.cameras = {}

        for i in range(self.num_robots):
            robot_id = f"robot_{i}"
            if robot_id in self.cfg.__dict__:
                self.robots[f"robot_{i}"] = Articulation(self.cfg.__dict__["robot_" + str(i)])
                self.scene.articulations[f"robot_{i}"] = self.robots[f"robot_{i}"]

        # SETUP CAMERAS #
        # self.cameras["robot_0"] = TiledCamera(self.cfg.camera_0)
        # self.scene.sensors["robot_0_camera"] = self.cameras["robot_0"]
        # SETUP CAMERAS #

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: dict):

        # PREPHYSICS FOR CRAZYFLIE #

        self.processed_actions["robot_0"] = self.processed_actions["robot_0"].clamp(-1.0, 1.0)
        self._thrust[:, 0, 2] = (
            self.cfg.thrust_to_weight * self._crazyflie_weight * (self.processed_actions["robot_0"][:, 0] + 1.0) / 2.0
        )
        self._moment[:, 0, :] = self.cfg.moment_scale * self.processed_actions["robot_0"][:, 1:]

        # PREPHYSICS FOR CRAZYFLIE #

    def _apply_action(self):
        self.robots["robot_0"].set_external_force_and_torque(
            self._thrust, self._moment, body_ids=self._crazyflie_body_ids
        )

    def _get_observations(self) -> dict:
        # drone_camera = self.cameras["robot_0"].data.output["depth"].to(self.device)
        # drone_camera_feat = self.cnnModel(drone_camera)

        desired_pos_b, _ = subtract_frame_transforms(
            self.robots["robot_0"].data.root_state_w[:, :3],
            self.robots["robot_0"].data.root_state_w[:, 3:7],
            self._desired_pos_w,
        )

        drone_obs = torch.cat(
            [
                self.robots["robot_0"].data.root_lin_vel_b,
                self.robots["robot_0"].data.root_ang_vel_b,
                self.robots["robot_0"].data.projected_gravity_b,
                desired_pos_b,
                # drone_camera_feat
            ],
            dim=-1,
        )
        self.previous_actions = copy.deepcopy(self.actions)
        obs = {"robot_0": drone_obs.to(self.device)}

        return obs

    def get_y_euler_from_quat(self, quaternion):
        w, x, y, z = quaternion[:, 0], quaternion[:, 1], quaternion[:, 2], quaternion[:, 3]
        y_euler_angle = torch.arcsin(2 * (w * y - z * x))
        return y_euler_angle

    def _get_rewards(self) -> dict:
        if not self.headless:
            self._draw_markers()

        lin_vel = torch.sum(torch.square(self.robots["robot_0"].data.root_lin_vel_b), dim=1)
        ang_vel = torch.sum(torch.square(self.robots["robot_0"].data.root_ang_vel_b), dim=1)
        distance_to_goal = torch.linalg.norm(self._desired_pos_w - self.robots["robot_0"].data.root_pos_w, dim=1)
        distance_to_goal_mapped = 1 - torch.tanh(distance_to_goal / 0.8)
        rewards = {
            "lin_vel": lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
            "distance_to_goal": distance_to_goal_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return {"robot_0": reward}

    def _get_dones(self) -> tuple[dict, dict]:
        time_out = (self.episode_length_buf >= self.max_episode_length - 1).to(self.device)
        died = self.robots["robot_0"].data.root_pos_w[:, 2] < 0.1
        dones = {}
        dones["robot_0"] = died.to(self.device)
        time_out = {robot_id: time_out for robot_id in self.robots.keys()}

        # dones = {robot_id: torch.zeros(self.num_envs).to(torch.int8).to(self.device) for robot_id in self.robots.keys()}

        return dones, time_out
        # return dones, dones

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robots["robot_0"]._ALL_INDICES

        # Logging
        final_distance_to_goal = torch.linalg.norm(
            self._desired_pos_w[env_ids] - self.robots["robot_0"].data.root_pos_w[env_ids], dim=1
        ).mean()
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        # extras["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        # extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        extras["Metrics/final_distance_to_goal"] = final_distance_to_goal.item()
        self.extras["log"].update(extras)

        self.robots["robot_0"].reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self.actions["robot_0"][env_ids] = 0.0
        # Sample new commands
        self._desired_pos_w[env_ids, :2] = self.robots["robot_0"].data.root_pos_w[env_ids, :2] + torch.zeros_like(
            self._desired_pos_w[env_ids, :2]
        ).uniform_(-5.0, 5.0)
        # self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
        self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2]).uniform_(0.5, 5.0)
        # Reset robot state
        joint_pos = self.robots["robot_0"].data.default_joint_pos[env_ids]
        joint_vel = self.robots["robot_0"].data.default_joint_vel[env_ids]
        default_root_state = self.robots["robot_0"].data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self.robots["robot_0"].write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robots["robot_0"].write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robots["robot_0"].write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
