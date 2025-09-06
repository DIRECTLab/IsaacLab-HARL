# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import torch

import isaacsim.core.utils.torch as torch_utils
from isaacsim.core.utils.torch.rotations import compute_heading_and_up, compute_rot, quat_conjugate

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
import torchvision.transforms as transforms
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import Camera, CameraCfg, TiledCamera, TiledCameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import subtract_frame_transforms, quat_from_angle_axis, quat_from_euler_xyz, quat_mul, euler_xyz_from_quat, normalize
from torch import nn


##
# Pre-defined configs
##
from isaaclab_assets.robots.minitank import MINITANK_CFG  # isort: skip
from isaaclab_assets.robots.anymal import ANYMAL_C_CFG  # isort: skip
from isaaclab_assets.robots.unitree import H1_CFG  # isort: skip
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip
from isaaclab_assets import CRAZYFLIE_CFG  # isort: skip


class SimpleCNN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, img_height: int = 480, img_width: int = 640):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.fc = nn.Linear(32 * img_height * img_width, out_channels)

    def forward(self, x):
        """forward pass of the CNN.

        Args:
            x (tensor): shape (batch_size, channels, height, width)

        Returns:
            _type_: _description_
        """
        x = x.nan_to_num(nan=0.0, posinf=100_000) # since values are depth values, we can set inf to a large number
        x = x.permute(0, 3, 1, 2)  # Change to (batch_size, channels, height, width)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x



def normalize_angle(x):
    return torch.atan2(torch.sin(x), torch.cos(x))

@configclass
class MinitankStage1EnvCfg(DirectMARLEnvCfg):
    # env
    episode_length_s = 20.0
    decimation = 4
    anymal_action_scale = 0.5
    action_space = 2
    action_spaces = {"robot_0": 2}

    # with camera
    # observation_spaces = {"robot_0": 1,  "robot_1": 1036}
    observation_spaces = {"robot_0": 10}
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

    ### MINITANK CONFIGURATION ###
    robot_0: ArticulationCfg = MINITANK_CFG.replace(prim_path="/World/envs/env_.*/Robot_0")
    # robot_0.init_state.rot = (1.0, 0.0, 0.0, 1.0)
    robot_0.init_state.pos = (0.0, 0.0, 0.2)

    # camera_0 = TiledCameraCfg(
    #     prim_path="/World/envs/env_.*/Robot_0/robot/arm/front_cam",
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
    #     # offset=CameraCfg.OffsetCfg(pos=(0, 0, .25), rot=(0,0,1,0), convention="opengl"),
    #     offset=CameraCfg.OffsetCfg(pos=(0, 0, 1), rot=(0,0,1,0), convention="opengl"),
    # )

    action_scale = .5
    max_vel = 2
    ### MINITANK CONFIGURATION ###

def define_markers() -> VisualizationMarkers:
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/myMarkers",
        markers={
            "arrow1": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                scale=(.1, .1, 1),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.0, 0.0)),
            ),
            "arrow2": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                scale=(.1, .1, 1),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.8, 0.0)),
            ),
            "sphere1": sim_utils.SphereCfg(
                radius=0.1,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),
            ),
        },
    )
    return VisualizationMarkers(marker_cfg)



class MinitankStage1Env(DirectMARLEnv):
    cfg: MinitankStage1EnvCfg
    def __init__(
        self,
        cfg: MinitankStage1EnvCfg,
        render_mode: str | None = None,
        headless: bool | None = False,
        **kwargs,
    ):
        super().__init__(cfg, render_mode, **kwargs)
        # self.cnnModel = SimpleCNN(1, 1024, 256, 256).to(self.device)
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
        self._desired_pos_w = torch.zeros((self.num_envs, 3), device=self.device)

        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "tank_angle_reward",
            ]
        }

        if not self.headless:
            self.my_visualizer = define_markers()


    def _draw_markers(self):

        marker_ids = torch.concat([
            torch.zeros(self.num_envs, dtype=torch.int32).to(self.device),
            torch.ones(self.num_envs, dtype=torch.int32).to(self.device),
            2 * torch.ones(self.num_envs, dtype=torch.int32).to(self.device)
        ], dim=0)

        desired_pos = self._desired_pos_w
        arm_pos = self.robots['robot_0'].data.body_com_pos_w[:, 1, :]
        base_pos = self.robots['robot_0'].data.body_com_pos_w[:, 0, :]
        base_pos_offset = torch.zeros_like(base_pos)
        base_pos_offset[:, 2] = 0.06
        base_pos = base_pos + base_pos_offset


        diff = desired_pos - arm_pos
        arm_diff = arm_pos - base_pos
        arm_direction = arm_diff / torch.linalg.norm(arm_diff, dim=1, keepdim=True)

        desired_direction = diff / torch.linalg.norm(diff, dim=1, keepdim=True)
        x_vector = torch.zeros_like(desired_direction)
        x_vector[:, 0] = 1.0

        r = torch.cross(x_vector, desired_direction)
        r = r / torch.linalg.norm(r, dim=1, keepdim=True)
        r_arm = torch.cross(x_vector, arm_direction)
        r_arm = r_arm / torch.linalg.norm(r_arm, dim=1, keepdim=True)

        dot_prod_angle = torch.sum(x_vector * desired_direction, dim=1)
        angle = dot_prod_angle / (x_vector.norm(dim=1) * desired_direction.norm(dim=1))
        angle = torch.acos(angle)


        dot_prod_angle_arm = torch.sum(x_vector * arm_direction, dim=1)
        angle_arm = dot_prod_angle_arm / (x_vector.norm(dim=1) * arm_direction.norm(dim=1))
        angle_arm = torch.acos(angle_arm)

        orientation = quat_from_angle_axis(angle, r)
        arm_orientation = quat_from_angle_axis(angle_arm, r_arm)
        sphere_orientation = torch.zeros_like(arm_orientation)
        positions = torch.concat([arm_pos, arm_pos, self._desired_pos_w], dim=0)
        orientations = torch.concat([orientation, arm_orientation, sphere_orientation], dim=0)

        self.my_visualizer.visualize(positions, orientations, marker_indices=marker_ids)


    def _get_vector_angle_reward(self):
        """Calculates the cosine of the angle between the quaternion vector from the minitank to the drone and the\
              actual quaternion vector of the arm of the minitank.
        """
        arm_pos = self.robots['robot_0'].data.body_com_pos_w[:, 1, :]
        base_pos = self.robots['robot_0'].data.body_com_pos_w[:, 0, :]
        base_pos_offset = torch.zeros_like(base_pos)
        base_pos_offset[:, 2] = 0.06
        base_pos = base_pos + base_pos_offset


        diff = self._desired_pos_w - arm_pos
        arm_diff = arm_pos - base_pos
        arm_direction = arm_diff / torch.linalg.norm(arm_diff, dim=1, keepdim=True)

        desired_direction = diff / torch.linalg.norm(diff, dim=1, keepdim=True)
        x_vector = torch.zeros_like(desired_direction)
        x_vector[:, 0] = 1.0

        r = torch.cross(x_vector, desired_direction)
        r = r / torch.linalg.norm(r, dim=1, keepdim=True)
        r_arm = torch.cross(x_vector, arm_direction)
        r_arm = r_arm / torch.linalg.norm(r_arm, dim=1, keepdim=True)

        dot_prod_angle = torch.sum(x_vector * desired_direction, dim=1)
        angle = dot_prod_angle / (x_vector.norm(dim=1) * desired_direction.norm(dim=1))
        angle = torch.acos(angle)


        dot_prod_angle_arm = torch.sum(x_vector * arm_direction, dim=1)
        angle_arm = dot_prod_angle_arm / (x_vector.norm(dim=1) * arm_direction.norm(dim=1))
        angle_arm = torch.acos(angle_arm)

        self.desired_orientation = normalize(quat_from_angle_axis(angle, r)) 
        self.arm_orientation = normalize(quat_from_angle_axis(angle_arm, r_arm))

        a = torch.sum(torch.abs(self.desired_orientation - self.arm_orientation), dim=1)
        b = torch.sum(torch.abs(self.desired_orientation + self.arm_orientation), dim=1)

        res = torch.stack([a,b], dim=1)
        reward = torch.min(res, dim=1)
        reward_mapped = torch.exp(-reward.values)
        return reward_mapped, self.arm_orientation, self.desired_orientation
        # return reward
 
    def _setup_scene(self):
        self.num_robots = sum(1 for key in self.cfg.__dict__.keys() if "robot_" in key)
        self.robots = {}
        # self.cameras = {}

        for i in range(self.num_robots):
            robot_id = f"robot_{i}"
            if robot_id in self.cfg.__dict__:
                self.robots[f"robot_{i}"] = Articulation(self.cfg.__dict__["robot_" + str(i)])
                self.scene.articulations[f"robot_{i}"] = self.robots[f"robot_{i}"]

        ### SETUP CAMERAS ###
        # self.cameras["robot_0"] = TiledCamera(self.cfg.camera_0)
        # self.scene.sensors["robot_0_camera"] = self.cameras["robot_0"]
        # self.cameras["robot_1"] = TiledCamera(self.cfg.camera_1)
        # self.scene.sensors["robot_1_camera"] = self.cameras["robot_1"]
        ### SETUP CAMERAS ###


        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: dict):

        ### PREPHYSICS FOR MINITANK ###

        self.processed_actions = copy.deepcopy(actions)
        self.processed_actions["robot_0"] = (
            torch.clip((self.cfg.action_scale * self.processed_actions["robot_0"]) + \
                       self.robots["robot_0"].data.joint_vel, -self.cfg.max_vel, self.cfg.max_vel)
        )
        ### PREPHYSICS FOR MINITANK ###

    def _apply_action(self):
        # self.robots["robot_0"].set_joint_velocity_target(self.processed_actions["robot_0"])
        # self.processed_actions["robot_0"] = torch.tensor([[0,1]]).to(self.device)
        self.robots["robot_0"].set_joint_velocity_target(self.processed_actions["robot_0"])

    def _get_observations(self) -> dict:
        # drone_camera = self.cameras["robot_1"].data.output["depth"].to(self.device)
        # drone_camera_feat = self.cnnModel(drone_camera)
        self.arm_orientation_reward, self.arm_orientation, self.desired_orientation = self._get_vector_angle_reward()

        tank_obs = torch.cat(
            [
                self.processed_actions["robot_0"],
                self.desired_orientation,
                self.arm_orientation
                # TODO add more padded observations if needed
            ],
            dim=-1,
        )

        self.previous_actions = copy.deepcopy(self.actions)
        obs = {"robot_0":tank_obs}
        
        return obs

    def get_y_euler_from_quat(self, quaternion):
        w, x, y, z = quaternion[:, 0], quaternion[:, 1], quaternion[:, 2], quaternion[:, 3]
        y_euler_angle = torch.arcsin(2 * (w * y - z * x))
        return y_euler_angle

    def _get_rewards(self) -> dict:
        if not self.headless:
            self._draw_markers()

        ### MINITANK REWARDS ###
        minitank_rewards = self.arm_orientation_reward * self.step_dt
        ### MINITANK REWARDS ###
    
        self._episode_sums["tank_angle_reward"] = minitank_rewards

        return {"robot_0": minitank_rewards.to(self.device)}

    def _get_dones(self) -> tuple[dict, dict]:
        time_out = (self.episode_length_buf >= self.max_episode_length - 1).to(self.device)
        dones = {}
        dones["robot_0"] = torch.zeros(self.num_envs).to(torch.int8).to(self.device)
        time_out = {robot_id:time_out for robot_id in self.robots.keys()}

        # dones = {robot_id: torch.zeros(self.num_envs).to(torch.int8).to(self.device) for robot_id in self.robots.keys()}

        return dones, time_out
        # return dones, dones

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()

        for agent, action_space in self.cfg.action_spaces.items():
            self.actions[agent][env_ids] = torch.zeros(env_ids.shape[0], action_space, device=self.device)
            self.previous_actions[agent][env_ids] = torch.zeros(env_ids.shape[0], action_space, device=self.device)

        for robot_id, robot in self.robots.items():
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
            # default_root_state[:, :2] += torch.zeros_like(default_root_state[:, :2]).uniform_(-5, 5)
            # if robot_id == "robot_1":
            #     default_root_state[:, 2] = torch.zeros_like(default_root_state[:, 2]).uniform_(1, 5)

            default_root_state[:, :3] += self._terrain.env_origins[env_ids]
            robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
            robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
            robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self._desired_pos_w[env_ids, :2] = self.robots["robot_0"].data.root_pos_w[env_ids, :2] + \
            torch.zeros_like(self._desired_pos_w[env_ids, :2]).uniform_(-10.0, 10.0)
        
        vals = self._desired_pos_w[env_ids, :2]

        # Set values between 0 and 3 to 3
        vals[(vals > 0) & (vals < 3)] = 3.0

        # Set values between -3 and 0 to -3
        vals[(vals < 0) & (vals > -3)] = -3.0

        self._desired_pos_w[env_ids, :2] = vals

        self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2]).uniform_(0.5, 10.0)


        
