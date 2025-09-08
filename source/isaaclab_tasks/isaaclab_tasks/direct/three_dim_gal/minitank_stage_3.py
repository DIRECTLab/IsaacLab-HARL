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
class MinitankStage3EnvCfg(DirectMARLEnvCfg):
    decimation = 4
    episode_length_s = 20.0

    # 2 minitanks, 2 drones
    action_spaces = {
        "robot_0": 2,   # e.g. throttle, steering
        "robot_3": 4,      # e.g. x, y, z, yaw
        "robot_1": 2,
        "robot_2": 4,
    }

    # Example: minitank obs = 18, drone obs = 20
    observation_spaces = {
        "robot_0": 18,
        "robot_3": 20,
        "robot_1": 18,
        "robot_2": 20,
    }

    state_space = 0
    state_spaces = {agent: 0 for agent in action_spaces.keys()}

    possible_agents = list(action_spaces.keys())

    teams = {
        "team_0": ["robot_0", "robot_3"],
        "team_1": ["robot_1", "robot_2"],
    }

    sim: SimulationCfg = SimulationCfg(dt=1 / 200, render_interval=decimation)

    # Minitank configs
    robot_0: ArticulationCfg = MINITANK_CFG.replace(prim_path="/World/envs/env_.*/Minitank_0")
    robot_0.init_state.pos = (0.0, 0.5, 0.2)

    robot_1: ArticulationCfg = MINITANK_CFG.replace(prim_path="/World/envs/env_.*/Minitank_1")
    robot_1.init_state.pos = (0.0, -0.5, 0.2)

    # Drone configs
    robot_3: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Drone_0")
    robot_3.init_state.pos = (2.0, 0.5, 3.5)

    robot_2: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Drone_1")
    robot_2.init_state.pos = (2.0, -0.5, 3.5)

    # Arena/physics
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

    env_spacing = 10.0
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=10.0, replicate_physics=True)

    action_scale = .5
    max_vel = 2
    ### MINITANK CONFIGURATION ###

def define_markers(agent_idx: int) -> VisualizationMarkers:
    palette = [
        (1.0, 0.0, 0.0),  # red
        (0.0, 0.5, 1.0),  # blue
        (1.0, 1.0, 0.0),  # yellow
        (1.0, 0.0, 1.0),  # magenta
        (0.0, 1.0, 1.0),  # cyan
        (1.0, 0.5, 0.0),  # orange
        (0.5, 0.0, 1.0),  # purple
    ]
    color = palette[agent_idx % len(palette)]
    marker_cfg = VisualizationMarkersCfg(
        prim_path=f"/Visuals/myMarkers/agent_{agent_idx}",
        markers={
            f"arrow_{agent_idx}": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                scale=(.1, .1, 1),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            ),
            f"laser_cylinder_{agent_idx}": sim_utils.CylinderCfg(
                radius=0.01,
                height=10,
                axis="x",
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            ),
            f"sphere_{agent_idx}": sim_utils.SphereCfg(
                radius=0.1,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            ),
        },
    )
    return VisualizationMarkers(marker_cfg)


def angle_between_vectors(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    """
    Computes the angle (in radians) between two vectors v1 and v2 along the last dimension.
    Args:
        v1 (torch.Tensor): shape (..., 3)
        v2 (torch.Tensor): shape (..., 3)
    Returns:
        torch.Tensor: angle in radians, shape (...,)
    """
    v1_norm = v1 / torch.linalg.norm(v1, dim=-1, keepdim=True)
    v2_norm = v2 / torch.linalg.norm(v2, dim=-1, keepdim=True)
    dot_prod = torch.sum(v1_norm * v2_norm, dim=-1)
    dot_prod = torch.clamp(dot_prod, -1.0, 1.0)  # Clamp for numerical stability
    angle = torch.acos(dot_prod)
    return angle

class MinitankStage3Env(DirectMARLEnv):
    cfg: MinitankStage3EnvCfg
    def __init__(
        self,
        cfg: MinitankStage3EnvCfg,
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
        self._desired_pos_w = {agent: torch.zeros((self.num_envs, 3), device=self.device) for agent in self.cfg.action_spaces}

        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "tank_angle_reward",
            ]
        }

        if not self.headless:
            self.agent_visualizers = {}
            for agent_idx, agent in enumerate(self.cfg.action_spaces):
                self.agent_visualizers[agent] = define_markers(agent_idx)


    def _draw_markers(self):
        # Color palette for agents
        palette = [
            (1.0, 0.0, 0.0),  # red
            (0.0, 0.5, 1.0),  # blue
            (1.0, 1.0, 0.0),  # yellow
            (1.0, 0.0, 1.0),  # magenta
            (0.0, 1.0, 1.0),  # cyan
            (1.0, 0.5, 0.0),  # orange
            (0.5, 0.0, 1.0),  # purple
        ]
        num_agents = len(self.cfg.action_spaces)
        # Each agent gets 3 markers: arrow, cylinder, sphere
        marker_ids = []
        for agent_idx in range(num_agents):
            marker_ids.extend([
                agent_idx * 3 + 0 for _ in range(self.num_envs)
            ])  # arrow
            marker_ids.extend([
                agent_idx * 3 + 1 for _ in range(self.num_envs)
            ])  # cylinder
            marker_ids.extend([
                agent_idx * 3 + 2 for _ in range(self.num_envs)
            ])  # sphere
        marker_ids = torch.tensor(marker_ids, dtype=torch.int32, device=self.device)

        positions_list = []
        orientations_list = []
        # For each agent, for each env, add [arrow, cylinder, sphere] in order
        for agent_idx, agent in enumerate(self.cfg.action_spaces):
            color = palette[agent_idx % len(palette)]
            desired_pos = self._desired_pos_w[agent]
            arm_pos = self.robots[agent].data.body_com_pos_w[:, 1, :]
            base_pos = self.robots[agent].data.body_com_pos_w[:, 0, :]
            base_pos_offset = torch.zeros_like(base_pos)
            base_pos_offset[:, 2] = 0.06
            base_pos = base_pos + base_pos_offset

            # Compute direction vectors
            diff = desired_pos - arm_pos
            arm_diff = arm_pos - base_pos
            arm_direction = arm_diff / torch.linalg.norm(arm_diff, dim=1, keepdim=True)
            desired_direction = diff / torch.linalg.norm(diff, dim=1, keepdim=True)
            
            # X axis reference vector
            x_vector = torch.zeros_like(desired_direction)
            x_vector[:, 0] = 1.0
            
            # Compute rotation axes for desired and arm directions
            r = torch.cross(x_vector, desired_direction)
            r = r / torch.linalg.norm(r, dim=1, keepdim=True)
            r_arm = torch.cross(x_vector, arm_direction)
            r_arm = r_arm / torch.linalg.norm(r_arm, dim=1, keepdim=True)
            
            # Compute angles for desired and arm directions using angle_between_vectors
            angle = angle_between_vectors(x_vector, desired_direction)
            angle_arm = angle_between_vectors(x_vector, arm_direction)
            
            # Compute quaternions for marker orientations
            orientation = quat_from_angle_axis(angle, r)
            arm_length = -0.25
            arm_offset = (arm_length / 2) * arm_direction
            arm_orientation = quat_from_angle_axis(angle_arm, r_arm)
            sphere_orientation = torch.zeros_like(arm_orientation)
            cylinder_length = 10.0
            cylinder_offset = (cylinder_length / 2) * arm_direction
            # For each env, add [arrow, cylinder, sphere]
            for env_idx in range(self.num_envs):
                positions_list.append(arm_pos[env_idx] + arm_offset[env_idx])      # arrow
                positions_list.append(arm_pos[env_idx] + cylinder_offset[env_idx]) # cylinder
                positions_list.append(desired_pos[env_idx])                        # sphere
                orientations_list.append(orientation[env_idx])
                orientations_list.append(arm_orientation[env_idx])
                orientations_list.append(sphere_orientation[env_idx])
        positions = torch.stack(positions_list, dim=0)
        orientations = torch.stack(orientations_list, dim=0)
        
        # Visualize markers in the scene
        if not self.headless:
            offset = 0
            for agent_idx, agent in enumerate(self.cfg.action_spaces):
                agent_marker_count = 3 * self.num_envs
                agent_positions = positions[offset:offset+agent_marker_count]
                agent_orientations = orientations[offset:offset+agent_marker_count]
                # For each env, marker indices should be [0, 1, 2] (arrow, cylinder, sphere)
                agent_marker_ids = torch.cat([
                    torch.tensor([0, 1, 2], dtype=torch.int32, device=self.device).repeat(self.num_envs)
                ])
                self.agent_visualizers[agent].visualize(agent_positions, agent_orientations, marker_indices=agent_marker_ids)
                offset += agent_marker_count


    def _get_vector_angle_reward(self, agent: str):
        """
        Calculates the cosine of the angle between the quaternion vector from the minitank to the drone and the
        actual quaternion vector of the arm of the minitank, for a given agent.
        """
        arm_pos = self.robots[agent].data.body_com_pos_w[:, 1, :]
        base_pos = self.robots[agent].data.body_com_pos_w[:, 0, :]
        base_pos_offset = torch.zeros_like(base_pos)
        base_pos_offset[:, 2] = 0.06
        base_pos = base_pos + base_pos_offset

        diff = self._desired_pos_w[agent] - arm_pos
        arm_diff = arm_pos - base_pos
        arm_direction = arm_diff / torch.linalg.norm(arm_diff, dim=1, keepdim=True)
        desired_direction = diff / torch.linalg.norm(diff, dim=1, keepdim=True)
        x_vector = torch.zeros_like(desired_direction)
        x_vector[:, 0] = 1.0

        r = torch.cross(x_vector, desired_direction)
        r = r / torch.linalg.norm(r, dim=1, keepdim=True)
        r_arm = torch.cross(x_vector, arm_direction)
        r_arm = r_arm / torch.linalg.norm(r_arm, dim=1, keepdim=True)

        # Use angle_between_vectors for both angles
        angle = angle_between_vectors(x_vector, desired_direction)
        angle_arm = angle_between_vectors(x_vector, arm_direction)

        desired_orientation = normalize(quat_from_angle_axis(angle, r))
        arm_orientation = normalize(quat_from_angle_axis(angle_arm, r_arm))

        a = torch.sum(torch.abs(desired_orientation - arm_orientation), dim=1)
        b = torch.sum(torch.abs(desired_orientation + arm_orientation), dim=1)

        res = torch.stack([a, b], dim=1)
        reward = torch.min(res, dim=1)
        reward_mapped = torch.exp(-reward.values)
        return reward_mapped, arm_orientation, desired_orientation
 
    def _setup_scene(self):

        self.robots = {}
        self.minitanks = {}
        self.drones = {}
        # Add minitanks
        for tank_id in ["robot_0", "robot_1"]:
            self.robots[tank_id] = Articulation(self.cfg.__dict__[tank_id])
            self.scene.articulations[tank_id] = self.robots[tank_id]
            self.minitanks[tank_id] = self.robots[tank_id]
        # Add drones
        for drone_id in ["robot_3", "robot_2"]:
            self.robots[drone_id] = Articulation(self.cfg.__dict__[drone_id])
            self.scene.articulations[drone_id] = self.robots[drone_id]
            self.drones[drone_id] = self.robots[drone_id]

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
        for agent in self.cfg.action_spaces:
            self.processed_actions[agent] = torch.clip(
                (self.cfg.action_scale * actions[agent]) + self.robots[agent].data.joint_vel,
                -self.cfg.max_vel, self.cfg.max_vel
            )
        ### PREPHYSICS FOR MINITANK ###

    def _apply_action(self):
        for agent in self.cfg.action_spaces:
            self.robots[agent].set_joint_velocity_target(self.processed_actions[agent])

    def _get_observations(self) -> dict:
        obs = {}
        # For now, use a fixed radius (can be made configurable)
        rcol = torch.full((self.num_envs, 1), 2.0, device=self.device)
        env_center = self._terrain.env_origins if hasattr(self._terrain, "env_origins") else torch.zeros((self.num_envs, 3), device=self.device)

        for i, (team, agents) in enumerate(self.cfg.teams.items()):
            obs[team] = {}
            for agent_id in agents:
                teammate_id = [a for a in agents if a != agent_id][0]
                enemy_team = [a for a in self.cfg.teams.keys() if a != team][0]
                enemies = self.cfg.teams[enemy_team]

                enemy_0_id = enemies[0]
                enemy_1_id = enemies[1]

                enemy_0_pos, _ = subtract_frame_transforms(
                    self.robots[agent_id].data.root_state_w[:, :3], self.robots[agent_id].data.root_state_w[:, 3:7],
                    self.robots[enemy_0_id].data.root_pos_w
                )

                teammate_pos, _ = subtract_frame_transforms(
                    self.robots[agent_id].data.root_state_w[:, :3], self.robots[agent_id].data.root_state_w[:, 3:7],
                    self.robots[teammate_id].data.root_pos_w
                )

                enemy_1_pos, _ = subtract_frame_transforms(
                    self.robots[agent_id].data.root_state_w[:, :3], self.robots[agent_id].data.root_state_w[:, 3:7],
                    self.robots[enemy_1_id].data.root_pos_w
                )

                dist_center = torch.norm(
                    self.robots[agent_id].data.root_pos_w - env_center, dim=-1, keepdim=True
                )

                # For tanks, use full state; for drones, use simpler obs
                if agent_id.startswith("minitank"):
                    obs_vec = torch.cat([
                        self.robots[agent_id].data.root_lin_vel_b,
                        self.robots[agent_id].data.root_ang_vel_b,
                        self.robots[agent_id].data.projected_gravity_b,
                        self.robots[agent_id].data.joint_pos - self.robots[agent_id].data.default_joint_pos,
                        self.robots[agent_id].data.joint_vel,
                        self.actions[agent_id],
                        enemy_1_pos,
                        teammate_pos,
                        enemy_0_pos,
                        dist_center,
                        rcol,
                    ], dim=-1)
                else:
                    robot_vel = self.robots[agent_id].data.root_lin_vel_b
                    obs_vec = torch.cat([
                        enemy_0_pos, teammate_pos, enemy_1_pos, dist_center, robot_vel, rcol
                    ], dim=1)

                obs_vec = torch.nan_to_num(obs_vec, nan=0.0, posinf=1e6, neginf=-1e6)
                # Ensure obs_vec matches expected shape
                target_size = self.cfg.observation_spaces[agent_id]
                obs_size = obs_vec.shape[1]
                if obs_size < target_size:
                    pad = torch.zeros((self.num_envs, target_size - obs_size), device=self.device)
                    obs_vec = torch.cat([obs_vec, pad], dim=1)
                elif obs_size > target_size:
                    obs_vec = obs_vec[:, :target_size]
                obs[team][agent_id] = obs_vec

        self.previous_actions = copy.deepcopy(self.actions)
        return obs

    def get_y_euler_from_quat(self, quaternion):
        w, x, y, z = quaternion[:, 0], quaternion[:, 1], quaternion[:, 2], quaternion[:, 3]
        y_euler_angle = torch.arcsin(2 * (w * y - z * x))
        return y_euler_angle

    def _get_rewards(self) -> dict:
        # Reward logic: teams are rewarded for simply existing together (no push-out or fallen logic)
        # Each team gets a constant reward per timestep as long as their agents exist
        reward_value = 1.0  # You can adjust this value as needed
        team_rewards = {team: torch.full((self.num_envs,), reward_value, device=self.device) for team in self.cfg.teams.keys()}

        # Optionally log the reward
        for team in self.cfg.teams.keys():
            self._episode_sums[f"{team}_existence_reward"] = team_rewards[team]

        return team_rewards


    def _get_dones(self) -> tuple[dict, dict]:
        # Multiagent done logic (adapted from sumo_stage_2_hetero)
        env_xy = self._terrain.env_origins[:, :2].to(self.device)
        out = {}
        for agent_id in self.robots.keys():
            pos_xy = self.robots[agent_id].data.root_pos_w[:, :2]
            dist = torch.linalg.norm(pos_xy - env_xy, dim=1)
            out[agent_id] = dist > 2.0

        team0_out = torch.any(torch.stack([out["robot_0"], out["robot_3"]]), dim=0)
        team1_out = torch.any(torch.stack([out["robot_1"], out["robot_2"]]), dim=0)
        fallen = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for agent_id, robot in self.minitanks.items():
            died = robot.data.root_pos_w[:, 2] < .1
            fallen |= died

        done = team0_out | team1_out | fallen
        dones = {team: done for team in self.cfg.teams.keys()}

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        timeouts = {team: time_out for team in self.cfg.teams.keys()}
        return dones, timeouts

    def _reset_idx(self, env_ids: torch.Tensor | None):
        # Multiagent reset logic (adapted from sumo_stage_2_hetero)
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        super()._reset_idx(env_ids)

        # Optionally randomize episode length
        if hasattr(self, "debug") and not self.debug:
            if len(env_ids) == self.num_envs:
                self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        # Reset robots and drones
        origins = self._terrain.env_origins[env_ids]  # (N, 3)
        for agent_id in self.robots.keys():
            robot = self.robots[agent_id]
            robot.reset(env_ids)
            default_root_state = robot.data.default_root_state[env_ids].clone()
            default_root_state[:, :3] = origins[:, :3]
            robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
            robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
            if agent_id in self.minitanks:
                joint_pos = robot.data.default_joint_pos[env_ids]
                joint_vel = robot.data.default_joint_vel[env_ids]
                robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Reset actions and sample new commands for all agents
        for agent_id in self.cfg.action_spaces:
            self.actions[agent_id][env_ids] = 0.0
            self._desired_pos_w[agent_id][env_ids, :2] = torch.zeros_like(self._desired_pos_w[agent_id][env_ids, :2]).uniform_(-2.0, 2.0)
            self._desired_pos_w[agent_id][env_ids, :2] += origins[:, :2]
            self._desired_pos_w[agent_id][env_ids, 2] = torch.zeros_like(self._desired_pos_w[agent_id][env_ids, 2]).uniform_(0.5, 1.5)

        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)


        
