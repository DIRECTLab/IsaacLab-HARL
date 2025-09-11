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
class MinitankStage3v2EnvCfg(DirectMARLEnvCfg):
    # env
    episode_length_s = 20.0
    decimation = 4
    anymal_action_scale = 0.5
    action_space = 2
    action_spaces = {f"minitank_{i}": 2 for i in range(2)} | \
                    {f"drone_{i}": 4 for i in range(2)}

    # Padded observation: 10 (original) + 3 (teammate_pos) + 3 (other_pos) + 1 (dist_to_center) + 1 (arena_radius) = 18
    observation_space = 18
    observation_spaces = {f"minitank_{i}": 18 for i in range(2)} | \
                        {f"drone_{i}": 20 for i in range(2)}
    state_space = 0
    state_spaces = {f"minitank_{i}": 0 for i in range(2)} | \
                    {f"drone_{i}": 0 for i in range(2)}
    # possible_agents = [f"minitank_{i}" for i in range(2)] + \
    #                 [f"drone_{i}" for i in range(2)]
    possible_agents = list(action_spaces.keys())
    # Teams for two agents
    # teams = {"team_0": ["minitank_0"], "team_1": ["minitank_1"]}
    teams = {"team_0": 
                [
                "minitank_0",
                "minitank_1"
            ], 
            "team_1": [
                "drone_0", 
                "drone_1",
            ],
        }
    
    wall_0 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object0",
        spawn=sim_utils.CuboidCfg(
            size=(20, 0.5, 2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=10.0),  
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 5.0, 1), rot=(1.0, 0.0, 0.0, 0.0) # Position originally was (0.0, 0, 0.61)
        ),
    )

    wall_1 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object1",
        spawn=sim_utils.CuboidCfg(
            size=(20, 0.5, 2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=10.0),  
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, -5.0, 1), rot=(1.0, 0.0, 0.0, 0.0) # Position originally was (0.0, 0, 0.61)
        ),
    )

    wall_2 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object2",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 10, 2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=10.0),  
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(10.0, 0.0, 1), rot=(1.0, 0.0, 0.0, 0.0) # Position originally was (0.0, 0, 0.61)
        ),
    )

    wall_3 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object3",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 10, 2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=10.0),  
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-10.0, 0.0, 1), rot=(1.0, 0.0, 0.0, 0.0) # Position originally was (0.0, 0, 0.61)
        ),
    )

        ### MINITANK CONFIGURATION ###
    minitank_0: ArticulationCfg = MINITANK_CFG.replace(prim_path="/World/envs/env_.*/Minitank_0")
    minitank_0.init_state.pos = (0.0, 0.5, 0.2)

    minitank_1: ArticulationCfg = MINITANK_CFG.replace(prim_path="/World/envs/env_.*/Minitank_1")
    minitank_1.init_state.pos = (0.0, -0.5, 0.2)

    # Drone configs
    drone_0: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Drone_0")
    drone_0.init_state.pos = (2.0, 0.5, 3.5)

    drone_1: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Drone_1")
    drone_1.init_state.pos = (2.0, -0.5, 3.5)

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
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=20.0, replicate_physics=True)

    # events
    # events: EventCfg = EventCfg()



    ### MINITANK CONFIGURATION ###
    action_scale = .5
    max_vel = 2

    ### DRONE CONFIGURATION ###
    thrust_to_weight = 1.9
    moment_scale = 0.01

    # reward scales
    lin_vel_reward_scale = -0.05
    ang_vel_reward_scale = -0.01
    distance_to_goal_reward_scale = 15.0
    debug_vis = True

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

class MinitankStage3v2Env(DirectMARLEnv):
    cfg: MinitankStage3v2EnvCfg
    def __init__(
        self,
        cfg: MinitankStage3v2EnvCfg,
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
        

        self._thrust = {agent: torch.zeros(self.num_envs, 1, 3, device=self.device) for agent in self.cfg.action_spaces}
        self._moment = {agent: torch.zeros(self.num_envs, 1, 3, device=self.device) for agent in self.cfg.action_spaces}
        # self._desired_pos_w = {agent: torch.zeros(self.num_envs, 3, device=self.device) for agent in self.cfg.action_spaces}

        self._body_id = {}
        self._robot_mass = {}
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = {}
        for agent in self.drones.keys():
            self._body_id[agent] = self.robots[agent].find_bodies("body")[0]
            self._robot_mass[agent] = self.robots[agent].root_physx_view.get_masses()[0].sum()
            self._robot_weight[agent] = (self._robot_mass[agent] * self._gravity_magnitude).item()


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
        self.wall_0 = RigidObject(self.cfg.wall_0)
        self.wall_1 = RigidObject(self.cfg.wall_1)
        self.wall_2 = RigidObject(self.cfg.wall_2)
        self.wall_3 = RigidObject(self.cfg.wall_3)
        self.robots = {}
        self.minitanks = {}
        self.drones = {}

        for tank_id in ["minitank_0", "minitank_1"]:
            self.robots[tank_id] = Articulation(self.cfg.__dict__[tank_id])
            self.scene.articulations[tank_id] = self.robots[tank_id]
            self.minitanks[tank_id] = self.robots[tank_id]
        # Add drones
        for drone_id in ["drone_0", "drone_1"]:
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
        for agent in self.minitanks.keys():
            self.processed_actions[agent] = torch.clip(
                (self.cfg.action_scale * actions[agent]) + self.robots[agent].data.joint_vel,
                -self.cfg.max_vel, self.cfg.max_vel
            )
        ### PREPHYSICS FOR MINITANK ###

        for agent in self.drones.keys():
            self.actions[agent] = actions[agent].clone().clamp(-1.0, 1.0)
            self._thrust[agent][:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight[agent] * (self.actions[agent][:, 0] + 1.0) / 2.0
            self._moment[agent][:, 0, :] = self.cfg.moment_scale * self.actions[agent][:, 1:]

    def _apply_action(self):
        for agent in self.minitanks.keys():
            self.robots[agent].set_joint_velocity_target(self.processed_actions[agent])

        for agent in self.drones.keys():
            self.robots[agent].set_external_force_and_torque(self._thrust[agent], self._moment[agent], body_ids=self._body_id[agent])



    def _get_observations(self) -> dict:
        # Arena radius and center
        arena_radius = torch.full((self.num_envs, 1), 2.0, device=self.device)
        env_center = self._terrain.env_origins if hasattr(self._terrain, "env_origins") else torch.zeros((self.num_envs, 3), device=self.device)

        obs_dict = {team: {} for team in self.cfg.teams}
        # for i, agent in enumerate(self.cfg.action_spaces):
        for team in self.cfg.teams:
            for agent in self.cfg.teams[team]:
                if "minitank" in agent:
                    # Desired position in body frame
                    desired_pos_b, _ = subtract_frame_transforms(
                        self.robots[agent].data.root_state_w[:, :3], self.robots[agent].data.root_state_w[:, 3:7], self._desired_pos_w[agent]
                    )
                    # Teammate and other agent positions (in body frame)
                    teammate_id = "minitank_1" if agent == "minitank_0" else "minitank_0"
                    teammate_pos, _ = subtract_frame_transforms(
                        self.robots[agent].data.root_state_w[:, :3], self.robots[agent].data.root_state_w[:, 3:7], self.robots[teammate_id].data.root_pos_w
                    )
                    # For two agents, other_pos is same as teammate_pos
                    other_pos = teammate_pos
                    dist_to_center = torch.norm(self.robots[agent].data.root_pos_w - env_center, dim=-1, keepdim=True)

                    # Use vector angle reward for each agent
                    arm_orientation_reward, arm_orientation, desired_orientation = self._get_vector_angle_reward(agent)
                    processed_actions = self.processed_actions[agent]
                    obs_parts = [
                        processed_actions,
                        desired_orientation,
                        arm_orientation,
                        desired_pos_b,
                        teammate_pos,
                        other_pos,
                        dist_to_center,
                        arena_radius,
                    ]
                    obs = torch.cat(obs_parts, dim=-1)
                    obs_size = obs.shape[1]
                    target_size = self.cfg.observation_spaces[agent]
                    if obs_size < target_size:
                        pad = torch.zeros((self.num_envs, target_size - obs_size), device=self.device)
                        obs = torch.cat([obs, pad], dim=-1)
                    elif obs_size > target_size:
                        obs = obs[:, :target_size]
                    obs = torch.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)
                    obs_dict[team][agent] = obs
                elif "drone" in agent:
                    # Desired position in body frame
                    desired_pos, _ = subtract_frame_transforms(
                        self.robots[agent].data.root_state_w[:, :3], self.robots[agent].data.root_state_w[:, 3:7], self._desired_pos_w[agent]
                    )
                    # Teammate and other agent positions (in body frame)
                    teammate_id = "drone_1" if agent == "drone_0" else "drone_0"
                    teammate_pos, _ = subtract_frame_transforms(
                        self.robots[agent].data.root_state_w[:, :3], self.robots[agent].data.root_state_w[:, 3:7], self.robots[teammate_id].data.root_pos_w
                    )
                    obs_parts = torch.cat([
                            self.robots[agent].data.root_lin_vel_b,
                            self.robots[agent].data.root_ang_vel_b,
                            self.robots[agent].data.projected_gravity_b,
                            desired_pos,
                            teammate_pos,
                            teammate_pos,
                            # other_pos,
                            dist_to_center,
                            arena_radius,
                        ],
                        dim=-1 )
                    obs_parts = torch.nan_to_num(obs_parts, nan=0.0, posinf=1e6, neginf=-1e6)
                    obs_dict[team][agent] = obs_parts
                else:
                    raise ValueError(f"Unknown agent type for {agent}")


        self.previous_actions = copy.deepcopy(self.actions)
        return obs_dict

    def get_y_euler_from_quat(self, quaternion):
        w, x, y, z = quaternion[:, 0], quaternion[:, 1], quaternion[:, 2], quaternion[:, 3]
        y_euler_angle = torch.arcsin(2 * (w * y - z * x))
        return y_euler_angle

    def _get_rewards(self) -> dict:
        if not self.headless:
            self._draw_markers()

            # self.draw_action_thrust("drone_1")
            # self.draw_action_thrust("drone_0")

        # Reward logic: teams are rewarded for simply existing together (no push-out or fallen logic)
        # Each team gets a constant reward per timestep as long as their agents exist
        reward_value = 1.0  # You can adjust this value as needed
        team_rewards = {team: torch.full((self.num_envs,), reward_value, device=self.device) for team in self.cfg.teams.keys()}

        # Optionally log the reward
        for team in self.cfg.teams.keys():
            self._episode_sums[f"{team}_existence_reward"] = team_rewards[team]


        # ### MINITANK REWARDS ###
        # all_rewards = {}
        # team_rewards = {team: torch.zeros(self.num_envs, device=self.device) for team in self.cfg.teams}
        # for team in self.cfg.teams:
        #     for agent in self.cfg.teams[team]:
        #         if "minitank" in agent:
        #             arm_orientation_reward, _, _ = self._get_vector_angle_reward(agent)
        #             minitank_rewards = arm_orientation_reward * self.step_dt
        #         ### MINITANK REWARDS ###
            
        #             self._episode_sums["tank_angle_reward"] = minitank_rewards
        #             rewards = {
        #                 "tank_angle_reward": minitank_rewards
        #             }
        #             reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        #             for key, value in rewards.items():
        #                 self._episode_sums[key] += value
        #             all_rewards[agent] = reward
        #         elif "drone" in agent:
        #             drone_rewards = torch.zeros(self.num_envs, device=self.device)
        #             all_rewards[agent] = drone_rewards
        #         else:
        #             raise ValueError(f"Unknown agent type for {agent}")
        #         team_rewards[team] += reward
        return team_rewards


    def _get_dones(self) -> tuple[dict, dict]:
        # time_out_tensor = (self.episode_length_buf >= self.max_episode_length - 1).to(self.device)
        # dones = {}
        # time_out = {}
        # for team in self.cfg.teams:
        #     for agent in self.cfg.teams[team]:
        #         if "minitank" in agent:           
        #             died_tensor = self.robots[agent].data.root_pos_w[:, 2] < 0.1
        #             dones[team] = died_tensor
        #             time_out[team] = time_out_tensor
        #         if "drone" in agent:
        #             died_tensor = self.robots[agent].data.root_pos_w[:, 2] < 0.0
        #             dones[team] = died_tensor
        # return dones, time_out
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        timeouts = {team: time_out for team in self.cfg.teams.keys()}
        return timeouts, timeouts

    def _reset_idx(self, env_ids: torch.Tensor | None):
        # Use the first robot to get ALL_INDICES, but reset all robots
        first_agent = list(self.cfg.action_spaces.keys())[0]
        if env_ids is None or (hasattr(env_ids, "__len__") and len(env_ids) == self.num_envs):
            env_ids = self.robots[first_agent]._ALL_INDICES
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        super()._reset_idx(env_ids)
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()

        env_size = env_ids.shape[0]
        for agent, action_space in self.cfg.action_spaces.items():
            self.actions[agent][env_ids] = torch.zeros(env_size, action_space, device=self.device)
            self.previous_actions[agent][env_ids] = torch.zeros(env_size, action_space, device=self.device)

        for agent in self.cfg.action_spaces:
            robot = self.robots[agent]
            robot.reset(env_ids)
            if env_ids is not None and hasattr(env_ids, "shape") and env_ids.shape[0] == self.num_envs:
                self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

            joint_pos = robot.data.default_joint_pos[env_ids]
            joint_vel = robot.data.default_joint_vel[env_ids]
            default_root_state = robot.data.default_root_state[env_ids]
            default_root_state[:, :3] += self._terrain.env_origins[env_ids]
            robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
            robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
            robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Reset actions and sample new commands for all robots
        for agent in self.cfg.action_spaces:
            self.actions[agent][env_ids] = 0.0
            self._desired_pos_w[agent][env_ids, :2] = torch.zeros_like(self._desired_pos_w[agent][env_ids, :2]).uniform_(-2.0, 2.0)
            self._desired_pos_w[agent][env_ids, :2] += self._terrain.env_origins[env_ids, :2]
            self._desired_pos_w[agent][env_ids, 2] = torch.zeros_like(self._desired_pos_w[agent][env_ids, 2]).uniform_(0.5, 1.5)


        
