# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy

pass
import torch

import omni.isaac.core.utils.torch as torch_utils
from omni.isaac.core.utils.torch.rotations import compute_heading_and_up, compute_rot, quat_conjugate

import omni.isaac.lab.envs.mdp as mdp
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from omni.isaac.lab.envs import DirectMARLEnv, DirectMARLEnvCfg
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.markers import VisualizationMarkers, VisualizationMarkersCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import ContactSensor, ContactSensorCfg, RayCasterCfg, patterns
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.utils.math import quat_from_angle_axis

##
# Pre-defined configs
##
from omni.isaac.lab_assets.anymal import ANYMAL_C_CFG  # isort: skip
from omni.isaac.lab_assets.unitree import H1_CFG  # isort: skip
from omni.isaac.lab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip


def normalize_angle(x):
    return torch.atan2(torch.sin(x), torch.cos(x))


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

    physics_material_1 = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot_1", body_names=".*"),
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

    add_base_mass_1 = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot_1", body_names="pelvis"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )
    
    add_base_mass_2 = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot_2", body_names="pelvis"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )

    add_base_mass_3 = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot_3", body_names="pelvis"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )
    


@configclass
class HeterogeneousMultiAgentFlatEnvCfg(DirectMARLEnvCfg):
    # env
    episode_length_s = 20.0
    decimation = 4
    anymal_action_scale = 0.5
    action_space = 12
    action_spaces = {"robot_0": 12, "robot_1": 19, "robot_2": 19, "robot_3": 19}

    observation_spaces = {"robot_0": 48, "robot_1": 72, "robot_2": 72, "robot_3": 72}
    state_space = 0
    possible_agents = ["robot_0", "robot_1", "robot_2", "robot_3"]
    state_spaces = {f"robot_{i}": 0 for i in range(len(possible_agents))}

    for agets in possible_agents:
        print(agets)

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        disable_contact_processing=True,
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
    robot_0.init_state.rot = (1.0, 0.0, 0.0, 1)
    robot_0.init_state.pos = (-1.0, 0.0, 0.5)

    robot_1: ArticulationCfg = H1_CFG.replace(prim_path="/World/envs/env_.*/Robot_1")
    robot_1.init_state.rot = (1.0, 0.0, 0.0, 1)
    robot_1.init_state.pos = (1.0, 0.0, 1.0)
    
    robot_2: ArticulationCfg = H1_CFG.replace(prim_path="/World/envs/env_.*/Robot_2")
    robot_2.init_state.rot = (1.0, 0.0, 0.0, 1)
    robot_2.init_state.pos = (2.0, 0.0, 1)

    robot_3: ArticulationCfg = H1_CFG.replace(prim_path="/World/envs/env_.*/Robot_3")
    robot_3.init_state.rot = (1.0, 0.0, 0.0, 1)
    robot_3.init_state.pos = (3.0, 0.0, 1)
    # # rec prism
    # cfg_rec_prism= RigidObjectCfg(
    #     prim_path="/World/envs/env_.*/Object",
    #     spawn=sim_utils.CuboidCfg(
    #         size=(2,1,2),
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(),
    #         mass_props=sim_utils.MassPropertiesCfg(mass=.5), # changed from 1.0 to 0.5
    #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0))
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 3, 2), rot=(1.0, 0.0, 0.0, 0.0)), #started the bar lower
    # )
    cfg_rec_prism: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.UsdFileCfg(
            # usd_path=f"assets/GrandPiano_instanceable_meshes.usd",
            # usd_path=f"assets/GrandPiano_instanceable_meshes.usda",
            usd_path="assets/ball_instanceable_meshes.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                # make it easier to move
                linear_damping=0.001,
                angular_damping=0.001,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.0001),
            scale=(1, 1, 1),
            # scale=(.01, .01, .01),
            # define the physics material
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 2, 0.1), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    # reward scales (override from flat config)
    flat_orientation_reward_scale = 0.0

    # reward scales
    lin_vel_reward_scale = 1.0
    yaw_rate_reward_scale = 0.5
    z_vel_reward_scale = -2.0
    ang_vel_reward_scale = -0.05
    joint_torque_reward_scale = -2.5e-5
    joint_accel_reward_scale = -2.5e-7
    action_rate_reward_scale = -0.01
    feet_air_time_reward_scale = 0.5
    undersired_contact_reward_scale = -1.0
    flat_orientation_reward_scale = -5.0
    flat_bar_roll_angle_reward_scale = -1.0
    angular_velocity_scale: float = 0.25
    dof_vel_scale: float = 0.1
    h1_action_scale = 1.0
    termination_height: float = 0.8
    anymal_min_z_pos = 0.3

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


@configclass
class HeterogeneousMultiAgentRoughEnvCfg(HeterogeneousMultiAgentFlatEnvCfg):
    # env
    observation_space = 235

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=9,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
        debug_vis=False,
    )

    # we add a height scanner for perceptive locomotion
    height_scanner_0 = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot_0/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    height_scanner_1 = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot_1/pelvis",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    
    height_scanner_2 = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot_2/pelvis",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    
    height_scanner_3 = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot_3/pelvis",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    # reward scales (override from flat config)
    flat_orientation_reward_scale = 0.0


def define_markers() -> VisualizationMarkers:
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/myMarkers",
        markers={
            "sphere1": sim_utils.SphereCfg(
                radius=0.15,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.0, 0.0)),
            ),
            "sphere2": sim_utils.SphereCfg(
                radius=0.15,
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


class HeterogeneousMultiAgentTeam(DirectMARLEnv):
    cfg: HeterogeneousMultiAgentFlatEnvCfg | HeterogeneousMultiAgentRoughEnvCfg

    def __init__(
        self,
        cfg: HeterogeneousMultiAgentFlatEnvCfg | HeterogeneousMultiAgentRoughEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ):
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
        # X/Y linear velocity and yaw angular velocity commands
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)
        self._joint_dof_idx, _ = self.robots["robot_1"].find_joints(".*")

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_lin_vel_xy_exp",
                "track_ang_vel_z_exp",
                "lin_vel_z_l2",
                "ang_vel_xy_l2",
                "dof_torques_l2",
                "dof_acc_l2",
                "action_rate_l2",
                "feet_air_time",
                "undesired_contacts",
                "flat_orientation_l2",
                "flat_bar_roll_angle",
            ]
        }

        self.base_ids = {}
        self.feet_ids = {}
        self.undesired_body_contact_ids = {}

        base_bodies = ["base", "pelvis", "pelvis", "pelvis"]
        for (robot_id, contact_sensor), base_body in zip(self.contact_sensors.items(), base_bodies):
            _base_id, _ = contact_sensor.find_bodies(base_body)
            _feet_ids, _ = contact_sensor.find_bodies(".*FOOT")
            _undesired_contact_body_ids, _ = contact_sensor.find_bodies(".*THIGH")
            self.base_ids[robot_id] = _base_id
            self.feet_ids[robot_id] = _feet_ids
            self.undesired_body_contact_ids[robot_id] = _undesired_contact_body_ids

        self.targets = torch.tensor([1000, 0, 0], dtype=torch.float32, device=self.sim.device).repeat(
            (self.num_envs, 1)
        )

        def apply_anymal_action(self, robot_id):
            self.robots[robot_id].set_joint_position_target(self.processed_actions[robot_id])

        def apply_h1_action(self, robot_id):
            forces = self.cfg.h1_action_scale * self.robot_data[robot_id]["joint_gears"] * self.actions[robot_id]
            self.robots[robot_id].set_joint_effort_target(forces, joint_ids=self._joint_dof_idx)
        
        self.robot_instance_types = ["anymal"]+ 3*["h1"]
        self.robot_data = {}
        for i, robot_type in enumerate(self.robot_instance_types):
            robot_id = f"robot_{i}"
            cur_robot_data = {"type": robot_type,"targets": self.scene.env_origins}

            if robot_type == "anymal":
                # cur_robot_data["apply_action"] = apply_anymal_action  # Assign function directly
                anymal_data = {"apply_action":apply_anymal_action,
                                "None": None,}
                cur_robot_data.update(anymal_data)

            if robot_type == "h1":
                h1_data = {
                    "apply_action": apply_h1_action,  # Assign function
                    "potentials": torch.zeros(self.num_envs, dtype=torch.float32, device=self.sim.device),
                    "prev_potentials": torch.zeros(self.num_envs, dtype=torch.float32, device=self.sim.device),
                    "targets": torch.tensor([1000, 0, 0], dtype=torch.float32, device=self.sim.device).repeat((self.num_envs, 1)),
                    "joint_gears": torch.tensor(self.cfg.joint_gears, dtype=torch.float32, device=self.sim.device),
                    "heading_vec": torch.tensor([1, 0, 0], dtype=torch.float32, device=self.sim.device).repeat((self.num_envs, 1)),
                    "up_vec": torch.tensor([0, 0, 1], dtype=torch.float32, device=self.sim.device).repeat((self.num_envs, 1)),
                    "start_rotation": torch.tensor([1, 0, 0, 0], device=self.sim.device, dtype=torch.float32),
                    "inv_start_rot": quat_conjugate(torch.tensor([1, 0, 0, 0], device=self.sim.device, dtype=torch.float32)).repeat((self.num_envs, 1)),
                    "basis_vec0": torch.tensor([1, 0, 0], dtype=torch.float32, device=self.sim.device).repeat((self.num_envs, 1)),
                    "basis_vec1": torch.tensor([0, 0, 1], dtype=torch.float32, device=self.sim.device).repeat((self.num_envs, 1)),
                }

                # Explicitly update key-by-key instead of using `update()`
                cur_robot_data.update(h1_data)


            self.robot_data[robot_id] = cur_robot_data 


    def _setup_scene(self):
        self.num_robots = sum(1 for key in self.cfg.__dict__.keys() if "robot_" in key)
        self.robots = {}
        self.contact_sensors = {}
        self.height_scanners = {}
        self.my_visualizer = define_markers()
        self.object = RigidObject(self.cfg.cfg_rec_prism)

        self.scene.rigid_objects["object"] = self.object

        for i in range(self.num_robots):
            robot_id = f"robot_{i}"
            if robot_id in self.cfg.__dict__:
                self.robots[f"robot_{i}"] = Articulation(self.cfg.__dict__["robot_" + str(i)])
                self.scene.articulations[f"robot_{i}"] = self.robots[f"robot_{i}"]

            contact_sensor_id = "contact_sensor_" + str(i)

            if contact_sensor_id in self.cfg.__dict__:
                self.contact_sensors[f"robot_{i}"] = ContactSensor(self.cfg.__dict__["contact_sensor_" + str(i)])
                self.scene.sensors[f"robot_{i}"] = self.contact_sensors[f"robot_{i}"]

            # height_scanner_id = "height_scanner_" + str(i)
            # if height_scanner_id in self.cfg.__dict__:
            #     self.height_scanners[f"robot_{i}"] = RayCaster(self.cfg.__dict__["height_scanner_" + str(i)])
            #     self.scene.sensors[f"robot_{i}"] = self.height_scanners[f"robot_{i}"]

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
        # We need to process the actions for each scene independently
        self.processed_actions = copy.deepcopy(actions)

        for i, robot_type in enumerate(self.robot_instance_types):
            robot_id = f"robot_{i}"
            if robot_type == "anymal":
                robot_action_space = self.action_spaces[robot_id].shape[0]
                self.actions[robot_id] = actions[robot_id][:, :robot_action_space].clone()
                self.processed_actions[robot_id] = (
                    self.cfg.anymal_action_scale * self.actions[robot_id] + self.robots[robot_id].data.default_joint_pos
                )
            if robot_type == "h1":
                self.actions[robot_id] = actions[robot_id].clone()

    def _get_anymal_fallen(self):
        agent_dones = []

        for _, robot in self.robots.items():
            died = robot.data.body_com_pos_w[:, 0, 2].view(-1) < self.cfg.anymal_min_z_pos
            agent_dones.append(died)

        return torch.any(torch.stack(agent_dones), dim=0)

    def _apply_action(self):

        for i in range(self.num_robots):
            robot_id = f"robot_{i}"
            self.robot_data[robot_id]["apply_action"](self, robot_id)



    def _compute_intermediate_values(self, robot_idx: int):
        robot_id = f"robot_{robot_idx}"
        cur_robot_data = self.robot_data[robot_id]
        robot = self.robots[robot_id]

        cur_robot_data["torso_position"], cur_robot_data["torso_rotation"] = robot.data.root_link_pos_w, robot.data.root_link_quat_w
        velocity, ang_velocity = robot.data.root_com_lin_vel_w, robot.data.root_com_ang_vel_w
        cur_robot_data["dof_pos"], cur_robot_data["dof_vel"] = robot.data.joint_pos, robot.data.joint_vel

        inv_start_rot = torch.tensor(list(cur_robot_data["inv_start_rot"]), device=self.device) if isinstance(cur_robot_data["inv_start_rot"], tuple) else cur_robot_data["inv_start_rot"]

        (
            cur_robot_data["up_proj"],
            cur_robot_data["heading_proj"],
            cur_robot_data["up_vec"],
            cur_robot_data["heading_vec"],
            cur_robot_data["vel_loc"],
            cur_robot_data["angvel_loc"],
            cur_robot_data["roll"],
            cur_robot_data["pitch"],
            cur_robot_data["yaw"],
            cur_robot_data["angle_to_target"],
            cur_robot_data["dof_pos_scaled"],
            cur_robot_data["prev_potentials"],
            cur_robot_data["potentials"],
        ) = compute_intermediate_values(
            cur_robot_data["targets"],
            cur_robot_data["torso_position"],
            cur_robot_data["torso_rotation"],
            velocity,
            ang_velocity,
            cur_robot_data["dof_pos"],
            robot.data.soft_joint_pos_limits[0, :, 0],
            robot.data.soft_joint_pos_limits[0, :, 1],
            cur_robot_data["inv_start_rot"],
            cur_robot_data["basis_vec0"],
            cur_robot_data["basis_vec1"],
            cur_robot_data["potentials"],
            cur_robot_data["prev_potentials"].resize_as_(cur_robot_data["potentials"]),
            self.cfg.sim.dt,
        )

    def _get_observations(self) -> dict:
        self.previous_actions = copy.deepcopy(self.actions)
        # height_datas = {}
        # for robot_id, robot in self.robots.items():
        #     height_data = None
        #     # if isinstance(self.cfg, HeterogeneousMultiAgentWalkingRoughEnvCfg):
        #     if robot_id in self.height_scanners:
        #         height_data = (
        #             self.height_scanners[robot_id].data.pos_w[:, 2].unsqueeze(1) - self.height_scanners[robot_id].data.ray_hits_w[..., 2] - 0.5
        #         ).clip(-1.0, 1.0)
        #         height_datas[robot_id] = (height_data)

        obs = {}

        robot_id = "robot_0"
        robot = self.robots[robot_id]
        # anymal_commands = torch.stack([self._commands[:, 1], -self._commands[:, 0], self._commands[:, 2]]).t()
        obs[robot_id] = torch.cat(
            [
                tensor
                for tensor in (
                    robot.data.root_com_lin_vel_b,
                    robot.data.root_com_ang_vel_b,
                    robot.data.projected_gravity_b,
                    # anymal_commands,
                    self._commands,
                    robot.data.joint_pos - robot.data.default_joint_pos,
                    robot.data.joint_vel,
                    None,
                    self.actions[robot_id],
                )
                if tensor is not None
            ],
            dim=-1,
        )

        for i in range(1, 4):
            robot_id = f"robot_{i}"
            robot = self.robots[robot_id]
            cur_robot_data = self.robot_data[robot_id]

            obs[robot_id] = torch.cat(
            (
                cur_robot_data["torso_position"][:, 2].view(-1, 1),
                cur_robot_data["vel_loc"],
                cur_robot_data["angvel_loc"] * self.cfg.angular_velocity_scale,
                normalize_angle(cur_robot_data["yaw"]).unsqueeze(-1),
                normalize_angle(cur_robot_data["roll"]).unsqueeze(-1),
                normalize_angle(cur_robot_data["angle_to_target"]).unsqueeze(-1),
                cur_robot_data["up_proj"].unsqueeze(-1),
                cur_robot_data["heading_proj"].unsqueeze(-1),
                cur_robot_data["dof_pos_scaled"],
                cur_robot_data["dof_vel"] * self.cfg.dof_vel_scale,
                self.actions[robot_id],
                self._commands,
            ),
            dim=-1,
            )


        # obs = torch.cat(obs, dim=0)
        # observations = {"policy": obs}
        return obs

    def get_y_euler_from_quat(self, quaternion):
        w, x, y, z = quaternion[:, 0], quaternion[:, 1], quaternion[:, 2], quaternion[:, 3]
        y_euler_angle = torch.arcsin(2 * (w * y - z * x))
        return y_euler_angle

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

        bar_pos = self.object.data.body_com_pos_w.squeeze(1).clone()
        bar_yaw = self.object.data.root_com_ang_vel_b[:, 2].clone()

        scale1 = torch.ones((self._commands.shape[0], 3), device=self.device)
        scale1[:, 0] = torch.abs(z_commands)

        scale2 = torch.ones((self._commands.shape[0], 3), device=self.device)
        scale2[:, 0] = torch.abs(bar_yaw)

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
                    torch.sign(bar_yaw) * _90,
                ],
                axis=0,
            ),
            torch.tensor([0.0, 1.0, 0.0], device=self.device),
        )

        marker_scales = torch.concat(
            [torch.ones((3 * self._commands.shape[0], 3), device=self.device), scale1, scale2], axis=0
        )

        obj_vel = self.object.data.root_com_lin_vel_b.clone()
        obj_vel[:, 2] = 0

        marker_locations = torch.concat(
            [bar_pos, bar_pos + xy_commands, bar_pos + obj_vel, bar_pos + offset1, bar_pos + offset2], axis=0
        )

        marker_locations[:, 2] += 1.0

        self.my_visualizer.visualize(
            marker_locations, marker_orientations, scales=marker_scales, marker_indices=marker_ids
        )

    def _get_rewards(self) -> dict:
        reward = {}

        bar_commands = torch.stack([-self._commands[:, 1], self._commands[:, 0], self._commands[:, 2]]).t()
        self._draw_markers(bar_commands)
        yaw_rate_error = torch.square(self._commands[:, 2] - self.object.data.root_com_ang_vel_b[:, 2])
        yaw_rate_error_mapped = torch.exp(-yaw_rate_error)

        for robot_id, _ in self.robots.items():
            # linear velocity tracking
            # lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - self.object.data.root_com_lin_vel_b[:, :2]), dim=1) #changing this to the bar
            lin_vel_error = torch.sum(
                torch.square(bar_commands[:, :2] - self.object.data.root_com_lin_vel_b[:, :2]), dim=1
            )  # changing this to the bar
            lin_vel_error_mapped = torch.exp(-lin_vel_error)

            rewards = {
                "track_ang_vel_z_exp": yaw_rate_error_mapped * self.cfg.yaw_rate_reward_scale * self.step_dt,
                "track_lin_vel_xy_exp": lin_vel_error_mapped * self.cfg.lin_vel_reward_scale * self.step_dt,
            }
            curr_reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

            reward[robot_id] = curr_reward

        return reward

    # def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
    #     all_dones = {}
    #     all_died = []
    #     for i in range(self.num_robots):
    #         time_out = self.episode_length_buf >= self.max_episode_length - 1
    #         net_contact_forces = self.contact_sensors[i].data.net_forces_w_history
    #         died = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self.base_ids[i]], dim=-1), dim=1)[0] > 1.0, dim=1)
    #         all_dones.append(time_out)
    #         all_died.append(died)

    #     return torch.any(torch.cat(all_dones), dim=0), torch.any(torch.cat(all_died), dim=0)

    # TODO: Implement a dones function that handles multiple robots
    def _get_dones(self) -> tuple[dict, dict]:
        self._compute_intermediate_values(1)
        self._compute_intermediate_values(2)
        self._compute_intermediate_values(3)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        h1_died = self.robot_data["robot_1"]["torso_position"][:, 2] < self.cfg.termination_height
        h1_died_2 = self.robot_data["robot_2"]["torso_position"][:, 2] < self.cfg.termination_height
        h1_died_3 = self.robot_data["robot_3"]["torso_position"][:, 2] < self.cfg.termination_height
        anymal_fallen = self._get_anymal_fallen()

        # dones = torch.logical_or(h1_died, anymal_fallen)
        # get all dones for all robots if any of the robots are done
        dones = torch.logical_or(h1_died, h1_died_2)
        dones = torch.logical_or(dones, h1_died_3)
        dones = torch.logical_or(dones, anymal_fallen)
        # dones = anymal_fallen

        # return {key:torch.zeros_like(time_out) for key in self.robots.keys()}, {key:torch.zeros_like(dones) for key in self.robots.keys()}
        return {key: time_out for key in self.robots.keys()}, {key: dones for key in self.robots.keys()}

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        object_default_state = self.object.data.default_root_state.clone()[env_ids]
        object_default_state[:, 0:3] += self.scene.env_origins[env_ids]
        self.object.write_root_state_to_sim(object_default_state, env_ids)
        self.object.reset(env_ids)

        # Reset joint position commands
        for agent, action_space in self.cfg.action_spaces.items():
            self.actions[agent][env_ids] = torch.zeros(env_ids.shape[0], action_space, device=self.device)
            self.previous_actions[agent][env_ids] = torch.zeros(env_ids.shape[0], action_space, device=self.device)

        # X/Y linear velocity and yaw angular velocity commands
        self._commands[env_ids] = torch.zeros_like(self._commands[env_ids]).uniform_(-1.0, 1.0)
        self._commands[env_ids, 2] = 0.0

        # Reset indexed robots
        for i in range(1, 4):
            robot_id = f"robot_{i}"
            robot_env_ids = env_ids if env_ids is not None else self.robots[robot_id]._ALL_INDICES
            self.robots[robot_id].reset(robot_env_ids)

            joint_pos = self.robots[robot_id].data.default_joint_pos[robot_env_ids]
            joint_vel = self.robots[robot_id].data.default_joint_vel[robot_env_ids]
            default_root_state = self.robots[robot_id].data.default_root_state[robot_env_ids]
            default_root_state[:, :3] += self.scene.env_origins[robot_env_ids]

            self.robots[robot_id].write_root_link_pose_to_sim(default_root_state[:, :7], robot_env_ids)
            self.robots[robot_id].write_root_com_velocity_to_sim(default_root_state[:, 7:], robot_env_ids)
            self.robots[robot_id].write_joint_state_to_sim(joint_pos, joint_vel, None, robot_env_ids)

            # Ensure that robot_data[robot_id]["targets"] exists
            if robot_id not in self.robot_data or "targets" not in self.robot_data[robot_id]:
                raise ValueError(f"Missing target data for {robot_id}")

            to_target = self.robot_data[robot_id]["targets"][robot_env_ids] - default_root_state[:, :3]
            to_target[:, 2] = 0.0
            self.robot_data[robot_id]["potentials"] = -torch.norm(to_target, p=2, dim=-1) / self.cfg.sim.dt

            # Dynamically call the correct compute function
            compute_func = getattr(self, f"_compute_intermediate_values", None)
            if compute_func is not None:
                compute_func(i)
            else:
                raise AttributeError(f"Missing function _compute_intermediate_values_{i}")

        # Reset idx for anymal (robot_0)
        robot = self.robots["robot_0"]
        robot_env_ids = env_ids if env_ids is not None else robot._ALL_INDICES
        robot.reset(robot_env_ids)

        if env_ids is None or len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        # Reset robot state
        joint_pos = robot.data.default_joint_pos[robot_env_ids]
        joint_vel = robot.data.default_joint_vel[robot_env_ids]
        default_root_state = robot.data.default_root_state[robot_env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[robot_env_ids]

        robot.write_root_pose_to_sim(default_root_state[:, :7], robot_env_ids)
        robot.write_root_velocity_to_sim(default_root_state[:, 7:], robot_env_ids)
        robot.write_joint_state_to_sim(joint_pos, joint_vel, None, robot_env_ids)

        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][robot_env_ids])
            extras[f"Episode_Reward/{key}"] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][robot_env_ids] = 0.0

        self.extras["log"] = dict()
        self.extras["log"].update(extras)



@torch.jit.script
def compute_intermediate_values(
    targets: torch.Tensor,
    torso_position: torch.Tensor,
    torso_rotation: torch.Tensor,
    velocity: torch.Tensor,
    ang_velocity: torch.Tensor,
    dof_pos: torch.Tensor,
    dof_lower_limits: torch.Tensor,
    dof_upper_limits: torch.Tensor,
    inv_start_rot: torch.Tensor,
    basis_vec0: torch.Tensor,
    basis_vec1: torch.Tensor,
    potentials: torch.Tensor,
    prev_potentials: torch.Tensor,
    dt: float,
):
    to_target = targets - torso_position
    to_target[:, 2] = 0.0
    torso_quat, up_proj, heading_proj, up_vec, heading_vec = compute_heading_and_up(
        torso_rotation, inv_start_rot, to_target, basis_vec0, basis_vec1, 2
    )

    vel_loc, angvel_loc, roll, pitch, yaw, angle_to_target = compute_rot(
        torso_quat, velocity, ang_velocity, targets, torso_position
    )

    dof_pos_scaled = torch_utils.maths.unscale(dof_pos, dof_lower_limits, dof_upper_limits)

    to_target = targets - torso_position
    to_target[:, 2] = 0.0
    prev_potentials[:] = potentials
    potentials = -torch.norm(to_target, p=2, dim=-1) / dt

    return (
        up_proj,
        heading_proj,
        up_vec,
        heading_vec,
        vel_loc,
        angvel_loc,
        roll,
        pitch,
        yaw,
        angle_to_target,
        dof_pos_scaled,
        prev_potentials,
        potentials,
    )
