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
from isaaclab.envs.common import ViewerCfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.anymal import ANYMAL_C_CFG  # isort: skip
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip

# import omni.usd
# from pxr import Usd, UsdPhysics, PhysxSchema

# def _iter_rigid_links(root_path: str):
#     """Yield (prim_path, prim) for rigid bodies under root_path."""
#     stage = omni.usd.get_context().get_stage()
#     root = stage.GetPrimAtPath(root_path)
#     if not root.IsValid():
#         return
#     for prim in Usd.PrimRange(root):
#         if PhysxSchema.PhysxRigidBodyAPI(prim):
#             yield prim.GetPath().pathString, prim

# def get_link_masses(robot_root_path: str):
#     """Return dict of link masses and total mass for an articulation."""
#     masses = {}
#     total = 0.0
#     for pth, prim in _iter_rigid_links(robot_root_path):
#         m_api = UsdPhysics.MassAPI.Apply(prim)
#         m = m_api.GetMassAttr().Get()
#         # If unset (None), the asset might be density-based; the randomizer should author Mass.
#         masses[pth] = float(m) if m is not None else None
#         if m is not None:
#             total += float(m)
#     return masses, total

# def log_robot_masses():
#     # Adjust paths if your envs are batched: e.g. f"/World/envs/env_{i}/Robot_0"
#     r0_root = "/World/envs/env_0/Robot_0"
#     r1_root = "/World/envs/env_0/Robot_1"
#     m0, tot0 = get_link_masses(r0_root)
#     m1, tot1 = get_link_masses(r1_root)
#     print("[Mass check] Robot_0 total:", tot0)
#     for k,v in m0.items(): print("  ", v, k)
#     print("[Mass check] Robot_1 total:", tot1)
#     for k,v in m1.items(): print("  ", v, k)

def chase_commands_holonomic(rel_pos_b, v_max=1.5, w_max=2.5, k_v=1.0, k_w=1.5, eps=1e-6):
    # rel_pos_b: [N, 3] = opponent position in robot body frame
    planar = rel_pos_b[:, :2]                               # [N, 2] (dx, dy)
    dist = torch.linalg.norm(planar, dim=-1, keepdim=True)  # [N, 1]
    dir_xy = planar / (dist + eps)                          # unit direction

    # speed that grows with distance and saturates
    speed = v_max * torch.tanh(k_v * dist)                  # [N, 1]

    # linear vel toward opponent
    x_cmd = (speed * dir_xy[:, [0]])        # [N]
    y_cmd = (speed * dir_xy[:, [1]])          # [N]

    # yaw rate to bias facing the opponent
    heading_err = torch.atan2(planar[:, 1], planar[:, 0]).unsqueeze(-1)   # [N]
    z_cmd = w_max * torch.tanh(k_w * heading_err)           # [N]

    return torch.cat([x_cmd, y_cmd, z_cmd], dim=-1)

def get_commands(robot_0, robot_1):
    robot_0_desired_pos, _ = subtract_frame_transforms(
        robot_0.data.root_state_w[:, :3], robot_0.data.root_state_w[:, 3:7], \
            robot_1.data.root_pos_w
    )

    robot_1_desired_pos, _ = subtract_frame_transforms(
        robot_1.data.root_state_w[:, :3], robot_1.data.root_state_w[:, 3:7], \
            robot_0.data.root_pos_w
    )

    robot_0_commands = chase_commands_holonomic(robot_0_desired_pos)
    robot_1_commands = chase_commands_holonomic(robot_1_desired_pos)

    return robot_0_commands, robot_1_commands

def get_distance_to_center(arena_center: torch.tensor, robot: Articulation):
    robot_pos_w = robot.data.root_state_w[:, :3]                # (num_envs, 3)
    # XY deltas to arena center
    robot_delta_xy = robot_pos_w[:, :2] - arena_center          # (num_envs, 2)
    # Distances (keepdim=True so we get (num_envs, 1) for concat)
    return torch.norm(robot_delta_xy, dim=1, keepdim=True)

@configclass
class EventCfg:
    """Configuration for randomization."""

    physics_material_0 = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot_0", body_names=".*"),
            "static_friction_range": (0.75, 0.85),
            "dynamic_friction_range": (0.55, 0.65),
            "restitution_range": (0.0, 0.2),
            "num_buckets": 64,
        },
    )

    mass_scale_0 = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot_0", body_names=".*"),
            # scale each link's mass by a factor in this range
            "mass_distribution_params": (0.90, 1.30),  # widen if you want more asymmetry
            "operation": "scale",                      # use multiplicative scaling
            # "scale_inertia": True,  # if your Isaac Lab version supports this flag
        },
    )

    # add_base_mass_0 = EventTerm(
    #     func=mdp.randomize_rigid_body_mass,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot_0", body_names="base"),
    #         "mass_distribution_params": (-5.0, 5.0),
    #         "operation": "add",
    #     },
    # )

    physics_material_1 = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot_1", body_names=".*"),
            "static_friction_range": (0.75, 0.85),
            "dynamic_friction_range": (0.55, 0.65),
            "restitution_range": (0.0, 0.2),
            "num_buckets": 64,
        },
    )

    mass_scale_1 = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot_1", body_names=".*"),
            # scale each link's mass by a factor in this range
            "mass_distribution_params": (0.90, 1.30),  # widen if you want more asymmetry
            "operation": "scale",                      # use multiplicative scaling
            # "scale_inertia": True,  # if your Isaac Lab version supports this flag
        },
    )

    # add_base_mass_1 = EventTerm(
    #     func=mdp.randomize_rigid_body_mass,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot_1", body_names="base"),
    #         "mass_distribution_params": (-5.0, 5.0),
    #         "operation": "add",
    #     },
    # )


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

def get_quaternion_tuple_from_xyz(x, y, z):
    quat_tensor = quat_from_euler_xyz(torch.tensor([x]), torch.tensor([y]), torch.tensor([z])).flatten()
    return (quat_tensor[0].item(), quat_tensor[1].item(), quat_tensor[2].item(), quat_tensor[3].item())

@configclass
class AnymalCAdversarialSumoStage2EnvCfg(DirectMARLEnvCfg):
    # env
    episode_length_s = 20.0
    decimation = 4
    action_scale = 0.5
    action_space = 12
    action_spaces = {f"robot_{i}": 12 for i in range(2)}
    observation_space = 52
    observation_spaces = {f"robot_{i}": 52 for i in range(2)}
    state_space = 0
    state_spaces = {f"robot_{i}": 0 for i in range(2)}
    possible_agents = ["robot_0", "robot_1"]
    arena_radius = 2.25
    ring_radius_min = 1.5
    ring_radius_max = 3

    teams = {
        "team_0": ["robot_0"],
        "team_1": ["robot_1"]
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
            static_friction=0.9,
            dynamic_friction=0.75,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=10.0, replicate_physics=True)

    # events
    events: EventCfg = EventCfg()

    # cfg_cylinder = RigidObjectCfg(
    #     prim_path="/World/envs/env_.*/Arena",
    #     spawn=sim_utils.CylinderCfg(
    #         radius=arena_radius,
    #         height=1,
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #             kinematic_enabled=True,
    #             disable_gravity=True
    #         ),
    #         mass_props=sim_utils.MassPropertiesCfg(mass=100),
    #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0)),
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(
    #         pos=(0.0, 0.0, 0.5), rot=(1.0, 0.0, 0.0, 0.0)
    #     ),
    # )

    # robot
    robot_0: ArticulationCfg = ANYMAL_C_CFG.replace(prim_path="/World/envs/env_.*/Robot_0")
    contact_sensor_0: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot_0/base*", history_length=3, update_period=0.005, track_air_time=True,
        filter_prim_paths_expr=["/World/envs/env_.*/Arena"]
    )
    
    robot_0.init_state.rot = get_quaternion_tuple_from_xyz(0,0,-torch.pi/2)
    robot_0.init_state.pos = (0.0, 1.0, 0.3)

    robot_1: ArticulationCfg = ANYMAL_C_CFG.replace(prim_path="/World/envs/env_.*/Robot_1")
    contact_sensor_1: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot_1/base*", history_length=3, update_period=0.005, track_air_time=True,
        filter_prim_paths_expr=["/World/envs/env_.*/Arena"]
    )
    robot_1.init_state.rot = get_quaternion_tuple_from_xyz(0,0,torch.pi/2)
    robot_1.init_state.pos = (0.0, -1.0, 0.3)


    # viewer = ViewerCfg(eye=(10.0, 10.0, 10.0), origin_type="asset_root", asset_name="robot_0", env_index=0)

    # reward scales (override from flat config)
    flat_orientation_reward_scale = 0.0

    # reward scales
    loser_scale = -10.0
    winner_scale = 10.0
    timeout_scale = -10.0
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



class AnymalCAdversarialSumoStage2Env(DirectMARLEnv):
    cfg: AnymalCAdversarialSumoStage2EnvCfg

    def __init__(
        self, cfg: AnymalCAdversarialSumoStage2EnvCfg, render_mode: str | None = None, debug=False, **kwargs
    ):
        self.debug = debug
        super().__init__(cfg, render_mode, **kwargs)
        # Joint position command (deviation from default joint positions)
        self.alpha = 0.1
        self.actions = {
            agent: torch.zeros(self.num_envs, action_space, device=self.device)
            for agent, action_space in self.cfg.action_spaces.items()
        }
        self.previous_actions = {
            agent: torch.zeros(self.num_envs, action_space, device=self.device)
            for agent, action_space in self.cfg.action_spaces.items()
        }
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "loser",
                "winner",
                "timeout",
            ]
        }

        self.base_ids = {}
        self.feet_ids = {}
        self.undesired_body_contact_ids = {}

        for robot_id, contact_sensor in self.contact_sensors.items():
            _base_id, _ = contact_sensor.find_bodies("base")
            # _feet_ids, _ = contact_sensor.find_bodies(".*FOOT")
            # _undesired_contact_body_ids, _ = contact_sensor.find_bodies(["base", ".*THIGH", ".*HIP", ".*SHANK"])
            self.base_ids[robot_id] = _base_id
            # self.feet_ids[robot_id] = _feet_ids
            self.undesired_body_contact_ids[robot_id] = _base_id

        self.model = torch.load("/home/jacobmorrey/Downloads/actor_agent_robot_0_full.pt", map_location=self.device, weights_only=False)
        self.model.eval()

        self.curr_step = 0
        self.max_steps = 1_000_000_000

        self.ring_radius = torch.full((self.num_envs,), (self.cfg.ring_radius_min + self.cfg.ring_radius_max) * 0.5,
                                      dtype=torch.float32, device=self.device)

        self._ring_segments = 64
        markers = {f"ring_{i}":sim_utils.SphereCfg(
                    radius=.05,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.0, 0.0)),
                ) for i in range(self._ring_segments)}

        ring_marker_cfg = VisualizationMarkersCfg(
            prim_path="/World/RingMarkers",
            markers=markers
        )
        self.ring_markers = VisualizationMarkers(ring_marker_cfg)

    @torch.no_grad()
    def _draw_ring_markers(self):
        """
        Draws a ring per environment by placing N tiny spheres on a circle
        centered at each env origin with radius given by self.ring_radius.
        Uses marker indices 0..N-1 that correspond to your ring_i markers.
        """
        device = self.device
        E = self.num_envs
        N = self._ring_segments
        z0 = float(getattr(self.cfg, "ring_z", 0.0))

        # Angles for the circle (excluded endpoint to avoid duplicate point)
        theta = torch.linspace(0, 2 * torch.pi, steps=N + 1, device=device)[:-1]  # (N,)
        cs = torch.cos(theta)  # (N,)
        sn = torch.sin(theta)  # (N,)

        # Env centers and radii
        origins_xy = self.scene.env_origins[:, :2].to(device)          # (E, 2)
        radii = self.ring_radius.view(E, 1)                             # (E, 1)

        # Build batched positions: stack per marker index (ring_k) across all envs.
        # For marker k, we place E positions at angle theta[k].
        # Result: positions shape = (N*E, 3), marker_indices length = N*E
        pos_chunks = []
        idx_chunks = []

        # Precompute z column for all envs
        z_col = torch.full((E, 1), z0, device=device)

        for k in range(N):
            dir_k = torch.tensor([cs[k].item(), sn[k].item()], device=device)  # (2,)
            xy_k = origins_xy + radii * dir_k                                  # (E, 2)
            pos_k = torch.cat([xy_k, z_col], dim=1)                            # (E, 3)
            pos_chunks.append(pos_k)
            idx_chunks.append(k * torch.ones(E, dtype=torch.long, device=device))

        marker_positions = torch.cat(pos_chunks, dim=0)   # (N*E, 3)
        marker_indices  = torch.cat(idx_chunks, dim=0)    # (N*E,)

        marker_scales = torch.ones((marker_positions.shape[0], 3), device=device)

        # Orientations: identity quaternions since these are spheres.
        marker_orientations = torch.zeros((marker_positions.shape[0], 4), device=device)
        marker_orientations[:, 0] = 1.0  # w=1, x=y=z=0

        # Visualize. Your API already accepts marker_indices like your velocity viz.
        self.ring_markers.visualize(
            marker_positions,
            marker_orientations,
            scales=marker_scales,
            marker_indices=marker_indices,
        )

    def _setup_scene(self):
        self.num_robots = sum(1 for key in self.cfg.__dict__.keys() if "robot_" in key)
        self.robots = {}
        self.contact_sensors = {}
        self.height_scanners = {}
        if self.debug:
            self.my_visualizer = define_markers()

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
            with torch.no_grad():
                model_actions, _, _ = self.model(self.recent_obs[robot_id], torch.zeros_like(self.recent_obs[robot_id]), torch.ones_like(self.recent_obs[robot_id]))
            combined_actions = model_actions + (min(max(self.curr_step/(self.max_steps/4), self.alpha), 1.0) * self.actions[robot_id])
            self.processed_actions[robot_id] = (
                self.cfg.action_scale * combined_actions + robot.data.default_joint_pos
            )

    def _apply_action(self):
        for robot_id, robot in self.robots.items():
            robot.set_joint_position_target(self.processed_actions[robot_id])

    def _get_observations(self) -> dict:
        self.curr_step += 1
        self.previous_actions = copy.deepcopy(self.actions)

        robot_0_desired_pos, _ = subtract_frame_transforms(
            self.robots["robot_0"].data.root_state_w[:, :3], self.robots["robot_0"].data.root_state_w[:, 3:7], \
                self.robots["robot_1"].data.root_pos_w
        )

        robot_1_desired_pos, _ = subtract_frame_transforms(
            self.robots["robot_1"].data.root_state_w[:, :3], self.robots["robot_1"].data.root_state_w[:, 3:7], \
                self.robots["robot_0"].data.root_pos_w
        )
        time_remaining = (self.max_episode_length - self.episode_length_buf).unsqueeze(-1)

        robot_0_commands = chase_commands_holonomic(robot_0_desired_pos)
        robot_1_commands = chase_commands_holonomic(robot_1_desired_pos)

        robot_0_dist_to_center = get_distance_to_center(self.scene.env_origins[:, :2].to(self.device), self.robots["robot_0"])
        robot_1_dist_to_center = get_distance_to_center(self.scene.env_origins[:, :2].to(self.device), self.robots["robot_1"])

        arena_radius = self.ring_radius.view(-1, 1)

        robot_0_obs = torch.cat(
                [
                    tensor
                    for tensor in (
                        self.robots["robot_0"].data.root_lin_vel_b,
                        self.robots["robot_0"].data.root_ang_vel_b,
                        self.robots["robot_0"].data.projected_gravity_b,
                        self.robots["robot_0"].data.joint_pos - self.robots["robot_0"].data.default_joint_pos,
                        self.robots["robot_0"].data.joint_vel,
                        self.actions["robot_0"],
                        robot_0_desired_pos,
                        time_remaining,
                        robot_0_dist_to_center,
                        robot_1_dist_to_center,
                        arena_radius
                    )
                    if tensor is not None
                ],
                dim=-1,
            )

        robot_1_obs = torch.cat(
            [
                tensor
                for tensor in (
                    self.robots["robot_1"].data.root_lin_vel_b,
                    self.robots["robot_1"].data.root_ang_vel_b,
                    self.robots["robot_1"].data.projected_gravity_b,
                    self.robots["robot_1"].data.joint_pos - self.robots["robot_1"].data.default_joint_pos,
                    self.robots["robot_1"].data.joint_vel,
                    self.actions["robot_1"],
                    robot_1_desired_pos,
                    time_remaining,
                    robot_1_dist_to_center,
                    robot_0_dist_to_center,
                    arena_radius
                )
                if tensor is not None
            ],
            dim=-1,
        )

        robot_0_obs_old = torch.cat(
                [
                    tensor
                    for tensor in (
                        self.robots["robot_0"].data.root_lin_vel_b,
                        self.robots["robot_0"].data.root_ang_vel_b,
                        self.robots["robot_0"].data.projected_gravity_b,
                        robot_0_commands,
                        self.robots["robot_0"].data.joint_pos - self.robots["robot_0"].data.default_joint_pos,
                        self.robots["robot_0"].data.joint_vel,
                        self.actions["robot_0"],
                    )
                    if tensor is not None
                ],
                dim=-1,
            )

        robot_1_obs_old = torch.cat(
            [
                tensor
                for tensor in (
                    self.robots["robot_1"].data.root_lin_vel_b,
                    self.robots["robot_1"].data.root_ang_vel_b,
                    self.robots["robot_1"].data.projected_gravity_b,
                    robot_1_commands,
                    self.robots["robot_1"].data.joint_pos - self.robots["robot_1"].data.default_joint_pos,
                    self.robots["robot_1"].data.joint_vel,
                    self.actions["robot_1"],
                )
                if tensor is not None
            ],
            dim=-1,
        )

        self.recent_obs = {"robot_0": robot_0_obs_old, "robot_1": robot_1_obs_old}
        return {"team_0": {"robot_0": robot_0_obs}, "team_1": {"robot_1": robot_1_obs}}

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
            dim=0,
        )

        robot_pos = self.robots["robot_0"].data.root_pos_w.clone()
        robot_yaw = self.robots["robot_0"].data.root_ang_vel_b[:, 2].clone()

        scale1 = torch.ones((self._commands.shape[0], 3), device=self.device)
        scale1[:, 0] = torch.abs(z_commands)

        scale2 = torch.ones((self._commands.shape[0], 3), device=self.device)
        scale2[:, 0] = torch.abs(robot_yaw)

        offset1 = torch.zeros((self._commands.shape[0], 3), device=self.device)
        offset1[:, 1] = 0

        offset2 = torch.zeros((self._commands.shape[0], 3), device=self.device)
        offset2[:, 1] = 0

        marker_scales = torch.concat(
            [torch.ones((3 * self._commands.shape[0], 3), device=self.device), scale1, scale2], dim=0
        )

        marker_locations = torch.concat(
            [
                robot_pos,
                robot_pos + xy_commands,
                robot_pos + self.robots["robot_0"].data.root_lin_vel_b,
                robot_pos + offset1,
                robot_pos + offset2,
            ],
            dim=0,
        )

        _90 = (-3.14 / 2) * torch.ones(self._commands.shape[0]).to(self.device)

        marker_orientations = quat_from_angle_axis(
            torch.concat(
                [
                    torch.zeros(3 * self._commands.shape[0]).to(self.device),
                    torch.sign(z_commands) * _90,
                    torch.sign(robot_yaw) * _90,
                ],
                dim=0,
            ),
            torch.tensor([0.0, 1.0, 0.0], device=self.device),
        )

        self.my_visualizer.visualize(
            marker_locations, marker_orientations, scales=marker_scales, marker_indices=marker_ids
        )

    def _get_rewards(self) -> dict:
        if self.debug:
            self._draw_markers(self._commands)
        all_rewards = {}

        time_out = self.episode_length_buf >= self.max_episode_length - 1

        dones = {}
        for team, robots in self.cfg.teams.items():
            team_lost = []
            for robot_id in robots:
                fallen = self.robots[robot_id].data.root_pos_w[:, 2] < 0.17
                dist_to_center = get_distance_to_center(self.scene.env_origins[:, :2].to(self.device), self.robots[robot_id])
                left_arena = dist_to_center.squeeze(-1) > self.ring_radius
                robot_lost = torch.logical_or(fallen, left_arena)
                team_lost.append(robot_lost.unsqueeze(-1))
            team_lost = torch.all(torch.cat(team_lost, dim=-1), dim=-1)
            dones[team] = team_lost
        for robot_id in self.robots.keys():
            for team, robots in self.cfg.teams.items():
                if robot_id in robots:
                    loser = dones[team]
                else:
                    winner = dones[team]

            rewards = {
                "timeout": time_out * self.cfg.timeout_scale,
                "loser": loser * self.cfg.loser_scale,
                "winner": winner * self.cfg.winner_scale,
            }
            reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
            all_rewards[robot_id] = reward
            # Logging
            for key, value in rewards.items():
                self._episode_sums[key] += value

        return {"team_0" : all_rewards["robot_0"], "team_1" : all_rewards["robot_1"]}

    def _get_dones(self) -> tuple[dict, dict]:

        anymal_left = []
        anymal_fell = []
        for robot_id, contact_sensor in self.contact_sensors.items():
            dist_to_center = get_distance_to_center(self.scene.env_origins[:, :2].to(self.device), self.robots[robot_id])
            left_arena = dist_to_center.squeeze(-1) > self.ring_radius
            fell = self.robots[robot_id].data.root_pos_w[:, 2] < 0.17
            anymal_left.append(left_arena)
            anymal_fell.append(fell)

        anymal_left = torch.any(torch.stack(anymal_left), dim=0)
        anymal_fell = torch.any(torch.stack(anymal_fell), dim=0)

        lost = torch.logical_or(anymal_left, anymal_fell)
        lost = {team:lost for team in self.cfg.teams.keys()}

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        time_out = {team:time_out for team in self.cfg.teams.keys()}

        return lost, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robots["robot_0"]._ALL_INDICES
        super()._reset_idx(env_ids)  # once

        # spread out resets
        if len(env_ids) == self.num_envs: # type:ignore
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        # log_robot_masses()

        # sample commands once
        self._commands[env_ids] = torch.zeros_like(self._commands[env_ids]).uniform_(-1.0, 1.0)

        low, high = self.cfg.ring_radius_min, self.cfg.ring_radius_max
    
        self.ring_radius[env_ids] = torch.zeros(env_ids.shape[0]).uniform_(low, high).to(self.device)

        for robot_id in self.robots.keys():
            self.robots[robot_id].reset(env_ids)
            self.actions[robot_id][env_ids] = 0.0
            self.previous_actions[robot_id][env_ids] = 0.0
            joint_pos = self.robots[robot_id].data.default_joint_pos[env_ids]
            joint_vel = self.robots[robot_id].data.default_joint_vel[env_ids]
            default_root_state = self.robots[robot_id].data.default_root_state[env_ids]
            default_root_state[:, :3] += self._terrain.env_origins[env_ids]
            self.robots[robot_id].write_root_pose_to_sim(default_root_state[:, :7], env_ids)
            self.robots[robot_id].write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
            self.robots[robot_id].write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self._draw_ring_markers()

        # logging
        extras = {}
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = extras