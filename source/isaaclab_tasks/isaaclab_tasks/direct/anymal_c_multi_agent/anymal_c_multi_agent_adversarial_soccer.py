# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import torch

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.utils.math import quat_rotate_inverse, quat_conjugate, subtract_frame_transforms
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
from isaaclab.utils.math import quat_from_angle_axis

##
# Pre-defined configs
##
from isaaclab_assets.robots.anymal import ANYMAL_C_CFG  # isort: skip
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip

import math
import colorsys

def set_robot_spacing_and_rot(index, total, spacing_mode="linear", orientation_mode="linear",linear_range=4.0, circle_radius=2.0, grid_shape=(2,2), grid_spacing=(2.0,2.0)):
    """
    Returns (x, y, z) position for a robot given its index and total number of robots.
    spacing_mode: 'linear', 'circular', or 'grid'
    linear_range: total range along y axis for linear spacing
    circle_radius: radius for circular arrangement
    grid_shape: (rows, cols) for grid
    grid_spacing: (dx, dy) for grid

    Returns (pos, rot) for a robot given its index and total number of robots.
    pos: (x, y, z)
    rot: quaternion (w, x, y, z)
    orientation_mode: 'random' or 'face_center'
    """
    import random
    from isaaclab.utils.math import quat_from_angle_axis

    # --- Position ---
    if spacing_mode == "linear":
        if total == 1:
            y = 0.0
        else:
            y = -linear_range / 2 + index * (linear_range / (total - 1))
        pos = (0.0, y, 0.5)
    elif spacing_mode == "circular":
        angle = 2 * math.pi * index / total
        x = circle_radius * math.cos(angle)
        y = circle_radius * math.sin(angle)
        pos = (x, y, 0.5)
    elif spacing_mode == "grid":
        rows, cols = grid_shape
        dx, dy = grid_spacing
        row = index // cols
        col = index % cols
        x = (col - (cols - 1) / 2) * dx
        y = (row - (rows - 1) / 2) * dy
        pos = (x, y, 0.5)
    elif spacing_mode == "random":
        # space the robots with in some distribution, but with spacing (w) from each other 
        # Divide the area into non-overlapping sectors for each robot
        angle_sector = 2 * math.pi / total
        angle_start = index * angle_sector
        angle_end = angle_start + angle_sector
        angle = random.uniform(angle_start, angle_end)
        r_min = circle_radius * 0.7
        r_max = circle_radius * 1.3
        r = random.uniform(r_min, r_max)
        x = r * math.cos(angle)
        y = r * math.sin(angle)
        pos = (x, y, 0.5)

    else:
        raise ValueError(f"Unknown spacing_mode: {spacing_mode}")
    
    # --- orientation ---
    if orientation_mode == "random":
        yaw = random.uniform(-math.pi, math.pi)
    elif orientation_mode == "face_center":
        # Compute the center of all robots (assuming all at same z)
        if spacing_mode == "linear":
            center_y = 0.0
            center = (0.0, center_y, 0.5)
        elif spacing_mode == "circular":
            center = (0.0, 0.0, 0.5)
        elif spacing_mode == "grid":
            rows, cols = grid_shape
            center = (0.0, 0.0, 0.5)
        else:
            center = (0.0, 0.0, 0.5)

        dx = center[0] - pos[0]
        dy = center[1] - pos[1]
        yaw = math.atan2(dy, dx)
    elif orientation_mode == "linear":
        yaw = 0.0
    else:
        raise ValueError(f"Unknown orientation_mode: {orientation_mode}")
    
    yaw_tensor = torch.tensor([yaw], dtype=torch.float32)
    axis_tensor = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
    rot = quat_from_angle_axis(yaw_tensor, axis_tensor)  # (w, x, y, z)
    rot = rot.squeeze(0)  # Remove batch dimension if present
    rot_tuple = tuple(rot.tolist())  # Ensure it's a tuple of length 4
    return pos, rot_tuple
    

@configclass
class EventCfg:
    """Configuration for randomization."""
    def __init__(self, possible_agents=None, material_scales=None):
        if material_scales is None:
            material_scales = {}
        if possible_agents is None:
            possible_agents = []
        for i, robot_id in enumerate(possible_agents):
            scales = material_scales.get(robot_id, {})
            setattr(
                self,
                f"physics_material_{i}",
                EventTerm(
                    func=mdp.randomize_rigid_body_material,
                    mode="startup",
                    params={
                        "asset_cfg": SceneEntityCfg(robot_id, body_names=".*"),
                        "static_friction_range": scales.get("static_friction_range", (0.8, 0.8)),
                        "dynamic_friction_range": scales.get("dynamic_friction_range", (0.6, 0.6)),
                        "restitution_range": scales.get("restitution_range", (0.0, 0.0)),
                        "num_buckets": 64,
                    },
                ),
            )
            setattr(
                self,
                f"add_base_mass_{i}",
                EventTerm(
                    func=mdp.randomize_rigid_body_mass,
                    mode="startup",
                    params={
                        "asset_cfg": SceneEntityCfg(robot_id, body_names="base"),
                        "mass_distribution_params": (-5.0, 5.0),
                        "operation": "add",
                    },
                ),
            )


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

# --- Team arrow marker utilities ---
def get_n_colors(n):
    """Returns n visually distinct RGB colors as (r,g,b) tuples in [0,1]."""
    return [colorsys.hsv_to_rgb(i / n, 0.8, 1.0) for i in range(n)]

def define_team_arrow_markers(num_teams):
    """Define a colored arrow marker for each team."""
    colors = get_n_colors(num_teams)
    scale_val = 0.007
    markers = {
        f"team_arrow_{i}": sim_utils.ConeCfg(
            radius=0.25,
            height=-1.0,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        )
        for i, color in enumerate(colors)
    }
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/teamFlag",
        markers=markers,
    )
    return VisualizationMarkers(marker_cfg)



@configclass
class AnymalCAdversarialSoccerEnvCfg(DirectMARLEnvCfg):
    def __init__(self, team_robot_counts=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Default: 2 robots per team for 2 teams
        if team_robot_counts is None:
            team_robot_counts = {
                "team_0": 2, 
                "team_1": 1, 
                # "team_2": 1
                }
        self.padded_dummy_obs_buffer_add = 0
        self.team_robot_counts = team_robot_counts
        self.num_teams = len(team_robot_counts)
        self.possible_agents = []
        self.teams = {}
        self.action_spaces = {}
        self.observation_spaces = {}
        self.state_spaces = {}
        self.physics_material_scales = {}
        spacing_mode = getattr(self, "robot_spacing_mode", "linear") # linear, circular, grid, random
        spacing_kwargs = getattr(self, "robot_spacing_kwargs", {})
        robot_idx = 0
        for team, num_robots in team_robot_counts.items():
            self.teams[team] = []
            for i in range(num_robots):
                robot_id = f"{team}_robot_{i}"
                self.possible_agents.append(robot_id)
                self.teams[team].append(robot_id)
                self.action_spaces[robot_id] = 12
                # Calculate observation space dynamically
                base_obs_dim = 48  # self state (13) + base_lin_vel (3) + base_ang_vel (3) + gravity_vec (3) + dof_pos (12) + dof_vel (12)]
                num_total_robots = sum(team_robot_counts.values())
                rel_T_dim = (num_total_robots + self.padded_dummy_obs_buffer_add) * 9  # rel_T is [num_total_robots, 9] (7 pose + 1 team mask + 1 neighbor id)
                total_obs_dim = base_obs_dim + rel_T_dim  # rel_T already includes team mask and neighbor id per robot
                self.observation_spaces[robot_id] = total_obs_dim
                # self.observation_spaces[robot_id] = base_obs_dim
                self.state_spaces[robot_id] = 0
                self.physics_material_scales[robot_id] = {
                    "static_friction_range": (0.7 + 0.02*robot_idx, 0.9 + 0.02*robot_idx),
                    "dynamic_friction_range": (0.5 + 0.01*robot_idx, 0.7 + 0.01*robot_idx),
                    "restitution_range": (0.0, 0.1*robot_idx),
                }
                setattr(self, robot_id, ANYMAL_C_CFG.replace(prim_path=f"/World/envs/env_.*/{robot_id}"))
                contact_sensor = ContactSensorCfg(
                    prim_path=f"/World/envs/env_.*/{robot_id}/.*", history_length=3, update_period=0.005, track_air_time=True
                )
                setattr(self, f"contact_sensor_{robot_id}", contact_sensor)
                # Set initial state
                pos, rot = set_robot_spacing_and_rot(robot_idx, sum(team_robot_counts.values()), spacing_mode=spacing_mode, **spacing_kwargs)
                getattr(self, robot_id).init_state.pos = pos
                getattr(self, robot_id).init_state.rot = rot
                robot_idx += 1

        self.events = EventCfg(possible_agents=self.possible_agents, material_scales=self.physics_material_scales)

        # simulation
        self.episode_length_s = 20.0
        self.decimation = 4
        self.action_scale = 0.5
        self.action_space = 12
        self.observation_space = 48
        self.state_space = 0
        self.sim = SimulationCfg(
            dt=1 / 200,
            render_interval=self.decimation,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
        )
        self.terrain = TerrainImporterCfg(
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
        self.scene = InteractiveSceneCfg(num_envs=1, env_spacing=7.0, replicate_physics=True)

        # reward scales (override from flat config)
        self.flat_orientation_reward_scale = 0.0
        self.lin_vel_reward_scale = 1.0
        self.yaw_rate_reward_scale = 0.5
        self.z_vel_reward_scale = -2.0
        self.ang_vel_reward_scale = -0.05
        self.joint_torque_reward_scale = -2.5e-5
        self.joint_accel_reward_scale = -2.5e-7
        self.action_rate_reward_scale = -0.01
        self.feet_air_time_reward_scale = 0.5
        self.undesired_contact_reward_scale = -1.0
        self.flat_orientation_reward_scale = -5.0



class AnymalCAdversarialSoccerEnv(DirectMARLEnv):
    cfg: AnymalCAdversarialSoccerEnvCfg

    def __init__(
        self, cfg: AnymalCAdversarialSoccerEnvCfg, render_mode: str | None = None, debug=False, **kwargs
    ):
        # self.debug = debug
        self.debug = True
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
        # self._commands = {agent : torch.zeros(self.num_envs, 3, device=self.device) for agent in self.cfg.possible_agents}
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)

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
                "body_ground_contact",
            ]
        }

        self.base_ids = {}
        self.feet_ids = {}
        self.undesired_body_contact_ids = {}

        if self.debug:
            self.my_visualizer = define_markers()
            self.team_arrow_visualizer = define_team_arrow_markers(self.cfg.num_teams)

        for robot_id, contact_sensor in self.contact_sensors.items():
            _base_id, _ = contact_sensor.find_bodies("base")
            _feet_ids, _ = contact_sensor.find_bodies(".*FOOT")
            _undesired_contact_body_ids, _ = contact_sensor.find_bodies(".*THIGH")
            self.base_ids[robot_id] = _base_id
            self.feet_ids[robot_id] = _feet_ids
            self.undesired_body_contact_ids[robot_id] = _undesired_contact_body_ids

    def _draw_team_arrows(self):
        # Draws a colored arrow marker above each robot, pointing down
        marker_locations = []
        marker_orientations = []
        marker_indices = []
        team_list = list(self.cfg.teams.keys())
        for team_idx, team in enumerate(team_list):
            for robot_id in self.cfg.teams[team]:
                pos = self.robots[robot_id].data.root_pos_w  # (num_envs, 3)
                arrow_pos = pos.clone()
                arrow_pos[:, 2] += .7
                marker_locations.append(arrow_pos)
                q = torch.tensor([0.0, 0.0, 0.0, 0.0], device=pos.device).expand(pos.shape[0], 4)
                marker_orientations.append(q)
                marker_indices.append(team_idx * torch.ones(pos.shape[0], device=pos.device, dtype=torch.long))
        if marker_locations:
            marker_locations = torch.cat(marker_locations, dim=0)
            marker_orientations = torch.cat(marker_orientations, dim=0)
            marker_indices = torch.cat(marker_indices, dim=0)
            self.team_arrow_visualizer.visualize(
                marker_locations, marker_orientations, marker_indices=marker_indices
            )

    def _setup_scene(self):
        self.robots = {}
        self.contact_sensors = {}
        self.height_scanners = {}
        if self.debug:
            self.my_visualizer = define_markers()
        for robot_id in self.cfg.possible_agents:
            self.robots[robot_id] = Articulation(getattr(self.cfg, robot_id))
            self.scene.articulations[robot_id] = self.robots[robot_id]
            self.contact_sensors[robot_id] = ContactSensor(getattr(self.cfg, f"contact_sensor_{robot_id}"))
            self.scene.sensors[robot_id] = self.contact_sensors[robot_id]
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
        # Compute neighbor obs for all robots (single call, efficient)

        def get_relative_obs(ref_robot_id, other_robot_id_list):
            # Ensure the reference robot is not in the other_robot_id_list
            filtered_other_robot_id_list = [r for r in other_robot_id_list if r != ref_robot_id]
            # Absolute states
            ref_pos  = self.robots[ref_robot_id].data.root_pos_w            # [num_envs, 3]
            ref_quat = self.robots[ref_robot_id].data.root_quat_w           # [num_envs, 4]  (w, x, y, z)

            if len(filtered_other_robot_id_list) > 0:
                other_pos  = torch.stack([self.robots[r].data.root_pos_w  for r in filtered_other_robot_id_list], dim=1)  # [num_envs, num_others, 3]
                other_quat = torch.stack([self.robots[r].data.root_quat_w for r in filtered_other_robot_id_list], dim=1)  # [num_envs, num_others, 4]
                # Include self at index 0 (unsorted)
                all_pos  = torch.cat([ref_pos.unsqueeze(1),  other_pos],  dim=1)  # [num_envs, num_robots, 3]
                all_quat = torch.cat([ref_quat.unsqueeze(1), other_quat], dim=1)  # [num_envs, num_robots, 4]
            else:
                # Only the ref robot present
                all_pos  = ref_pos.unsqueeze(1)                                  # [num_envs, 1, 3]
                all_quat = ref_quat.unsqueeze(1)                                 # [num_envs, 1, 4]

            num_envs, num_robots = all_pos.shape[:2]

            # Local IDs prior to sorting (0=self, 1..others)
            base_ids = torch.arange(num_robots, device=all_pos.device).view(1, -1).expand(num_envs, -1)  # [num_envs, num_robots]

            # --- Compute relatives FIRST (unsorted) ---
            t01 = ref_pos.unsqueeze(1).expand(-1, num_robots, 3).reshape(-1, 3)   # [num_envs*num_robots, 3]
            q01 = ref_quat.unsqueeze(1).expand(-1, num_robots, 4).reshape(-1, 4)  # [num_envs*num_robots, 4]
            t02 = all_pos.reshape(-1, 3)                                          # [num_envs*num_robots, 3]
            q02 = all_quat.reshape(-1, 4)                                         # [num_envs*num_robots, 4]

            rel_pos_flat, rel_quat_flat = subtract_frame_transforms(t01, q01, t02, q02)  # [N*R,3], [N*R,4]
            rel_pos  = rel_pos_flat.view(num_envs, num_robots, 3)                 # [num_envs, num_robots, 3]
            rel_quat = rel_quat_flat.view(num_envs, num_robots, 4)                # [num_envs, num_robots, 4] (w,x,y,z)
            rel_tq   = torch.cat([rel_pos, rel_quat], dim=-1)                     # [num_envs, num_robots, 7]

            # --- Sort by distance in reference frame (equivalent to world) ---
            d2  = (rel_pos ** 2).sum(dim=-1)                                      # [num_envs, num_robots]
            idx = torch.argsort(d2, dim=1)                                        # [num_envs, num_robots]

            # Apply permutation to features and ids
            idx7 = idx.unsqueeze(-1).expand(-1, -1, 7)
            rel_tq_sorted = torch.gather(rel_tq, 1, idx7)                         # [num_envs, num_robots, 7]
            neighbor_ids  = torch.gather(base_ids, 1, idx)                        # [num_envs, num_robots]

            # Team mask (False for self), then permute
            team_name = ref_robot_id.split("_robot_")[0]
            raw_mask = torch.tensor(
                [False] + [not rid.startswith(team_name) for rid in filtered_other_robot_id_list],
                device=all_pos.device, dtype=torch.bool
            ).view(1, -1).expand(num_envs, -1)                                    # [num_envs, num_robots]
            team_mask = torch.gather(raw_mask, 1, idx)                             # [num_envs, num_robots] (bool)

            # Single per-robot row: [tx,ty,tz,qw,qx,qy,qz, team_mask(float), neighbor_id(float)]
            rel_T = torch.cat(
                [
                    rel_tq_sorted,                                 # [num_envs, num_robots, 7]
                    team_mask.unsqueeze(-1).to(rel_tq.dtype),      # [num_envs, num_robots, 1]
                    neighbor_ids.unsqueeze(-1).to(rel_tq.dtype),   # [num_envs, num_robots, 1]
                ],
                dim=-1,
            )  # -> [num_envs, num_robots, 9]

            return rel_T  # [num_envs, num_robots, 9]

        robot_id_list = list(self.robots.keys())
        # if self.padded_obs_buffer_add > 0:
        #     robot_id_list = robot_id_list + ["fake_robot"] * self.padded_obs_buffer_add
        robot_id_to_idx = {rid: i for i, rid in enumerate(robot_id_list)}
        for team, robot_ids in self.cfg.teams.items():
            obs[team] = {}
            for robot_id in robot_ids:
                # Gather observation components for this robot
                root_lin_vel_b = self.robots[robot_id].data.root_lin_vel_b # [num_envs, 3]
                root_ang_vel_b = self.robots[robot_id].data.root_ang_vel_b # [num_envs, 3]
                projected_gravity_b = self.robots[robot_id].data.projected_gravity_b # [num_envs, 3]
                commands = self._commands # [num_envs, 3], shared command for all robots
                joint_pos_delta = self.robots[robot_id].data.joint_pos - self.robots[robot_id].data.default_joint_pos # [num_envs, 12]
                joint_vel = self.robots[robot_id].data.joint_vel # [num_envs, 12]
                actions = self.actions[robot_id] # [num_envs, 12]

                rel_T = get_relative_obs(robot_id, robot_id_list)  # [num_envs, num_robots, 9]
                rel_T_flat = rel_T.reshape(rel_T.shape[0], -1)     # [num_envs, num_robots * 9]

                # Concatenate all observation components
                obs_vec = torch.cat(
                    [
                        root_lin_vel_b,           # [num_envs, 3]
                        root_ang_vel_b,           # [num_envs, 3]
                        projected_gravity_b,      # [num_envs, 3]
                        commands,                 # [num_envs, 3]
                        joint_pos_delta,          # [num_envs, 12]
                        joint_vel,                # [num_envs, 12]
                        actions,                  # [num_envs, 12]
                        rel_T_flat,               # [num_envs, num_robots * 9] 
                        # Note: (9 = [tx,ty,tz, qw,qx,qy,qz, team_mask, neighbor_id])
                    ],
                    dim=-1,
                )
                obs[team][robot_id] = obs_vec # [num_envs, obs_dim=48 + num_robots*9]
        return obs

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

        # Use the first robot id from possible_agents (e.g., "team_0_robot_0")
        first_robot_id = self.cfg.possible_agents[0]
        robot_pos = self.robots[first_robot_id].data.root_pos_w.clone()
        robot_yaw = self.robots[first_robot_id].data.root_ang_vel_b[:, 2].clone()

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
                robot_pos + self.robots[first_robot_id].data.root_lin_vel_b,
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
            self._draw_team_arrows()
        all_rewards = {}
        for robot_id in self.robots.keys():
            # linear velocity tracking
            lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - self.robots[robot_id].data.root_lin_vel_b[:, :2]), dim=1)
            lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25)
            # yaw rate tracking
            yaw_rate_error = torch.square(self._commands[:, 2] - self.robots[robot_id].data.root_ang_vel_b[:, 2])
            yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.25)
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
            air_time = torch.sum((last_air_time - 0.5) * first_contact, dim=1) * (
                torch.norm(self._commands[:, :2], dim=1) > 0.1
            )
            # undesired contacts
            net_contact_forces = self.contact_sensors[robot_id].data.net_forces_w_history
            is_contact = (
                torch.max(torch.norm(net_contact_forces[:, :, self.undesired_body_contact_ids[robot_id]], dim=-1), dim=1)[0] > 1.0
            )
            contacts = torch.sum(is_contact, dim=1)
            # flat orientation
            flat_orientation = torch.sum(torch.square(self.robots[robot_id].data.projected_gravity_b[:, :2]), dim=1)

            # --- Negative reward for touching the ground with the body (base) ---
            # Check if the base is in contact with the ground
            base_contact_forces = self.contact_sensors[robot_id].data.net_forces_w_history[:, :, self.base_ids[robot_id]]
            base_contact = (torch.max(torch.norm(base_contact_forces, dim=-1), dim=1)[0] > 1.0)
            base_contact_penalty = base_contact.float() * -2.0 * self.step_dt  # scale as needed

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
                "body_ground_contact": base_contact_penalty.squeeze(-1),
            }
            reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
            all_rewards[robot_id] = reward
            # Logging
            for key, value in rewards.items():
                self._episode_sums[key] += value
        # Group rewards by team
        rewards = {team: torch.stack([all_rewards[robot_id] for robot_id in robot_ids]).sum(dim=0)
                    for team, robot_ids in self.cfg.teams.items()}
        return rewards

    def _get_dones(self) -> tuple[dict, dict]:
        # anymal_died = []
        # for robot_id, contact_sensor in self.contact_sensors.items():
        #     net_contact_forces = contact_sensor.data.net_forces_w_history
        #     died = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self.base_ids[robot_id]], dim=-1), dim=1)[0] > 1.0, dim=1)
        #     anymal_died.append(died)

        # anymal_died = torch.any(torch.stack(anymal_died), dim=1)

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        time_out_dict = {team: time_out for team in self.cfg.teams.keys()}
        return time_out_dict, time_out_dict

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            # get 1st robot name
            name = list(self.robots.keys())[0]
            env_ids = self.robots[name]._ALL_INDICES
        super()._reset_idx(env_ids)  # once

        # spread out resets
        if len(env_ids) == self.num_envs:
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        # sample commands once
        self._commands[env_ids] = torch.zeros_like(self._commands[env_ids]).uniform_(-1.0, 1.0)
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

        # logging
        extras = {}
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = extras