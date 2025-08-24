# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import math
import colorsys
import torch

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.utils.math import (
    quat_rotate_inverse,
    quat_conjugate,
    quat_apply,
    subtract_frame_transforms,
    quat_from_angle_axis,
)
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

##
# Pre-defined configs
##
from isaaclab_assets.robots.anymal import ANYMAL_C_CFG  # isort: skip
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip


# ----------------------------------------------------------------------------------------
# Position & orientation helpers for initial robot placement
# ----------------------------------------------------------------------------------------

def set_robot_spacing_and_rot(
    index,
    total,
    spacing_mode: str = "linear",
    orientation_mode: str = "linear",
    linear_range: float = 4.0,
    circle_radius: float = 2.0,
    grid_shape=(2, 2),
    grid_spacing=(2.0, 2.0),
):
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
        # Divide the ring area into per-robot sectors to reduce collisions
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

    # --- Orientation (yaw) ---
    if orientation_mode == "random":
        yaw = random.uniform(-math.pi, math.pi)
    elif orientation_mode == "face_center":
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


def define_markers_single() -> VisualizationMarkers:
    arrow_usd = f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd"  # (NO_Idea)

    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/combined_arrows",
        markers={
            "cmd_arrow": sim_utils.UsdFileCfg(
                usd_path=arrow_usd,
                scale=(1.0, 1.0, 1.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.2, 0.2)),  # red
            ),
            "vel_arrow": sim_utils.UsdFileCfg(
                usd_path=arrow_usd,
                scale=(1.0, 1.0, 1.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.35, 1.0)),  # blue
            ),
            "cmd_rot_arrow": sim_utils.UsdFileCfg(
                usd_path=arrow_usd,
                scale=(1.0, 1.0, 1.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.7, 0.2)),  # orange/yellow
            ),
            "vel_rot_arrow": sim_utils.UsdFileCfg(
                usd_path=arrow_usd,
                scale=(1.0, 1.0, 1.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.95, 0.4)),  # green
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
            height=-.50,
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

        # Hard cap for number of relative robot slots included in the observation
        self.max_rel_robots_in_obs = getattr(self, "max_rel_robots_in_obs", 8)
        # How to pad if fewer real robots exist than the cap: "random" | "zeros"
        self.dummy_pad_mode = getattr(self, "dummy_pad_mode", "random")

        # Default: 2 robots per team for 2 teams
        if team_robot_counts is None:
            team_robot_counts = {
                "team_0": 2, 
                "team_1": 1, 
                # "team_2": 1
                }
        self.padded_dummy_obs_buffer_add = max(0, self.max_rel_robots_in_obs - max(team_robot_counts.values()) - 1)
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

            self.team_arrow_visualizer = define_team_arrow_markers(self.cfg.num_teams)
            self.arrows = define_markers_single()

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
                arrow_pos[:, 2] += 1.0
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
            self.my_visualizer = define_markers_single()
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
    def _ensure_dummy_cache(self):
        cap = self.cfg.max_rel_robots_in_obs
        R = len(self.robots)  # includes self
        self._pad_per_obs = max(0, cap - R)
        if self._pad_per_obs == 0:
            self._dummy_rel_T_cache = None
            return
        # Allocate if missing or wrong shape
        need_alloc = (
            not hasattr(self, "_dummy_rel_T_cache")
            or self._dummy_rel_T_cache.shape != (self.num_envs, self._pad_per_obs, 9)
        )
        if need_alloc:
            self._dummy_rel_T_cache = torch.zeros(self.num_envs, self._pad_per_obs, 9, device=self.device)

    def _sample_dummy_rel_T(self, n_envs: int, m_slots: int, *, device, dtype,
                            r_range=(10.0, 30.0), z_height=0.50) -> torch.Tensor:
        """Create [n_envs, m_slots, 9] dummy rows: [tx,ty,tz,qw,qx,qy,qz, team_mask, neighbor_id]."""
        if m_slots == 0:
            return torch.zeros((n_envs, 0, 9), device=device, dtype=dtype)

        # Random polar position in XY
        u = torch.rand((n_envs, m_slots, 2), device=device, dtype=dtype)
        r = r_range[0] + (r_range[1] - r_range[0]) * u[..., 0]
        ang = -math.pi + 2 * math.pi * u[..., 1]
        tx = r * torch.cos(ang)
        ty = r * torch.sin(ang)
        tz = torch.full_like(tx, z_height)
        t = torch.stack([tx, ty, tz], dim=-1)  # [N,M,3]

        # Random yaw-only quaternion about +Z
        yaw = -math.pi + 2 * math.pi * torch.rand((n_envs, m_slots), device=device, dtype=dtype)
        half = 0.5 * yaw
        qw = torch.cos(half)
        qx = torch.zeros_like(qw)
        qy = torch.zeros_like(qw)
        qz = torch.sin(half)
        q = torch.stack([qw, qx, qy, qz], dim=-1)  # [N,M,4]

        # team_mask=1.0 (treat as opponent), neighbor_id=-1.0 sentinel
        team_mask = torch.ones((n_envs, m_slots, 1), device=device, dtype=dtype)
        # Create random unique neighbor ids in [real_robots, 1000)
        num_real_robots = len(self.robots)
        neighbor_id = torch.randint(num_real_robots, 1000, (n_envs, m_slots, 1), device=device, dtype=dtype)

        return torch.cat([t, q, team_mask, neighbor_id], dim=-1)  # [N,M,9]

    # ------------------------
    # Observations
    # ------------------------
    def _get_observations(self) -> dict:
        self.previous_actions = copy.deepcopy(self.actions)
        obs = {}
        # Compute neighbor obs for all robots (single call, efficient)

        def get_relative_obs(ref_robot_id: str, other_robot_id_list: list[str], num_slots: int) -> torch.Tensor:
            """
            Returns [N, num_slots, 9] for the reference robot, sorted by distance (self included if present),
            truncated to K=min(R, num_slots), and padded (random or zeros) to num_slots.
            """
            # Remove ref from candidate list if present (we will add self explicitly below)
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

            # Keep K nearest
            R = rel_tq_sorted.shape[1]
            K = min(R, num_slots)
            if num_slots == 0:
                # No relative info requested
                return torch.zeros((num_envs, 0, 9), device=rel_tq_sorted.device, dtype=rel_tq_sorted.dtype)

            rel_tq_k = rel_tq_sorted[:, :K, :]  # [N,K,7]
            team_mask_k = team_mask[:, :K].unsqueeze(-1).to(rel_tq_k.dtype)  # [N,K,1]
            neighbor_ids_k = neighbor_ids[:, :K].unsqueeze(-1).to(rel_tq_k.dtype)  # [N,K,1]
            kept = torch.cat([rel_tq_k, team_mask_k, neighbor_ids_k], dim=-1)  # [N,K,9]

            if K == num_slots:
                return kept  # [N,num_slots,9]

            # Need padding
            pad_count = num_slots - K
            if pad_count > 0:
                if self._dummy_rel_T_cache is not None:
                    pad = self._dummy_rel_T_cache[:, :pad_count, :]  # stable across the episode
                else:
                    pad = torch.zeros((num_envs, pad_count, 9), device=kept.device, dtype=kept.dtype)
                rel_T = torch.cat([kept, pad], dim=1)
            else:
                rel_T = kept
                
            return rel_T

        robot_id_list = list(self.robots.keys())
        for team, robot_ids in self.cfg.teams.items():
            obs[team] = {}
            for robot_id in robot_ids:
                # Gather observation components for this robot
                root_lin_vel_b = self.robots[robot_id].data.root_lin_vel_b # [num_envs, 3]
                root_ang_vel_b = self.robots[robot_id].data.root_ang_vel_b # [num_envs, 3]
                projected_gravity_b = self.robots[robot_id].data.projected_gravity_b # [num_envs, 3]
                commands = self._commands # [num_envs, 3], shared command for all robots
                joint_pos_delta = (
                    self.robots[robot_id].data.joint_pos - self.robots[robot_id].data.default_joint_pos
                )  # [N,12]
                joint_vel = self.robots[robot_id].data.joint_vel # [num_envs, 12]
                actions = self.actions[robot_id] # [num_envs, 12]

                # Relative block with cap + padding
                num_slots = self.cfg.max_rel_robots_in_obs
                rel_T = get_relative_obs(robot_id, robot_id_list, num_slots)  # [num_envs, num_robots, 9]
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

    def _draw_cmd_and_vel(self,
                        gain_lin=3.0, min_len_lin=0.05, z_cmd=0.25, z_vel=0.30, thickness=0.1,
                        gain_rot=2.0, min_len_rot=0.04, z_rot=0.5):
        """
        (single visualizer with two prototypes):
        - marker_indices=0 -> 'cmd_arrow' (red)
        - marker_indices=1 -> 'vel_arrow' (blue)

        Draw per robot, per env:
        1) Command XY arrow   (red)
        2) Velocity XY arrow  (blue)
        3) Command yaw arrow  (red, up for +yaw, down for -yaw)
        4) Velocity yaw arrow (blue, up for +yaw, down for -yaw)
        """
        first_robot = next(iter(self.robots.values()))
        N      = first_robot.data.root_pos_w.shape[0]
        device = first_robot.data.root_pos_w.device
        dtype  = first_robot.data.root_pos_w.dtype

        # Shared command in world (XY + yaw rate)
        cmd_w = self._commands.to(device=device, dtype=dtype).clone()  # [N,3]
        cmd_w[:, 2] = cmd_w[:, 2]  # yaw rate (rad/s), keep as-is
        cmd_xy = cmd_w[:, :2]
        omega_cmd = cmd_w[:, 2]                                        # [N]

        # Unit axes (batched) for angle-axis
        z_axis = torch.zeros((N, 3), device=device, dtype=dtype); z_axis[:, 2] = 1.0  # +Z
        y_axis = torch.zeros((N, 3), device=device, dtype=dtype); y_axis[:, 1] = 1.0  # +Y

        all_loc, all_ori, all_scl, all_idx = [], [], [], []

        for _, robot_ids in self.cfg.teams.items():
            for rid in robot_ids:
                pos_w  = self.robots[rid].data.root_pos_w.to(device=device, dtype=dtype)   # [N,3]
                quat_w = self.robots[rid].data.root_quat_w.to(device=device, dtype=dtype)  # [N,4] (w,x,y,z)

                # Linear velocity in world (fallback body->world)
                if hasattr(self.robots[rid].data, "root_lin_vel_w"):
                    vel_w = self.robots[rid].data.root_lin_vel_w.to(device=device, dtype=dtype).clone()
                else:
                    v_b = self.robots[rid].data.root_lin_vel_b.to(device=device, dtype=dtype)
                    vel_w = quat_apply(quat_w, v_b)
                vel_w[:, 2] = 0.0

                # Angular velocity about world Z (fallback body->world then take z)
                if hasattr(self.robots[rid].data, "root_ang_vel_w"):
                    omega_w = self.robots[rid].data.root_ang_vel_w.to(device=device, dtype=dtype)  # [N,3]
                else:
                    # rotate ω_b to world via q v q*
                    omega_b = self.robots[rid].data.root_ang_vel_b.to(device=device, dtype=dtype)  # [N,3]
                    omega_w = quat_apply(quat_w, omega_b)
                omega_world_z = omega_w[:, 2]  # [N]

                # ---------- 1) Command XY (index 0, red) ----------
                len_cmd = (cmd_xy.pow(2).sum(-1).sqrt() * gain_lin).clamp_min(min_len_lin)
                yaw_cmd = torch.atan2(cmd_xy[:, 1], cmd_xy[:, 0]).to(dtype)
                q_cmd_xy = quat_from_angle_axis(yaw_cmd, z_axis)                  # yaw about +Z
                p_cmd_xy = pos_w.clone(); p_cmd_xy[:, 2] += z_cmd
                s_cmd_xy = torch.stack([len_cmd,
                                        torch.full_like(len_cmd, thickness),
                                        torch.full_like(len_cmd, thickness)], dim=-1)
                idx_cmd_xy = torch.zeros(N, device=device, dtype=torch.long)

                # ---------- 2) Velocity XY (index 1, blue) ----------
                v_xy   = vel_w[:, :2]
                len_v  = (v_xy.pow(2).sum(-1).sqrt() * gain_lin).clamp_min(min_len_lin)
                yaw_v  = torch.atan2(v_xy[:, 1], v_xy[:, 0]).to(dtype)
                q_vel_xy = quat_from_angle_axis(yaw_v, z_axis)
                p_vel_xy = pos_w.clone(); p_vel_xy[:, 2] += z_vel
                s_vel_xy = torch.stack([len_v,
                                        torch.full_like(len_v, thickness),
                                        torch.full_like(len_v, thickness)], dim=-1)
                idx_vel_xy = torch.ones(N, device=device, dtype=torch.long)

                # ---------- 3) Command yaw (index 0, red; up for +, down for -) ----------
                len_cmd_yaw = (omega_cmd.abs() * gain_rot).clamp_min(min_len_rot)
                # rotate +X to ±Z via ±π/2 about +Y
                ang_cmd_yaw = torch.sign(omega_cmd).to(dtype) * (math.pi / 2)
                q_cmd_yaw   = quat_from_angle_axis(ang_cmd_yaw, y_axis)          # up (+), down (-)
                p_cmd_yaw   = pos_w.clone(); p_cmd_yaw[:, 2] += z_rot
                s_cmd_yaw   = torch.stack([len_cmd_yaw,
                                        torch.full_like(len_cmd_yaw, thickness),
                                        torch.full_like(len_cmd_yaw, thickness)], dim=-1)
                idx_cmd_yaw = torch.zeros(N, device=device, dtype=torch.long)

                # ---------- 4) Velocity yaw (index 1, blue; up for +, down for -) ----------
                len_vel_yaw = (omega_world_z.abs() * gain_rot).clamp_min(min_len_rot)
                ang_vel_yaw = torch.sign(omega_world_z).to(dtype) * (math.pi / 2)
                q_vel_yaw   = quat_from_angle_axis(ang_vel_yaw, y_axis)
                p_vel_yaw   = pos_w.clone(); p_vel_yaw[:, 2] += (z_rot + 0.05)    # slight offset to avoid overlap
                s_vel_yaw   = torch.stack([len_vel_yaw,
                                        torch.full_like(len_vel_yaw, thickness),
                                        torch.full_like(len_vel_yaw, thickness)], dim=-1)
                idx_vel_yaw = torch.ones(N, device=device, dtype=torch.long)

                # Append all four arrows (2 linear + 2 rotational)
                all_loc.append(torch.cat([p_cmd_xy, p_vel_xy, p_cmd_yaw, p_vel_yaw], dim=0))
                all_ori.append(torch.cat([q_cmd_xy, q_vel_xy, q_cmd_yaw, q_vel_yaw], dim=0))
                all_scl.append(torch.cat([s_cmd_xy, s_vel_xy, s_cmd_yaw, s_vel_yaw], dim=0))
                # Use marker indices: 0=cmd_arrow, 1=vel_arrow, 2=cmd_rot_arrow, 3=vel_rot_arrow
                all_idx.append(torch.cat([
                    torch.zeros(N, device=device, dtype=torch.long),      # cmd_arrow
                    torch.ones(N, device=device, dtype=torch.long),       # vel_arrow
                    torch.full((N,), 2, device=device, dtype=torch.long), # cmd_rot_arrow
                    torch.full((N,), 3, device=device, dtype=torch.long)  # vel_rot_arrow
                ], dim=0))

        if all_loc:
            self.arrows.visualize(
                torch.cat(all_loc, dim=0),
                torch.cat(all_ori, dim=0),
                scales=torch.cat(all_scl, dim=0),
                marker_indices=torch.cat(all_idx, dim=0),
            )

    def _get_rewards(self) -> dict:
        if self.debug:
            # self._draw_markers(self._commands)
            self._draw_cmd_and_vel()
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
            base_contact_penalty = base_contact.float() * -5.0 * self.step_dt  # scale as needed

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

        self._ensure_dummy_cache()
        if self._dummy_rel_T_cache is not None:
            if self.cfg.dummy_pad_mode == "zeros":
                self._dummy_rel_T_cache[env_ids] = 0.0
            else:
                pad = self._sample_dummy_rel_T(
                    n_envs=len(env_ids),
                    m_slots=self._pad_per_obs,
                    device=self.device,
                    dtype=torch.float32,
                )
                self._dummy_rel_T_cache[env_ids] = pad
        # logging
        extras = {}
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = extras