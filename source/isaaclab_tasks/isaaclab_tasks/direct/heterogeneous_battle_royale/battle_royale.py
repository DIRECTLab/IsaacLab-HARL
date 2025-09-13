from __future__ import annotations

import torch
import copy
import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab_assets.robots.leatherback import LEATHERBACK_CFG  # isort: skip
from isaaclab_assets.robots.anymal import ANYMAL_C_CFG  # isort: skip
from isaaclab_assets.robots.unitree import H1_CFG  # isort: skip
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.utils.math import quat_from_euler_xyz
import torch
import math

def get_quaternion_tuple_from_xyz(x, y, z):
    quat_tensor = quat_from_euler_xyz(torch.tensor([x]), torch.tensor([y]), torch.tensor([z])).flatten()
    return (quat_tensor[0].item(), quat_tensor[1].item(), quat_tensor[2].item(), quat_tensor[3].item())

def get_quaternion_facing_center(x, y):
    """Compute quaternion to face (0,0) from position (x,y)."""
    yaw = math.atan2(-y, -x)  # vector pointing toward center
    return get_quaternion_tuple_from_xyz(0, 0, yaw)

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

    add_base_mass_0 = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot_0", body_names="base"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )

    physics_material_1 = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot_1", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.2),
            "num_buckets": 64,
        },
    )

    add_base_mass_1 = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot_1", body_names="base"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )

@configclass
class BattleRoyaleHeteroByTeamEnvCfg(DirectMARLEnvCfg):
    decimation = 4
    episode_length_s = 1

    # 4 robots, each with throttle + steering (2 actions)

    action_spaces = {"anymal_0": 1,
                     "anymal_1": 1,
                     "h1_0": 1,
                     "leatherback_0": 1,
                    "anymal_2": 1,
                    "leatherback_1": 1,
                    "h1_1": 1,
    }

    # Observation: teammate (3) + opp1 (3) + opp2 (3) + rcol(1) + dist_center(1) + velocity(3)
    # = 14 per robot
    observation_spaces = {
        "anymal_0": 1,
        "anymal_1": 1,
        "h1_0": 1,
        "leatherback_0": 1,
        "anymal_2": 1,
        "leatherback_1": 1,
        "h1_1": 1,
    }
    state_space = 0
    state_spaces = {f"robot_{i}": 0 for i in range(4)} 
    possible_agents = []

    # Teams
    teams = {
        "team_0": ["anymal_0", "anymal_1"],
        "team_1": ["h1_0", "leatherback_0"],
        "team_2": ["anymal_2", "leatherback_1", "h1_1"],
    }
    num_robots = sum(len(members) for members in teams.values())
    
    for team_name, members in teams.items():
        for agent_id in members:
            possible_agents.append(agent_id)


    sim: SimulationCfg = SimulationCfg(dt=1 / 200, render_interval=decimation)

    radius = 1.5

    # Robot configs (prim paths unique per robot)
    # --- Group A (center ~ ( +radius, 0 )) ---
    anymal_0: ArticulationCfg = ANYMAL_C_CFG.replace(prim_path="/World/envs/env_.*/Robot_0")
    anymal_0.init_state.pos = (radius, +0.3, 0.3)
    anymal_0.init_state.rot = get_quaternion_tuple_from_xyz(0.0, 0.0, math.atan2(-(+0.3), -(radius)))

    anymal_1: ArticulationCfg = ANYMAL_C_CFG.replace(prim_path="/World/envs/env_.*/Robot_1")
    anymal_1.init_state.pos = (radius, -0.3, 0.3)
    anymal_1.init_state.rot = get_quaternion_tuple_from_xyz(0.0, 0.0, math.atan2(-(-0.3), -(radius)))

    # --- Group B (center ~ ( -0.5*radius, +sqrt(3)/2*radius )) ---
    h1_0: ArticulationCfg = H1_CFG.replace(prim_path="/World/envs/env_.*/Robot_2")
    h1_0.init_state.pos = (-0.5 * radius + 0.3, (math.sqrt(3) / 2) * radius, 1.0)
    h1_0.init_state.rot = get_quaternion_tuple_from_xyz(
        0.0, 0.0,
        math.atan2(-((math.sqrt(3) / 2) * radius), -(-0.5 * radius + 0.3))
    )

    leatherback_0: ArticulationCfg = LEATHERBACK_CFG.replace(prim_path="/World/envs/env_.*/Robot_3")
    leatherback_0.init_state.pos = (-0.5 * radius - 0.3, (math.sqrt(3) / 2) * radius, 0.1)
    leatherback_0.init_state.rot = get_quaternion_tuple_from_xyz(
        0.0, 0.0,
        math.atan2(-((math.sqrt(3) / 2) * radius), -(-0.5 * radius - 0.3))
    )

    # --- Group C (center ~ ( -0.5*radius, -sqrt(3)/2*radius )) ---
    anymal_2: ArticulationCfg = ANYMAL_C_CFG.replace(prim_path="/World/envs/env_.*/Robot_4")
    anymal_2.init_state.pos = (-0.5 * radius + 0.0, -(math.sqrt(3) / 2) * radius + 0.3, 0.3)
    anymal_2.init_state.rot = get_quaternion_tuple_from_xyz(
        0.0, 0.0,
        math.atan2(-(-(math.sqrt(3) / 2) * radius + 0.3), -(-0.5 * radius + 0.0))
    )

    leatherback_1: ArticulationCfg = LEATHERBACK_CFG.replace(prim_path="/World/envs/env_.*/Robot_5")
    leatherback_1.init_state.pos = (-0.5 * radius - 0.3, -(math.sqrt(3) / 2) * radius - 0.3, 0.1)
    leatherback_1.init_state.rot = get_quaternion_tuple_from_xyz(
        0.0, 0.0,
        math.atan2(-(-(math.sqrt(3) / 2) * radius - 0.3), -(-0.5 * radius - 0.3))
    )

    h1_1: ArticulationCfg = H1_CFG.replace(prim_path="/World/envs/env_.*/Robot_6")
    h1_1.init_state.pos = (-0.5 * radius + 0.3, -(math.sqrt(3) / 2) * radius - 0.3, 1.0)
    h1_1.init_state.rot = get_quaternion_tuple_from_xyz(
        0.0, 0.0,
        math.atan2(-(-(math.sqrt(3) / 2) * radius - 0.3), -(-0.5 * radius + 0.3))
    )



    throttle_dof_name = [
        "Wheel__Knuckle__Front_Left",
        "Wheel__Knuckle__Front_Right",
        "Wheel__Upright__Rear_Right",
        "Wheel__Upright__Rear_Left"
    ]
    steering_dof_name = [
        "Knuckle__Upright__Front_Right",
        "Knuckle__Upright__Front_Left",
    ]

    throttle_scale = 10
    throttle_max = 50
    steering_scale = 0.1
    steering_max = 0.75

    action_scale = 0.5
    ring_radius_min = 3
    ring_radius_max = 6
    reward_scale = 10
    # time penalty
    time_penalty = -0.01

    env_spacing = 10.0
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=ring_radius_max * 2, replicate_physics=True)

class BattleRoyaleHeteroByTeamEnv(DirectMARLEnv):
    cfg: BattleRoyaleHeteroByTeamEnvCfg

    def __init__(self, cfg: BattleRoyaleHeteroByTeamEnvCfg, render_mode: str | None = None, headless: bool | None = None, debug: bool | None = False, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.headless = headless
        self.debug = debug


        self.env_spacing = self.cfg.env_spacing

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

        team_dot_markers = {
            "team_0": sim_utils.SphereCfg(
                radius=0.08,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.8)),  # blue
            ),
            "team_1": sim_utils.SphereCfg(
                radius=0.08,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.0, 0.0)),  # red
            ),
            "team_2": sim_utils.SphereCfg(
                radius=0.08,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.8, 0.0)),  # green
            ),
        }

        self.team_markers = VisualizationMarkers(
            VisualizationMarkersCfg(prim_path="/World/TeamDots", markers=team_dot_markers)
        )


        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "team_0_push_out_reward",
                "team_1_push_out_reward"
            ]
        }

        self.base_ids = {}

        for robot_id, contact_sensor in self.contact_sensors.items():
            _base_id, _ = contact_sensor.find_bodies("base")
            self.base_ids[robot_id] = _base_id


    @torch.no_grad()
    def _draw_team_dots(self):
        positions, indices = [], []

        # Map each robot_id -> team_index
        robot_to_team = {}
        for team_idx, (team_name, members) in enumerate(self.cfg.teams.items()):
            for member in members:
                robot_to_team[member] = team_idx

        for robot_id, robot in self.robots.items():
            pos = robot.data.root_pos_w.clone()
            pos[:, 2] += 0.5  # hover above robot
            positions.append(pos)

            team_idx = robot_to_team.get(robot_id, 0)  # default to 0 if not found
            indices.append(torch.full((self.num_envs,), team_idx, device=self.device))

        # Stack all data for markers
        marker_positions = torch.cat(positions, dim=0)
        marker_indices = torch.cat(indices, dim=0)

        # Identity quaternions
        marker_orientations = torch.zeros((marker_positions.shape[0], 4), device=self.device)
        marker_orientations[:, 0] = 1.0  

        # Uniform scales
        marker_scales = torch.ones((marker_positions.shape[0], 3), device=self.device)

        self.team_markers.visualize(
            marker_positions, marker_orientations, scales=marker_scales, marker_indices=marker_indices
        )

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
        
        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(
                size=(500.0, 500.0),  # Much larger ground plane (500m x 500m)
                color=(0.2, 0.2, 0.2),  # Dark gray color
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode="multiply",
                    restitution_combine_mode="multiply",
                    static_friction=1.0,
                    dynamic_friction=1.0,
                    restitution=0.0,
                ),
            ),
        )
        # Setup robots
        self.robots = {}
        self.num_robots = sum(1 for key in self.cfg.__dict__.keys() if "robot_" in key)
        self.contact_sensors = {}
        for team_name, members in self.cfg.teams.items():
            for agent_id in members:
                cfg = getattr(self.cfg, agent_id)
                self.robots[agent_id] = Articulation(cfg)
                self.scene.articulations[agent_id] = self.robots[agent_id]

        # Clone environments and filter collisions
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])

        # Add lighting
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: dict) -> None:
        pass

    def _apply_action(self) -> None:
        pass

    def _get_observations(self) -> dict:
        obs = {}
        for team_name, members in self.cfg.teams.items():
            obs[team_name] = {}
            for agent_id in members:
                obs[team_name][agent_id] = torch.zeros((self.num_envs, self.cfg.observation_spaces[agent_id]), device=self.device)
        return obs
    
    def _get_rewards(self) -> dict:
        self._draw_team_dots()


        return {
            "team_0": torch.zeros(self.num_envs, device=self.device),
            "team_1": torch.zeros(self.num_envs, device=self.device),
            "team_2": torch.zeros(self.num_envs, device=self.device),
        }

    def _robots_out_of_ring(self) -> dict[str, torch.Tensor]:
        env_xy = self.scene.env_origins[:, :2].to(self.device)  
        out = {}
        for robot_id in self.robots.keys():
            pos_xy = self.robots[robot_id].data.root_pos_w[:, :2]  
            dist = torch.linalg.norm(pos_xy - env_xy, dim=1)
            out[robot_id] = dist > self.ring_radius
        return out

    def _get_dones(self) -> tuple[dict, dict]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        timeouts = {team: time_out for team in self.cfg.teams.keys()}
        return timeouts, timeouts

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        super()._reset_idx(env_ids)

        # spread out the updates
        if not self.debug:
            if len(env_ids) == self.num_envs: # type: ignore
                self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        # Randomize ring radius per env
        low, high = self.cfg.ring_radius_min, self.cfg.ring_radius_max
        self.ring_radius[env_ids] = (
            torch.empty(env_ids.shape[0], device=self.device).uniform_(low, high) # type: ignore
        )

        origins = self.scene.env_origins[env_ids]  # (N, 3)


        for i, robot_id in enumerate(self.robots):
            self.robots[robot_id].reset(env_ids)
            default_root_state = self.robots[robot_id].data.default_root_state[env_ids].clone()
            default_root_state[:, :2] += origins[:, :2]
            self.robots[robot_id].write_root_pose_to_sim(default_root_state[:, :7], env_ids)
            self.robots[robot_id].write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
            joint_pos = self.robots[robot_id].data.default_joint_pos[env_ids]
            joint_vel = self.robots[robot_id].data.default_joint_vel[env_ids]
            self.robots[robot_id].write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self._draw_ring_markers()
        self._draw_team_dots()
        extras = dict()
        
        self.extras["log"] = dict()
        self.extras["log"].update(extras)

