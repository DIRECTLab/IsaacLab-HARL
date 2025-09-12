from __future__ import annotations

import torch
import copy
import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab_assets.robots.leatherback import LEATHERBACK_CFG  # isort: skip
from isaaclab_assets.robots.anymal import ANYMAL_C_CFG  # isort: skip
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.utils.math import quat_from_angle_axis, quat_from_euler_xyz, quat_rotate_inverse
from isaaclab_assets.custom.soccer_ball import SOCCERBALL_CFG  # isort: skip
from isaaclab.envs.common import ViewerCfg
import random

def get_quaternion_tuple_from_xyz(x, y, z):
    quat_tensor = quat_from_euler_xyz(torch.tensor([x]), torch.tensor([y]), torch.tensor([z])).flatten()
    return (quat_tensor[0].item(), quat_tensor[1].item(), quat_tensor[2].item(), quat_tensor[3].item())

@configclass
class EventCfg:
    """Configuration for randomization."""

    physics_material_0 = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("anymal_0", body_names=".*"),
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
            "asset_cfg": SceneEntityCfg("anymal_0", body_names="base"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )

    physics_material_1 = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("anymal_1", body_names=".*"),
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
            "asset_cfg": SceneEntityCfg("anymal_1", body_names="base"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )

@configclass
class AnymalSoccerHeteroByTeamEnvCfg(DirectMARLEnvCfg):
    decimation = 4
    episode_length_s = 30.0
    action_scale = 0.5
    anymal_action_spaces = {f"anymal_{i}": 12 for i in range(2)}
    leatherback_action_spaces = {f"leatherback_{i}": 2 for i in range(2)}
    action_spaces = anymal_action_spaces | leatherback_action_spaces
    anymal_observation_spaces = {f"anymal_{i}": 66 for i in range(2)}
    leatherback_observation_spaces = {f"leatherback_{i}": 24 for i in range(2)}
    observation_spaces = anymal_observation_spaces | leatherback_observation_spaces
    state_space = 0
    state_spaces = {robot: 0 for robot in observation_spaces.keys()}
    possible_agents = ["anymal_0", "anymal_1", "leatherback_0", "leatherback_1"]

    teams = {
        "team_0": ["anymal_0", "anymal_1"],
        "team_1": ["leatherback_0", "leatherback_1"],
    }

    sim: SimulationCfg = SimulationCfg(dt=1 / 200, render_interval=decimation)

    anymal_0: ArticulationCfg = ANYMAL_C_CFG.replace(prim_path="/World/envs/env_.*/Anymal_0")
    anymal_0.init_state.rot = get_quaternion_tuple_from_xyz(0,0,torch.pi)
    anymal_0.init_state.pos = (0, 0, .5)

    anymal_1: ArticulationCfg = ANYMAL_C_CFG.replace(prim_path="/World/envs/env_.*/Anymal_1")
    anymal_1.init_state.rot = get_quaternion_tuple_from_xyz(0,0,torch.pi)
    anymal_1.init_state.pos = (0, 0, .5)

    leatherback_0: ArticulationCfg = LEATHERBACK_CFG.replace(prim_path="/World/envs/env_.*/Leatherback_0")
    leatherback_0.init_state.rot = get_quaternion_tuple_from_xyz(0,0,torch.pi)
    leatherback_0.init_state.pos = (0, 0, .3)

    leatherback_1: ArticulationCfg = LEATHERBACK_CFG.replace(prim_path="/World/envs/env_.*/Leatherback_1")
    leatherback_1.init_state.rot = get_quaternion_tuple_from_xyz(0,0,torch.pi)
    leatherback_1.init_state.pos = (0, 0, .3)

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
            pos=(0.0, 5.0, 1), rot=(1.0, 0.0, 0.0, 0.0)
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
            pos=(0.0, -5.0, 1), rot=( 0, 0, 0, 1)
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
            pos=(10.0, 0.0, 1), rot=(1.0, 0.0, 0.0, 0.0)
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
            pos=(-10.0, 0.0, 1), rot=(1.0, 0.0, 0.0, 0.0)
        ),
    )

    ball = SOCCERBALL_CFG.replace(prim_path="/World/envs/env_.*/Object4")
    ball.init_state.pos = (0.0, 0.0, 0.1)

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

    env_spacing = 20.0
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=env_spacing, replicate_physics=True)
    viewer = ViewerCfg(eye=(10.0, 10.0, 10.0), env_index=0, origin_type="env")

    goal_reward_scale = 20
    ball_to_goal_reward_scale = 1.0
    dist_to_ball_reward_scale = 1.0

class AnymalSoccerHeteroByTeamEnv(DirectMARLEnv):
    cfg: AnymalSoccerHeteroByTeamEnvCfg

    def __init__(self, cfg: AnymalSoccerHeteroByTeamEnvCfg, render_mode: str | None = None, headless: bool | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.headless = headless

        self._throttle_dof_idx, _ = self.robots["leatherback_0"].find_joints(self.cfg.throttle_dof_name)
        self._steering_dof_idx, _ = self.robots["leatherback_0"].find_joints(self.cfg.steering_dof_name)

        self._throttle_state = {robot_id:torch.zeros((self.num_envs,4), device=self.device, dtype=torch.float32) for robot_id in self.leatherbacks.keys()}
        self._steering_state = {robot_id:torch.zeros((self.num_envs,2), device=self.device, dtype=torch.float32) for robot_id in self.leatherbacks.keys()}
        
        self.actions = {
            agent: torch.zeros(self.num_envs, action_space, device=self.device)
            for agent, action_space in self.cfg.action_spaces.items()
        }
        self.previous_actions = {
            agent: torch.zeros(self.num_envs, action_space, device=self.device)
            for agent, action_space in self.cfg.action_spaces.items()
        }
        self.penalty_box = {
            agent: torch.zeros(self.num_envs, 2, device=self.device)
            for agent  in self.robots.keys()
        }

        self.env_spacing = self.cfg.env_spacing

        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "team_0_score_reward",
                "team_0_ball_to_goal_reward",
                "team_0_timestep_reward",
                "team_1_score_reward",
                "team_1_ball_to_goal_reward",
                "team_1_timestep_reward",
            ]
        }

        marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/myMarkers",
            markers={
                    "goal1": sim_utils.CuboidCfg(
                        size=(1, 3, 0.1),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
                    ),
                    "goal2": sim_utils.CuboidCfg(
                        size=(1, 3, 0.1),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                    ),
                },
        )
        self.goal_area = VisualizationMarkers(marker_cfg)
        self.goal0_pos, self.goal1_pos, self.goal0_area, self.goal1_area = self._get_goal_areas()
        self.target_goal = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        team_dot_markers = {
            "blue": sim_utils.SphereCfg(
                radius=0.08,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.8)),
            ),
            "red": sim_utils.SphereCfg(
                radius=0.08,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.0, 0.0)),
            ),
        }
        self.team_markers = VisualizationMarkers(
            VisualizationMarkersCfg(prim_path="/World/TeamDots", markers=team_dot_markers)
        )


    def _setup_scene(self):
        self.wall_0 = RigidObject(self.cfg.wall_0)
        self.wall_1 = RigidObject(self.cfg.wall_1)
        self.wall_2 = RigidObject(self.cfg.wall_2)
        self.wall_3 = RigidObject(self.cfg.wall_3)
        self.ball = RigidObject(self.cfg.ball)

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
        # Setup rest of the scene
        self.robots = {}
        self.anymals = {}
        self.leatherbacks = {}
        self.num_robots = len(self.cfg.possible_agents)
        for agent_id in self.cfg.possible_agents:
            self.robots[agent_id] = Articulation(self.cfg.__dict__[agent_id])
            self.scene.articulations[agent_id] = self.robots[agent_id]
            if 'anymal' in agent_id:
                self.anymals[agent_id] = self.robots[agent_id]
            elif 'leatherback' in agent_id:
                self.leatherbacks[agent_id] = self.robots[agent_id]

        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])
        # Add lighting
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    @torch.no_grad()
    def _draw_team_dots(self):
        positions, indices, orientations, scales = [], [], [], []
        for team, agents in self.cfg.teams.items():
            for robot_id in agents:
                pos = self.robots[robot_id].data.root_pos_w.clone()
                pos[:, 2] += 0.5  # hover above robot
                positions.append(pos)
    
                team_color = "blue" if "0" in team else "red"
                indices.append(torch.full((self.num_envs,), 0 if team_color=="blue" else 1, device=self.device))

        marker_positions = torch.cat(positions, dim=0)
        marker_indices = torch.cat(indices, dim=0)
        marker_orientations = torch.zeros((marker_positions.shape[0], 4), device=self.device); marker_orientations[:,0]=1.0
        marker_scales = torch.ones((marker_positions.shape[0], 3), device=self.device)

        self.team_markers.visualize(marker_positions, marker_orientations, scales=marker_scales, marker_indices=marker_indices)

    def _draw_goal_areas(self):
        marker_ids0 = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        marker_ids1 = torch.ones(self.num_envs, dtype=torch.int32, device=self.device)

        marker_ids = torch.concat([marker_ids0, marker_ids1], dim=0)

        marker_locations = torch.concat([self.goal0_pos, self.goal1_pos], dim=0)

        self.goal_area.visualize(marker_locations, marker_indices=marker_ids)

    def _pre_physics_step(self, actions: dict) -> None:

        for robot_id in self.leatherbacks.keys():
            self._throttle_action = actions[robot_id][:, 0].repeat_interleave(4).reshape((-1, 4)) * self.cfg.throttle_scale
            self.throttle_action = torch.clamp(self._throttle_action, -self.cfg.throttle_max, self.cfg.throttle_max)
            self._throttle_state[robot_id] = self._throttle_action
            
            self._steering_action = actions[robot_id][:, 1].repeat_interleave(2).reshape((-1, 2)) * self.cfg.steering_scale
            self._steering_action = torch.clamp(self._steering_action, -self.cfg.steering_max, self.cfg.steering_max)
            self._steering_state[robot_id] = self._steering_action

        self.processed_actions = {}
        self.actions = copy.deepcopy(actions)
        for robot_id, robot in self.anymals.items():
            self.processed_actions[robot_id] = (
                self.cfg.action_scale * actions[robot_id] + robot.data.default_joint_pos
            )
        self.ball.update(self.step_dt)

    def _apply_action(self) -> None:
        for robot_id, robot in self.leatherbacks.items():
            robot.set_joint_velocity_target(self._throttle_state[robot_id], joint_ids=self._throttle_dof_idx)
            robot.set_joint_position_target(self._steering_state[robot_id], joint_ids=self._steering_dof_idx)

        for robot_id, robot in self.anymals.items():
            robot.set_joint_position_target(self.processed_actions[robot_id])
    
    def _get_observations(self) -> dict:
        all_obs = {}
        for team in self.cfg.teams.keys():
            all_obs[team] = {}
            for robot_id in self.cfg.teams[team]:
                if robot_id in self.anymals.keys():
                    robot_state = (
                        self.robots[robot_id].data.root_lin_vel_b,
                        self.robots[robot_id].data.root_ang_vel_b,
                        self.robots[robot_id].data.projected_gravity_b,
                        self.robots[robot_id].data.joint_pos - self.robots[robot_id].data.default_joint_pos,
                        self.robots[robot_id].data.joint_vel,
                        self.actions[robot_id],
                    )
                else:
                    robot_state = (
                        self.robots[robot_id].data.root_lin_vel_b,
                    )

                ball_pos, _ = subtract_frame_transforms(
                    self.robots[robot_id].data.root_state_w[:, :3], self.robots[robot_id].data.root_state_w[:, 3:7],
                    self.ball.data.root_pos_w
                )

                # ball velocity in robot frame
                ball_vel = quat_rotate_inverse(
                    self.robots[robot_id].data.root_state_w[:, 3:7],
                    self.ball.data.root_vel_w[:, :3] - self.robots[robot_id].data.root_vel_w[:, :3]
                )

                target_goal_pos, _ = subtract_frame_transforms(
                    self.robots[robot_id].data.root_state_w[:, :3],
                    self.robots[robot_id].data.root_state_w[:, 3:7],
                    self.goal1_pos if team == "team_0" else self.goal0_pos  
                )

                other_goal_pos, _ = subtract_frame_transforms(
                    self.robots[robot_id].data.root_state_w[:, :3],
                    self.robots[robot_id].data.root_state_w[:, 3:7],
                    self.goal0_pos if team == "team_0" else self.goal1_pos
                )

                teammate_pos, _ = subtract_frame_transforms(
                    self.robots[robot_id].data.root_state_w[:, :3],
                    self.robots[robot_id].data.root_state_w[:, 3:7],
                    self.robots[self.cfg.teams[team][1] if robot_id == self.cfg.teams[team][0] else self.cfg.teams[team][0]].data.root_pos_w
                )

                enemy_0_pos, _ = subtract_frame_transforms(
                    self.robots[robot_id].data.root_state_w[:, :3],
                    self.robots[robot_id].data.root_state_w[:, 3:7],
                    self.robots[self.cfg.teams["team_1" if team == "team_0" else "team_0"][0]].data.root_pos_w
                )
                enemy_1_pos, _ = subtract_frame_transforms(
                    self.robots[robot_id].data.root_state_w[:, :3],
                    self.robots[robot_id].data.root_state_w[:, 3:7],
                    self.robots[self.cfg.teams["team_1" if team == "team_0" else "team_0"][1]].data.root_pos_w
                )

                # buf0 = torch.zeros((self.num_envs, 9), device = self.device)



                # obs = torch.cat(
                #     robot_state + (
                #     ball_pos,  # Ball position in robot frame (3)
                #     ball_vel, # Ball velocity in robot frame (3)
                #     target_goal_pos, # Target goal position in robot frame (3)
                #     other_goal_pos,  # other goal position in robot frame (3)
                #     buf0
                # ), dim=-1)

                obs = torch.cat(
                    robot_state + (
                    ball_pos,  # Ball position in robot frame (3)
                    ball_vel, # Ball velocity in robot frame (3)
                    target_goal_pos, # Target goal position in robot frame (3)
                    other_goal_pos,  # other goal position in robot frame (3)
                    teammate_pos,  # Teammate position in robot frame (3)
                    enemy_0_pos,  # Enemy 0 position in robot frame (3)
                    enemy_1_pos,  # Enemy 1 position in robot frame (3)
                ), dim=-1)

                obs = torch.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)

                all_obs[team][robot_id] = obs

        return all_obs
    
    def _get_rewards(self) -> dict:
        self._draw_team_dots()
        ball_in_goal1, ball_in_goal2 = self._ball_in_goal_area()
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        out_of_arena = self._get_out_of_arena()

        time_step_reward = -0.01 * torch.ones(self.num_envs, device=self.device)

        # Distances to opponent’s goal
        team_0_ball_distance_to_goal = torch.linalg.norm(self.ball.data.root_pos_w - self.goal0_pos, dim=1)
        team_0_ball_distance_to_goal_mapped = 1 - torch.tanh(team_0_ball_distance_to_goal / 0.8)

        team_1_ball_distance_to_goal = torch.linalg.norm(self.ball.data.root_pos_w - self.goal0_pos, dim=1)
        team_1_ball_distance_to_goal_mapped = 1 - torch.tanh(team_1_ball_distance_to_goal / 0.8)

        # Score rewards
        team_0_score_reward = (ball_in_goal2.to(torch.float32) - ball_in_goal1.to(torch.float32)) * self.cfg.goal_reward_scale
        team_1_score_reward = (ball_in_goal1.to(torch.float32) - ball_in_goal2.to(torch.float32)) * self.cfg.goal_reward_scale

        # Team 0 rewards
        reward_team_0 = {
            "team_0_score_reward": team_0_score_reward,
            "team_0_ball_to_goal_reward": team_0_ball_distance_to_goal_mapped,
            "team_0_timestep_reward": time_step_reward,
        }
        reward_team_0 = {k: torch.nan_to_num(v, nan=0.0, posinf=1e6, neginf=-1e6)
                        for k, v in reward_team_0.items()}

        for k, v in reward_team_0.items():
            self._episode_sums[k] += v

        total_reward_team_0 = torch.sum(torch.stack(list(reward_team_0.values()), dim=0), dim=0)

        # Team 1 rewards
        reward_team_1 = {
            "team_1_score_reward": team_1_score_reward,
            "team_1_ball_to_goal_reward": team_1_ball_distance_to_goal_mapped,
            "team_1_timestep_reward": time_step_reward,
        }
        reward_team_1 = {k: torch.nan_to_num(v, nan=0.0, posinf=1e6, neginf=-1e6)
                        for k, v in reward_team_1.items()}

        for k, v in reward_team_1.items():
            self._episode_sums[k] += v

        total_reward_team_1 = torch.sum(torch.stack(list(reward_team_1.values()), dim=0), dim=0)

        return {
            "team_0": total_reward_team_0,
            "team_1": total_reward_team_1,
        }
    
    def _get_goal_areas(self):
        goal1_size = self.goal_area.cfg.markers['goal1'].size
        goal0_pos = self.scene.env_origins.clone() + torch.tensor([-9.25, 0.0, 0.05], device=self.device)

        goal2_size = self.goal_area.cfg.markers['goal2'].size
        goal1_pos = self.scene.env_origins.clone() + torch.tensor([9.25, 0.0, 0.05], device=self.device)

        # Extract goal area from goal post positions
        goal1_min = goal0_pos + torch.tensor([-goal1_size[0]/2, -goal1_size[1]/2, 0], device=self.device)
        goal1_max = goal0_pos + torch.tensor([goal1_size[0]/2, goal1_size[1]/2, 0], device=self.device)   
        goal2_min = goal1_pos + torch.tensor([-goal2_size[0]/2, -goal2_size[1]/2, 0], device=self.device)
        goal2_max = goal1_pos + torch.tensor([goal2_size[0]/2, goal2_size[1]/2, 0], device=self.device)

        return goal0_pos, goal1_pos, (goal1_min, goal1_max), (goal2_min, goal2_max)
    
    def _get_out_of_arena(self):
        out_of_arena = torch.zeros(self.num_envs, dtype=torch.int8, device=self.device)

        for robot in self.robots.values():
            out_of_arena |= robot.data.root_pos_w[:, 2] > 5
        
        return out_of_arena
    
    def _get_fallen_robots(self, robot_ids):
        fallen = {}
        for robot_id in robot_ids:
            fallen[robot_id] = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            fallen[robot_id] = self.robots[robot_id].data.root_com_pos_w[:, 2] < .09
        return fallen
    
    def _ball_in_goal_area(self):
        ball_pos = self.ball.data.root_pos_w[:, :2]
        in_goal1 = torch.all((ball_pos >= self.goal0_area[0][:,:2]) & (ball_pos <= self.goal0_area[1][:,:2]), dim=1)
        in_goal2 = torch.all((ball_pos >= self.goal1_area[0][:,:2]) & (ball_pos <= self.goal1_area[1][:,:2]), dim=1)
        return in_goal1, in_goal2

    def _get_dones(self) -> tuple[dict, dict]:
        ball_in_goal1, ball_in_goal2 = self._ball_in_goal_area()

        ball_in_any_goal = ball_in_goal1 | ball_in_goal2
        out_of_arena = self._get_out_of_arena()

        # fallen = self._get_fallen_robots(self.robots.keys())

        done = out_of_arena | ball_in_any_goal
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        dones = {team: done for team in self.cfg.teams.keys()}
        time_outs = {team: time_out for team in self.cfg.teams.keys()}

        return dones, time_outs

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        episode_lengths = self.episode_length_buf[env_ids].to(torch.float32).clone() + 1
        super()._reset_idx(env_ids)

        num_reset_envs = len(env_ids) # type: ignore
        if num_reset_envs == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(
                self.episode_length_buf, high=int(self.max_episode_length)
            )

        ball_in_goal1, ball_in_goal2 = self._ball_in_goal_area()

        team_0_percent_scored = torch.sum(ball_in_goal2.to(torch.float32)) / num_reset_envs
        team_1_percent_scored = torch.sum(ball_in_goal1.to(torch.float32)) / num_reset_envs

        self.target_goal[env_ids] = torch.randint(0, 2, (num_reset_envs,), device=self.device).to(torch.int32)

        self._draw_goal_areas()

        sampled_grid_pos = self._sample_positions_grid(env_ids, self.num_robots+1, 1, 1)

        # Cache for convenience
        origins = self.scene.env_origins[env_ids].clone()  # (N, 3)

        # We’ll assign robots sequentially
        robot_ids = list(self.robots.keys())

        ball_default_state = self.ball.data.default_root_state.clone()[env_ids]
        ball_default_state[:, :2] = ball_default_state[:, :2] + self.scene.env_origins[env_ids][:,:2] +\
        sampled_grid_pos[:, self.num_robots]
        self.ball.write_root_state_to_sim(ball_default_state, env_ids)
        self.ball.reset(env_ids)

        for i, robot_id in enumerate(robot_ids):
            # Reset robot internals
            self.robots[robot_id].reset(env_ids)
            self.actions[robot_id][env_ids] = 0.0

            # Get default states for these envs
            joint_pos = self.robots[robot_id].data.default_joint_pos[env_ids]
            joint_vel = self.robots[robot_id].data.default_joint_vel[env_ids]
            default_root_state = self.robots[robot_id].data.default_root_state[env_ids].clone()

            # Place robot
            default_root_state[:, :2] += origins[:, :2]
            default_root_state[:, :2] += sampled_grid_pos[:, i]

            # Write to sim
            self.robots[robot_id].write_root_pose_to_sim(default_root_state[:, :7], env_ids)
            self.robots[robot_id].write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
            self.robots[robot_id].write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Draw markers & reset episode logs
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids] / episode_lengths)
            extras["Episode_Reward/"+key] = episodic_sum_avg
            self._episode_sums[key][env_ids] = 0.0

        extras["Team0_Percent_Scored"] = team_0_percent_scored
        extras["Team1_Percent_Scored"] = team_1_percent_scored
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()


    def _sample_positions_grid(self, env_ids, num_samples, min_dist=1.0, grid_spacing=1.0):
        device = self.scene.env_origins.device
        N = len(env_ids)

        offsets = torch.zeros((N, num_samples, 2), device=device)
        env_origins = self.scene.env_origins[env_ids][:, :2].clone()

        _, _, goal0_area, goal1_area = self._get_goal_areas()
        goal1_min, goal1_max = goal0_area
        goal2_min, goal2_max = goal1_area

        all_valid_points = []  # collect per-env lists

        for i in range(N):
            xs = torch.arange(env_origins[i, 0] - 9, env_origins[i, 0] + 10,
                            grid_spacing, device=device)
            ys = torch.arange(env_origins[i, 1] - 4, env_origins[i, 1] + 5,
                            grid_spacing, device=device)
            xv, yv = torch.meshgrid(xs, ys, indexing="ij")
            grid_points = torch.stack([xv.flatten(), yv.flatten()], dim=-1)

            # mask out goal areas
            in_goal1 = (grid_points[:, 0] >= goal1_min[i, 0]) & (grid_points[:, 0] <= goal1_max[i, 0]) & \
                    (grid_points[:, 1] >= goal1_min[i, 1]) & (grid_points[:, 1] <= goal1_max[i, 1])
            in_goal2 = (grid_points[:, 0] >= goal2_min[i, 0]) & (grid_points[:, 0] <= goal2_max[i, 0]) & \
                    (grid_points[:, 1] >= goal2_min[i, 1]) & (grid_points[:, 1] <= goal2_max[i, 1])
            mask = ~(in_goal1 | in_goal2)

            valid_points = grid_points[mask]
            all_valid_points.append(valid_points)

            if valid_points.shape[0] < num_samples:
                raise ValueError(f"Not enough valid grid points outside goals for env {i}")

            idx = torch.randperm(valid_points.shape[0], device=device)[:num_samples]
            offsets[i] = valid_points[idx] - env_origins[i]

        # save for visualization
        self.valid_points = all_valid_points  
        # self._draw_grid_markers()

        return offsets
    

    @torch.no_grad()
    def _draw_grid_markers(self):
        """
        Draws green dots at every valid grid point for every environment
        stored in self.valid_points (populated by _sample_positions_grid).
        """
        device = self.device
        if not hasattr(self, "valid_points"):
            raise RuntimeError("Run _sample_positions_grid first to populate valid_points")

        pos_chunks = []
        idx_chunks = []
        marker_counter = 0

        for i, pts in enumerate(self.valid_points):
            z_col = torch.full((pts.shape[0], 1), 0.05, device=device)
            pos_i = torch.cat([pts, z_col], dim=1)

            pos_chunks.append(pos_i)
            idx_chunks.append(marker_counter + torch.arange(pts.shape[0],
                                                            device=device,
                                                            dtype=torch.long))
            marker_counter += pts.shape[0]

        marker_positions = torch.cat(pos_chunks, dim=0)      # (M, 3)
        marker_indices  = torch.cat(idx_chunks, dim=0)       # (M,)
        marker_scales   = 10 * torch.ones((marker_positions.shape[0], 3), device=device)

        marker_orientations = torch.zeros((marker_positions.shape[0], 4), device=device)
        marker_orientations[:, 0] = 1.0  # identity quaternion

        if not hasattr(self, "grid_markers"):
            markers = {f"grid_{i}": sim_utils.SphereCfg(
                radius=0.05,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            ) for i in range(marker_counter)}

            grid_marker_cfg = VisualizationMarkersCfg(
                prim_path="/World/GridMarkers",
                markers=markers
            )
            self.grid_markers = VisualizationMarkers(grid_marker_cfg)

        self.grid_markers.visualize(
            marker_positions,
            marker_orientations,
            scales=marker_scales,
            marker_indices=marker_indices,
        )