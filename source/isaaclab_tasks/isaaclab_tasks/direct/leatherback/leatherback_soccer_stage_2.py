from __future__ import annotations

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab_assets.robots.leatherback import LEATHERBACK_CFG  # isort: skip
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
class LeatherbackStage2AdversarialSoccerEnvCfg(DirectMARLEnvCfg):
    decimation = 4
    episode_length_s = 30.0
    action_spaces = {f"robot_{i}": 2 for i in range(4)}
    observation_spaces = {f"robot_{i}": 24 for i in range(4)}
    state_space = 0
    state_spaces = {f"robot_{i}": 0 for i in range(4)}
    possible_agents = ["robot_0", "robot_1", "robot_2", "robot_3"]

    teams = {
        "team_0": ["robot_0", "robot_1"],
        "team_1": ["robot_2", "robot_3"],
    }

    sim: SimulationCfg = SimulationCfg(dt=1 / 200, render_interval=decimation)
    robot_0: ArticulationCfg = LEATHERBACK_CFG.replace(prim_path="/World/envs/env_.*/Robot_0")
    robot_0.init_state.pos = (-1.0, -2.0, .1)
    robot_1: ArticulationCfg = LEATHERBACK_CFG.replace(prim_path="/World/envs/env_.*/Robot_1")
    robot_1.init_state.pos = (-1.0, 2.0, .1)
    robot_2: ArticulationCfg = LEATHERBACK_CFG.replace(prim_path="/World/envs/env_.*/Robot_2")
    robot_2.init_state.pos = (1.0, -2.0, .1)
    robot_3: ArticulationCfg = LEATHERBACK_CFG.replace(prim_path="/World/envs/env_.*/Robot_3")
    robot_3.init_state.pos = (1.0, 2.0, .1)

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

    env_spacing = 20.0
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=env_spacing, replicate_physics=True)
    viewer = ViewerCfg(eye=(10.0, 10.0, 10.0), env_index=0, origin_type="env")

    throttle_scale = 10
    throttle_max = 50
    steering_scale = 0.1
    steering_max = 0.75

    goal_reward_scale = 20
    ball_to_goal_reward_scale = 1.0
    dist_to_ball_reward_scale = 1.0

class LeatherbackStage2AdversarialSoccerEnv(DirectMARLEnv):
    cfg: LeatherbackStage2AdversarialSoccerEnvCfg

    def __init__(self, cfg: LeatherbackStage2AdversarialSoccerEnvCfg, render_mode: str | None = None, headless: bool | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.headless = headless
        
        self._throttle_dof_idx, _ = self.robots["robot_0"].find_joints(self.cfg.throttle_dof_name)
        self._steering_dof_idx, _ = self.robots["robot_0"].find_joints(self.cfg.steering_dof_name)

        self._throttle_state = {robot_id:torch.zeros((self.num_envs,4), device=self.device, dtype=torch.float32) for robot_id in self.robots.keys()}
        self._steering_state = {robot_id:torch.zeros((self.num_envs,2), device=self.device, dtype=torch.float32) for robot_id in self.robots.keys()}

        self.env_spacing = self.cfg.env_spacing


        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "goal_reward_team0",
                "goal_reward_team1",
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
        self.goal1_pos, self.goal2_pos, self.goal1_area, self.goal2_area = self._get_goal_areas()
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
        self.num_robots = sum(1 for key in self.cfg.__dict__.keys() if "robot_" in key)
        for i in range(self.num_robots):
            self.robots[f"robot_{i}"] = Articulation(self.cfg.__dict__["robot_" + str(i)])
            self.scene.articulations[f"robot_{i}"] = self.robots[f"robot_{i}"]

        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])
        # Add lighting
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    @torch.no_grad()
    def _draw_team_dots(self):
        positions, indices, orientations, scales = [], [], [], []
        for robot_id, robot in self.robots.items():
            pos = robot.data.root_pos_w.clone()
            pos[:, 2] += 0.5  # hover above robot
            positions.append(pos)

            team = "blue" if "0" in robot_id or "1" in robot_id else "red"
            indices.append(torch.full((self.num_envs,), 0 if team=="blue" else 1, device=self.device))

        marker_positions = torch.cat(positions, dim=0)
        marker_indices = torch.cat(indices, dim=0)
        marker_orientations = torch.zeros((marker_positions.shape[0], 4), device=self.device); marker_orientations[:,0]=1.0
        marker_scales = torch.ones((marker_positions.shape[0], 3), device=self.device)

        self.team_markers.visualize(marker_positions, marker_orientations, scales=marker_scales, marker_indices=marker_indices)

    def _draw_goal_areas(self):
        marker_ids0 = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        marker_ids1 = torch.ones(self.num_envs, dtype=torch.int32, device=self.device)

        marker_ids = torch.concat([marker_ids0, marker_ids1], dim=0)

        marker_locations = torch.concat([self.goal1_pos, self.goal2_pos], dim=0)

        self.goal_area.visualize(marker_locations, marker_indices=marker_ids)

    def _pre_physics_step(self, actions: dict) -> None:

        for robot_id in self.robots.keys():
            self._throttle_action = actions[robot_id][:, 0].repeat_interleave(4).reshape((-1, 4)) * self.cfg.throttle_scale
            self._throttle_action = torch.clamp(self._throttle_action, -self.cfg.throttle_max, self.cfg.throttle_max)
            self._throttle_state[robot_id] = self._throttle_action
            
            self._steering_action = actions[robot_id][:, 1].repeat_interleave(2).reshape((-1, 2)) * self.cfg.steering_scale
            self._steering_action = torch.clamp(self._steering_action, -self.cfg.steering_max, self.cfg.steering_max)
            self._steering_state[robot_id] = self._steering_action
            self.ball.update(self.step_dt)

    def _apply_action(self) -> None:
        for robot_id in self.robots.keys():
            # self._throttle_state[robot_id] = -5*torch.ones_like(self._throttle_state[robot_id], device=self.device)
            self.robots[robot_id].set_joint_velocity_target(self._throttle_state[robot_id], joint_ids=self._throttle_dof_idx)
            self.robots[robot_id].set_joint_position_target(self._steering_state[robot_id], joint_ids=self._steering_dof_idx)


    def relative_velocity(obj_vel_w, ref_rot_w):
        """Convert world-frame velocity into reference-frame coordinates.
        
        Args:
            obj_vel_w (torch.Tensor): (E, 3) linear velocity of object in world frame
            ref_rot_w (torch.Tensor): (E, 4) quaternion of reference orientation in world frame
        
        Returns:
            torch.Tensor: (E, 3) velocity in the reference's local frame
        """
        return quat_rotate_inverse(ref_rot_w, obj_vel_w)
    
    def _get_observations(self) -> dict:
        all_obs = {}
        for team in self.cfg.teams.keys():
            all_obs[team] = {}
            for robot_id in self.cfg.teams[team]:
                robot_vel = self.robots[robot_id].data.root_lin_vel_b

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
                    self.goal2_pos if team == "team_0" else self.goal1_pos  
                )

                other_goal_pos, _ = subtract_frame_transforms(
                    self.robots[robot_id].data.root_state_w[:, :3],
                    self.robots[robot_id].data.root_state_w[:, 3:7],
                    self.goal1_pos if team == "team_0" else self.goal2_pos
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

                obs = torch.cat((
                    robot_vel, # Robot velocity in world frame (3)
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

        team_0_reward = (ball_in_goal2.to(torch.int8) - ball_in_goal1.to(torch.int8)) * (time_out.to(torch.int8) ^ 1)
        team_1_reward = ball_in_goal1.to(torch.int8) - ball_in_goal2.to(torch.int8) * (time_out.to(torch.int8) ^ 1)

        rewards = {
            "team_0": team_0_reward * self.cfg.goal_reward_scale,
            "team_1": team_1_reward * self.cfg.goal_reward_scale,
        }

        rewards = {k: torch.nan_to_num(v, nan=0.0, posinf=1e6, neginf=-1e6)
                for k, v in rewards.items()}
        
        self._episode_sums["goal_reward_team0"] += rewards["team_0"]
        self._episode_sums["goal_reward_team1"] += rewards["team_1"]

        return rewards
    
    def _get_goal_areas(self):
        goal1_size = self.goal_area.cfg.markers['goal1'].size
        goal1_pos = self.scene.env_origins.clone() + torch.tensor([-9, 0.0, 0.05], device=self.device)

        goal2_size = self.goal_area.cfg.markers['goal2'].size
        goal2_pos = self.scene.env_origins.clone() + torch.tensor([9, 0.0, 0.05], device=self.device)

        # Extract goal area from goal post positions
        goal1_min = goal1_pos + torch.tensor([-goal1_size[0]/2, -goal1_size[1]/2, 0], device=self.device)
        goal1_max = goal1_pos + torch.tensor([goal1_size[0]/2, goal1_size[1]/2, 0], device=self.device)   
        goal2_min = goal2_pos + torch.tensor([-goal2_size[0]/2, -goal2_size[1]/2, 0], device=self.device)
        goal2_max = goal2_pos + torch.tensor([goal2_size[0]/2, goal2_size[1]/2, 0], device=self.device)

        return goal1_pos, goal2_pos, (goal1_min, goal1_max), (goal2_min, goal2_max)
    
    def _ball_in_goal_area(self):
        ball_pos = self.ball.data.root_pos_w[:, :2]
        in_goal1 = torch.all((ball_pos >= self.goal1_area[0][:,:2]) & (ball_pos <= self.goal1_area[1][:,:2]), dim=1)
        in_goal2 = torch.all((ball_pos >= self.goal2_area[0][:,:2]) & (ball_pos <= self.goal2_area[1][:,:2]), dim=1)
        return in_goal1, in_goal2

    def _get_dones(self) -> tuple[dict, dict]:
        ball_in_goal1, ball_in_goal2 = self._ball_in_goal_area()

        ball_in_any_goal = ball_in_goal1 | ball_in_goal2

        time_out = self.episode_length_buf >= self.max_episode_length - 1


        dones = {team: ball_in_any_goal for team in self.cfg.teams.keys()}
        time_outs = {team: time_out for team in self.cfg.teams.keys()}

        return dones, time_outs

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        episode_lengths = self.episode_length_buf[env_ids].to(torch.float32).clone() + 1
        super()._reset_idx(env_ids)

        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(
                self.episode_length_buf, high=int(self.max_episode_length)
            )

        ball_in_goal1, ball_in_goal2 = self._ball_in_goal_area()

        team_0_percent_scored = torch.sum(ball_in_goal2.to(torch.float32)) / len(env_ids)
        team_1_percent_scored = torch.sum(ball_in_goal1.to(torch.float32)) / len(env_ids)

        self.target_goal[env_ids] = torch.randint(0, 2, (len(env_ids),), device=self.device).to(torch.int32)

        self._draw_goal_areas()

        # Cache for convenience
        origins = self.scene.env_origins[env_ids]  # (N, 3)

        # We’ll assign robots sequentially
        robot_ids = list(self.robots.keys())

        ball_default_state = self.ball.data.default_root_state.clone()[env_ids]
        ball_default_state[:, :2] = ball_default_state[:, :2] + self.scene.env_origins[env_ids][:,:2]
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
            default_root_state[:, :2] = origins[:, :2] + torch.zeros_like(default_root_state[:, :2], device=self.device).uniform_(-3, 3)

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
