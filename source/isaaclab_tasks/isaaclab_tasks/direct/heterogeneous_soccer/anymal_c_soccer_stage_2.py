from __future__ import annotations

import torch
import copy
import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab_assets.robots.anymal import ANYMAL_C_CFG  # isort: skip
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.utils.math import quat_from_angle_axis, quat_from_euler_xyz, quat_rotate_inverse
from isaaclab_assets.custom.soccer_ball import SOCCERBALL_CFG  # isort: skip
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

@configclass
class AnymalStage2SoccerEnvCfg(DirectMARLEnvCfg):
    decimation = 4
    episode_length_s = 20.0
    action_scale = 0.5
    action_spaces = {f"robot_{i}": 12 for i in range(1)}
    observation_spaces = {f"robot_{i}": 66 for i in range(1)}
    state_space = 0
    state_spaces = {f"robot_{i}": 0 for i in range(1)}
    possible_agents = ["robot_0"]

    events: EventCfg = EventCfg()
    sim: SimulationCfg = SimulationCfg(dt=1 / 200, render_interval=decimation)
    robot_0: ArticulationCfg = ANYMAL_C_CFG.replace(prim_path="/World/envs/env_.*/Robot_0")
    # robot_0.init_state.rot = get_quaternion_tuple_from_xyz(0,torch.pi,0)
    robot_0.init_state.pos = (0.0, 0.0, .3)

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

    ball = SOCCERBALL_CFG.replace(prim_path="/World/envs/env_.*/Object4")
    ball.init_state.pos = (0.0, 0.0, 0.1)

    env_spacing = 20.0
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=env_spacing, replicate_physics=True)

    goal_reward_scale = 20
    ball_to_goal_reward_scale = 1.0
    dist_to_ball_reward_scale = 1.0
    ball_velocity_scale = 1.0

class AnymalStage2SoccerEnv(DirectMARLEnv):
    cfg: AnymalStage2SoccerEnvCfg

    def __init__(self, cfg: AnymalStage2SoccerEnvCfg, render_mode: str | None = None, headless: bool | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.headless = headless
        
        self.env_spacing = self.cfg.env_spacing

        self.actions = {
            agent: torch.zeros(self.num_envs, action_space, device=self.device)
            for agent, action_space in self.cfg.action_spaces.items()
        }
        self.previous_actions = {
            agent: torch.zeros(self.num_envs, action_space, device=self.device)
            for agent, action_space in self.cfg.action_spaces.items()
        }

        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                # "dist_to_ball_reward",
                "ball_velocity_reward",
                "ball_to_goal_reward",
                "goal_reward",
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
                    "goal_to_score": sim_utils.CuboidCfg(
                        size=(.5, .5, .5),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                    ),

                },
        )
        self.goal_area = VisualizationMarkers(marker_cfg)
        self.goal1_pos, self.goal2_pos, self.goal1_area, self.goal2_area = self._get_goal_areas()
        self.target_goal = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)


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

    def _draw_goal_areas(self):
        marker_ids0 = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        marker_ids1 = torch.ones(self.num_envs, dtype=torch.int32, device=self.device)
        marker_ids2 = 2 * torch.ones(self.num_envs, dtype=torch.int32, device=self.device)

        marker_ids = torch.concat([marker_ids0, marker_ids1, marker_ids2], dim=0)

        goal_to_score_pos = torch.where(self.target_goal.unsqueeze(1) == 0, self.goal1_pos, self.goal2_pos)

        marker_locations = torch.concat([self.goal1_pos, self.goal2_pos, goal_to_score_pos], dim=0)

        self.goal_area.visualize(marker_locations, marker_indices=marker_ids)

    def _pre_physics_step(self, actions: dict) -> None:
        self.processed_actions = {}
        self.actions = copy.deepcopy(actions)
        for robot_id, robot in self.robots.items():
            self.processed_actions[robot_id] = (
                self.cfg.action_scale * actions[robot_id] + robot.data.default_joint_pos
            )
        self.ball.update(self.step_dt)

    def _apply_action(self) -> None:
        for robot_id, robot in self.robots.items():
            robot.set_joint_position_target(self.processed_actions[robot_id])
    
    def _get_observations(self) -> dict:
        robot_state = (
            self.robots["robot_0"].data.root_lin_vel_b,
            self.robots["robot_0"].data.root_ang_vel_b,
            self.robots["robot_0"].data.projected_gravity_b,
            self.robots["robot_0"].data.joint_pos - self.robots["robot_0"].data.default_joint_pos,
            self.robots["robot_0"].data.joint_vel,
            self.actions["robot_0"],
        )

        ball_pos, _ = subtract_frame_transforms(
            self.robots["robot_0"].data.root_state_w[:, :3], self.robots["robot_0"].data.root_state_w[:, 3:7],
            self.ball.data.root_pos_w
        )

        # ball velocity in robot frame
        ball_vel = quat_rotate_inverse(
            self.robots["robot_0"].data.root_state_w[:, 3:7],
            self.ball.data.root_vel_w[:, :3] - self.robots["robot_0"].data.root_vel_w[:, :3]
        )

        target_goal_pos, _ = subtract_frame_transforms(
            self.robots["robot_0"].data.root_state_w[:, :3],
            self.robots["robot_0"].data.root_state_w[:, 3:7],
            torch.where(self.target_goal.unsqueeze(1) == 0, self.goal1_pos, self.goal2_pos)
        )

        other_goal_pos, _ = subtract_frame_transforms(
            self.robots["robot_0"].data.root_state_w[:, :3],
            self.robots["robot_0"].data.root_state_w[:, 3:7],
            torch.where(self.target_goal.unsqueeze(1) == 0, self.goal2_pos, self.goal1_pos)
        )

        teammate_buffer = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        enemy_0_buffer = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        enemy_1_buffer = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)

        obs = torch.cat(
            robot_state + (
            ball_pos,  # Ball position in robot frame (3)
            ball_vel, # Ball velocity in robot frame (3)
            target_goal_pos, # Target goal position in robot frame (3)
            other_goal_pos,  # other goal position in robot frame (3)
            teammate_buffer,  # Teammate position in robot frame (3)
            enemy_0_buffer,  # Enemy 0 position in robot frame (3)
            enemy_1_buffer,  # Enemy 1 position in robot frame (3)
        ), dim=-1)

        obs = torch.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)

        return {"robot_0": obs}
    
    def _get_rewards(self) -> dict:
        ball_in_goal1, ball_in_goal2 = self._ball_in_goal_area()

        goal_pos = torch.zeros_like(self.goal1_pos)

        goal_pos[self.target_goal == 0] = self.goal1_pos[self.target_goal == 0]
        goal_pos[self.target_goal == 1] = self.goal2_pos[self.target_goal == 1]

        ball_distance_to_goal = torch.linalg.norm(self.ball.data.root_pos_w - goal_pos, dim=1)
        ball_distance_to_goal_mapped = 1 - torch.tanh(ball_distance_to_goal / .8)

        ball_vel = torch.norm(self.ball.data.root_lin_vel_w, dim=1)
        ball_vel_reward = torch.tanh(ball_vel / .8)

        # robot_distance_to_ball = torch.linalg.norm(self.robots["robot_0"].data.root_pos_w[:, :3] - self.ball.data.root_pos_w, dim=1)
        # robot_distance_to_ball_mapped = 1 - torch.tanh(robot_distance_to_ball / .8)
        
        goal_reward = torch.zeros(self.num_envs, device=self.device)
        # Reward is 1 if ball is in target goal area, if in other goal area, reward is -1
        goal_reward[ball_in_goal1 & (self.target_goal == 0)] = 1.0
        goal_reward[ball_in_goal2 & (self.target_goal == 1)] = 1.0
        goal_reward[ball_in_goal1 & (self.target_goal == 1)] = -1.0
        goal_reward[ball_in_goal2 & (self.target_goal == 0)] = -1.0

        rewards = {
            # "dist_to_ball_reward": robot_distance_to_ball_mapped * self.cfg.dist_to_ball_reward_scale * self.step_dt,
            "ball_velocity_reward": ball_vel_reward  * self.cfg.ball_velocity_scale * self.step_dt,
            "ball_to_goal_reward": ball_distance_to_goal_mapped  * self.cfg.ball_to_goal_reward_scale * self.step_dt,
            "goal_reward": goal_reward * self.cfg.goal_reward_scale,
        }

        rewards = {k: torch.nan_to_num(v, nan=0.0, posinf=1e6, neginf=-1e6)
                for k, v in rewards.items()}

        reward = torch.sum(torch.stack([rewards[key] for key in rewards.keys()]), dim=0)

        for key in rewards.keys():
            self._episode_sums[key] += rewards[key]

        return {"robot_0": reward}
    
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

        fallen = self.robots["robot_0"].data.root_com_pos_w[:, 2] < .1
        dones = ball_in_any_goal | fallen

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return {"robot_0": dones}, {"robot_0": time_out}

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        episode_lengths = self.episode_length_buf[env_ids].to(torch.float32).clone() + 1
        super()._reset_idx(env_ids)

        num_reset_ids = len(env_ids) # type: ignore
        if num_reset_ids == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(
                self.episode_length_buf, high=int(self.max_episode_length)
            )

        ball_in_goal1, ball_in_goal2 = self._ball_in_goal_area()

        target_goals = self.target_goal[env_ids]

        goals_scored = ((ball_in_goal1[env_ids] & (target_goals == 0))
              | (ball_in_goal2[env_ids] & (target_goals == 1))).to(torch.float32).sum().item()
        
        percent_scored = goals_scored / num_reset_ids if num_reset_ids > 0 else 0

        self.target_goal[env_ids] = torch.randint(0, 2, (num_reset_ids,), device=self.device).to(torch.int32)

        self._draw_goal_areas()

        sampled_grid_pos = self._sample_positions_grid(env_ids, 2, 1, 1)

        # Cache for convenience
        origins = self.scene.env_origins[env_ids]  # (N, 3)

        # We’ll assign robots sequentially
        robot_ids = list(self.robots.keys())

        ball_default_state = self.ball.data.default_root_state.clone()[env_ids]
        ball_default_state[:, :2] = ball_default_state[:, :2] + self.scene.env_origins[env_ids][:,:2] +\
        sampled_grid_pos[:, 1]
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
            default_root_state[:, :2] = origins[:, :2]
            default_root_state[:, 2] += self.robots[robot_id].data.default_root_state[env_ids][:, 2]
            default_root_state[:, :2] += sampled_grid_pos[:, 0]

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

        extras["Percent_Scored"] = percent_scored
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()


    def _sample_positions_grid(self, env_ids, num_samples, min_dist=1.0, grid_spacing=1.0):
        device = self.scene.env_origins.device
        N = len(env_ids)

        offsets = torch.zeros((N, num_samples, 2), device=device)
        env_origins = self.scene.env_origins[env_ids][:, :2].clone()

        _, _, goal1_area, goal2_area = self._get_goal_areas()
        goal1_min, goal1_max = goal1_area
        goal2_min, goal2_max = goal2_area

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