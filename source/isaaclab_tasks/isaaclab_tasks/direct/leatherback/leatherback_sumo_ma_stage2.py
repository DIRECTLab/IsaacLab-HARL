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
from isaaclab.utils.math import quat_from_angle_axis, quat_from_euler_xyz

def get_quaternion_tuple_from_xyz(x, y, z):
    quat_tensor = quat_from_euler_xyz(torch.tensor([x]), torch.tensor([y]), torch.tensor([z])).flatten()
    return (quat_tensor[0].item(), quat_tensor[1].item(), quat_tensor[2].item(), quat_tensor[3].item())

@configclass
class LeatherbackSumoMAStage2EnvCfg(DirectMARLEnvCfg):
    decimation = 4
    episode_length_s = 30.0

    # 4 robots, each with throttle + steering (2 actions)
    action_spaces = {f"robot_{i}": 2 for i in range(4)}

    # Observation: teammate (3) + opp1 (3) + opp2 (3) + rcol(1) + dist_center(1) + velocity(3)
    # = 14 per robot
    observation_spaces = {f"robot_{i}": 14 for i in range(4)}

    state_space = 0
    state_spaces = {f"robot_{i}": 0 for i in range(4)}

    possible_agents = [f"robot_{i}" for i in range(4)]

    # Teams
    teams = {
        "team_0": ["robot_0", "robot_1"],
        "team_1": ["robot_2", "robot_3"],
    }

    sim: SimulationCfg = SimulationCfg(dt=1 / 200, render_interval=decimation)

    # Robot configs (prim paths unique per robot)
    robot_0: ArticulationCfg = LEATHERBACK_CFG.replace(prim_path="/World/envs/env_.*/Robot_0")
    robot_0.init_state.rot = get_quaternion_tuple_from_xyz(0, 0, -torch.pi/2)
    robot_0.init_state.pos = (0.0, 0.25, 0.1)

    robot_1: ArticulationCfg = LEATHERBACK_CFG.replace(prim_path="/World/envs/env_.*/Robot_1")
    robot_1.init_state.rot = get_quaternion_tuple_from_xyz(0, 0, -torch.pi/2)
    robot_1.init_state.pos = (0.0, 0.75, 0.1)

    robot_2: ArticulationCfg = LEATHERBACK_CFG.replace(prim_path="/World/envs/env_.*/Robot_2")
    robot_2.init_state.rot = get_quaternion_tuple_from_xyz(0, 0, torch.pi/2)
    robot_2.init_state.pos = (0.0, -0.25, 0.1)

    robot_3: ArticulationCfg = LEATHERBACK_CFG.replace(prim_path="/World/envs/env_.*/Robot_3")
    robot_3.init_state.rot = get_quaternion_tuple_from_xyz(0, 0, torch.pi/2)
    robot_3.init_state.pos = (0.0, -0.75, 0.1)

    # DOF mappings (same for all robots)
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

    env_spacing = 10.0
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=env_spacing, replicate_physics=True)

    throttle_scale = 10
    throttle_max = 50
    steering_scale = 0.1
    steering_max = 0.75

    ring_radius_min = 1
    ring_radius_max = 5
    reward_scale = 10
    # time penalty
    time_penalty = -0.01


class LeatherbackSumoMAStage2Env(DirectMARLEnv):
    cfg: LeatherbackSumoMAStage2EnvCfg

    def __init__(self, cfg: LeatherbackSumoMAStage2EnvCfg, render_mode: str | None = None, headless: bool | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.headless = headless
        
        self._throttle_dof_idx, _ = self.robots["robot_0"].find_joints(self.cfg.throttle_dof_name)
        self._steering_dof_idx, _ = self.robots["robot_0"].find_joints(self.cfg.steering_dof_name)

        self._throttle_state = {robot_id:torch.zeros((self.num_envs,4), device=self.device, dtype=torch.float32) for robot_id in self.robots.keys()}
        self._steering_state = {robot_id:torch.zeros((self.num_envs,2), device=self.device, dtype=torch.float32) for robot_id in self.robots.keys()}

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

        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "other_dist_from_center_reward",
                "dist_to_other_robot_reward",
                "other_dist_from_center",
                "dist_to_other_robot",
                "time_out_reward",
                "push_out_reward"
            ]
        }

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

    def _pre_physics_step(self, actions: dict) -> None:
        for robot_id in self.robots.keys():
            self._throttle_action = actions[robot_id][:, 0].repeat_interleave(4).reshape((-1, 4)) * self.cfg.throttle_scale
            self.throttle_action = torch.clamp(self._throttle_action, -self.cfg.throttle_max, self.cfg.throttle_max)
            self._throttle_state[robot_id] = self._throttle_action
            
            self._steering_action = actions[robot_id][:, 1].repeat_interleave(2).reshape((-1, 2)) * self.cfg.steering_scale
            self._steering_action = torch.clamp(self._steering_action, -self.cfg.steering_max, self.cfg.steering_max)
            self._steering_state[robot_id] = self._steering_action

    def _apply_action(self) -> None:
        for robot_id in self.robots.keys():
            # self._throttle_state[robot_id] = -5*torch.ones_like(self._throttle_state[robot_id], device=self.device)
            self.robots[robot_id].set_joint_velocity_target(self._throttle_state[robot_id], joint_ids=self._throttle_dof_idx)
            self.robots[robot_id].set_joint_position_target(self._steering_state[robot_id], joint_ids=self._steering_dof_idx)

    def _get_observations(self) -> dict:
        obs = {}
        for team_name, robots in self.cfg.teams.items():
            team_obs = {}
            for i, robot_id in enumerate(robots):
                teammate_id = robots[1-i]
                enemies = [r for r in self.robots.keys() if r not in robots]

                # relative positions
                teammate_pos, _ = subtract_frame_transforms(
                    self.robots[robot_id].data.root_state_w[:, :3],
                    self.robots[robot_id].data.root_state_w[:, 3:7],
                    self.robots[teammate_id].data.root_pos_w,
                )
                enemy_pos = []
                for enemy_id in enemies:
                    epos, _ = subtract_frame_transforms(
                        self.robots[robot_id].data.root_state_w[:, :3],
                        self.robots[robot_id].data.root_state_w[:, 3:7],
                        self.robots[enemy_id].data.root_pos_w,
                    )
                    enemy_pos.append(epos)

                # center dist + vel
                dist_center = torch.norm(self.robots[robot_id].data.root_pos_w - self.scene.env_origins, dim=-1, keepdim=True)
                vel = self.robots[robot_id].data.root_lin_vel_b
                rcol = self.ring_radius.view(-1,1)

                obs_vec = torch.cat([teammate_pos] + enemy_pos + [rcol, dist_center, vel], dim=1)
                team_obs[robot_id] = obs_vec
            obs[team_name] = team_obs
        return obs
    
    def _get_rewards(self) -> dict:
        self._draw_team_dots()

        circle_centers = self.scene.env_origins

        # --- Distances from center (per robot) ---
        team0_dists = torch.stack([
            torch.norm(self.robots["robot_0"].data.root_pos_w - circle_centers, dim=-1),
            torch.norm(self.robots["robot_1"].data.root_pos_w - circle_centers, dim=-1),
        ], dim=1)  # (E, 2)
        team1_dists = torch.stack([
            torch.norm(self.robots["robot_2"].data.root_pos_w - circle_centers, dim=-1),
            torch.norm(self.robots["robot_3"].data.root_pos_w - circle_centers, dim=-1),
        ], dim=1)  # (E, 2)

        # Team averages
        team0_avg_dist = team0_dists.mean(dim=1)  # (E,)
        team1_avg_dist = team1_dists.mean(dim=1)  # (E,)

        # Map through tanh like before
        team0_avg_mapped = torch.tanh(team0_avg_dist / 0.8)
        team1_avg_mapped = torch.tanh(team1_avg_dist / 0.8)

        # --- Inter-team distances (average of pairwise) ---
        team0_positions = torch.stack([
            self.robots["robot_0"].data.root_pos_w,
            self.robots["robot_1"].data.root_pos_w,
        ], dim=1)  # (E, 2, 3)
        team1_positions = torch.stack([
            self.robots["robot_2"].data.root_pos_w,
            self.robots["robot_3"].data.root_pos_w,
        ], dim=1)  # (E, 2, 3)

        # Compute all pairwise distances between robots of different teams
        diff = team0_positions.unsqueeze(2) - team1_positions.unsqueeze(1)  # (E, 2, 2, 3)
        pairwise_dists = torch.norm(diff, dim=-1)  # (E, 2, 2)
        avg_team_dist = pairwise_dists.mean(dim=(1, 2))  # (E,)

        # Map like before
        avg_team_dist_mapped = 1 - torch.tanh(avg_team_dist / 0.8)

        # --- Time penalty ---
        time_penalty = self.cfg.time_penalty * torch.ones_like(avg_team_dist, device=self.device)

        # --- Push out reward (adversarial) ---
        out = self._robots_out_of_ring()
        team0_out = torch.any(torch.stack([out["robot_0"], out["robot_1"]]), dim=0)
        team1_out = torch.any(torch.stack([out["robot_2"], out["robot_3"]]), dim=0)

        team0_lost = team0_out.to(torch.float32)
        team1_lost = team1_out.to(torch.float32)
        push_out_reward_0 = (team1_lost - team0_lost) * self.cfg.reward_scale
        push_out_reward_1 = (team0_lost - team1_lost) * self.cfg.reward_scale

        # --- Combine rewards ---
        rewards_team0 = (
            (-team0_avg_mapped + team1_avg_mapped) * self.step_dt   # other team further
            + (-avg_team_dist_mapped) * self.step_dt               # closer to opponents
            + time_penalty
            + push_out_reward_0
        )
        rewards_team1 = (
            (-team1_avg_mapped + team0_avg_mapped) * self.step_dt
            + (-avg_team_dist_mapped) * self.step_dt
            + time_penalty
            + push_out_reward_1
        )

        # Log components
        self._episode_sums["other_dist_from_center_reward"] += (team1_avg_mapped - team0_avg_mapped) * self.step_dt
        self._episode_sums["dist_to_other_robot_reward"] += avg_team_dist_mapped * self.step_dt
        self._episode_sums["time_out_reward"] += time_penalty
        self._episode_sums["push_out_reward"] += push_out_reward_0 + push_out_reward_1
        self._episode_sums["other_dist_from_center"] = torch.cat([team0_avg_dist.unsqueeze(1), team1_avg_dist.unsqueeze(1)], dim=1).mean(dim=1)
        self._episode_sums["dist_to_other_robot"] = avg_team_dist

        return {
            "team_0": rewards_team0,
            "team_1": rewards_team1,
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
        out_map = self._robots_out_of_ring()
        team0_out = torch.any(torch.stack([out_map["robot_0"], out_map["robot_1"]]), dim=0)
        team1_out = torch.any(torch.stack([out_map["robot_2"], out_map["robot_3"]]), dim=0)

        done = team0_out | team1_out
        dones = {team: done for team in self.cfg.teams.keys()}

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        timeouts = {team: time_out for team in self.cfg.teams.keys()}
        return dones, timeouts

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        super()._reset_idx(env_ids)

        # spread out the updates
        if len(env_ids) == self.num_envs:
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        # Randomize ring radius per env
        low, high = self.cfg.ring_radius_min, self.cfg.ring_radius_max
        self.ring_radius[env_ids] = (
            torch.empty(env_ids.shape[0], device=self.device).uniform_(low, high)
        )

        origins = self.scene.env_origins[env_ids]  # (N, 3)
        N = env_ids.shape[0]

        # Teams
        team0 = ["robot_0", "robot_1"]
        team1 = ["robot_2", "robot_3"]
        all_teams = [team0, team1]

        # spacing for teammates
        teammate_spacing = 0.5  # meters apart

        for team in all_teams:
            # sample a random base angle for the team
            theta = 2.0 * torch.pi * torch.rand(N, device=self.device)
            r = 0.5 * self.ring_radius[env_ids]  # place near half radius (inside ring)

            # base position for team center
            base_offsets = torch.zeros((N, 3), device=self.device)
            base_offsets[:, 0] = r * torch.cos(theta)
            base_offsets[:, 1] = r * torch.sin(theta)

            # tangent direction (perpendicular to radial vector) for teammate spacing
            tangent = torch.zeros((N, 2), device=self.device)
            tangent[:, 0] = -torch.sin(theta)
            tangent[:, 1] = torch.cos(theta)

            # offsets for teammates
            teammate_offsets = [
                base_offsets.clone(),
                base_offsets.clone()
            ]
            teammate_offsets[0][:, 0:2] += 0.5 * teammate_spacing * tangent
            teammate_offsets[1][:, 0:2] -= 0.5 * teammate_spacing * tangent

            # assign to robots
            for robot_id, offsets in zip(team, teammate_offsets):
                self.robots[robot_id].reset(env_ids)
                self.actions[robot_id][env_ids] = 0.0

                joint_pos = self.robots[robot_id].data.default_joint_pos[env_ids]
                joint_vel = self.robots[robot_id].data.default_joint_vel[env_ids]
                default_root_state = self.robots[robot_id].data.default_root_state[env_ids].clone()

                default_root_state[:, :3] = origins
                default_root_state[:, 0:2] += offsets[:, 0:2]
                default_root_state[:, 2] += self.robots[robot_id].data.default_root_state[env_ids][:, 2]

                self.robots[robot_id].write_root_pose_to_sim(default_root_state[:, :7], env_ids)
                self.robots[robot_id].write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
                self.robots[robot_id].write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Draw markers & reset episode logs
        self._draw_ring_markers()
        self._draw_team_dots()
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras[key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
