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


    throttle_scale = 10
    throttle_max = 50
    steering_scale = 0.1
    steering_max = 0.75

    ring_radius_min = 3
    ring_radius_max = 6
    reward_scale = 10
    # time penalty
    time_penalty = -0.01

    env_spacing = 10.0
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=ring_radius_max * 2, replicate_physics=True)

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


        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "team_0_push_out_reward",
                "team_1_push_out_reward"
            ]
        }


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
            # self._throttle_state[robot_id] = torch.zeros_like(self._throttle_state[robot_id], device=self.device)
            self.robots[robot_id].set_joint_velocity_target(self._throttle_state[robot_id], joint_ids=self._throttle_dof_idx)
            self.robots[robot_id].set_joint_position_target(self._steering_state[robot_id], joint_ids=self._steering_dof_idx)

    def _get_observations(self) -> dict:
        obs = {}
        rcol = self.ring_radius.view(-1, 1)

        for i, (team, agents) in enumerate(self.cfg.teams.items()):
            obs[team] = {}
            for robot_id in agents:
                teammate_id = [a for a in agents if a != robot_id][0]
                enemy_team = [a for a in self.cfg.teams.keys() if a != team][0]
                enemies = self.cfg.teams[enemy_team]

                enemy_0_id = enemies[0]
                enemy_1_id = enemies[1]

                enemy_0_pos, _ = subtract_frame_transforms(
                    self.robots[robot_id].data.root_state_w[:, :3], self.robots[robot_id].data.root_state_w[:, 3:7],
                    self.robots[enemy_0_id].data.root_pos_w
                )

                teammate_pos, _ = subtract_frame_transforms(
                    self.robots[robot_id].data.root_state_w[:, :3], self.robots[robot_id].data.root_state_w[:, 3:7],
                    self.robots[teammate_id].data.root_pos_w
                )

                enemy_1_pos, _ = subtract_frame_transforms(
                    self.robots[robot_id].data.root_state_w[:, :3], self.robots[robot_id].data.root_state_w[:, 3:7],
                    self.robots[enemy_1_id].data.root_pos_w
                )

                dist_center = torch.norm(
                    self.robots[robot_id].data.root_pos_w - self.scene.env_origins, dim=-1, keepdim=True
                )

                robot_vel = self.robots[robot_id].data.root_lin_vel_b

                robot_obs = torch.cat([enemy_0_pos, teammate_pos, enemy_1_pos, dist_center, robot_vel, rcol], dim=1)
                robot_obs = torch.nan_to_num(robot_obs, nan=0.0, posinf=1e6, neginf=-1e6)

                obs[team][robot_id] = robot_obs

        return obs
    
    def _get_rewards(self) -> dict:
        self._draw_team_dots()

        out = self._robots_out_of_ring()
        team0_out = torch.any(torch.stack([out["robot_0"], out["robot_1"]]), dim=0)
        team1_out = torch.any(torch.stack([out["robot_2"], out["robot_3"]]), dim=0)

        team0_lost = team0_out.to(torch.float32)
        team1_lost = team1_out.to(torch.float32)
        time_out = (self.episode_length_buf >= self.max_episode_length - 1).to(torch.int8)
        
        push_out_reward_0 = (team1_lost - team0_lost - time_out) * self.cfg.reward_scale
        push_out_reward_1 = (team0_lost - team1_lost - time_out) * self.cfg.reward_scale

        self._episode_sums["team_0_push_out_reward"] += push_out_reward_0
        self._episode_sums["team_1_push_out_reward"] += push_out_reward_1

        return {
            "team_0": push_out_reward_0,
            "team_1": push_out_reward_1,
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

        self.team0_win = team1_out
        self.team1_win = team0_out

        done = team0_out | team1_out
        dones = {team: done for team in self.cfg.teams.keys()}

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        timeouts = {team: time_out for team in self.cfg.teams.keys()}
        return dones, timeouts

    def _reset_idx(self, env_ids: torch.Tensor | None):
        avg_episode_length = torch.mean(self.episode_length_buf[env_ids].to(torch.float32) + 1)

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

        grid_offsets = self._sample_positions_grid(N, self.ring_radius[env_ids], self.num_robots)

        for i, robot_id in enumerate(self.robots):
            default_root_state = self.robots[robot_id].data.default_root_state[env_ids].clone()
            default_root_state[:, :3] = origins
            default_root_state[:, 0:2] += grid_offsets[:, i, 0:2]
            self.robots[robot_id].write_root_pose_to_sim(default_root_state[:, :7], env_ids)
            self.robots[robot_id].write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)

        self._draw_ring_markers()
        self._draw_team_dots()
        extras = dict()
        
        if hasattr(self, "team0_win"):
            extras["team0_win_percentage"] = (torch.count_nonzero(self.team0_win[env_ids]).item() / N) if N > 0 else 0
            extras["team1_win_percentage"] = (torch.count_nonzero(self.team1_win[env_ids]).item() / N) if N > 0 else 0

        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/"+key] = episodic_sum_avg / avg_episode_length
            self._episode_sums[key][env_ids] = 0.0

        self.extras["log"] = dict()
        self.extras["log"].update(extras)


    def _sample_positions_grid(self, N, radii, num_samples, min_dist=0.6, grid_spacing=0.5):
        """
        Sample positions from a grid inside the circle.

        Args:
            N: number of environments
            radii: tensor of ring radii (N,)
            num_samples: number of positions needed per environment
            min_dist: minimum allowed spacing (still enforced between chosen grid spots)
            grid_spacing: spacing between candidate grid points
        Returns:
            (N, num_samples, 3) tensor of sampled positions
        """
        device = radii.device
        offsets = torch.zeros((N, num_samples, 3), device=device)

        for i in range(N):
            r = radii[i].item()

            # Build grid within bounding square [-r, r]
            xs = torch.arange(-r, r + grid_spacing, grid_spacing, device=device)
            ys = torch.arange(-r, r + grid_spacing, grid_spacing, device=device)
            grid_x, grid_y = torch.meshgrid(xs, ys, indexing="xy")
            grid = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)

            # Keep only points inside circle
            mask = torch.linalg.norm(grid, dim=1) < r
            candidates = grid[mask]

            # Shuffle candidates
            perm = torch.randperm(candidates.shape[0], device=device)
            candidates = candidates[perm]

            chosen = []
            for pt in candidates:
                if len(chosen) == num_samples:
                    break
                if all(torch.norm(pt - c) >= min_dist for c in chosen):
                    chosen.append(pt)

            if len(chosen) < num_samples:
                raise RuntimeError(f"Not enough valid grid positions for env {i}.")

            offsets[i, :, 0:2] = torch.stack(chosen, dim=0)

        return offsets
