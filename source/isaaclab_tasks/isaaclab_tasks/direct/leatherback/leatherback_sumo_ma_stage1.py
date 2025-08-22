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
class LeatherbackSumoMAStage1EnvCfg(DirectMARLEnvCfg):
    decimation = 4
    episode_length_s = 30.0

    # 4 robots, each with throttle + steering (2 actions)
    action_spaces = {f"robot_{i}": 2 for i in range(2)}

    # Observation: teammate (3) + opp1 (3) + opp2 (3) + rcol(1) + dist_center(1) + velocity(3)
    # = 14 per robot
    observation_spaces = {f"robot_{i}": 8 for i in range(2)}

    state_space = 0
    state_spaces = {f"robot_{i}": 0 for i in range(2)}

    possible_agents = [f"robot_{i}" for i in range(2)]

    # Teams
    teams = {
        "team_0": ["robot_0"],
        "team_1": ["robot_1"]
    }

    sim: SimulationCfg = SimulationCfg(dt=1 / 200, render_interval=decimation)

    # Robot configs (prim paths unique per robot)
    robot_0: ArticulationCfg = LEATHERBACK_CFG.replace(prim_path="/World/envs/env_.*/Robot_0")
    robot_0.init_state.pos = (0.0, 0.5, 0.1)

    robot_1: ArticulationCfg = LEATHERBACK_CFG.replace(prim_path="/World/envs/env_.*/Robot_1")
    robot_1.init_state.pos = (0.0, -0.5, 0.1)

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

    ring_radius_min = 2
    ring_radius_max = 5
    reward_scale = 10
    # time penalty
    time_penalty = -0.01
    box_velocity_scale = 0.01

    block_0 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/block_0",
        spawn=sim_utils.CuboidCfg(
            size=(.4,.4,.4),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.01),  # changed from 1.0 to 0.5
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(1.0, .5, 0.1), rot=(1.0, 0.0, 0.0, 0.0)
        ),  # started the bar lower
    )


    block_1 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/block_1",
        spawn=sim_utils.CuboidCfg(
            size=(.4,.4,.4),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.01),  # changed from 1.0 to 0.5
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(1.0, -.5, 0.1), rot=(1.0, 0.0, 0.0, 0.0)
        ),
    )


class LeatherbackSumoMAStage1Env(DirectMARLEnv):
    cfg: LeatherbackSumoMAStage1EnvCfg

    def __init__(self, cfg: LeatherbackSumoMAStage1EnvCfg, render_mode: str | None = None, headless: bool | None = None, **kwargs):
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
                # Team 0
                "team_0_block0_center_reward",
                "team_0_dist_r0_b0_reward",
                "team_0_time_penalty",
                "team_0_push_out_reward",

                # Team 1
                "team_1_block1_center_reward",
                "team_1_dist_r1_b1_reward",
                "team_1_time_penalty",
                "team_1_push_out_reward",
            ]
        }

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
        self.blocks = {}

        for i in range(2):
            self.blocks[f"block_{i}"] = RigidObject(self.cfg.__dict__[f"block_{i}"])

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
            _throttle_action = actions[robot_id][:, 0].repeat_interleave(4).reshape((-1, 4)) * self.cfg.throttle_scale
            _throttle_action = torch.clamp(_throttle_action, -self.cfg.throttle_max, self.cfg.throttle_max)
            self._throttle_state[robot_id] = _throttle_action
            
            self._steering_action = actions[robot_id][:, 1].repeat_interleave(2).reshape((-1, 2)) * self.cfg.steering_scale
            self._steering_action = torch.clamp(self._steering_action, -self.cfg.steering_max, self.cfg.steering_max)
            self._steering_state[robot_id] = self._steering_action

        for block in self.blocks.values():
            block.update(self.step_dt)

    def _apply_action(self) -> None:
        for robot_id in self.robots.keys():
            # self._throttle_state[robot_id] = -5*torch.ones_like(self._throttle_state[robot_id], device=self.device)
            self.robots[robot_id].set_joint_velocity_target(self._throttle_state[robot_id], joint_ids=self._throttle_dof_idx)
            self.robots[robot_id].set_joint_position_target(self._steering_state[robot_id], joint_ids=self._steering_dof_idx)

    def _get_observations(self) -> dict:
        rcol = self.ring_radius.view(-1, 1)

        robot_0_desired_pos, _ = subtract_frame_transforms(
            self.robots["robot_0"].data.root_state_w[:, :3], self.robots["robot_0"].data.root_state_w[:, 3:7],
            self.blocks["block_0"].data.root_pos_w
        )
        robot_0_dist_center = torch.norm(
            self.robots["robot_0"].data.root_pos_w - self.scene.env_origins, dim=-1, keepdim=True
        )
        robot_0_vel = self.robots["robot_0"].data.root_lin_vel_b

        robot_0_obs = torch.cat([robot_0_desired_pos, robot_0_dist_center, robot_0_vel, rcol], dim=1)

        robot_1_desired_pos, _ = subtract_frame_transforms(
            self.robots["robot_1"].data.root_state_w[:, :3], self.robots["robot_1"].data.root_state_w[:, 3:7],
            self.blocks["block_1"].data.root_pos_w
        )
        robot_1_dist_center = torch.norm(
            self.robots["robot_1"].data.root_pos_w - self.scene.env_origins, dim=-1, keepdim=True
        )
        robot_1_vel = self.robots["robot_1"].data.root_lin_vel_b

        robot_1_obs = torch.cat([robot_1_desired_pos, robot_1_dist_center, robot_1_vel, rcol], dim=1)

        robot_0_obs = torch.nan_to_num(robot_0_obs, nan=0.0, posinf=1e6, neginf=-1e6)
        robot_1_obs = torch.nan_to_num(robot_1_obs, nan=0.0, posinf=1e6, neginf=-1e6)

        return {"team_0":{"robot_0":robot_0_obs}, "team_1":{"robot_1":robot_1_obs}}
    
    def _get_rewards(self) -> dict:
        circle_centers = self.scene.env_origins

        # --- Distance of each block from center ---
        block0_dist = torch.norm(self.blocks["block_0"].data.root_pos_w - circle_centers, dim=-1)
        block1_dist = torch.norm(self.blocks["block_1"].data.root_pos_w - circle_centers, dim=-1)

        block0_center_mapped = torch.tanh(block0_dist / self.cfg.ring_radius_max)
        block1_center_mapped = torch.tanh(block1_dist / self.cfg.ring_radius_max)

        # --- Distances robot<->its block ---
        robot0_pos = self.robots["robot_0"].data.root_pos_w
        robot1_pos = self.robots["robot_1"].data.root_pos_w
        block0_pos = self.blocks["block_0"].data.root_pos_w
        block1_pos = self.blocks["block_1"].data.root_pos_w

        dist_r0_b0 = torch.norm(robot0_pos - block0_pos, dim=-1)
        dist_r1_b1 = torch.norm(robot1_pos - block1_pos, dim=-1)

        dist_r0_b0_mapped = 1 - torch.tanh(dist_r0_b0 / self.cfg.ring_radius_max)
        dist_r1_b1_mapped = 1 - torch.tanh(dist_r1_b1 / self.cfg.ring_radius_max)

        # --- Time penalty (shared) ---
        time_penalty = self.cfg.time_penalty * torch.ones_like(block0_dist, device=self.device)

        out = self._robots_out_of_ring()
        block0_out = out["block_0"].to(torch.float32)
        block1_out = out["block_1"].to(torch.float32)
        robot0_out = out["robot_0"].to(torch.float32)
        robot1_out = out["robot_1"].to(torch.float32)

        push_out_reward_team0 = (block0_out - robot0_out) * self.cfg.reward_scale
        push_out_reward_team1 = (block1_out - robot1_out) * self.cfg.reward_scale

        # --- Rewards per team ---
        rewards_team0 = {
            "block0_center_reward": block0_center_mapped * self.step_dt,
            "dist_r0_b0_reward": dist_r0_b0_mapped * self.step_dt,
            "time_penalty": time_penalty,
            "push_out_reward": push_out_reward_team0,
        }

        rewards_team1 = {
            "block1_center_reward": block1_center_mapped * self.step_dt,
            "dist_r1_b1_reward": dist_r1_b1_mapped * self.step_dt,
            "time_penalty": time_penalty,
            "push_out_reward": push_out_reward_team1,
        }

        # Sum per team
        reward_team0 = torch.sum(torch.stack(list(rewards_team0.values())), dim=0)
        reward_team1 = torch.sum(torch.stack(list(rewards_team1.values())), dim=0)

        # Log episode sums
        for key, val in rewards_team0.items():
            self._episode_sums[f"team_0_{key}"] += val
        for key, val in rewards_team1.items():
            self._episode_sums[f"team_1_{key}"] += val

        return {
            "team_0": reward_team0,
            "team_1": reward_team1,
        }

    def _robots_out_of_ring(self) -> dict[str, torch.Tensor]:
        env_xy = self.scene.env_origins[:, :2].to(self.device)  
        out = {}
        for robot_id in self.robots.keys():
            pos_xy = self.robots[robot_id].data.root_pos_w[:, :2]  
            dist = torch.linalg.norm(pos_xy - env_xy, dim=1)
            out[robot_id] = dist > self.ring_radius
        for block_id in self.blocks.keys():
            pos_xy = self.blocks[block_id].data.root_com_pos_w[:, :2]  
            dist = torch.linalg.norm(pos_xy - env_xy, dim=1)
            out[block_id] = dist > self.ring_radius

        return out

    def _get_dones(self) -> tuple[dict, dict]:
        out_map = self._robots_out_of_ring()
        team0_out = torch.any(torch.stack([out_map["robot_0"], out_map["robot_1"]]), dim=0)
        team1_out = torch.any(torch.stack([out_map["block_0"], out_map["block_1"]]), dim=0)

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
            self.episode_length_buf[:] = torch.randint_like(
                self.episode_length_buf, high=int(self.max_episode_length)
            )

        # Randomize ring radius per env
        low, high = self.cfg.ring_radius_min, self.cfg.ring_radius_max
        self.ring_radius[env_ids] = (
            torch.empty(env_ids.shape[0], device=self.device).uniform_(low, high)
        )

        origins = self.scene.env_origins[env_ids]  # (N, 3)
        N = env_ids.shape[0]

        # Example: need 3 positions per env (2 robots + 1 block)
        num_samples = len(self.robots) + len(self.blocks)
        grid_offsets = self._sample_positions_grid(N, self.ring_radius[env_ids], num_samples)

        # Assign slots
        robot_slots = grid_offsets[:, :len(self.robots), :]
        block_slots = grid_offsets[:, len(self.robots):, :]

        # Apply robot positions
        for i, robot_id in enumerate(self.robots):
            default_root_state = self.robots[robot_id].data.default_root_state[env_ids].clone()
            default_root_state[:, :3] = origins
            default_root_state[:, 0:2] += robot_slots[:, i, 0:2]
            self.robots[robot_id].write_root_pose_to_sim(default_root_state[:, :7], env_ids)
            self.robots[robot_id].write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)

        # Apply block positions
        for j, (block_id, block) in enumerate(self.blocks.items()):
            default_root_state = block.data.default_root_state[env_ids].clone()
            default_root_state[:, :3] = origins
            default_root_state[:, 0:2] += block_slots[:, j, 0:2]
            block.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
            block.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)

        # Draw markers & reset episode logs
        self._draw_ring_markers()
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/"+key] = episodic_sum_avg / self.max_episode_length_s
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

