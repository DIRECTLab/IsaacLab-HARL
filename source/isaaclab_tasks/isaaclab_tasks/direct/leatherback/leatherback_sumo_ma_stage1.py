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
    observation_spaces = {f"robot_{i}": 14 for i in range(2)}

    state_space = 0
    state_spaces = {f"robot_{i}": 0 for i in range(2)}

    possible_agents = [f"robot_{i}" for i in range(2)]

    # Teams
    teams = {
        "team_0": ["robot_0", "robot_1"],
    }

    sim: SimulationCfg = SimulationCfg(dt=1 / 200, render_interval=decimation)

    # Robot configs (prim paths unique per robot)
    robot_0: ArticulationCfg = LEATHERBACK_CFG.replace(prim_path="/World/envs/env_.*/Robot_0")
    robot_0.init_state.rot = get_quaternion_tuple_from_xyz(0, 0, -torch.pi/2)
    robot_0.init_state.pos = (0.0, 0.25, 0.1)

    robot_1: ArticulationCfg = LEATHERBACK_CFG.replace(prim_path="/World/envs/env_.*/Robot_1")
    robot_1.init_state.rot = get_quaternion_tuple_from_xyz(0, 0, -torch.pi/2)
    robot_1.init_state.pos = (0.0, 0.75, 0.1)

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
            pos=(0.0, .5, 0.1), rot=(1.0, 0.0, 0.0, 0.0)
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
            pos=(0.0, -.5, 0.1), rot=(1.0, 0.0, 0.0, 0.0)
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
                "blocks_dist_from_center_reward",
                "dist_to_blocks_reward",
                "time_out_reward",
                "push_out_reward"
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
                blocks = [r for r in self.blocks.keys()]

                # relative positions
                teammate_pos, _ = subtract_frame_transforms(
                    self.robots[robot_id].data.root_state_w[:, :3],
                    self.robots[robot_id].data.root_state_w[:, 3:7],
                    self.robots[teammate_id].data.root_pos_w,
                )
                enemy_pos = []
                for block_id in blocks:
                    epos, _ = subtract_frame_transforms(
                        self.robots[robot_id].data.root_state_w[:, :3],
                        self.robots[robot_id].data.root_state_w[:, 3:7],
                        self.blocks[block_id].data.root_pos_w,
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
        circle_centers = self.scene.env_origins

        # --- Distance of blocks from center ---
        block_dists = torch.stack([
            torch.norm(self.blocks["block_0"].data.root_pos_w - circle_centers, dim=-1),
            torch.norm(self.blocks["block_1"].data.root_pos_w - circle_centers, dim=-1),
        ], dim=1)  # (E, 2)
        block_avg_dist = block_dists.mean(dim=1)  # (E,)
        block_dist_from_center_mapped = torch.tanh(block_avg_dist / self.cfg.ring_radius_max)

        # --- Min distance between robots and blocks ---
        robot_positions = torch.stack([
            self.robots["robot_0"].data.root_pos_w,
            self.robots["robot_1"].data.root_pos_w,
        ], dim=1)  # (E, 2, 3)

        block_positions = torch.stack([
            self.blocks["block_0"].data.root_pos_w,
            self.blocks["block_1"].data.root_pos_w,
        ], dim=1)  # (E, 2, 3)

        # Pairwise distances between robots and blocks
        diff = robot_positions.unsqueeze(2) - block_positions.unsqueeze(1)  # (E, 2, 2, 3)
        pairwise_dists = torch.norm(diff, dim=-1)  # (E, 2, 2)

        # For each robot, take min dist to any block → (E, 2)
        min_dists_per_robot = pairwise_dists.min(dim=2).values

        # Average over robots → (E,)
        avg_min_dist = min_dists_per_robot.mean(dim=1)

        # Map closer = higher reward
        avg_min_dist_mapped = 1 - torch.tanh(avg_min_dist / 0.8)

        # --- Time penalty ---
        time_penalty = self.cfg.time_penalty * torch.ones_like(avg_min_dist, device=self.device)

        # --- Push out reward (adversarial) ---
        out = self._robots_out_of_ring()
        team0_out = torch.any(torch.stack([out["robot_0"], out["robot_1"]]), dim=0)
        team1_out = torch.any(torch.stack([out["block_0"], out["block_1"]]), dim=0)

        team0_lost = team0_out.to(torch.float32)
        team1_lost = team1_out.to(torch.float32)
        push_out_reward_0 = (team1_lost - team0_lost) * self.cfg.reward_scale

        rewards = {
            "blocks_dist_from_center_reward": block_dist_from_center_mapped * self.step_dt,
            "dist_to_blocks_reward": avg_min_dist_mapped * self.step_dt,
            "time_out_reward": time_penalty,
            "push_out_reward": push_out_reward_0,
        }

        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        for key, val in rewards.items():
            self._episode_sums[key] += val

        return {
            "team_0": reward,
        }

    def _robots_out_of_ring(self) -> dict[str, torch.Tensor]:
        env_xy = self.scene.env_origins[:, :2].to(self.device)  
        out = {}
        for robot_id in self.robots.keys():
            pos_xy = self.robots[robot_id].data.root_pos_w[:, :2]  
            dist = torch.linalg.norm(pos_xy - env_xy, dim=1)
            out[robot_id] = dist > self.ring_radius
        for block_id in self.blocks.keys():
            pos_xy = self.blocks[block_id].data.root_pos_w[:, :2]  
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

        # Place robots
        teammate_spacing = 1.0  # meters apart
        for team in self.cfg.teams.values():
            theta = 2.0 * torch.pi * torch.rand(N, device=self.device)
            r = 0.5 * self.ring_radius[env_ids]

            base_offsets = torch.zeros((N, 3), device=self.device)
            base_offsets[:, 0] = r * torch.cos(theta)
            base_offsets[:, 1] = r * torch.sin(theta)

            tangent = torch.zeros((N, 2), device=self.device)
            tangent[:, 0] = -torch.sin(theta)
            tangent[:, 1] = torch.cos(theta)

            teammate_offsets = [
                base_offsets.clone(),
                base_offsets.clone()
            ]
            teammate_offsets[0][:, 0:2] += 0.5 * teammate_spacing * tangent
            teammate_offsets[1][:, 0:2] -= 0.5 * teammate_spacing * tangent

            for robot_id, offsets in zip(team, teammate_offsets):
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

        # Place blocks with spacing checks
        for block_id, block in self.blocks.items():
            # collect robot root positions (N, 2)
            robot_positions = []
            for robot_id in self.robots:
                robot_pos = self.robots[robot_id].data.root_state_w[env_ids, :2]
                robot_positions.append(robot_pos)

            # also include already placed blocks to avoid overlaps
            placed_blocks = []
            for other_id, other_block in self.blocks.items():
                if other_id == block_id:
                    continue
                placed_blocks.append(other_block.data.root_state_w[env_ids, :2])

            existing = robot_positions + placed_blocks
            offsets = self._sample_positions(N, self.ring_radius[env_ids], existing_positions=existing)

            default_root_state = block.data.default_root_state[env_ids].clone()
            default_root_state[:, :3] = origins
            default_root_state[:, 0:2] += offsets[:, 0:2]
            default_root_state[:, 2] += block.data.default_root_state[env_ids][:, 2]

            block.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
            block.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)

        # Draw markers & reset episode logs
        self._draw_ring_markers()
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras[key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0

        team_pos = (self.robots["robot_0"].data.root_pos_w[env_ids, :2] + self.robots["robot_1"].data.root_pos_w[env_ids, :2]) / 2
        block_pos = (self.blocks["block_0"].data.root_pos_w[env_ids, :2] + self.blocks["block_1"].data.root_pos_w[env_ids, :2]) / 2
        extras["avg_difference_between_teams"] = torch.norm(team_pos - block_pos, dim=-1).mean()
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
    
    def _sample_positions(self, N, radii, existing_positions=None, min_dist=0.6):
        """
        Sample random (x, y) offsets inside the ring, avoiding overlaps.
        
        Args:
            N: number of environments
            radii: tensor of ring radii (N,)
            existing_positions: list of tensors (N, 2) for already placed objects
            min_dist: minimum allowed distance (meters)
        Returns:
            offsets: (N, 3) tensor of sampled positions
        """
        device = radii.device
        offsets = torch.zeros((N, 3), device=device)

        for i in range(N):
            tries = 0
            while True:
                theta = 2.0 * torch.pi * torch.rand(1, device=device)
                r = 0.5 * radii[i] * torch.rand(1, device=device)  # anywhere within half radius
                candidate = torch.tensor([r * torch.cos(theta), r * torch.sin(theta)], device=device)

                # check against existing positions for this env
                ok = True
                if existing_positions is not None:
                    for pos in existing_positions:
                        if torch.norm(candidate - pos[i]) < min_dist:
                            ok = False
                            break

                if ok or tries > 20:  # fallback after 20 tries
                    offsets[i, 0:2] = candidate
                    break
                tries += 1
        return offsets
