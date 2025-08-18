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
class LeatherbackSumoEnvCfg(DirectMARLEnvCfg):
    decimation = 4
    episode_length_s = 30.0
    action_spaces = {f"robot_{i}": 2 for i in range(2)}
    observation_spaces = {f"robot_{i}": 3 for i in range(2)}
    state_space = 0
    state_spaces = {f"robot_{i}": 0 for i in range(2)}
    possible_agents = ["robot_0", "robot_1"]

    teams = {
        "team_0":["robot_0"],
        "team_1":["robot_1"],
    }

    sim: SimulationCfg = SimulationCfg(dt=1 / 200, render_interval=decimation)
    robot_0: ArticulationCfg = LEATHERBACK_CFG.replace(prim_path="/World/envs/env_.*/Robot_0")
    robot_0.init_state.rot = get_quaternion_tuple_from_xyz(0,0,-torch.pi/2)
    robot_0.init_state.pos = (0.0, .25, .1)
    robot_1: ArticulationCfg = LEATHERBACK_CFG.replace(prim_path="/World/envs/env_.*/Robot_1")
    robot_1.init_state.rot = get_quaternion_tuple_from_xyz(0,0,torch.pi/2)
    robot_1.init_state.pos = (0.0, -.25, .1)

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

    ring_radius_min = 0.5
    ring_radius_max = 3
    reward_scale = 10

class LeatherbackSumoEnv(DirectMARLEnv):
    cfg: LeatherbackSumoEnvCfg

    def __init__(self, cfg: LeatherbackSumoEnvCfg, render_mode: str | None = None, headless: bool | None = None, **kwargs):
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
        self._steering_state = {}
        self._throttle_state = {}
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
        robot_0_desired_pos, _ = subtract_frame_transforms(
            self.robots["robot_0"].data.root_state_w[:, :3], self.robots["robot_0"].data.root_state_w[:, 3:7],
            self.robots["robot_1"].data.root_pos_w
        )
        robot_1_desired_pos, _ = subtract_frame_transforms(
            self.robots["robot_1"].data.root_state_w[:, :3], self.robots["robot_1"].data.root_state_w[:, 3:7],
            self.robots["robot_0"].data.root_pos_w
        )

        rcol = self.ring_radius.view(-1, 1)
        obs0 = torch.cat([robot_0_desired_pos, rcol], dim=1)
        obs1 = torch.cat([robot_1_desired_pos, rcol], dim=1)

        return {"team_0": {"robot_0": obs0}, "team_1": {"robot_1": obs1}}
    
    def _get_rewards(self) -> dict:
        out = self._robots_out_of_ring()
        r0_lost = out["robot_0"].to(torch.int8)
        r1_lost = out["robot_1"].to(torch.int8)
        time_out_reward = (self.episode_length_buf >= self.max_episode_length - 1).to(torch.int8)

        team_0_reward = (r1_lost - r0_lost - time_out_reward) * self.cfg.reward_scale
        team_1_reward = (r0_lost - r1_lost - time_out_reward) * self.cfg.reward_scale

        return {"team_0": {"robot_0": team_0_reward},
                "team_1": {"robot_1": team_1_reward}}

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
        out_any = torch.any(torch.stack([out_map[r] for r in self.robots.keys()]), dim=0)

        dones = {team: out_any for team in self.cfg.teams.keys()}

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        timeouts = {team: time_out for team in self.cfg.teams.keys()}
        return dones, timeouts

    def _reset_idx(self, env_ids: torch.Tensor | None):

        if env_ids is None:
            env_ids = self.leatherback._ALL_INDICES
        super()._reset_idx(env_ids)

        low, high = self.cfg.ring_radius_min, self.cfg.ring_radius_max
    
        self.ring_radius[env_ids] = torch.zeros(env_ids.shape[0]).uniform_(low, high).to(self.device)

        for robot_id in self.robots.keys():
            self.robots[robot_id].reset(env_ids)
            self.actions[robot_id][env_ids] = 0.0
            joint_pos = self.robots[robot_id].data.default_joint_pos[env_ids]
            joint_vel = self.robots[robot_id].data.default_joint_vel[env_ids]
            default_root_state = self.robots[robot_id].data.default_root_state[env_ids]
            default_root_state[:, :3] += self.scene.env_origins[env_ids]
            self.robots[robot_id].write_root_pose_to_sim(default_root_state[:, :7], env_ids)
            self.robots[robot_id].write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
            self.robots[robot_id].write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self._draw_ring_markers()
        self.extras["log"] = {}

