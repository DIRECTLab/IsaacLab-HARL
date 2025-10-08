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
from isaaclab.sensors import Camera, CameraCfg, TiledCamera, TiledCameraCfg
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.utils.math import quat_from_angle_axis, quat_from_euler_xyz, quat_rotate_inverse
from isaaclab_assets.custom.soccer_ball import SOCCERBALL_CFG  # isort: skip
import numpy as np
import matplotlib.pyplot as plt

def get_quaternion_tuple_from_xyz(x, y, z):
    quat_tensor = quat_from_euler_xyz(torch.tensor([x]), torch.tensor([y]), torch.tensor([z])).flatten()
    return (quat_tensor[0].item(), quat_tensor[1].item(), quat_tensor[2].item(), quat_tensor[3].item())

@configclass
class YahboomSearchAndRescueEnvCfg(DirectMARLEnvCfg):
    decimation = 4
    episode_length_s = 20.0
    action_spaces = {f"robot_{i}": 2 for i in range(1)}
    observation_spaces = {f"robot_{i}": 1600 for i in range(1)}
    state_space = 0
    state_spaces = {f"robot_{i}": 0 for i in range(1)}
    possible_agents = ["robot_0"]

    sim: SimulationCfg = SimulationCfg(dt=1 / 200, render_interval=decimation)
    robot_0: ArticulationCfg = LEATHERBACK_CFG.replace(prim_path="/World/envs/env_.*/Robot_0")
    robot_0.init_state.pos = (-1.0, 0.0, .5)

    # Camera on the intel realsense D435 has a depth camera with w 848, h 480, and an rgb camera with w 480, h 640
    camera_0 = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot_0/Rigid_Bodies/Chassis/Camera",
        update_period=0.1,
        height=40,
        width=40,
        data_types=["depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
        ),
        # spawn=sim_utils.PinholeCameraCfg(
        #     focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        # ),
        # offset=CameraCfg.OffsetCfg(pos=(0, 0, .25), rot=(0,0,1,0), convention="opengl"),
        offset=CameraCfg.OffsetCfg(pos=(20, 0, 30), rot=get_quaternion_tuple_from_xyz(torch.pi/2,  0, -torch.pi/2), convention="opengl"),
    )

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

    block = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Block_.*",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 0.5, 0.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.25),  # base height above ground
            rot=(1.0, 0.0, 0.0, 0.0)
        ),
    )

    ball = SOCCERBALL_CFG.replace(prim_path="/World/envs/env_.*/Object4")
    ball.init_state.pos = (1.0, 0.0, 0.5)

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

    env_spacing = 30.0
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=env_spacing, replicate_physics=True)

    throttle_scale = 10
    throttle_max = 50
    steering_scale = 0.1
    steering_max = 10

    goal_reward_scale = 20
    ball_to_goal_reward_scale = 1.0
    dist_to_ball_reward_scale = 1.0

class YahboomSearchAndRescueEnv(DirectMARLEnv):
    cfg: YahboomSearchAndRescueEnvCfg

    def __init__(self, cfg: YahboomSearchAndRescueEnvCfg, render_mode: str | None = None, headless: bool | None = None, debug: bool = False, **kwargs):
        self.debug = debug
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
                "dist_to_ball_reward",
            ]
        }


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

        self.camera = TiledCamera(self.cfg.camera_0)
        self.scene.sensors["camera_0"] = self.camera
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])
        # Add lighting
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)


        self.blocks = []
        num_blocks_per_env = 10  # you can adjust this
        for i in range(num_blocks_per_env):
            block_cfg = self.cfg.block.replace(
                prim_path=f"/World/envs/env_.*/Block_{i}"
            )
            block = RigidObject(block_cfg)
            self.blocks.append(block)

        if self.debug:
            plt.ion()
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            self.actual_depth_im = ax.imshow(torch.zeros((40, 40)), vmin=0, vmax=10, cmap='plasma')
            # self.image_slice = ax[1].imshow(torch.zeros((1,40)), vmin=0, vmax=10, cmap='plasma')

    def _pre_physics_step(self, actions: dict) -> None:
        self._throttle_action = actions["robot_0"][:, 0].repeat_interleave(4).reshape((-1, 4)) * self.cfg.throttle_scale
        self._throttle_action = torch.clamp(self._throttle_action, -self.cfg.throttle_max, self.cfg.throttle_max)
        self._throttle_state["robot_0"] = self._throttle_action
        
        self._steering_action = actions["robot_0"][:, 1].repeat_interleave(2).reshape((-1, 2)) * self.cfg.steering_scale
        self._steering_action = torch.clamp(self._steering_action, -self.cfg.steering_max, self.cfg.steering_max)
        self._steering_state["robot_0"] = self._steering_action
        self.ball.update(self.step_dt)

    def _apply_action(self) -> None:
        for robot_id in self.robots.keys():
            self.robots[robot_id].set_joint_velocity_target(self._throttle_state[robot_id], joint_ids=self._throttle_dof_idx)
            self.robots[robot_id].set_joint_position_target(self._steering_state[robot_id], joint_ids=self._steering_dof_idx)
    
    def _get_observations(self) -> dict:
        img = self.camera.data.output["depth"]
        if self.debug:
            self.actual_depth_im.set_data(self.camera.data.output["depth"][0].cpu().numpy())
            # self.image_slice.set_data(img_slice[0].unsqueeze(0).cpu().numpy())
            plt.draw()
            plt.pause(0.001)

        return {"robot_0": torch.nan_to_num(img.reshape((self.num_envs, -1)).to(torch.float32), nan=0.0, posinf=1e6, neginf=-1e6)}
    
    def _get_rewards(self) -> dict:
        robot_pos = self.robots["robot_0"].data.root_pos_w[:, :3]
        ball_pos = self.ball.data.root_pos_w
        robot_distance_to_ball = torch.linalg.norm(robot_pos - ball_pos, dim=1)
        robot_distance_to_ball_mapped = robot_distance_to_ball
        robot_distance_to_ball_mapped = robot_distance_to_ball_mapped * self.cfg.dist_to_ball_reward_scale * self.step_dt
        robot_distance_to_ball_mapped = -1 * torch.nan_to_num(robot_distance_to_ball_mapped, nan=0.0, posinf=1e6, neginf=-1e6)

        self._episode_sums["dist_to_ball_reward"] += robot_distance_to_ball_mapped
        return {"robot_0": robot_distance_to_ball_mapped}

    def _get_dones(self) -> tuple[dict, dict]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return {"robot_0": time_out}, {"robot_0": time_out}

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

        # sample random grid positions for blocks
        num_blocks = len(self.blocks)
        offsets = self._sample_positions_grid(env_ids, num_blocks + 2, min_dist=1.0, grid_spacing=1.0)
        # offsets = self._sample_positions_grid(env_ids, 2, min_dist=1.0, grid_spacing=1.0)

        # set each block position
        for i, block in enumerate(self.blocks):
            block_state = block.data.default_root_state.clone()[env_ids]
            block_state[:, :2] = self.scene.env_origins[env_ids][:, :2] + offsets[:, i]
            block_state[:, 2] = 0.25  # fixed height
            block.write_root_state_to_sim(block_state, env_ids)
            block.reset(env_ids)

        # Cache for convenience
        origins = self.scene.env_origins[env_ids]  # (N, 3)

        # We’ll assign robots sequentially
        robot_ids = list(self.robots.keys())

        ball_default_state = self.ball.data.default_root_state.clone()[env_ids]
        ball_default_state[:, :2] = ball_default_state[:, :2] + self.scene.env_origins[env_ids][:,:2]\
        + offsets[:, -2]
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
            default_root_state[:, :2] += offsets[:, -1]

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

        final_dist_to_ball = torch.mean(torch.linalg.norm(self.robots["robot_0"].data.root_pos_w[:, :3] - self.ball.data.root_pos_w, dim=1)[env_ids]).item()
        extras["Final_Dist_To_Ball"] = final_dist_to_ball
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()


    def _sample_positions_grid(self, env_ids, num_samples, min_dist=1.0, grid_spacing=2.0):
        """
        Samples well-separated 2D positions per environment using a coarse meshgrid
        so blocks don't overlap. Super fast for large env counts.
        """
        device = self.scene.env_origins.device
        N = len(env_ids)

        offsets = torch.zeros((N, num_samples, 2), device=device)
        env_origins = self.scene.env_origins[env_ids][:, :2]

        # Use grid_spacing >= min_dist to guarantee spacing
        for i in range(N):
            # define grid area (tight bounds)
            xs = torch.arange(env_origins[i, 0] - 8, env_origins[i, 0] + 8, grid_spacing, device=device)
            ys = torch.arange(env_origins[i, 1] - 4, env_origins[i, 1] + 4, grid_spacing, device=device)
            xv, yv = torch.meshgrid(xs, ys, indexing="ij")

            grid_points = torch.stack([xv.flatten(), yv.flatten()], dim=-1)

            # randomly pick num_samples without replacement
            perm = torch.randperm(grid_points.shape[0], device=device)
            chosen = grid_points[perm[:num_samples]]

            offsets[i] = chosen - env_origins[i]

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