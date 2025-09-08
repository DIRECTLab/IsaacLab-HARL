# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObjectCfg, RigidObject
from isaaclab.envs import  DirectMARLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, TiledCameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.envs import DirectMARLEnv
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sensors import TiledCamera
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import quat_from_angle_axis

##
# Pre-defined configs
##
from isaaclab_assets.robots.anymal import ANYMAL_C_CFG  # isort: skip

@configclass
class AnymalCSoccerCfg(DirectMARLEnvCfg):
    # env
    episode_length_s = 20.0
    decimation = 4
    action_scale = 0.5 
    action_space = 12
    action_spaces = {f"robot_{i}": 3 for i in range(1)}
    observation_space = 48
    observation_spaces = {f"robot_{i}": (3,80,80) for i in range(1)}
    state_space = 0
    state_spaces = {f"robot_{i}": 0 for i in range(1)}
    possible_agents = ["robot_0"]

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=100.0, replicate_physics=True)

    # events
    # events: EventCfg = EventCfg()

    # robot
    robot_0: ArticulationCfg = ANYMAL_C_CFG.replace(prim_path="/World/envs/env_.*/Robot_0")
    contact_sensor_0: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot_0/.*", history_length=3, update_period=0.005, track_air_time=True
    )
    robot_0.init_state.rot = (1.0, 0.0, 0.0, 0.0)
    robot_0.init_state.pos = (0.0, 0.0, 0.5)

    walking_policy_dir = "/home/isaacp/repos/IsaacLab-HARL/results/isaaclab/Isaac-Velocity-Flat-Anymal-C-Happo-Direct-v0/happo/anymal_c_walking_policy_happo/seed-00001-2025-07-25-16-41-42/best_model/actor_agent0_full.pt"

    # Goal Posts
    cfg_goalpost0 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object0",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 0.5, 1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=10.0),  
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(10.0, 1.0, 0.5), rot=(1.0, 0.0, 0.0, 0.0) # Position originally was (0.0, 0, 0.61)
        ),
    )
    cfg_goalpost1 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object1",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 0.5, 1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(10.0, -1.0, 0.5), rot=(1.0, 0.0, 0.0, 0.0) # Position originally was (0.0, 0, 0.61)
        ),
    )

    # Soccer Ball
    cfg_soccer_ball = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object2",
        spawn=sim_utils.SphereCfg(
            radius=0.5,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="average",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.3,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(1.5, 0.0, 1.0),
            rot=(1.0, 0.0, 0.0, 0.0)
        )
    )

    # camera
    rgb_camera_0 = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot_0/base/front_cam",
        update_period=0.1,
        height=80,
        width=80,
        # spawn=sim_utils.FisheyeCameraCfg(
        #     projection_type="fisheyePolynomial",
        # ),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
        ),

        offset=TiledCameraCfg.OffsetCfg(pos=(0, 0, .5), rot=(0.9945, 0.0, 0.1045, 0.0), convention="world"),
    )

    # reward scales
    anymal_min_z = 0.2
    anymal_dist_to_ball_scale = 0.1
    ball_dist_to_goal_scale = 1.0

    command_action_scale = .1
    max_action = 1.0
    min_action = -1.0

def define_markers(cfg: AnymalCSoccerCfg, goal_depth:float, goal_width:float) -> VisualizationMarkers:
    # Extract goal area from goal post positions

    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/myMarkers",
        markers={
            "sphere1": sim_utils.SphereCfg(
                radius=0.1,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.0, 0.0)),
            ),
            "sphere2": sim_utils.SphereCfg(
                radius=0.1,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),
            ),
            "arrow1": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                scale=(0.1, 0.1, 1.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.0, 0.0)),
            ),
            "arrow2": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                scale=(0.1, 0.1, 1.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),
            ),
            "sphere3": sim_utils.SphereCfg(
                radius=.1,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            ),
            "cube1": sim_utils.CuboidCfg(
                size=(goal_depth, goal_width, 0.1),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
            ),
            "cube2": sim_utils.CuboidCfg(
                size=(0.1, 0.1, 3),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
            ),
            "cube3": sim_utils.CuboidCfg(
                size=(0.1, 0.1, 3),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
            ),
        },
    )
    return VisualizationMarkers(marker_cfg)

class AnymalCPlaySoccer(DirectMARLEnv):
    cfg: AnymalCSoccerCfg

    def __init__(
        self, cfg: AnymalCSoccerCfg, render_mode: str | None = None, debug=False, **kwargs
    ):
        super().__init__(cfg, render_mode, **kwargs)

        self.debug = debug
        self.actions = {
            agent: torch.zeros(self.num_envs, action_space, device=self.device)
            for agent, action_space in self.cfg.action_spaces.items()
        }
        self.walking_actions = {
            agent: torch.zeros(self.num_envs, 12, device=self.device)
            for agent, _ in self.cfg.action_spaces.items()
        }
        self.previous_actions = {
            agent: torch.zeros(self.num_envs, action_space, device=self.device)
            for agent, action_space in self.cfg.action_spaces.items()
        }
        # X/Y linear velocity and yaw angular velocity commands
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)
        self.walking_model = torch.load(self.cfg.walking_policy_dir, map_location=self.device)
        self.walking_model.eval()

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "ball_dist_to_goal_reward",
                "anymal_dist_to_ball_reward"
            ]
        }

        self.base_ids = {}
        self.feet_ids = {}
        self.undesired_body_contact_ids = {}

        goal_post_0_pos_y = self.cfg.cfg_goalpost0.init_state.pos[1]
        goal_post_1_pos_y = self.cfg.cfg_goalpost1.init_state.pos[1]

        self.goal_width = abs(goal_post_0_pos_y - goal_post_1_pos_y)
        self.goal_depth = self.cfg.cfg_goalpost0.spawn.size[0]

        if self.debug:
            self.my_visualizer = define_markers(self.cfg, self.goal_depth, self.goal_width)

        self.mid_goal_pos = self.object.data.root_pos_w.clone()
        self.mid_goal_pos[:, 1] += ((self.object1.data.root_pos_w[:, 1] - self.object.data.root_pos_w[:, 1]) / 2.0)
        self.mid_goal_pos[:, 2] = 0.0

        self.goal_top_left_corner = self.mid_goal_pos.clone()
        self.goal_top_left_corner[:, 0] -= self.goal_depth / 2.0
        self.goal_top_left_corner[:, 1] += self.goal_width / 2.0

        self.goal_bottom_right_corner = self.mid_goal_pos.clone()
        self.goal_bottom_right_corner[:, 0] += self.goal_depth / 2.0
        self.goal_bottom_right_corner[:, 1] -= self.goal_width / 2.0

        self.previous_ball_pos = self.object2.data.root_pos_w.clone()
        self.curr_ball_pos = self.object2.data.root_pos_w.clone()

        self.previous_anymal_pos = self.robots["robot_0"].data.root_pos_w.clone()
        self.curr_anymal_pos = self.robots["robot_0"].data.root_pos_w.clone()
    
    def _setup_scene(self):
        self.num_robots = sum(1 for key in self.cfg.__dict__.keys() if "robot_" in key)
        self.robots = {}
        self.height_scanners = {}
        self.rgb_camera = {}
        self.object = RigidObject(self.cfg.cfg_goalpost0)  # Goal Post 0
        self.object1 = RigidObject(self.cfg.cfg_goalpost1)  # Goal Post 1
        self.object2 = RigidObject(self.cfg.cfg_soccer_ball)  # Soccer Ball

        for i in range(self.num_robots):
            self.robots[f"robot_{i}"] = Articulation(self.cfg.__dict__["robot_" + str(i)])
            self.scene.articulations[f"robot_{i}"] = self.robots[f"robot_{i}"]

            self.rgb_camera[f"robot_{i}"] = TiledCamera(self.cfg.__dict__["rgb_camera_" + str(i)])
            self.scene.sensors[f"robot_{i}"] = self.rgb_camera[f"robot_{i}"]

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        # We need to process the actions for each scene independently
        self._commands = torch.clip(self._commands + actions["robot_0"] * self.cfg.command_action_scale, self.cfg.min_action, self.cfg.max_action)
        # self._commands = torch.zeros_like(self._commands)
        # self._commands[:, 0] = 1

        self.walking_processed_actions = {}        
        for robot_id, robot in self.robots.items():
            walking_obs = torch.cat(
                [
                    tensor
                    for tensor in (
                        robot.data.root_com_lin_vel_b,
                        robot.data.root_com_ang_vel_b,
                        robot.data.projected_gravity_b,
                        self._commands,
                        robot.data.joint_pos - robot.data.default_joint_pos,
                        robot.data.joint_vel,
                        # height_datas[robot_id],
                        self.walking_actions[robot_id],
                    )
                    if tensor is not None
                ],
                dim=-1,
            )

            # we only need the first element of the output, which is the action for the robot
            self.walking_actions[robot_id] = self.walking_model(walking_obs, torch.zeros(1), torch.zeros(1))[0]

            self.walking_processed_actions[robot_id] = self.walking_actions[robot_id].clone()
            self.walking_processed_actions[robot_id] = self.cfg.action_scale * self.walking_processed_actions[robot_id] + robot.data.default_joint_pos

    def _apply_action(self):
        for robot_id, robot in self.robots.items():
            robot.set_joint_position_target(self.walking_processed_actions[robot_id])


    def _draw_markers(self, command):
        xy_commands = command.clone()
        z_commands = xy_commands[:, 2].clone()
        xy_commands[:, 2] = 0

        marker_ids = torch.concat(
            [
                0 * torch.zeros(2 * self._commands.shape[0]),
                1 * torch.ones(self._commands.shape[0]),
                2 * torch.ones(self._commands.shape[0]),
                3 * torch.ones(self._commands.shape[0]),
                4 * torch.ones(self._commands.shape[0]),
                5 * torch.ones(self._commands.shape[0]),
                6 * torch.ones(self._commands.shape[0]),
                7 * torch.ones(self._commands.shape[0]),
            ],
            dim=0,
        )

        robot_pos = self.robots["robot_0"].data.root_pos_w
        robot_yaw = self.robots["robot_0"].data.root_com_ang_vel_b[:, 2]

        scale1 = torch.ones((self._commands.shape[0], 3), device=self.device)
        scale1[:, 0] = torch.abs(z_commands)

        scale2 = torch.ones((self._commands.shape[0], 3), device=self.device)
        scale2[:, 0] = torch.abs(robot_yaw)

        offset1 = torch.zeros((self._commands.shape[0], 3), device=self.device)
        offset1[:, 1] = 0

        offset2 = torch.zeros((self._commands.shape[0], 3), device=self.device)
        offset2[:, 1] = 0

        # marker_orientations = torch.zeros(, 3)

        marker_scales = torch.concat(
            [torch.ones((3 * self.num_envs, 3), device=self.device), 
             scale1, 
             scale2, 
             torch.ones((4 * self.num_envs, 3), device=self.device)], dim=0
        )

        marker_locations = torch.concat(
            [
                robot_pos,
                robot_pos + xy_commands,
                robot_pos + self.robots["robot_0"].data.root_com_lin_vel_b,
                robot_pos + offset1,
                robot_pos + offset2,
                self.mid_goal_pos,
                self.mid_goal_pos,
                self.goal_top_left_corner,
                self.goal_bottom_right_corner
            ],
            dim=0,
        )

        _90 = (-3.14 / 2) * torch.ones(self._commands.shape[0]).to(self.device)

        marker_orientations = quat_from_angle_axis(
            torch.concat(
                [
                    torch.zeros(3 * self.num_envs).to(self.device),
                    torch.sign(z_commands) * _90,
                    torch.sign(robot_yaw) * _90,
                    torch.zeros(4 * self.num_envs).to(self.device)
                ],
                dim=0,
            ),
            torch.tensor([0.0, 1.0, 0.0], device=self.device),
        )

        self.my_visualizer.visualize(
            marker_locations, marker_orientations, scales=marker_scales, marker_indices=marker_ids
        )


    def _get_observations(self) -> dict:
        self.previous_actions = copy.deepcopy(self.actions)
        features = self.rgb_camera["robot_0"].data.output['rgb']
        features = features.permute(0, 3, 1, 2)  # Change to (N, C, H, W)
        obs = {"robot_0": features}

        return obs

    def _get_rewards(self) -> dict:
        reward = {}

        self.curr_ball_pos = self.object2.data.root_link_pos_w.clone()
        self.curr_anymal_pos = self.robots["robot_0"].data.root_pos_w.clone()

        previous_dist_to_goal = torch.linalg.norm(
            self.mid_goal_pos - self.previous_ball_pos, dim=-1
        )
        current_dist_to_goal = torch.linalg.norm(
            self.mid_goal_pos - self.curr_ball_pos, dim=-1
        )

        ball_dist_reward = (previous_dist_to_goal - current_dist_to_goal) * self.cfg.ball_dist_to_goal_scale

        self.ball_dist_to_goal = current_dist_to_goal.clone()

        previous_anymal_dist_to_ball = torch.linalg.norm(
            self.previous_anymal_pos - self.previous_ball_pos, dim=-1
        )
        current_anymal_dist_to_ball = torch.linalg.norm(
            self.curr_anymal_pos - self.curr_ball_pos, dim=-1
        )
        anymal_dist_reward = (previous_anymal_dist_to_ball - current_anymal_dist_to_ball) * self.cfg.anymal_dist_to_ball_scale

        self._episode_sums["ball_dist_to_goal_reward"] += ball_dist_reward
        self._episode_sums["anymal_dist_to_ball_reward"] += anymal_dist_reward

        scored_goal = self._get_scored_goal().float() * 100

        reward["robot_0"] = scored_goal + ball_dist_reward + anymal_dist_reward

        self.previous_ball_pos = self.curr_ball_pos.clone()
        self.previous_anymal_pos = self.curr_anymal_pos.clone()

        return reward

    def _get_scored_goal(self):
        # check if the ball is within the goal area
        corner_pos = getattr(self, "goal_top_left_corner", None)
        if corner_pos is not None:
            ball_pos = self.object2.data.root_link_pos_w.squeeze(1)

            bool_1 = torch.logical_and(
                ball_pos[:,0] > self.goal_top_left_corner[:,0],
                ball_pos[:,0] < self.goal_bottom_right_corner[:,0]
            )
            bool_2 = torch.logical_and(
                ball_pos[:,1] < self.goal_top_left_corner[:,1],
                ball_pos[:,1] > self.goal_bottom_right_corner[:,1]
            )
            scored_goal = torch.logical_and(bool_1, bool_2)
        else:
            scored_goal = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return scored_goal

    def _get_timeouts(self):
        return self.episode_length_buf >= self.max_episode_length - 1

    def _get_dones(self) -> tuple[dict, dict]:
        time_out = self._get_timeouts()
        scored_goal = self._get_scored_goal()
        self.scored_goal = scored_goal

        return {key: scored_goal for key in self.robots.keys()}, {key: time_out for key in self.robots.keys()}

    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)

        object_default_state = self.object.data.default_root_state.clone()[env_ids]
        object_default_state[:, 0:3] = object_default_state[:, 0:3] + self.scene.env_origins[env_ids]
        self.object.write_root_state_to_sim(object_default_state, env_ids)
        self.object.reset(env_ids)

        object_default_state1 = self.object1.data.default_root_state.clone()[env_ids]
        object_default_state1[:, 0:3] = object_default_state1[:, 0:3] + self.scene.env_origins[env_ids]
        self.object1.write_root_state_to_sim(object_default_state1, env_ids)
        self.object1.reset(env_ids)

        object_default_state2 = self.object2.data.default_root_state.clone()[env_ids]
        object_default_state2[:, 0:3] = object_default_state2[:, 0:3] + self.scene.env_origins[env_ids]
        object_default_state2[:, :2] += torch.zeros_like(object_default_state2[:, :2], device=self.device).uniform_(-5, 5)
        self.object2.write_root_state_to_sim(object_default_state2, env_ids)
        self.object2.reset(env_ids)


        # Joint position command (deviation from default joint positions)
        for agent, action_space in self.cfg.action_spaces.items():
            self.actions[agent][env_ids] = torch.zeros(env_ids.shape[0], action_space, device=self.device)
            self.previous_actions[agent][env_ids] = torch.zeros(env_ids.shape[0], action_space, device=self.device)

        for _, robot in self.robots.items():
            if env_ids is None or len(env_ids) == self.num_envs:
                env_ids = robot._ALL_INDICES
            robot.reset(env_ids)
            if len(env_ids) == self.num_envs:
                # Spread out the resets to avoid spikes in training when many environments reset at a similar time
                self.episode_length_buf[:] = torch.randint_like(
                    self.episode_length_buf, high=int(self.max_episode_length)
                )

            # Reset robot state
            joint_pos = robot.data.default_joint_pos[env_ids]
            joint_vel = robot.data.default_joint_vel[env_ids]
            default_root_state = robot.data.default_root_state[env_ids]
            default_root_state[:, :3] += self._terrain.env_origins[env_ids]
            robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
            robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
            robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self.previous_anymal_pos = self.robots["robot_0"].data.root_pos_w.clone()
        self.previous_ball_pos = self.object2.data.root_link_pos_w.clone()
        self.curr_anymal_pos = self.robots["robot_0"].data.root_pos_w.clone()
        self.curr_ball_pos = self.object2.data.root_link_pos_w.clone()
        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0

        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        
        ball_dist_to_goal = getattr(self, "ball_dist_to_goal", None)
        extras["ball_final_distance_to_goal"] = ball_dist_to_goal[env_ids].mean().item() if ball_dist_to_goal is not None else 0.0

        scored_goal = getattr(self, "scored_goal", None)
        if scored_goal is not None:
            extras["goal_score_percentage"] = torch.count_nonzero(scored_goal[env_ids]).item() / len(env_ids) if len(env_ids) > 0 else 0.0
        else:
            extras["goal_score_percentage"] = 0.0
        self.extras["log"].update(extras)
