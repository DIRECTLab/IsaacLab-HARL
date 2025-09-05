# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers import VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

##
# Pre-defined configs
##
from isaaclab_assets import CRAZYFLIE_CFG  # isort: skip
from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip




@configclass
class DroneStage1EnvSingleAgentCfg(DirectMARLEnvCfg):
    # env
    episode_length_s = 20.0
    decimation = 4
    anymal_action_scale = 0.5
    action_space = 4
    action_spaces = {"robot_0": 4}

    # Padded observation: 12 (original) + 3 (teammate_pos) + 3 (other_pos) + 1 (dist_to_center) + 1 (arena_radius) = 20
    observation_space = 20
    observation_spaces = {"robot_0": 20 for i in range(1)}
    state_space = 0
    state_spaces = {f"robot_{i}": 0 for i in range(1)}
    possible_agents = ["robot_0" for i in range(1)]

    # Teams for single agent
    teams = {"team_0": ["robot_0"]}

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
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=10.0, replicate_physics=True)

    # events
    # events: EventCfg = EventCfg()

    ### CRAZYFLIE CONFIGURATION ###
    robot_0: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot_0")
    robot_0.init_state.pos = (0.0, 0.0, 2.0)

    # camera_0 = TiledCameraCfg(
    #     prim_path="/World/envs/env_.*/Robot_0/body/front_cam",
    #     update_period=0.1,
    #     height=256,
    #     width=256,
    #     data_types=["depth"],
    #     spawn=sim_utils.FisheyeCameraCfg(
    #         projection_type="fisheyePolynomial",
    #     ),
    #     # spawn=sim_utils.PinholeCameraCfg(
    #     #     focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
    #     # ),
    #     offset=CameraCfg.OffsetCfg(pos=(0, 0, 0), rot=(1,0,0,0), convention="opengl"),
    # )

    thrust_to_weight = 1.9
    moment_scale = 0.01

    # reward scales
    lin_vel_reward_scale = -0.05
    ang_vel_reward_scale = -0.01
    distance_to_goal_reward_scale = 15.0
    debug_vis = True

    ### CRAZYFLIE CONFIGURATION ###


class DroneStage1EnvSingleAgentMARLEnv(DirectMARLEnv):
    cfg: DroneStage1EnvSingleAgentCfg

    def __init__(self, cfg: DroneStage1EnvSingleAgentCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Total thrust and moment applied to the base of the quadcopter
        self.actions = {
            agent: torch.zeros(self.num_envs, action_space, device=self.device)
            for agent, action_space in self.cfg.action_spaces.items()
        }
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

        crazyflie_mass = self.robots["robot_0"].root_physx_view.get_masses()[0].sum()
        self._crazyflie_body_ids = self.robots["robot_0"].find_bodies("body")[0]
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._crazyflie_weight = (crazyflie_mass * self._gravity_magnitude).item()
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "lin_vel",
                "ang_vel",
                "distance_to_goal",
                "crazyflie_cosine_reward",
                "tank_angle_reward",
            ]
        }
        # Get specific body indices
        self._body_id = self.robots["robot_0"].find_bodies("body")[0]
        self._robot_mass = self.robots["robot_0"].root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        self.num_robots = sum(1 for key in self.cfg.__dict__.keys() if "robot_" in key)
        self.robots = {}
        # self.cameras = {}

        for i in range(self.num_robots):
            robot_id = f"robot_{i}"
            if robot_id in self.cfg.__dict__:
                self.robots[f"robot_{i}"] = Articulation(self.cfg.__dict__["robot_" + str(i)])
                self.scene.articulations[f"robot_{i}"] = self.robots[f"robot_{i}"]

        ### SETUP CAMERAS ###
        # self.cameras["robot_0"] = TiledCamera(self.cfg.camera_0)
        # self.scene.sensors["robot_0_camera"] = self.cameras["robot_0"]
        ### SETUP CAMERAS ###


        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions["robot_0"] = actions["robot_0"].clone().clamp(-1.0, 1.0)
        self._thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (self.actions["robot_0"][:, 0] + 1.0) / 2.0
        self._moment[:, 0, :] = self.cfg.moment_scale * self.actions["robot_0"][:, 1:]

    def _apply_action(self):
        self.robots["robot_0"].set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)

    def _get_observations(self) -> dict:
        desired_pos_b, _ = subtract_frame_transforms(
            self.robots["robot_0"].data.root_state_w[:, :3], self.robots["robot_0"].data.root_state_w[:, 3:7], self._desired_pos_w
        )
        # Padding: teammate_pos (3), other_pos (3), dist_to_center (1), arena_radius (1)
        teammate_pos = torch.zeros((self.num_envs, 3), device=self.device)
        other_pos = torch.zeros((self.num_envs, 3), device=self.device)
        dist_to_center = torch.zeros((self.num_envs, 1), device=self.device)
        arena_radius = torch.zeros((self.num_envs, 1), device=self.device)
        obs = torch.cat(
            [
                self.robots["robot_0"].data.root_lin_vel_b,
                self.robots["robot_0"].data.root_ang_vel_b,
                self.robots["robot_0"].data.projected_gravity_b,
                desired_pos_b,
                teammate_pos,
                other_pos,
                dist_to_center,
                arena_radius,
            ],
            dim=-1,
        )
        # Return as team dict for compatibility
        observations = {"team_0": {"robot_0": obs}}
        return observations

    def _get_rewards(self) -> dict:
        all_rewards = {}
        # Compute reward for each agent (here, just robot_0)
        lin_vel = torch.sum(torch.square(self.robots["robot_0"].data.root_lin_vel_b), dim=1)
        ang_vel = torch.sum(torch.square(self.robots["robot_0"].data.root_ang_vel_b), dim=1)
        distance_to_goal = torch.linalg.norm(self._desired_pos_w - self.robots["robot_0"].data.root_pos_w, dim=1)
        distance_to_goal_mapped = 1 - torch.tanh(distance_to_goal / 0.8)
        rewards = {
            "lin_vel": lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
            "distance_to_goal": distance_to_goal_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        for key, value in rewards.items():
            self._episode_sums[key] += value
        all_rewards["robot_0"] = reward
        return {"team_0": all_rewards["robot_0"]}

    def _get_dones(self) -> tuple[dict, dict]:
        time_out_tensor = (self.episode_length_buf >= self.max_episode_length - 1).to(self.device)
        died_tensor = self.robots["robot_0"].data.root_pos_w[:, 2] < 0.1
        # Flatten for single-agent team
        dones = {"team_0": died_tensor}
        time_out = {"team_0": time_out_tensor}
        return dones, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robots["robot_0"]._ALL_INDICES

        # Logging
        final_distance_to_goal = torch.linalg.norm(
            self._desired_pos_w[env_ids] - self.robots["robot_0"].data.root_pos_w[env_ids], dim=1
        ).mean()
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        # extras["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        # extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        extras["Metrics/final_distance_to_goal"] = final_distance_to_goal.item()
        self.extras["log"].update(extras)

        self.robots["robot_0"].reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self.actions["robot_0"][env_ids] = 0.0
        # Sample new commands
        self._desired_pos_w[env_ids, :2] = torch.zeros_like(self._desired_pos_w[env_ids, :2]).uniform_(-2.0, 2.0)
        self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
        self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2]).uniform_(0.5, 1.5)
        # Reset robot state
        joint_pos = self.robots["robot_0"].data.default_joint_pos[env_ids]
        joint_vel = self.robots["robot_0"].data.default_joint_vel[env_ids]
        default_root_state = self.robots["robot_0"].data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self.robots["robot_0"].write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robots["robot_0"].write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robots["robot_0"].write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
                # -- goal pose
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            self.goal_pos_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # update the markers
        self.goal_pos_visualizer.visualize(self._desired_pos_w)
