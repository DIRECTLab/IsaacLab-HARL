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


class QuadcopterEnvWindow(BaseEnvWindow):
    """Window manager for the Quadcopter environment."""

    def __init__(self, env: QuadcopterMARLEnv, window_name: str = "IsaacLab"):
        """Initialize the window.

        Args:
            env: The environment object.
            window_name: The name of the window. Defaults to "IsaacLab".
        """
        # initialize base window
        super().__init__(env, window_name)
        # add custom UI elements
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    # add command manager visualization
                    self._create_debug_vis_ui_element("targets", self.env)


@configclass
class QuadcopterMARLEnvTeamCfg(DirectMARLEnvCfg):
    # env
    episode_length_s = 20.0
    decimation = 4
    anymal_action_scale = 0.5
    action_space = 4

    # Number of robots per team
    num_robots_per_team = 3  # You can change this number as needed
    action_spaces = {f"robot_{i}": 4 for i in range(num_robots_per_team)}
    
    base_obs_size = 3 + 3 + 3 + 3
    _other_pos_size = 3 * (num_robots_per_team - 1)
    _total_obs_size = base_obs_size + _other_pos_size
    observation_spaces = {f"robot_{i}": 18 for i in range(num_robots_per_team)}
    state_space = 0
    state_spaces = {f"robot_{i}": 0 for i in range(num_robots_per_team)}
    possible_agents = [f"robot_{i}" for i in range(num_robots_per_team)]
    teams = {
        "team_0": [f"robot_{i}" for i in range(num_robots_per_team)],
    }

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

    # Dynamically add robot configs
    for i in range(num_robots_per_team):
        cfg = CRAZYFLIE_CFG.replace(prim_path=f"/World/envs/env_.*/Robot_{i}")
        cfg.init_state.pos = (float(i), 0.0, 2.0)
        locals()[f"robot_{i}"] = cfg

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
    # Knockout negative reward scale
    knockout_negative_reward = -50.0

    ### CRAZYFLIE CONFIGURATION ###


class QuadcopterMARLEnvTeam(DirectMARLEnv):
    @property
    def agent_active_masks(self):
        """
        Returns a dict of agent active masks (1.0 if agent is alive, 0.0 if dead), shape [num_envs, 1] per agent.
        """
        agent_active_masks = {}
        for team, agents in self.cfg.teams.items():
            agent_active_masks[team] = {}
            for agent in agents:
                agent_active_masks[team][agent] = self.alive[agent].float().unsqueeze(-1)
        return agent_active_masks
    
    def _update_alive(self):
        # Mark agents as dead (alive=False) if they are below 0.2m from the ground
        for agent in self.cfg.action_spaces:
            # active_masks use 0 to mask out agents that have died
            below_ground = self.robots[agent].data.root_pos_w[:, 2] < 0.2
            self.alive[agent] = ~below_ground

    def _init_alive(self):
        # Initialize alive attribute for each agent (True=alive, False=dead)
        self.alive = {}
        for agent in self.cfg.action_spaces:
            self.alive[agent] = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

    def _draw_drone_color_markers(self):
        # Draw a colored marker above each drone, matching its goal color
        if not hasattr(self, "drone_pos_visualizer"):
            return
        num_envs = self.num_envs
        num_robots = self.cfg.num_robots_per_team
        marker_positions = []
        marker_indices = []
        for env_i in range(num_envs):
            for robot_i, agent in enumerate(self.cfg.action_spaces):
                pos = self.robots[agent].data.root_pos_w[env_i].clone()
                pos[2] += 0.1  # raise marker above drone
                marker_positions.append(pos)
                marker_indices.append(robot_i)
        marker_positions = torch.stack(marker_positions, dim=0)
        marker_indices = torch.tensor(marker_indices, device=marker_positions.device, dtype=torch.long)
        self.drone_pos_visualizer.visualize(marker_positions, marker_indices=marker_indices)
    cfg: QuadcopterMARLEnvTeamCfg

    def __init__(self, cfg: QuadcopterMARLEnvTeamCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Total thrust and moment applied to the base of each quadcopter
        self.actions = {
            agent: torch.zeros(self.num_envs, action_space, device=self.device)
            for agent, action_space in self.cfg.action_spaces.items()
        }
        self._thrust = {agent: torch.zeros(self.num_envs, 1, 3, device=self.device) for agent in self.cfg.action_spaces}
        self._moment = {agent: torch.zeros(self.num_envs, 1, 3, device=self.device) for agent in self.cfg.action_spaces}
        self._desired_pos_w = torch.zeros(self.num_envs, self.cfg.num_robots_per_team, 3, device=self.device)  # [env, robot, xyz]

        # Store per-robot body ids, mass, etc.
        self._body_id = {}
        self._robot_mass = {}
        self._robot_weight = {}
        self.robots = getattr(self, 'robots', {})
        for agent in self.cfg.action_spaces:
            self._body_id[agent] = self.robots[agent].find_bodies("body")[0]
            self._robot_mass[agent] = self.robots[agent].root_physx_view.get_masses()[0].sum()
            gravity_mag = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
            self._robot_weight[agent] = (self._robot_mass[agent] * gravity_mag).item()

        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "lin_vel",
                "ang_vel",
                "distance_to_goal",
                "crazyflie_cosine_reward",
                "tank_angle_reward",
                "knockout",  # This now tracks deaths (alive=False)
            ]
        }

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

        # Initialize alive attribute after all relevant attributes are set
        self._init_alive()

    def _setup_scene(self):
        self.num_robots = self.cfg.num_robots_per_team
        self.robots = {}
        for i in range(self.num_robots):
            robot_id = f"robot_{i}"
            if robot_id in self.cfg.__dict__:
                self.robots[robot_id] = Articulation(self.cfg.__dict__[robot_id])
                self.scene.articulations[robot_id] = self.robots[robot_id]

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

    def _pre_physics_step(self, actions: dict):
        # Update alive status before each physics step
        self._update_alive()
        # Apply actions for each robot
        for agent in self.cfg.action_spaces:
            # If agent is dead, set all actions to zero
            agent_actions = actions[agent].clone().clamp(-1.0, 1.0)
            mask = (~self.alive[agent]).unsqueeze(1)  # [num_envs, 1]
            agent_actions = torch.where(mask, torch.zeros_like(agent_actions), agent_actions)
            self.actions[agent] = agent_actions
            self._thrust[agent][:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight[agent] * (self.actions[agent][:, 0] + 1.0) / 2.0
            self._moment[agent][:, 0, :] = self.cfg.moment_scale * self.actions[agent][:, 1:]

    def _apply_action(self):
        for agent in self.cfg.action_spaces:
            self.robots[agent].set_external_force_and_torque(self._thrust[agent], self._moment[agent], body_ids=self._body_id[agent])

    def _get_observations(self) -> dict:
        obs = {}
        # Collect all robot positions for all envs: shape [num_envs, num_robots, 3]
        all_positions = torch.stack([
            self.robots[agent].data.root_state_w[:, :3] for agent in self.cfg.action_spaces
        ], dim=1)  # [num_envs, num_robots, 3]
        for i, agent in enumerate(self.cfg.action_spaces):
            desired_pos_b, _ = subtract_frame_transforms(
                self.robots[agent].data.root_state_w[:, :3],
                self.robots[agent].data.root_state_w[:, 3:7],
                self._desired_pos_w[:, i, :]
            )
            # Get positions of other robots (exclude self)
            other_indices = [j for j in range(len(self.cfg.action_spaces)) if j != i]
            other_pos = all_positions[:, other_indices, :].reshape(self.num_envs, -1)  # [num_envs, (num_robots-1)*3]
            obs[agent] = torch.cat([
                self.robots[agent].data.root_lin_vel_b,
                self.robots[agent].data.root_ang_vel_b,
                self.robots[agent].data.projected_gravity_b,
                desired_pos_b,
                other_pos,
            ], dim=-1)
        return {"team_0": obs}

    def _get_rewards(self) -> dict:
        rewards = {}
        for i, agent in enumerate(self.cfg.action_spaces):
            lin_vel = torch.sum(torch.square(self.robots[agent].data.root_lin_vel_b), dim=1)
            ang_vel = torch.sum(torch.square(self.robots[agent].data.root_ang_vel_b), dim=1)
            distance_to_goal = torch.linalg.norm(self._desired_pos_w[:, i, :] - self.robots[agent].data.root_pos_w, dim=1)
            distance_to_goal_mapped = 1 - torch.tanh(distance_to_goal / 0.8)
            # Death negative reward
            death_penalty = (~self.alive[agent]).float()
            rewards[agent] = (
                lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt
                + ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt
                + distance_to_goal_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt
                + death_penalty * self.cfg.knockout_negative_reward * self.step_dt
            )
            # Logging
            self._episode_sums["lin_vel"] += lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt
            self._episode_sums["ang_vel"] += ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt
            self._episode_sums["distance_to_goal"] += distance_to_goal_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt
            self._episode_sums["knockout"] += death_penalty * self.cfg.knockout_negative_reward * self.step_dt
        # Return rewards in team structure
        return {"team_0": torch.stack(list(rewards.values()), dim=1).sum(dim=1)}

    def _get_dones(self) -> tuple[dict, dict]:
        time_out = (self.episode_length_buf >= self.max_episode_length - 1).to(self.device)
        # Team is done if all team members are dead (alive=False)
        all_dead = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        for agent in self.cfg.action_spaces:
            all_dead &= ~self.alive[agent]
        dones = {"team_0": all_dead}
        time_outs = {"team_0": time_out}
        return dones, time_outs

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or (env_ids is not None and hasattr(env_ids, 'numel') and env_ids.numel() == self.num_envs):
            env_ids = self.robots[next(iter(self.robots))]._ALL_INDICES

        # Logging
        # Compute average final distance to goal over all agents (absolute 3D distance)
        num_robots = self.cfg.num_robots_per_team
        all_robot_pos = torch.stack([
            self.robots[f"robot_{i}"].data.root_pos_w[env_ids] for i in range(num_robots)
        ], dim=1)  # [num_envs, num_robots, 3]
        final_distances = torch.linalg.norm(self._desired_pos_w[env_ids, :, :] - all_robot_pos, dim=2)  # [num_envs, num_robots]
        final_distance_to_goal = final_distances.mean()
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0

            # NEW: log death counts
            death_counts = {}
            total_deaths = 0
            for agent in self.cfg.action_spaces:
                agent_deaths = (~self.alive[agent][env_ids]).sum().item()
                death_counts[f"Metrics/deaths/{agent}"] = agent_deaths
                total_deaths += agent_deaths
            death_counts["Metrics/deaths/total"] = total_deaths
            extras.update(death_counts)
        
        # Reset alive status for all agents
        self._init_alive()
        # Log per-agent final distance to goal
        num_robots = self.cfg.num_robots_per_team
        per_agent_final_dist = final_distances.mean(dim=0)  # [num_robots]
        for i in range(num_robots):
            extras[f"Metrics/final_distance_to_goal/robot_{i}"] = per_agent_final_dist[i].item()
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        # extras["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        # extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        extras["Metrics/final_distance_to_goal"] = final_distance_to_goal.item()
        self.extras["log"].update(extras)

        for agent in self.cfg.action_spaces:
            self.robots[agent].reset(env_ids)
        super()._reset_idx(env_ids)
        if hasattr(env_ids, 'numel') and env_ids.numel() == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        for agent in self.cfg.action_spaces:
            self.actions[agent][env_ids] = 0.0

        # Sample new unique goals for each robot in each env
        # Ensure goals are at least 0.5 apart in L2 norm
        min_dist = 0.5
        num_robots = self.cfg.num_robots_per_team
        # Ensure env_ids is a tensor
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        for env_i in env_ids.tolist():
            # Try up to 100 times to find unique goals
            for _ in range(100):
                candidate_goals = torch.zeros(num_robots, 3, device=self.device)
                candidate_goals[:, :2] = torch.empty(num_robots, 2, device=self.device).uniform_(-2.0, 2.0)
                candidate_goals[:, 2] = torch.empty(num_robots, device=self.device).uniform_(0.5, 1.5)
                # Add env origin offset
                candidate_goals[:, :2] += self._terrain.env_origins[env_i, :2]
                # Check pairwise distances
                dists = torch.cdist(candidate_goals[:, :2], candidate_goals[:, :2], p=2)
                if (dists + torch.eye(num_robots, device=self.device) * 9999).min() >= min_dist:
                    self._desired_pos_w[env_i, :, :] = candidate_goals
                    break
            else:
                # fallback: just assign, even if not unique
                self._desired_pos_w[env_i, :, :] = candidate_goals

        # Reset robot state
        for i, agent in enumerate(self.cfg.action_spaces):
            joint_pos = self.robots[agent].data.default_joint_pos[env_ids]
            joint_vel = self.robots[agent].data.default_joint_vel[env_ids]
            default_root_state = self.robots[agent].data.default_root_state[env_ids]
            default_root_state[:, :3] += self._terrain.env_origins[env_ids]
            self.robots[agent].write_root_pose_to_sim(default_root_state[:, :7], env_ids)
            self.robots[agent].write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
            self.robots[agent].write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first time (original logic)
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                palette = [
                    (1.0, 0.0, 0.0),  # red
                    # (0.0, 1.0, 0.0),  # green
                    (0.0, 0.5, 1.0),  # blue
                    (1.0, 1.0, 0.0),  # yellow
                    (1.0, 0.0, 1.0),  # magenta
                    (0.0, 1.0, 1.0),  # cyan
                    (1.0, 0.5, 0.0),  # orange
                    (0.5, 0.0, 1.0),  # purple
                ]
                num_robots = self.cfg.num_robots_per_team
                # Goal markers
                goal_marker_dict = {}
                for i in range(num_robots):
                    marker = CUBOID_MARKER_CFG.markers["cuboid"].copy()
                    marker.size = (0.05, 0.05, 0.05)
                    marker.visual_material = marker.visual_material.copy()
                    marker.visual_material.diffuse_color = palette[i % len(palette)]
                    marker.prim_path = f"/Visuals/Command/goal_position/goal_cuboid_{i}"
                    goal_marker_dict[f"goal_cuboid_{i}"] = marker
                from isaaclab.markers import VisualizationMarkersCfg
                goal_marker_cfg = VisualizationMarkersCfg(
                    prim_path="/Visuals/Command/goal_position",
                    markers=goal_marker_dict,
                )
                self.goal_pos_visualizer = VisualizationMarkers(goal_marker_cfg)
                # Drone markers
                drone_marker_dict = {}
                for i in range(num_robots):
                    marker = CUBOID_MARKER_CFG.markers["cuboid"].copy()
                    marker.size = (0.02, 0.02, 0.02)
                    marker.visual_material = marker.visual_material.copy()
                    marker.visual_material.diffuse_color = palette[i % len(palette)]
                    marker.prim_path = f"/Visuals/Command/drone_position/drone_cuboid_{i}"
                    drone_marker_dict[f"drone_cuboid_{i}"] = marker
                drone_marker_cfg = VisualizationMarkersCfg(
                    prim_path="/Visuals/Command/drone_position",
                    markers=drone_marker_dict,
                )
                self.drone_pos_visualizer = VisualizationMarkers(drone_marker_cfg)
                self._num_goal_markers = num_robots
            # set their visibility to true
            self.goal_pos_visualizer.set_visibility(True)
            self.drone_pos_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)
            if hasattr(self, "drone_pos_visualizer"):
                self.drone_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # update the goal markers for all robots in all envs
        # flatten to [num_envs * num_robots, 3]
        goal_positions = self._desired_pos_w.reshape(-1, 3)
        num_envs = self.num_envs
        num_robots = self.cfg.num_robots_per_team
        marker_indices = []
        for _ in range(num_envs):
            marker_indices.extend(list(range(num_robots)))
        marker_indices = torch.tensor(marker_indices, device=goal_positions.device if hasattr(goal_positions, 'device') else 'cpu', dtype=torch.long)
        self.goal_pos_visualizer.visualize(goal_positions, marker_indices=marker_indices)
        # Draw color markers above each drone (using the drone_cuboid markers)
        self._draw_drone_color_markers()
