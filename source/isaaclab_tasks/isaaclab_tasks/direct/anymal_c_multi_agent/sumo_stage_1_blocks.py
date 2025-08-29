from __future__ import annotations

import torch
import copy
import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab_assets.robots.leatherback import LEATHERBACK_CFG  # isort: skip
from isaaclab_assets.robots.anymal import ANYMAL_C_CFG  # isort: skip
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.utils.math import quat_from_angle_axis, quat_from_euler_xyz

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

    physics_material_1 = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot_1", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.2),
            "num_buckets": 64,
        },
    )

    add_base_mass_1 = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot_1", body_names="base"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )

@configclass
class SumoStage1BlocksEnvCfg(DirectMARLEnvCfg):
    decimation = 4
    episode_length_s = 20.0

    # 4 robots, each with throttle + steering (2 actions)
    action_spaces = {f"robot_{i}": 12 for i in range(2)}

    # Observation: teammate (3) + opp1 (3) + opp2 (3) + rcol(1) + dist_center(1) + velocity(3)
    # = 14 per robot
    observation_spaces = {f"robot_{i}": 48 for i in range(2)}

    state_space = 0
    state_spaces = {f"robot_{i}": 0 for i in range(2)}

    possible_agents = [f"robot_{i}" for i in range(2)]

    # Teams
    teams = {
        "team_0": ["robot_0"],
        "team_1": ["robot_1"]
    }

    sim: SimulationCfg = SimulationCfg(dt=1 / 200, render_interval=decimation)

    events: EventCfg = EventCfg()

    robot_0: ArticulationCfg = ANYMAL_C_CFG.replace(prim_path="/World/envs/env_.*/Robot_0")
    robot_0.init_state.rot = get_quaternion_tuple_from_xyz(0,0,torch.pi/2)
    robot_0.init_state.pos = (0.0, 1.0, 0.5)

    robot_1: ArticulationCfg = ANYMAL_C_CFG.replace(prim_path="/World/envs/env_.*/Robot_1")
    robot_1.init_state.rot = get_quaternion_tuple_from_xyz(0,0,torch.pi/2)
    robot_1.init_state.pos = (0.0, -1.0, 0.5)

    contact_sensor_0: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot_0/.*", history_length=3, update_period=0.005, track_air_time=True,
    )
    contact_sensor_1: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot_1/.*", history_length=3, update_period=0.005, track_air_time=True,
    )

    env_spacing = 10.0
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=env_spacing, replicate_physics=True)

    goal_reach_radius: float = 1.0         # within this distance counts as "reached"
    action_scale = 0.5
    ring_radius_min = 6
    ring_radius_max = 8
    reward_scale = 10
    # time penalty
    time_penalty = -0.01
    box_velocity_scale = 0.01
    reached_goal_reward = 40.0
    dist_to_goal_reward_scale = 2.0
    lin_vel_reward_scale = 1.0
    yaw_rate_reward_scale = 0.5
    z_vel_reward_scale = -2.0
    ang_vel_reward_scale = -0.05
    joint_torque_reward_scale = -2.5e-5
    joint_accel_reward_scale = -2.5e-7
    action_rate_reward_scale = -0.01
    feet_air_time_reward_scale = 0.5
    undesired_contact_reward_scale = -1.0
    flat_orientation_reward_scale = -5.0

    block_0 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/block_0",
        spawn=sim_utils.CuboidCfg(
            size=(.25,.25,.25),
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
            size=(.25,.25,.25),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.01),  # changed from 1.0 to 0.5
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(1.0, -.5, 0.1), rot=(1.0, 0.0, 0.0, 0.0)
        ),
    )


class SumoStage1BlocksEnv(DirectMARLEnv):
    cfg: SumoStage1BlocksEnvCfg

    def __init__(self, cfg: SumoStage1BlocksEnvCfg, render_mode: str | None = None, headless: bool | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.headless = headless
        
        self.actions = {
            agent: torch.zeros(self.num_envs, action_space, device=self.device)
            for agent, action_space in self.cfg.action_spaces.items()
        }
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
                "team_0_goal_reached",
                "team_0_distance_to_goal",
                "team_0_lin_vel_z_l2",
                "team_0_ang_vel_xy_l2",
                "team_0_dof_torques_l2",
                "team_0_dof_acc_l2",
                "team_0_action_rate_l2",
                "team_0_feet_air_time",
                "team_0_undesired_contacts",
                "team_0_flat_orientation_l2",
                # "team_0_block0_center_reward",
                # "team_0_dist_r0_b0_reward",
                # "team_0_time_penalty",
                # "team_0_push_out_reward",

                # Team 1
                "team_1_goal_reached",
                "team_1_distance_to_goal",
                "team_1_lin_vel_z_l2",
                "team_1_ang_vel_xy_l2",
                "team_1_dof_torques_l2",
                "team_1_dof_acc_l2",
                "team_1_action_rate_l2",
                "team_1_feet_air_time",
                "team_1_undesired_contacts",
                "team_1_flat_orientation_l2",
                # "team_1_block1_center_reward",
                # "team_1_dist_r1_b1_reward",
                # "team_1_time_penalty",
                # "team_1_push_out_reward",
            ]
        }

        self.base_ids = {}
        self.feet_ids = {}
        self.undesired_body_contact_ids = {}

        for robot_id, contact_sensor in self.contact_sensors.items():
            _base_id, _ = contact_sensor.find_bodies("base")
            _feet_ids, _ = contact_sensor.find_bodies(".*FOOT")
            _undesired_contact_body_ids, _ = contact_sensor.find_bodies([".*THIGH"])
            self.base_ids[robot_id] = _base_id
            self.feet_ids[robot_id] = _feet_ids
            self.undesired_body_contact_ids[robot_id] = _undesired_contact_body_ids

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
        self.contact_sensors = {}
        for i in range(self.num_robots):
            self.robots[f"robot_{i}"] = Articulation(self.cfg.__dict__["robot_" + str(i)])
            self.scene.articulations[f"robot_{i}"] = self.robots[f"robot_{i}"]
            self.contact_sensors[f"robot_{i}"] = ContactSensor(self.cfg.__dict__["contact_sensor_" + str(i)])
            self.scene.sensors[f"robot_{i}"] = self.contact_sensors[f"robot_{i}"]

        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])
        # Add lighting
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: dict) -> None:
        self.processed_actions = {}
        self.actions = copy.deepcopy(actions)
        for robot_id, robot in self.robots.items():
            self.processed_actions[robot_id] = (
                self.cfg.action_scale * actions[robot_id] + robot.data.default_joint_pos
            )

        for block in self.blocks.values():
            block.update(self.step_dt)

    def _apply_action(self) -> None:
        for robot_id, robot in self.robots.items():
            robot.set_joint_position_target(self.processed_actions[robot_id])

    def _get_observations(self) -> dict:
        self.previous_actions = copy.deepcopy(self.actions)
        time_remaining = (self.max_episode_length - self.episode_length_buf).unsqueeze(-1)
        rcol = self.ring_radius.view(-1, 1)

        robot_0_desired_pos, _ = subtract_frame_transforms(
            self.robots["robot_0"].data.root_state_w[:, :3], self.robots["robot_0"].data.root_state_w[:, 3:7],
            self.blocks["block_0"].data.root_pos_w
        )
        robot_0_teammate_pos, _ = subtract_frame_transforms(
            self.robots["robot_0"].data.root_state_w[:, :3], self.robots["robot_0"].data.root_state_w[:, 3:7],
            self.robots["robot_1"].data.root_pos_w
        )
        robot_0_other_block_pos, _ = subtract_frame_transforms(
            self.robots["robot_0"].data.root_state_w[:, :3], self.robots["robot_0"].data.root_state_w[:, 3:7],
            self.blocks["block_1"].data.root_pos_w
        )
        robot_0_dist_center = torch.norm(
            self.robots["robot_0"].data.root_pos_w - self.scene.env_origins, dim=-1, keepdim=True
        )
        robot_0_obs = torch.cat(
            [
                self.robots["robot_0"].data.root_lin_vel_b,
                self.robots["robot_0"].data.root_ang_vel_b,
                self.robots["robot_0"].data.projected_gravity_b,
                self.robots["robot_0"].data.joint_pos - self.robots["robot_0"].data.default_joint_pos,
                self.robots["robot_0"].data.joint_vel,
                self.actions["robot_0"],
                # robot_0_teammate_pos,
                robot_0_desired_pos,
                # robot_0_other_block_pos,
                # robot_0_dist_center,
                # rcol,
                # time_remaining
            ],
            dim=-1,
        )


        robot_1_desired_pos, _ = subtract_frame_transforms(
            self.robots["robot_1"].data.root_state_w[:, :3], self.robots["robot_1"].data.root_state_w[:, 3:7],
            self.blocks["block_1"].data.root_pos_w
        )
        robot_1_teammate_pos, _ = subtract_frame_transforms(
            self.robots["robot_1"].data.root_state_w[:, :3], self.robots["robot_1"].data.root_state_w[:, 3:7],
            self.robots["robot_0"].data.root_pos_w
        )
        robot_1_other_block_pos, _ = subtract_frame_transforms(
            self.robots["robot_1"].data.root_state_w[:, :3], self.robots["robot_1"].data.root_state_w[:, 3:7],
            self.blocks["block_0"].data.root_pos_w
        )
        robot_1_dist_center = torch.norm(
            self.robots["robot_1"].data.root_pos_w - self.scene.env_origins, dim=-1, keepdim=True
        )
        robot_1_obs = torch.cat(
            [
                self.robots["robot_1"].data.root_lin_vel_b,
                self.robots["robot_1"].data.root_ang_vel_b,
                self.robots["robot_1"].data.projected_gravity_b,
                self.robots["robot_1"].data.joint_pos - self.robots["robot_1"].data.default_joint_pos,
                self.robots["robot_1"].data.joint_vel,
                self.actions["robot_1"],
                # robot_1_teammate_pos,
                robot_1_desired_pos,
                # robot_1_other_block_pos,
                # robot_1_dist_center,
                # rcol,
                # time_remaining
            ],
            dim=-1,
        )

        robot_0_obs = torch.nan_to_num(robot_0_obs, nan=0.0, posinf=1e6, neginf=-1e6)
        robot_1_obs = torch.nan_to_num(robot_1_obs, nan=0.0, posinf=1e6, neginf=-1e6)

        return {"team_0":{"robot_0":robot_0_obs}, "team_1":{"robot_1":robot_1_obs}}
    
    def _get_rewards(self) -> dict:
        circle_centers = self.scene.env_origins
        reach_r = self.cfg.goal_reach_radius

        all_rewards = {}

        for i, robot_id in enumerate(self.robots.keys()):
            robot_pos = self.robots[f"robot_{i}"].data.root_pos_w
            block_pos = self.blocks[f"block_{i}"].data.root_pos_w
            distance_to_block = torch.norm(robot_pos - block_pos, dim=-1)
            distance_reward = 1 - torch.tanh(distance_to_block / self.cfg.ring_radius_max)
            hit = distance_to_block <= reach_r
            # z velocity tracking
            z_vel_error = torch.square(self.robots[robot_id].data.root_lin_vel_b[:, 2])
            # angular velocity x/y
            ang_vel_error = torch.sum(torch.square(self.robots[robot_id].data.root_ang_vel_b[:, :2]), dim=1)
            # joint torques
            joint_torques = torch.sum(torch.square(self.robots[robot_id].data.applied_torque), dim=1)
            # joint acceleration
            joint_accel = torch.sum(torch.square(self.robots[robot_id].data.joint_acc), dim=1)
            # action rate
            action_rate = torch.sum(torch.square(self.actions[robot_id] - self.previous_actions[robot_id]), dim=1)
            # feet air time
            first_contact = self.contact_sensors[robot_id].compute_first_contact(self.step_dt)[:, self.feet_ids[robot_id]]
            last_air_time = self.contact_sensors[robot_id].data.last_air_time[:, self.feet_ids[robot_id]]
            air_time = torch.sum((last_air_time - 0.5) * first_contact, dim=1)
            # undesired contacts
            net_contact_forces = self.contact_sensors[robot_id].data.net_forces_w_history
            is_contact = (
                torch.max(torch.norm(net_contact_forces[:, :, self.undesired_body_contact_ids[robot_id]], dim=-1), dim=1)[0] > 1.0
            )
            contacts = torch.sum(is_contact, dim=1)
            # flat orientation
            flat_orientation = torch.sum(torch.square(self.robots[robot_id].data.projected_gravity_b[:, :2]), dim=1)

            rewards = {
                "goal_reached": hit.float() * self.cfg.reached_goal_reward,
                "distance_to_goal": distance_reward * self.cfg.dist_to_goal_reward_scale * self.step_dt,
                "lin_vel_z_l2": z_vel_error * self.cfg.z_vel_reward_scale * self.step_dt,
                "ang_vel_xy_l2": ang_vel_error * self.cfg.ang_vel_reward_scale * self.step_dt,
                "dof_torques_l2": joint_torques * self.cfg.joint_torque_reward_scale * self.step_dt,
                "dof_acc_l2": joint_accel * self.cfg.joint_accel_reward_scale * self.step_dt,
                "action_rate_l2": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
                "feet_air_time": air_time * self.cfg.feet_air_time_reward_scale * self.step_dt,
                "undesired_contacts": contacts * self.cfg.undesired_contact_reward_scale * self.step_dt,
                "flat_orientation_l2": flat_orientation * self.cfg.flat_orientation_reward_scale * self.step_dt,
            }

            all_rewards[robot_id] = rewards

        # --- Rewards per team ---
        rewards_team0 = all_rewards["robot_0"]
        rewards_team1 = all_rewards["robot_1"]

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
        reach_r = self.cfg.goal_reach_radius
        any_robot_reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        any_robot_died = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        for i, (robot_id, robot) in enumerate(self.robots.items()):
            blocks_xy = self.blocks[f"block_{i}"].data.root_pos_w[:, :2]
            net_contact_forces = self.contact_sensors[robot_id].data.net_forces_w_history
            died = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self.base_ids[robot_id]], dim=-1), dim=1)[0] > 1.0, dim=1)
            # robot base XY: (N, 2)
            robot_xy = robot.data.root_pos_w[:, :2]
            # pairwise dists to both goals: (N, 2)
            dists = torch.norm(blocks_xy - robot_xy, dim=-1)
            # did this robot hit any goal? (N,)
            hit = (dists <= reach_r)
            any_robot_reached |= hit
            any_robot_died |= died

        dones = {team: torch.logical_or(any_robot_died.clone(), any_robot_reached.clone()) for team in self.cfg.teams.keys()}

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        timeouts = {team: time_out for team in self.cfg.teams.keys()}
        return dones, timeouts

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        episode_times = self.episode_length_buf[env_ids].to(torch.float32) + 1
        super()._reset_idx(env_ids)

        # spread out the updates
        if len(env_ids) == self.num_envs: #type:ignore
            self.episode_length_buf[:] = torch.randint_like(
                self.episode_length_buf, high=int(self.max_episode_length)
            )

        # Randomize ring radius per env
        low, high = self.cfg.ring_radius_min, self.cfg.ring_radius_max
        self.ring_radius[env_ids] = (
            torch.empty(env_ids.shape[0], device=self.device).uniform_(low, high) #type:ignore
        )

        origins = self.scene.env_origins[env_ids]  # (N, 3)
        N = env_ids.shape[0] #type:ignore

        # Example: need 3 positions per env (2 robots + 1 block)
        num_samples = len(self.robots) + len(self.blocks)
        grid_offsets = self._sample_positions_grid(N, self.ring_radius[env_ids], num_samples)

        # Assign slots
        robot_slots = grid_offsets[:, :len(self.robots), :]
        block_slots = grid_offsets[:, len(self.robots):, :]

        # Apply robot positions
        for i, robot_id in enumerate(self.robots):
            self.robots[robot_id].reset(env_ids)
            default_root_state = self.robots[robot_id].data.default_root_state[env_ids].clone()
            default_root_state[:, :2] = origins[:, :2]
            default_root_state[:, 0:2] += robot_slots[:, i, 0:2]
            joint_pos = self.robots[robot_id].data.default_joint_pos[env_ids]
            joint_vel = self.robots[robot_id].data.default_joint_vel[env_ids]
            self.robots[robot_id].write_root_pose_to_sim(default_root_state[:, :7], env_ids)
            self.robots[robot_id].write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
            self.robots[robot_id].write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Apply block positions
        for j, (block_id, block) in enumerate(self.blocks.items()):
            default_root_state = block.data.default_root_state[env_ids].clone()
            default_root_state[:, :2] = origins[:, :2]
            default_root_state[:, 0:2] += block_slots[:, j, 0:2]
            block.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
            block.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)

        # Draw markers & reset episode logs
        self._draw_ring_markers()
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids] / (episode_times))
            extras["Episode_Reward/"+key] = episodic_sum_avg
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
            r = radii[i].item() * 0.9

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

