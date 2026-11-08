"""World-fixed reaching with a goal-derived velocity command and learned whole-body control."""
from __future__ import annotations

import torch

from isaaclab.envs.mdp import UniformVelocityCommand
from isaaclab.managers import RewardTermCfg, ObservationTermCfg
from isaaclab.utils.math import quat_apply_inverse, yaw_quat
from isaaclab_tasks.manager_based.locomanipulation.go2_piper.go2_piper_walk_env_cfg import Go2PiperWalkEnvCfg
from isaaclab_tasks.manager_based.locomanipulation.go2_piper.mdp.commands import EeGoalCommand


def tensor(value):
    return value.torch if hasattr(value, "torch") else value


def tcp_world(env):
    robot = env.scene["robot"]
    if not hasattr(env, "_tcp_ids"):
        env._tcp_ids = [robot.body_names.index(n) for n in ("piper_link7", "piper_link8")]
    return tensor(robot.data.body_pos_w)[:, env._tcp_ids].mean(dim=1)


def tcp_base(env):
    robot = env.scene["robot"]
    return quat_apply_inverse(yaw_quat(tensor(robot.data.root_quat_w)), tcp_world(env) - tensor(robot.data.root_pos_w))


def tcp_error_base(env):
    return env.command_manager.get_term("ee_goal").command - tcp_base(env)


def reach_reward(env, std: float):
    distance = torch.linalg.vector_norm(tcp_world(env) - env.command_manager.get_term("ee_goal").goal_pos_w, dim=-1)
    return torch.exp(-distance / std)


def reach_distance(env):
    return torch.linalg.vector_norm(tcp_world(env) - env.command_manager.get_term("ee_goal").goal_pos_w, dim=-1).clamp(max=2.0)


class FixedGoalCommand(EeGoalCommand):
    """Sample once at reset, without interpolation from a previous episode."""

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        self._start_pos_w[env_ids] = self._target_pos_w[env_ids]
        self._start_pos_b[env_ids] = self._target_pos_b[env_ids]
        self.goal_pos_w[env_ids] = self._target_pos_w[env_ids]
        self.goal_pos_b[env_ids] = self._target_pos_b[env_ids]

    def _update_metrics(self):
        self.metrics["position_error"] = reach_distance(self._env)


class GoalVelocityCommand(UniformVelocityCommand):
    """Slow down at a reachable standoff; commands depend on the arm goal [m/s, rad/s]."""

    def _resample_command(self, env_ids):
        self.vel_command_b[env_ids] = 0.0

    def _update_command(self):
        goal = self._env.command_manager.get_term("ee_goal").goal_pos_w
        robot = self._env.scene["robot"]
        relative = quat_apply_inverse(yaw_quat(tensor(robot.data.root_quat_w)), goal - tensor(robot.data.root_pos_w))
        heading = torch.atan2(relative[:, 1], relative[:, 0])
        # The arm works in front of the body. Back away if the target is too close.
        target = torch.zeros_like(self.vel_command_b)
        target[:, 0] = ((relative[:, 0] - 0.40) * 1.2).clamp(-0.20, 0.55)
        target[:, 0] *= torch.cos(heading).clamp(min=0.0)
        target[:, 1] = (relative[:, 1] * 0.7).clamp(-0.15, 0.15)
        target[:, 2] = (heading * 1.3).clamp(-0.65, 0.65)
        delta = torch.tensor([0.8, 0.6, 1.2], device=self.device) * self._env.step_dt
        self.vel_command_b[:] += (target - self.vel_command_b).clamp(-delta, delta)


def build_config(num_envs: int, device: str, stage: str, seed: int, video: bool = False):
    cfg = Go2PiperWalkEnvCfg()
    cfg.seed = seed
    cfg.scene.num_envs = num_envs
    cfg.scene.env_spacing = 4.0
    cfg.sim.device = device
    cfg.episode_length_s = 12.0
    goal = cfg.commands.ee_goal
    goal.class_type = FixedGoalCommand
    goal.world_frame = True
    goal.sample_in_env_frame = True
    goal.resampling_time_range = (100.0, 100.0)
    goal.traj_time = (0.02, 0.02)
    goal.debug_vis = False
    goal.ranges.pos_x = (0.45, 0.80) if stage == "near" else (0.95, 1.55)
    goal.ranges.pos_y = (-0.15, 0.15) if stage == "near" else (-0.40, 0.40)
    goal.ranges.pos_z = (0.50, 0.70)
    velocity = cfg.commands.base_velocity
    velocity.class_type = GoalVelocityCommand
    velocity.heading_command = False
    velocity.rel_heading_envs = 0.0
    velocity.rel_standing_envs = 0.0
    velocity.debug_vis = False
    cfg.observations.policy.ee_pos_b = ObservationTermCfg(func=tcp_base)
    cfg.observations.policy.ee_error_b = ObservationTermCfg(func=tcp_error_base)
    # Preserve 81 observation dimensions, 18 actions, and the pretrained gait action scale.
    cfg.rewards.arm_pose_tracking = None
    cfg.rewards.ee_goal_tracking = RewardTermCfg(func=reach_reward, weight=7.0, params={"std": 0.25})
    cfg.rewards.tcp_precision = RewardTermCfg(func=reach_reward, weight=5.0, params={"std": 0.06})
    cfg.rewards.tcp_distance = RewardTermCfg(func=reach_distance, weight=-1.0)
    cfg.events.push_robot = None
    cfg.events.base_external_force_torque = None
    cfg.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (-0.15, 0.15)}
    if stage == "eval":
        cfg.observations.policy.enable_corruption = False
        cfg.events.reset_base.params["pose_range"]["yaw"] = (-0.25, 0.25)
    if video:
        from isaaclab_tasks.manager_based.locomanipulation.go2_piper.go2_piper_reach_env_cfg import ReachPlaySceneCfg
        cfg.scene.ee_goal_marker = ReachPlaySceneCfg(num_envs=num_envs, env_spacing=4.0).ee_goal_marker.copy()
        goal.marker_asset_name = "ee_goal_marker"
        cfg.viewer.eye = (1.3, -2.7, 1.5)
        cfg.viewer.lookat = (0.7, 0.0, 0.5)
        cfg.viewer.resolution = (960, 540)
    return cfg
