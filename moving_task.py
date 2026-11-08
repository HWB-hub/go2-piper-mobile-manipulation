"""Three-dimensional moving targets with whole-body Go2+Piper tracking."""
from __future__ import annotations

import math

import torch
from isaaclab.envs.mdp import UniformVelocityCommand
from isaaclab.managers import ObservationTermCfg, RewardTermCfg
from isaaclab.utils.math import quat_apply_inverse, yaw_quat
from isaaclab_tasks.manager_based.locomanipulation.go2_piper.mdp.commands import EeGoalCommand

from cooperative_task import build_config, reach_distance, tcp_world, tensor


class MovingGoalCommand(EeGoalCommand):
    """Analytic world-space trajectories independent of the robot's motion."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.center_w = torch.zeros_like(self.goal_pos_w)
        self.amplitude = torch.zeros_like(self.goal_pos_w)
        self.phase = torch.zeros_like(self.goal_pos_w)
        self.harmonic = torch.ones_like(self.goal_pos_w)
        self.omega = torch.zeros(self.num_envs, 1, device=self.device)
        self.goal_vel_w = torch.zeros_like(self.goal_pos_w)
        self.goal_acc_w = torch.zeros_like(self.goal_pos_w)

    def _resample_command(self, env_ids):
        ids = torch.as_tensor(env_ids, device=self.device)
        count = len(ids)
        r = torch.empty(count, device=self.device)
        self.center_w[ids] = self._env.scene.env_origins[ids]
        self.center_w[ids, 0] += r.uniform_(1.15, 1.35)
        self.center_w[ids, 1] += r.uniform_(-0.08, 0.08)
        self.center_w[ids, 2] += r.uniform_(0.61, 0.65)
        self.amplitude[ids, 0] = r.uniform_(0.48, 0.62)
        self.amplitude[ids, 1] = r.uniform_(0.32, 0.48)
        self.amplitude[ids, 2] = r.uniform_(0.07, 0.10)
        self.phase[ids, 0] = r.uniform_(-0.4, 0.4)
        self.phase[ids, 1] = self.phase[ids, 0] * 2.0
        self.phase[ids, 2] = r.uniform_(-math.pi, math.pi)
        self.harmonic[ids] = torch.tensor([1.0, 2.0, 1.5], device=self.device)
        pattern = self.cfg.motion_pattern
        ellipse = torch.rand(count, device=self.device) < 0.5 if pattern == "mixed" else torch.full(
            (count,), pattern == "ellipse", device=self.device, dtype=torch.bool
        )
        ellipse_ids = ids[ellipse]
        self.harmonic[ellipse_ids, 1] = 1.0
        self.phase[ellipse_ids, 1] = self.phase[ellipse_ids, 0] + math.pi / 2.0
        periods = r.uniform_(*self.cfg.motion_period_range)
        direction = torch.where(torch.rand(count, device=self.device) < 0.5, -1.0, 1.0)
        self.omega[ids, 0] = direction * (2.0 * math.pi / periods)
        self.update_targets()

    def update_targets(self):
        # Episode time makes this method idempotent when the velocity command calls it too.
        elapsed = tensor(self._env.episode_length_buf).float().unsqueeze(-1) * self._env.step_dt
        u = (elapsed - 2.0).clamp(min=0.0)
        ramp = 1.0
        tau = torch.where(u < ramp, u.square() / (2.0 * ramp), u - ramp / 2.0)
        rate = (u / ramp).clamp(max=1.0)
        accel = ((u > 0.0) & (u < ramp)).float() / ramp
        frequency = self.omega * self.harmonic
        angle = tau * frequency + self.phase
        self.goal_pos_w[:] = self.center_w + self.amplitude * torch.sin(angle)
        self.goal_vel_w[:] = self.amplitude * torch.cos(angle) * frequency * rate
        self.goal_acc_w[:] = self.amplitude * (
            -torch.sin(angle) * frequency.square() * rate.square() + torch.cos(angle) * frequency * accel
        )
        robot = self.robot
        self.goal_pos_b[:] = quat_apply_inverse(
            yaw_quat(tensor(robot.data.root_quat_w)), self.goal_pos_w - tensor(robot.data.root_pos_w)
        )
        if self._marker is not None:
            pose = torch.zeros(self.num_envs, 7, device=self.device)
            pose[:, :3] = self.goal_pos_w
            pose[:, 3] = 1.0
            self._marker.write_root_pose_to_sim(pose)

    def _update_command(self):
        self.update_targets()

    def _update_metrics(self):
        self.metrics["position_error"] = reach_distance(self._env)


class TrackingVelocityCommand(UniformVelocityCommand):
    """Track a reachable arm standoff, including target-velocity feedforward [m/s]."""

    def _resample_command(self, env_ids):
        self.vel_command_b[env_ids] = 0.0

    def _update_command(self):
        command = self._env.command_manager.get_term("ee_goal")
        command.update_targets()
        robot = self._env.scene["robot"]
        quat = yaw_quat(tensor(robot.data.root_quat_w))
        relative = quat_apply_inverse(quat, command.goal_pos_w - tensor(robot.data.root_pos_w))
        feedforward = quat_apply_inverse(quat, command.goal_vel_w)
        tcp_relative = quat_apply_inverse(quat, tcp_world(self._env) - tensor(robot.data.root_pos_w))
        tcp_error = relative - tcp_relative
        heading = torch.atan2(relative[:, 1], relative[:, 0])
        target = torch.zeros_like(self.vel_command_b)
        # Move the base into a comfortable arm workspace and close persistent TCP error.
        correction = ((relative[:, 0] - 0.35) + tcp_error[:, 0]).clamp(-0.25, 0.55)
        target[:, 0] = (correction * torch.cos(heading).clamp(min=0.0) + feedforward[:, 0]).clamp(-0.25, 0.60)
        target[:, 1] = (relative[:, 1] * 0.7 + feedforward[:, 1] * 0.6).clamp(-0.20, 0.20)
        target[:, 2] = (heading * 1.3).clamp(-0.75, 0.75)
        limit = torch.tensor([0.8, 0.6, 1.2], device=self.device) * self._env.step_dt
        self.vel_command_b[:] += (target - self.vel_command_b).clamp(-limit, limit)


def target_velocity_base(env):
    robot = env.scene["robot"]
    return quat_apply_inverse(
        yaw_quat(tensor(robot.data.root_quat_w)), env.command_manager.get_term("ee_goal").goal_vel_w
    )


def tcp_velocity_world(env):
    tcp_world(env)  # Resolve the same two finger bodies used for position tracking.
    return tensor(env.scene["robot"].data.body_link_lin_vel_w)[:, env._tcp_ids].mean(dim=1)


def velocity_tracking_reward(env):
    error = torch.linalg.vector_norm(
        tcp_velocity_world(env) - env.command_manager.get_term("ee_goal").goal_vel_w, dim=-1
    )
    return torch.exp(-error / 0.25)


def moving_config(num_envs: int, device: str, seed: int, stage: str, pattern: str, video: bool, precision_refine: bool = False):
    cfg = build_config(num_envs, device, "eval" if stage == "eval" else "far", seed, video)
    cfg.episode_length_s = 26.0
    cfg.scene.env_spacing = 5.0
    cfg.commands.ee_goal.class_type = MovingGoalCommand
    cfg.commands.ee_goal.motion_pattern = pattern
    cfg.commands.ee_goal.motion_period_range = (28.0, 36.0) if stage == "slow" else (20.0, 28.0)
    cfg.commands.base_velocity.class_type = TrackingVelocityCommand
    # Append target velocity; the original 81 observation columns retain their ordering.
    cfg.observations.policy.target_velocity_b = ObservationTermCfg(func=target_velocity_base)
    cfg.rewards.tcp_velocity_tracking = RewardTermCfg(func=velocity_tracking_reward, weight=0.7)
    if precision_refine:
        cfg.rewards.ee_goal_tracking.weight = 10.0
        cfg.rewards.tcp_precision.weight = 16.0
        cfg.rewards.tcp_distance.weight = -3.0
        cfg.rewards.tcp_velocity_tracking.weight = 3.0
    if video:
        cfg.viewer.origin_type = "env"
        cfg.viewer.env_index = 0
        cfg.viewer.eye = (1.3, -3.4, 1.8)
        cfg.viewer.lookat = (0.9, 0.0, 0.5)
    return cfg
