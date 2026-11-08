"""Smooth curved locomotion and target-conditioned whole-body posture in SI units."""
from __future__ import annotations

import math
import torch
from isaaclab.managers import ObservationTermCfg, RewardTermCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse, yaw_quat, euler_xyz_from_quat
from moving_task import MovingGoalCommand, TrackingVelocityCommand, moving_config
from cooperative_task import tensor, tcp_world


class CurvedGoalCommand(MovingGoalCommand):
    """World-fixed trajectory generator. No robot pose is used to generate targets."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        n = self.num_envs
        self.speed = torch.zeros(n, device=self.device)
        self.modulation = torch.zeros_like(self.speed)
        self.kind = torch.zeros(n, dtype=torch.long, device=self.device)
        self.start_w = torch.zeros_like(self.goal_pos_w)

    def _resample_command(self, env_ids):
        ids = torch.as_tensor(env_ids, device=self.device)
        count = len(ids)
        r = torch.empty(count, device=self.device)
        profile = self.cfg.profile
        profiles = {
            "curves": ((0.14, 0.27), (0.32, 0.48), (0.08, 0.12), (16.0, 22.0)),
            "adaptive": ((0.15, 0.40), (0.40, 0.65), (0.16, 0.25), (14.0, 20.0)),
            "dynamic": ((0.20, 0.58), (0.40, 0.70), (0.16, 0.26), (13.0, 19.0)),
        }
        speeds, lateral, vertical, periods = profiles[profile]
        self.start_w[ids] = tensor(self._env.scene.env_origins)[ids]
        self.start_w[ids, 0] += r.uniform_(0.85, 1.10)
        self.start_w[ids, 1] += r.uniform_(-0.10, 0.10)
        self.start_w[ids, 2] += 0.64
        self.amplitude[ids, 0] = r.uniform_(0.65, 1.0)
        self.amplitude[ids, 1] = r.uniform_(*lateral)
        self.amplitude[ids, 2] = r.uniform_(*vertical)
        self.phase[ids, 0] = r.uniform_(-0.2, 0.2)
        self.phase[ids, 1] = r.uniform_(-0.15, 0.15)
        self.phase[ids, 2] = r.uniform_(-math.pi, math.pi)
        self.omega[ids, 0] = 2 * math.pi / r.uniform_(*periods)
        self.speed[ids] = r.uniform_(*speeds)
        self.modulation[ids] = r.uniform_(0.25, 0.55)
        pattern = self.cfg.motion_pattern
        self.kind[ids] = torch.randint(0, 3, (count,), device=self.device) if pattern == "mixed" else {
            "serpentine": 0, "orbit": 1, "figure8": 2
        }[pattern]
        # Random mirrored curves, while keeping forward-progress S paths.
        self.amplitude[ids, 1] *= torch.where(torch.rand(count, device=self.device) < 0.5, -1., 1.)
        self.center_w[ids] = self.start_w[ids]
        self.harmonic[ids] = 1.0
        self.update_targets()

    def update_targets(self):
        elapsed = tensor(self._env.episode_length_buf).float() * self._env.step_dt
        u = (elapsed - 2.0).clamp(min=0.)
        tau = torch.where(u < 2., u.square() / 4., u - 1.)
        rate = (u / 2.).clamp(max=1.)
        accel = ((u > 0.) & (u < 2.)).float() / 2.
        w = self.omega[:, 0]
        a = w * tau
        p = self.phase
        ax, ay, az = self.amplitude.unbind(-1)
        v, m = self.speed, self.modulation
        # Derivatives below are with respect to tau; apply the smooth start chain rule.
        x = v * (tau + m * torch.sin(a) / w)
        dx = v * (1. + m * torch.cos(a))
        ddx = -v * m * w * torch.sin(a)
        y = ay * (torch.sin(a + p[:, 1]) - torch.sin(p[:, 1]))
        dy = ay * w * torch.cos(a + p[:, 1])
        ddy = -ay * w.square() * torch.sin(a + p[:, 1])
        orbit = self.kind == 1
        eight = self.kind == 2
        x = torch.where(orbit | eight, ax * (torch.sin(a + p[:, 0]) - torch.sin(p[:, 0])), x)
        dx = torch.where(orbit | eight, ax * w * torch.cos(a + p[:, 0]), dx)
        ddx = torch.where(orbit | eight, -ax * w.square() * torch.sin(a + p[:, 0]), ddx)
        y = torch.where(orbit, ay * (torch.cos(p[:, 0]) - torch.cos(a + p[:, 0])), y)
        dy = torch.where(orbit, ay * w * torch.sin(a + p[:, 0]), dy)
        ddy = torch.where(orbit, ay * w.square() * torch.cos(a + p[:, 0]), ddy)
        y = torch.where(eight, ay * torch.sin(2 * a), y)
        dy = torch.where(eight, 2 * ay * w * torch.cos(2 * a), dy)
        ddy = torch.where(eight, -4 * ay * w.square() * torch.sin(2 * a), ddy)
        za = 0.7 * a + p[:, 2]
        z = az * torch.sin(za)
        dz = az * 0.7 * w * torch.cos(za)
        ddz = -az * (0.7 * w).square() * torch.sin(za)
        d = torch.stack((dx, dy, dz), -1)
        dd = torch.stack((ddx, ddy, ddz), -1)
        self.goal_pos_w[:] = self.start_w + torch.stack((x, y, z), -1)
        self.goal_vel_w[:] = d * rate[:, None]
        self.goal_acc_w[:] = dd * rate[:, None].square() + d * accel[:, None]
        robot = self.robot
        self.goal_pos_b[:] = quat_apply_inverse(yaw_quat(tensor(robot.data.root_quat_w)),
                                                self.goal_pos_w - tensor(robot.data.root_pos_w))
        if self._marker is not None:
            pose = torch.zeros(self.num_envs, 7, device=self.device)
            pose[:, :3] = self.goal_pos_w
            pose[:, 3] = 1.
            self._marker.write_root_pose_to_sim(pose)


class AdaptiveVelocityCommand(TrackingVelocityCommand):
    """Continuous feedback steering plus a phase clock, height and lean preferences."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.phase = torch.zeros(self.num_envs, device=self.device)
        self.frequency = torch.full_like(self.phase, 1. / .65)
        self.last_step = torch.full_like(self.phase, -1, dtype=torch.long)
        self.posture = torch.zeros(self.num_envs, 3, device=self.device)
        self.posture[:, 0] = .30

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        if hasattr(self, "phase"):
            self.phase[env_ids] = 0.
            self.frequency[env_ids] = 1. / .65
            self.last_step[env_ids] = -1
            self.posture[env_ids] = torch.tensor([.30, 0., 0.], device=self.device)

    def _update_command(self):
        goal = self._env.command_manager.get_term("ee_goal")
        goal.update_targets()
        robot = self._env.scene["robot"]
        quat = yaw_quat(tensor(robot.data.root_quat_w))
        relative = quat_apply_inverse(quat, goal.goal_pos_w - tensor(robot.data.root_pos_w))
        ff = quat_apply_inverse(quat, goal.goal_vel_w)
        error = quat_apply_inverse(quat, goal.goal_pos_w - tcp_world(self._env))
        heading = torch.atan2(relative[:, 1], relative[:, 0])
        target = torch.zeros_like(self.vel_command_b)
        target[:, 0] = (((relative[:, 0] - .35) + error[:, 0]).clamp(-.30, .65)
                         * torch.cos(heading).clamp(min=0.) + ff[:, 0]).clamp(
                             -.30, getattr(self.cfg, "forward_speed_limit", .95))
        # Prefer turning over crabbing; allow lateral corrections for a moving gripper target.
        target[:, 1] = (.45 * relative[:, 1] + .45 * ff[:, 1]).clamp(-.22, .22)
        target[:, 2] = (1.6 * heading + .45 * ff[:, 1]).clamp(-1.15, 1.15)
        if hasattr(goal,"spin_mask"):
            anchor_error=goal.anchor-tensor(robot.data.root_pos_w)
            anchor_error[:,2]=0.
            hold=quat_apply_inverse(quat,goal.anchor_velocity+1.5*anchor_error)
            assistance = getattr(self, "tcp_assistance_gain", 0.)
            if assistance:
                # During travel, let the base help an arm that is lagging its target.
                # The world-space target and anchor remain independent of the robot.
                travel_blend = (goal.anchor_velocity[:,:2].norm(dim=-1)/.10).clamp(0.,1.)
                hold[:,:2] += assistance * travel_blend[:,None] * error[:,:2].clamp(-.20,.20)
            _,_,actual_yaw=euler_xyz_from_quat(tensor(robot.data.root_quat_w))
            yaw_error=torch.atan2(torch.sin(goal.heading-actual_yaw),torch.cos(goal.heading-actual_yaw))
            spin_target=torch.stack((hold[:,0].clamp(-.4,.6),hold[:,1].clamp(-.25,.25),
                                      (goal.heading_rate+1.5*yaw_error).clamp(-.85,.85)),-1)
            target[goal.spin_mask]=spin_target[goal.spin_mask]
        curve_speed_limit = getattr(self.cfg, 'curve_speed_limit', None)
        if curve_speed_limit is not None:
            curves = ~goal.spin_mask
            speed = target[curves, :2].norm(dim=-1).clamp(min=1e-6)
            cap = curve_speed_limit
            if getattr(self.cfg, 'adaptive_curve_speed_limit', False):
                cap = (ff[curves, :2].norm(dim=-1) + .10).clamp(min=cap, max=.66)
            target[curves, :2] *= (cap / speed).clamp(max=1.)[:, None]
        if getattr(self.cfg, 'tilt_speed_guard', False):
            gravity = tensor(robot.data.projected_gravity_b)
            tilt = gravity[:, :2].norm(dim=-1)
            scale = (1. - .8 * ((tilt - .14) / .26).clamp(0., 1.))
            target[:, :2] *= scale[:, None]
        dt = self._env.step_dt
        limits = torch.tensor([.65, .5, 1.1], device=self.device) * dt
        step = tensor(self._env.episode_length_buf)
        fresh = step != self.last_step
        self.vel_command_b[fresh] += (target - self.vel_command_b).clamp(-limits, limits)[fresh]
        self.last_step.copy_(step)
        speed = torch.linalg.vector_norm(self.vel_command_b[:, :2], dim=-1) + .12 * self.vel_command_b[:, 2].abs()
        wanted = (1.20 + 1.35 * speed).clamp(1.2, 2.25)
        blend = self.cfg.adaptation
        wanted = (1. - blend) / .65 + blend * wanted
        self.frequency[fresh] += (wanted - self.frequency).clamp(-.5 * dt, .5 * dt)[fresh]
        self.phase[fresh] = torch.remainder(self.phase[fresh] + self.frequency[fresh] * dt, 1.)
        h_goal = goal.goal_pos_w[:, 2] - tensor(self._env.scene.env_origins)[:, 2]
        desired = torch.zeros_like(self.posture)
        desired[:, 0] = (.30 + .34 * (h_goal - .64)).clamp(.24, .38)
        desired[:, 1] = (-.32 * (h_goal - .64)).clamp(-.12, .12)
        desired[:, 2] = (-self.vel_command_b[:, 0] * self.vel_command_b[:, 2] / 9.81).clamp(-.09, .09)
        p_limits = torch.tensor([.035, .08, .1], device=self.device) * dt
        self.posture[fresh] += (desired - self.posture).clamp(-p_limits, p_limits)[fresh]


def command(env):
    return env.command_manager.get_term("base_velocity")


def gait_clock(env):
    angle = 2 * math.pi * command(env).phase
    return torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1)


def posture_context(env):
    height = tensor(env.scene["robot"].data.root_pos_w)[:, 2] - tensor(env.scene.env_origins)[:, 2]
    return torch.cat((height[:, None], command(env).posture), dim=-1)


def target_acceleration(env):
    return quat_apply_inverse(yaw_quat(tensor(env.scene["robot"].data.root_quat_w)),
                              env.command_manager.get_term("ee_goal").goal_acc_w)


def gait_frequency(env):
    return command(env).frequency[:, None]


def feet_state(env):
    """Physical sole-center positions and velocities [m, m/s], not calf origins."""
    robot = env.scene["robot"]
    if not hasattr(env, "_adaptive_foot_ids"):
        env._adaptive_foot_ids = [robot.body_names.index(f"{x}_calf") for x in ("FL", "FR", "RL", "RR")]
        sensor = env.scene.sensors["contact_forces"]
        env._adaptive_sensor_ids = [sensor.body_names.index(f"{x}_calf") for x in ("FL", "FR", "RL", "RR")]
    ids = env._adaptive_foot_ids
    quat = tensor(robot.data.body_quat_w)[:, ids]
    local = torch.zeros(env.num_envs, 4, 3, device=env.device)
    local[:, :, 2] = -.213  # Verified Go2 URDF fixed foot joint, sphere radius .022 m.
    offset = quat_apply(quat, local)
    pos = tensor(robot.data.body_pos_w)[:, ids] + offset
    vel = tensor(robot.data.body_link_lin_vel_w)[:, ids] + torch.cross(
        tensor(robot.data.body_ang_vel_w)[:, ids], offset, dim=-1)
    force = tensor(env.scene.sensors["contact_forces"].data.net_forces_w_history).norm(dim=-1).max(dim=1)[0]
    contact = force[:, env._adaptive_sensor_ids] > 3.
    return pos, vel, contact


def adaptive_height_reward(env):
    state = posture_context(env)
    return torch.exp(-((state[:, 0] - state[:, 1]) / .045).square())


def adaptive_attitude_reward(env):
    r, p, _ = euler_xyz_from_quat(tensor(env.scene["robot"].data.root_quat_w))
    r = torch.atan2(torch.sin(r), torch.cos(r))
    p = torch.atan2(torch.sin(p), torch.cos(p))
    ref = command(env).posture
    return torch.exp(-((p - ref[:, 1]) / .16).square() - ((r - ref[:, 2]) / .12).square())


def excessive_tilt(env):
    grav = tensor(env.scene["robot"].data.projected_gravity_b)
    return (torch.linalg.vector_norm(grav[:, :2], dim=-1) - .20).clamp(min=0.).square()


def adaptive_contact_reward(env):
    _, _, contact = feet_state(env)
    c = command(env)
    speed = torch.linalg.vector_norm(c.vel_command_b[:, :2], dim=-1) + .12 * c.vel_command_b[:, 2].abs()
    duty = (.66 - .22 * speed).clamp(.50, .66)
    phases = torch.remainder(c.phase[:, None] + torch.tensor([0., .5, .5, 0.], device=env.device), 1.)
    stance = phases < duty[:, None]
    moving_score = (stance == contact).float().mean(-1)
    return torch.where(speed > .06, moving_score, contact.float().mean(-1))


def foot_slip(env):
    _, vel, contact = feet_state(env)
    return (vel[:, :, :2].square().sum(-1) * contact).sum(-1)


def actual_foot_impact(env):
    """Sole vertical impact speed squared [m^2/s^2] at first foot contact."""
    _, vel, _ = feet_state(env)
    first = tensor(env.scene.sensors["contact_forces"].compute_first_contact(env.step_dt))[:, env._adaptive_sensor_ids]
    return (((-vel[:, :, 2] - .10).clamp(min=0.)).square() * first).sum(-1)


def foot_clearance_reward(env):
    pos, _, contact = feet_state(env)
    speed = command(env).vel_command_b[:, :2].norm(dim=-1)+.25*command(env).vel_command_b[:,2].abs()
    height = pos[:, :, 2] - tensor(env.scene.env_origins)[:, None, 2] - .022
    desired = (.035 + .035 * speed).clamp(max=.07)
    score = torch.exp(-((height - desired[:, None]) / .03).square())
    return (score * (~contact)).mean(-1) * (speed > .06)


def support_penalty(env):
    _, _, contact = feet_state(env)
    speed = command(env).vel_command_b[:, :2].norm(dim=-1)
    minimum = torch.where(speed > .55, 1., 2.)
    return (minimum - contact.sum(-1)).clamp(min=0.)


def stagnant_foot_penalty(env):
    """Discourage leaving one leg planted or airborne over multiple gait cycles."""
    feet_state(env)
    data = env.scene.sensors["contact_forces"].data
    ids = env._adaptive_sensor_ids
    stance = tensor(data.current_contact_time)[:, ids]
    air = tensor(data.current_air_time)[:, ids]
    c = command(env)
    moving = c.vel_command_b[:, :2].norm(dim=-1) + .15 * c.vel_command_b[:, 2].abs() > .10
    return ((stance - .85).clamp(0., 2.).square() + (air - .55).clamp(0., 2.).square()).mean(-1) * moving


def adaptive_config(num_envs, device, seed, profile, pattern, video=False, evaluate=False, legacy=False):
    cfg = moving_config(num_envs, device, seed, "eval" if evaluate else "normal", "mixed", video)
    cfg.episode_length_s = 28.
    cfg.scene.env_spacing = 18.
    cfg.scene.ground.spawn.size = (1000., 1000.)
    cfg.commands.ee_goal.class_type = CurvedGoalCommand
    cfg.commands.ee_goal.profile = profile
    cfg.commands.ee_goal.motion_pattern = pattern
    # Reduce unphysical launch impulses, retain randomized joints, arm load, friction, gains and COM.
    cfg.events.reset_base.params["velocity_range"] = {k: (-.08, .08) for k in ("x", "y", "z", "roll", "pitch", "yaw")}
    cfg.events.reset_robot_joints.params["position_range"] = (.85, 1.15)
    cfg.events.reset_arm_joints.params["position_range"] = (-.20, .20)
    if legacy:
        return cfg
    cfg.commands.base_velocity.class_type = AdaptiveVelocityCommand
    # These commands are continuously derived from the target, so periodic random
    # resampling must not reset speed, gait phase, or body-height preferences.
    cfg.commands.base_velocity.resampling_time_range = (1000., 1000.)
    cfg.commands.base_velocity.adaptation = .35 if profile == "curves" else 1.
    cfg.observations.policy.gait_phase = ObservationTermCfg(func=gait_clock)
    cfg.observations.policy.posture_context = ObservationTermCfg(func=posture_context)
    cfg.observations.policy.target_acceleration = ObservationTermCfg(func=target_acceleration)
    cfg.observations.policy.gait_frequency = ObservationTermCfg(func=gait_frequency)
    cfg.rewards.base_height = RewardTermCfg(func=adaptive_height_reward, weight=2.5)
    cfg.rewards.flat_orientation_l2 = RewardTermCfg(func=excessive_tilt, weight=-16.)
    cfg.rewards.adaptive_attitude = RewardTermCfg(func=adaptive_attitude_reward, weight=1.0)
    cfg.rewards.lin_vel_z_l2.weight = -3.
    cfg.rewards.ang_vel_xy_l2.weight = -.15
    cfg.rewards.action_rate_l2.weight = -.12
    cfg.rewards.action_acc_l2.weight = -.015
    cfg.rewards.joint_mirror.weight = -.05
    cfg.rewards.hip_abduction_l2.weight = -.6
    cfg.rewards.phase_gait_soft = RewardTermCfg(func=adaptive_contact_reward, weight=3.)
    cfg.rewards.stagnant_foot = RewardTermCfg(func=stagnant_foot_penalty, weight=-1.5)
    cfg.rewards.foot_placement_and_turning = None
    cfg.rewards.rear_leg_support_and_push = None
    cfg.rewards.foot_clearance = RewardTermCfg(func=foot_clearance_reward, weight=.8)
    cfg.rewards.feet_slide = RewardTermCfg(func=foot_slip, weight=-1.)
    cfg.rewards.foot_impact_velocity = RewardTermCfg(func=actual_foot_impact, weight=-2.)
    cfg.rewards.stance_duty = RewardTermCfg(func=support_penalty, weight=-1.5)
    cfg.rewards.ee_goal_tracking.weight = 8.
    cfg.rewards.tcp_precision.weight = 7.
    cfg.rewards.tcp_distance.weight = -2.
    cfg.rewards.tcp_velocity_tracking.weight = 1.
    return cfg
