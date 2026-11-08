"""Phase-shaped whole-body locomotion without kinematic pose overrides.

References: Improbable-AI/walk-these-ways and unitreerobotics/unitree_rl_lab.
See REFERENCES.md for exact versions and the changes made for Go2+Piper.
"""
import torch
from isaaclab.managers import EventTermCfg, RewardTermCfg
from isaaclab.envs.mdp import push_by_setting_velocity
from isaaclab.utils.math import quat_apply_inverse, yaw_quat
from adaptive_task import AdaptiveVelocityCommand, feet_state, command
from cooperative_task import tensor
from spin_task import spin_config
from gait_reference import references, equivalent_speed


def planned_gait_velocity(env):
    """Use rotation feedforward for foot placement while the body follows a turn.

    Heading feedback can temporarily oppose a turn when the body leads its reference;
    this must not reverse the stepping pattern or disable lifting every other step.
    """
    velocity = command(env).vel_command_b.clone()
    goal = env.command_manager.get_term('ee_goal')
    rotating = goal.spin_mask & (goal.heading_rate.abs() > .05)
    velocity[rotating,2] = goal.heading_rate[rotating]
    return velocity


class NaturalVelocityCommand(AdaptiveVelocityCommand):
    """Keep navigation feedback; use an observable, smoothly changing trot cadence."""
    tcp_assistance_gain = 1.0

    def _update_command(self):
        step = tensor(self._env.episode_length_buf)
        fresh = step != self.last_step
        old_phase, old_frequency = self.phase.clone(), self.frequency.clone()
        super()._update_command()
        target = (1.65 + .90 * equivalent_speed(planned_gait_velocity(self._env))).clamp(1.65, 2.3)
        if getattr(self.cfg, "running_gait", False):
            target = (1.65 + .98 * equivalent_speed(planned_gait_velocity(self._env))).clamp(1.65, 3.25)
        dt = self._env.step_dt
        self.frequency[fresh] = (old_frequency + (target-old_frequency).clamp(-.45*dt, .45*dt))[fresh]
        self.phase[fresh] = torch.remainder(old_phase + self.frequency * dt, 1.)[fresh]


def gait_state(env):
    """Cache one physical-foot read and reference per environment step."""
    key = env.common_step_counter
    if getattr(env, '_natural_state_key', None) != key:
        pos, vel, contact = feet_state(env)
        c = command(env)
        desired, height, xy, moving = references(
            c.phase, c.frequency, planned_gait_velocity(env),
            running=getattr(c.cfg, "running_gait", False),
        )
        sensor = env.scene.sensors['contact_forces']
        force = tensor(sensor.data.net_forces_w)[:, env._adaptive_sensor_ids].norm(dim=-1)
        env._natural_state = (pos, vel, contact, desired, height, xy, moving, force)
        env._natural_state_key = key
    return env._natural_state


def swing_force_cost(env):
    """Penalize force [N] on feet that should be in swing, including pure yaw."""
    _, _, _, desired, _, _, _, force = gait_state(env)
    return ((1. - desired) * (1. - torch.exp(-force.square() / 35.**2))).mean(-1)


def stance_velocity_cost(env):
    """Penalize moving support feet using physical sole velocity [m/s]."""
    _, vel, _, desired, _, _, _, _ = gait_state(env)
    return (desired * (1. - torch.exp(-vel.square().sum(-1) / .20**2))).mean(-1)


def phase_clearance_cost(env):
    """Track lift-and-lower height [m]; reward is not gated by actual liftoff."""
    pos, _, _, _, height, _, _, _ = gait_state(env)
    actual = pos[:, :, 2] - tensor(env.scene.env_origins)[:, None, 2] - .022
    cost = ((actual - height) / .04).square().clamp(max=6.)
    return .6 * cost.mean(-1) + .4 * cost.max(-1).values


def turning_placement_cost(env):
    """Match a relaxed xy foot reference including both yaw-induced components."""
    pos, _, _, _, _, xy, _, _ = gait_state(env)
    robot = env.scene['robot']
    relative = pos - tensor(robot.data.root_pos_w)[:, None, :]
    quat = yaw_quat(tensor(robot.data.root_quat_w))[:, None, :].expand(-1, 4, -1)
    actual = quat_apply_inverse(quat.reshape(-1,4), relative.reshape(-1,3)).reshape(env.num_envs,4,3)
    return ((actual[:, :, :2] - xy) / .075).square().clamp(max=6.).mean((1,2))


def stuck_foot_cost(env):
    """Every leg must complete cycles while translating or turning [s]."""
    _, _, _, _, _, _, moving, _ = gait_state(env)
    data = env.scene.sensors['contact_forces'].data
    stance = tensor(data.current_contact_time)[:, env._adaptive_sensor_ids]
    air = tensor(data.current_air_time)[:, env._adaptive_sensor_ids]
    max_stance = .95 / command(env).frequency[:, None]
    max_air = .70 / command(env).frequency[:, None]
    return (((stance-max_stance).clamp(0., 1.)/.25).square() + ((air-max_air).clamp(0.,1.)/.25).square()).mean(-1) * moving


def air_time_balance_cost(env):
    """Discourage one leg using persistently different cycle times [s]."""
    _, _, _, _, _, _, moving, _ = gait_state(env)
    data = env.scene.sensors['contact_forces'].data
    air = tensor(data.last_air_time)[:, env._adaptive_sensor_ids].clamp(max=.7)
    stance = tensor(data.last_contact_time)[:, env._adaptive_sensor_ids].clamp(max=.7)
    return (air.var(-1, correction=0) + stance.var(-1, correction=0)) * command(env).frequency.square() * moving


def yaw_smoothness_cost(env):
    """Penalize changes in actual yaw rate [rad/s], excluding reset discontinuities."""
    rate = tensor(env.scene['robot'].data.root_ang_vel_w)[:,2]
    previous = getattr(env, '_natural_previous_yaw_rate', rate)
    cost = ((rate-previous)/.12).square().clamp(max=6.)
    env._natural_previous_yaw_rate = rate.clone()
    return cost * (tensor(env.episode_length_buf)>1)


def genuine_lift_cost(env):
    """A tiny loss of contact does not count as a completed foot lift [m, s]."""
    pos, _, _, _, _, _, moving, _ = gait_state(env)
    height = pos[:,:,2] - tensor(env.scene.env_origins)[:,None,2] - .022
    timer = getattr(env, '_natural_low_clearance_time', torch.zeros_like(height))
    clear = (height > .025) | (moving[:,None] < .2) | (tensor(env.episode_length_buf)[:,None] <= 1)
    timer = torch.where(clear, 0., timer + env.step_dt)
    env._natural_low_clearance_time = timer
    excess = ((timer-1.3/command(env).frequency[:,None]).clamp(0.,1.)/.3).square()
    return (.5*excess.mean(-1)+.5*excess.max(-1).values)*moving


def pre_contact_impact_cost(env):
    """Use previous-step downward sole speed [m/s], before the contact impulse."""
    _, vel, _, _, _, _, _, _ = gait_state(env)
    previous = getattr(env, '_natural_previous_foot_vz', vel[:,:,2])
    first = tensor(env.scene.sensors['contact_forces'].compute_first_contact(env.step_dt))[:,env._adaptive_sensor_ids]
    cost = ((-previous-.10).clamp(min=0.).square()*first).sum(-1)
    env._natural_previous_foot_vz = vel[:,:,2].clone()
    return cost * (tensor(env.episode_length_buf)>1)


def excess_travel_speed_cost(env):
    actual = tensor(env.scene['robot'].data.root_lin_vel_w)[:, :2].norm(dim=-1)
    requested = command(env).vel_command_b[:, :2].norm(dim=-1)
    return (actual - requested - .10).clamp(min=0.).square()


def run_contact_cost(env):
    _, _, contact, desired, _, _, _, _ = gait_state(env)
    speed = command(env).vel_command_b[:, 0]
    running = ((speed - .7) / .65).clamp(0., 1.)
    # Penalize actual support during the intended flight intervals.
    return running * ((1. - desired).square() * contact.float()).sum(dim=1)


def run_flight_reward(env):
    pos, _, contact, _, _, _, _, _ = gait_state(env)
    c = command(env)
    speed = tensor(env.scene['robot'].data.root_lin_vel_w)[:, 0]
    running = ((torch.minimum(c.vel_command_b[:, 0], speed) - 1.) / .35).clamp(0., 1.)
    phase = torch.remainder(c.phase, .5)
    flight_window = (1. - (phase - .44).abs() / .085).clamp(0., 1.)
    floor = tensor(env.scene.env_origins)[:, 2]
    clearance = (pos[:, :, 2] - floor[:, None] - .022).min(dim=1).values
    # Dense lead-in rewards the lowest foot; the bonus requires all contacts clear.
    return running * flight_window * (
        .5 * (clearance / .025).clamp(0., 1.) + (~contact.any(dim=1)).float()
    )


def unstable_body_motion_cost(env):
    robot = env.scene['robot'].data
    tilt = tensor(robot.projected_gravity_b)[:, :2].norm(dim=-1)
    angular_speed = tensor(robot.root_ang_vel_w)[:, :2].square().sum(dim=-1)
    return ((tilt - .15) / .25).clamp(0., 1.) * angular_speed


def natural_config(num_envs, device, seed, profile, pattern, video=False, evaluate=False, legacy=False, recipe='natural'):
    cfg = spin_config(num_envs, device, seed, profile, pattern, video, evaluate, legacy)
    if recipe == 'baseline':
        return cfg
    cfg.commands.base_velocity.class_type = NaturalVelocityCommand
    if pattern == 'walk_run':
        cfg.commands.base_velocity.forward_speed_limit = 2.15
        cfg.commands.base_velocity.running_gait = True
        if not evaluate:
            cfg.rewards.flat_orientation_l2.weight = -40.
            cfg.rewards.terminating.weight = -250.
    if recipe == 'stability_capped':
        cfg.commands.base_velocity.curve_speed_limit = .42
        cfg.commands.base_velocity.adaptive_curve_speed_limit = True
    if recipe == 'stability_guard':
        cfg.commands.base_velocity.tilt_speed_guard = True
    if not evaluate:
        # Rehearse curved travel sufficiently while retaining both turn skills.
        cfg.commands.ee_goal.spin_probability = .40
        cfg.commands.ee_goal.transition_probability = .20
    # Replace conflicting or inactive swing terms; retain whole-body task feedback.
    cfg.rewards.phase_gait_soft = None
    cfg.rewards.feet_air_time = None
    cfg.rewards.foot_clearance = None
    cfg.rewards.stagnant_foot = None
    cfg.rewards.joint_mirror = None
    cfg.rewards.contact_load_balance.weight = .2
    cfg.rewards.dof_torques_l2 = None  # Same function was also charged as joint_torques_l2.
    cfg.rewards.action_rate_l2.weight = -.10
    cfg.rewards.action_acc_l2.weight = -.025
    # Restore whole-body tracking accuracy after the gait acquisition stage.
    # Keep the per-leg clearance and smooth-turn constraints active.
    cfg.rewards.tcp_precision.weight = 6.
    cfg.rewards.tcp_distance.weight = -8.
    # Low-speed manipulation needs closer velocity tracking than the broad
    # original 0.5 m/s tolerance; otherwise gait can propel past the target.
    cfg.rewards.track_lin_vel_xy_exp.params['std'] = .25
    cfg.rewards.track_lin_vel_xy_exp.weight = 6.
    cfg.rewards.track_ang_vel_z_exp.params['std'] = .45
    cfg.rewards.track_ang_vel_z_exp.weight = 3.
    cfg.rewards.swing_force = RewardTermCfg(func=swing_force_cost, weight=-6.)
    cfg.rewards.stance_velocity = RewardTermCfg(func=stance_velocity_cost, weight=-3.)
    cfg.rewards.phase_clearance = RewardTermCfg(func=phase_clearance_cost, weight=-7.)
    cfg.rewards.turning_placement = RewardTermCfg(func=turning_placement_cost, weight=-1.)
    cfg.rewards.cycle_completion = RewardTermCfg(func=stuck_foot_cost, weight=-1.0)
    cfg.rewards.air_time_balance = RewardTermCfg(func=air_time_balance_cost, weight=-.8)
    cfg.rewards.yaw_smoothness = RewardTermCfg(func=yaw_smoothness_cost, weight=-.3)
    cfg.rewards.genuine_lift = RewardTermCfg(func=genuine_lift_cost, weight=-.8)
    cfg.rewards.foot_impact_velocity = RewardTermCfg(func=pre_contact_impact_cost, weight=-8.)
    if pattern == 'walk_run' and not evaluate:
        cfg.rewards.run_contact_timing = RewardTermCfg(func=run_contact_cost, weight=-3.)
        cfg.rewards.run_flight = RewardTermCfg(func=run_flight_reward, weight=9.)
    if recipe in ('stability', 'stability_transition', 'stability_balanced', 'stability_recovery', 'stability_guard') and not evaluate:
        cfg.rewards.flat_orientation_l2.weight = -40.
        cfg.rewards.ang_vel_xy_l2.weight = -.4
        cfg.rewards.terminating.weight = -250.
        if recipe == 'stability_transition':
            cfg.commands.ee_goal.spin_probability = .65
            cfg.commands.ee_goal.transition_probability = .45
        if recipe in ('stability_balanced', 'stability_recovery', 'stability_guard'):
            cfg.commands.ee_goal.spin_probability = .50
            cfg.commands.ee_goal.transition_probability = .30
            cfg.rewards.flat_orientation_l2.weight = -60.
            cfg.rewards.terminating.weight = -400.
            cfg.rewards.excess_travel_speed = RewardTermCfg(func=excess_travel_speed_cost, weight=-35.)
        if recipe == 'stability_recovery':
            cfg.commands.ee_goal.spin_probability = .40
            cfg.commands.ee_goal.transition_probability = .25
            cfg.rewards.unstable_body_motion = RewardTermCfg(func=unstable_body_motion_cost, weight=-2.)
            cfg.events.push_robot = EventTermCfg(
                func=push_by_setting_velocity,
                mode='interval',
                interval_range_s=(2., 5.),
                params={'velocity_range': {'x': (-.08, .08), 'y': (-.18, .18)}},
            )
    if recipe in ('stability_support', 'stability_capped') and not evaluate:
        cfg.commands.ee_goal.spin_probability = .40
        cfg.commands.ee_goal.transition_probability = .25
        cfg.rewards.flat_orientation_l2.weight = -55.
        cfg.rewards.ang_vel_xy_l2.weight = -.65
        cfg.rewards.terminating.weight = -350.
        cfg.rewards.stance_velocity.weight = -5.
        cfg.rewards.excess_travel_speed = RewardTermCfg(func=excess_travel_speed_cost, weight=-25.)
    if pattern == 'serpentine' and not evaluate:
        cfg.rewards.yaw_smoothness.weight = -.60
        cfg.rewards.action_acc_l2.weight = -.045
    return cfg
