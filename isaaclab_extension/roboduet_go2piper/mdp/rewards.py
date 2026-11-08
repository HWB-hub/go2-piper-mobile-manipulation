"""RoboDuet 的奖励函数（逐条移植 spec/02_rewards.md 的 33 项表）。

约定：
    - 每一项返回 (num_envs,) 的【原始值】，权重写在 env cfg 的 RewardsCfg 里；
      Isaac Lab 的 RewardManager 会自动乘 weight × dt（dt = 0.02），
      正好等于作者 `_prepare_reward_function` 里把 scale 乘 dt 的写法。
    - 作者表里两项权重为 0 的（jump / raibert_heuristic）不在这里实现，注释说明。
    - dog / arm 两个奖励缓冲怎么切分不在本文件里做：见 train_roboduet.py 的 split_rewards()
      （规则：dog = 全部 − 两个 vis_*；arm = dog − tracking_lin_vel − tracking_ang_vel）。

关节索引（与作者一致，20 DOF）：
    0..11  腿；12..17 臂的 6 个转动关节；18..19 夹爪（prismatic，无动作）
"""

from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat, quat_from_euler_xyz, quat_rotate_inverse

from . import arm_kinematics as ak

# 作者 asset_config.soft_dof_pos_limit = 0.9（config_go1 设的）
SOFT_DOF_POS_LIMIT = 0.9
# wtw_config: sigma_rew_neg
SIGMA_REW_NEG = 0.02


# ----------------------------------------------------------------------
# 小工具
# ----------------------------------------------------------------------
def _robot(env):
    return env.scene["robot"]


def _rd(env):
    return env.rd


def _leg_ids(env) -> torch.Tensor:
    """★不能再用 [:12] 这类硬切片：作者的 Isaac Gym DOF 顺序是"按腿分组"
    （FL_hip, FL_thigh, FL_calf, FR_hip, ...），而 Isaac Lab 里的关节顺序是
    "按关节类型分组"（4 个 hip → piper_joint1 → 4 个 thigh → ...）。
    所以腿/臂的取数一律按名字解析出来的下标。"""
    return env.rd_leg_joint_ids


def _arm_ids(env) -> torch.Tensor:
    return env.rd_arm_joint_ids


def _piper_ids(env) -> torch.Tensor:
    """作者用 [12:] 覆盖的 8 个 piper 关节（6 转动 + 2 夹爪）。"""
    return env.rd_piper_joint_ids


def _leg_action_ids(env) -> torch.Tensor:
    """腿动作在 18 维动作向量里的位置（= 0..11，动作项按 腿→臂 声明）。"""
    return torch.arange(12, device=env.device)


def _hip_action_ids(env) -> torch.Tensor:
    """4 个髋关节在【动作向量】里的位置（作者写的是 [0,3,6,9]，在 Isaac Lab 里顺序不同）。"""
    ids = getattr(env, "_rd_hip_action_ids", None)
    if ids is None:
        names = _robot(env).joint_names
        ids = torch.tensor(
            [i for i, j in enumerate(_leg_ids(env).tolist()) if names[j].endswith("_hip_joint")], device=env.device
        )
        env._rd_hip_action_ids = ids
    return ids


def _command(env, name: str) -> torch.Tensor:
    return env.command_manager.get_term(name).command


def _split_torque(env):
    return _robot(env).data.applied_torque.torch


def _soft_limits(env) -> torch.Tensor:
    """按 0.9 收缩后的软限位（作者 _process_dof_props 的算法）。"""
    lim = _robot(env).data.joint_pos_limits.torch  # (1, 20, 2)
    m = (lim[..., 0] + lim[..., 1]) / 2
    r = lim[..., 1] - lim[..., 0]
    lo = m - 0.5 * r * SOFT_DOF_POS_LIMIT
    hi = m + 0.5 * r * SOFT_DOF_POS_LIMIT
    return torch.stack([lo, hi], dim=-1)


# ----------------------------------------------------------------------
# 主跟踪项
# ----------------------------------------------------------------------
def tracking_lin_vel(env) -> torch.Tensor:
    cmd = _command(env, "velocity")
    robot = _robot(env)
    return torch.exp(-torch.sum(torch.square(cmd[:, :2] - robot.data.root_lin_vel_b.torch[:, :2]), dim=1) / 0.25)


def tracking_ang_vel(env) -> torch.Tensor:
    cmd = _command(env, "velocity")
    robot = _robot(env)
    return torch.exp(-torch.square(cmd[:, 2] - robot.data.root_ang_vel_b.torch[:, 2]) / 0.25)


def orientation_control(env) -> torch.Tensor:
    """惩罚机身实际 roll/pitch 与指令不一致（作者 §5.1）。"""
    cmd = _command(env, "body_pose")
    robot = _robot(env)
    q = robot.data.root_quat_w.torch
    n = q.shape[0]
    dev = q.device
    ones = torch.ones(n, device=dev)
    zeros = torch.zeros(n, device=dev)
    # q_roll = quatFromAngleAxis(-roll, [1,0,0]); q_pitch = quatFromAngleAxis(-pitch, [0,1,0])
    half_r = -cmd[:, 1] / 2
    q_roll = torch.stack([torch.cos(half_r), torch.sin(half_r), zeros, zeros], dim=-1)
    half_p = -cmd[:, 0] / 2
    q_pitch = torch.stack([torch.cos(half_p), zeros, torch.sin(half_p), zeros], dim=-1)
    from isaaclab.utils.math import quat_mul

    q_des = quat_mul(q_roll, q_pitch)
    g_hat = torch.tensor([0.0, 0.0, -1.0], device=dev).unsqueeze(0).repeat(n, 1)
    g_des = quat_rotate_inverse(q_des, g_hat)
    pg = robot.data.projected_gravity_b.torch
    return torch.sum(torch.square(pg[:, :2] - g_des[:, :2]), dim=1)


def orientation_heuristic(env) -> torch.Tensor:
    """作者 §5.2：按 pitch 与"臂伸展方向"的分段启发式惩罚。"""
    from .observations import base_roll_pitch

    arm_cmd = _command(env, "arm_pose")
    _, pitch = base_roll_pitch(env)
    delta_z = arm_cmd[:, 0] * torch.sin(arm_cmd[:, 1]) + ak.NOMINAL_BASE_HEIGHT - _robot(env).data.root_pos_w.torch[:, 2]
    thres = 0.1
    r = torch.zeros_like(pitch)
    low = delta_z < -thres
    high = delta_z > thres + 0.3
    r[low] = torch.square(pitch[low] - 0.4)
    r[high] = torch.square(pitch[high] + 0.3)
    return r


def arm_manip_commands_tracking_combine(env) -> torch.Tensor:
    """机械臂末端 6D 跟踪主奖励（带位置/姿态权重课程）。"""
    cmd = _command(env, "arm_pose")
    lpy = ak.lpy_in_base(env)
    lpy_err = torch.sum(torch.abs(lpy - cmd[:, 0:3]) / torch.tensor(ak.LPY_RANGE, device=lpy.device), dim=1)
    abg = ak.abg_in_base(env)
    rpy_err = torch.sum(torch.abs(abg - cmd[:, 3:6]) / torch.tensor(ak.RPY_RANGE, device=lpy.device), dim=1)

    lpy_start, lpy_end = env.rd_manip_w["lpy_start"], env.rd_manip_w["lpy_end"]
    rpy_start = env.rd_manip_w["rpy_start"]
    trans = env.rd_manip_w["transition_iters"]
    arm_iter = max(0, env.rd_iteration - env.rd_switch_iter)
    if trans <= 0:
        alpha = 1.0
    else:
        alpha = max(0.0, min(1.0, arm_iter / float(trans))) ** env.rd_manip_w["transition_power"]
    w_lpy = lpy_start + (lpy_end - lpy_start) * alpha
    if env.rd_manip_w["keep_sum_constant"]:
        w_rpy = (lpy_start + rpy_start) - w_lpy
    else:
        w_rpy = rpy_start + (env.rd_manip_w["rpy_end"] - rpy_start) * alpha
    return torch.exp(-(w_lpy * lpy_err + w_rpy * rpy_err))


def vis_manip_commands_tracking_lpy(env) -> torch.Tensor:
    cmd = _command(env, "arm_pose")
    lpy = ak.lpy_in_base(env)
    return torch.exp(-torch.sum(torch.abs(lpy - cmd[:, 0:3]) / torch.tensor(ak.LPY_RANGE, device=lpy.device), dim=1))


def vis_manip_commands_tracking_rpy(env) -> torch.Tensor:
    cmd = _command(env, "arm_pose")
    abg = ak.abg_in_base(env)
    return torch.exp(-torch.sum(torch.abs(abg - cmd[:, 3:6]) / torch.tensor(ak.RPY_RANGE, device=abg.device), dim=1))


# ----------------------------------------------------------------------
# 机身 / 腿的惩罚项
# ----------------------------------------------------------------------
def lin_vel_z(env) -> torch.Tensor:
    return torch.square(_robot(env).data.root_lin_vel_b.torch[:, 2])


def ang_vel_xy(env) -> torch.Tensor:
    return torch.sum(torch.square(_robot(env).data.root_ang_vel_b.torch[:, :2]), dim=1)


def torques(env) -> torch.Tensor:
    # 作者没有切片：20 个 DOF 全算
    return torch.sum(torch.square(_split_torque(env)), dim=1)


def dof_vel(env) -> torch.Tensor:
    return torch.sum(torch.square(_robot(env).data.joint_vel.torch[:, _leg_ids(env)]), dim=1)


def dof_acc(env) -> torch.Tensor:
    robot = _robot(env)
    return torch.sum(
        torch.square((_rd(env).last_dof_vel - robot.data.joint_vel.torch)[:, _leg_ids(env)] / env.step_dt), dim=1
    )


def action_rate(env) -> torch.Tensor:
    rd = _rd(env)
    ids = _leg_action_ids(env)
    return torch.sum(torch.square(rd.last_actions[:, ids] - rd.raw_actions[:, ids]), dim=1)


def hip_action_l2(env) -> torch.Tensor:
    a = _rd(env).raw_actions
    return torch.sum(torch.square(a[:, _hip_action_ids(env)]), dim=1)


def loco_energy(env) -> torch.Tensor:
    tau = _split_torque(env)[:, _leg_ids(env)]
    qd = _robot(env).data.joint_vel.torch[:, _leg_ids(env)]
    return torch.sum(torch.square(tau * qd), dim=1)


def dof_pos_limits(env) -> torch.Tensor:
    q = _robot(env).data.joint_pos.torch
    lim = _soft_limits(env)
    return torch.sum(
        torch.clamp(lim[..., 0] - q, min=0.0) + torch.clamp(q - lim[..., 1], min=0.0), dim=1
    )


def action_smoothness_1(env) -> torch.Tensor:
    robot = _robot(env)
    rd = _rd(env)
    ids = _leg_ids(env)
    diff = torch.square(robot.data.joint_pos_target.torch[:, ids] - rd.last_joint_pos_target[:, ids])
    diff = diff * (rd.last_actions[:, _leg_action_ids(env)] != 0)
    return torch.sum(diff, dim=1)


def action_smoothness_2(env) -> torch.Tensor:
    robot = _robot(env)
    rd = _rd(env)
    ids = _leg_ids(env)
    diff = torch.square(
        robot.data.joint_pos_target.torch[:, ids]
        - 2 * rd.last_joint_pos_target[:, ids]
        + rd.last_last_joint_pos_target[:, ids]
    )
    diff = diff * (rd.last_actions[:, _leg_action_ids(env)] != 0) * (rd.last_last_actions[:, _leg_action_ids(env)] != 0)
    return torch.sum(diff, dim=1)


# ----------------------------------------------------------------------
# 机械臂惩罚项（作者的 [num_actions_loco:] 在 20 维上覆盖 8 个 piper 关节；[:−2] 只覆盖前 6 个）
# ----------------------------------------------------------------------
def arm_energy(env) -> torch.Tensor:
    ids = _piper_ids(env)
    tau = _split_torque(env)[:, ids]
    qd = _robot(env).data.joint_vel.torch[:, ids]
    return torch.sum(torch.square(tau * qd), dim=1)


def arm_dof_vel(env) -> torch.Tensor:
    return torch.sum(torch.square(_robot(env).data.joint_vel.torch[:, _piper_ids(env)]), dim=1)


def arm_dof_acc(env) -> torch.Tensor:
    robot = _robot(env)
    return torch.sum(
        torch.square((_rd(env).last_dof_vel - robot.data.joint_vel.torch)[:, _piper_ids(env)] / env.step_dt), dim=1
    )


def arm_action_rate(env) -> torch.Tensor:
    rd = _rd(env)
    ids = torch.arange(12, 18, device=env.device)  # 动作向量里臂的 6 维
    return torch.sum(torch.square(rd.last_actions[:, ids] - rd.raw_actions[:, ids]), dim=1)


def arm_action_smoothness_1(env) -> torch.Tensor:
    robot = _robot(env)
    rd = _rd(env)
    ids = _arm_ids(env)  # 6 个 piper 转动关节（不含夹爪）
    diff = torch.square(robot.data.joint_pos_target.torch[:, ids] - rd.last_joint_pos_target[:, ids])
    diff = diff * (rd.last_actions[:, 12:18] != 0)
    return torch.sum(diff, dim=1)


def arm_action_smoothness_2(env) -> torch.Tensor:
    robot = _robot(env)
    rd = _rd(env)
    ids = _arm_ids(env)
    diff = torch.square(
        robot.data.joint_pos_target.torch[:, ids]
        - 2 * rd.last_joint_pos_target[:, ids]
        + rd.last_last_joint_pos_target[:, ids]
    )
    diff = diff * (rd.last_actions[:, 12:18] != 0) * (rd.last_last_actions[:, 12:18] != 0)
    return torch.sum(diff, dim=1)


def arm_control_smoothness_1(env) -> torch.Tensor:
    rd = _rd(env)
    plan = env.rd_plan_actions
    diff = torch.square(plan - rd.last_plan_actions) * (rd.last_plan_actions != 0)
    return torch.sum(diff, dim=1)


def arm_control_limits(env) -> torch.Tensor:
    plan = env.rd_plan_actions
    lim_p = env.cfg.commands.body_pose.pitch_range
    lim_r = env.cfg.commands.body_pose.roll_range
    return (
        torch.clamp(lim_p[0] - plan[:, 0], min=0.0)
        + torch.clamp(plan[:, 0] - lim_p[1], min=0.0)
        + torch.clamp(lim_r[0] - plan[:, 1], min=0.0)
        + torch.clamp(plan[:, 1] - lim_r[1], min=0.0)
    )


# ----------------------------------------------------------------------
# 接触 / 步态相关
# ----------------------------------------------------------------------
def _foot_points(env):
    """足端接触点（足底球心）的世界坐标与速度。

    本 USD 里足端刚体被并进了小腿（Isaac Gym 那边 foot 是独立刚体），
    所以要从 calf 刚体推算：URDF 里 FL_foot_joint 的 origin = (0, 0, -0.213)（calf 坐标系），
    足底球心再偏 (-0.002, 0, 0)。这里取球心近似足端接触点。
    """
    from isaaclab.utils.math import quat_apply

    robot = _robot(env)
    ids = env.rd_feet_body_ids
    pos = robot.data.body_pos_w.torch[:, ids]
    quat = robot.data.body_quat_w.torch[:, ids]
    ang = robot.data.body_ang_vel_w.torch[:, ids]
    lin = robot.data.body_lin_vel_w.torch[:, ids]
    off = torch.tensor([-0.002, 0.0, -0.213], device=pos.device).expand(pos.shape[0], pos.shape[1], 3)
    off_w = quat_apply(quat.reshape(-1, 4), off.reshape(-1, 3)).reshape(pos.shape)
    return pos + off_w, lin + torch.cross(ang, off_w, dim=-1)


def collision(env) -> torch.Tensor:
    """被惩罚刚体上的接触力范数 > 0.1 N 的个数（作者阈值 <0.1 视为无接触）。"""
    sensor = env.scene["contact_forces"]
    forces = sensor.data.net_forces_w_history.torch
    mag = torch.norm(forces, dim=-1).amax(dim=1)  # (N, B_sensor)
    ids = env.rd_penalised_sensor_ids
    return torch.sum((mag[:, ids] > 0.1).float(), dim=1)


def feet_slip(env) -> torch.Tensor:
    sensor = env.scene["contact_forces"]
    s_ids = env.rd_feet_sensor_ids
    rd = _rd(env)
    contact = sensor.data.net_forces_w.torch[:, s_ids, 2] > 1.0
    contact_filt = contact | rd.last_contacts
    rd.last_contacts.copy_(contact)
    _, foot_vel_w = _foot_points(env)
    foot_vel_xy = foot_vel_w[..., :2]
    return torch.sum(torch.norm(foot_vel_xy, dim=-1) ** 2 * contact_filt.float(), dim=1)


def feet_clearance_cmd_linear(env) -> torch.Tensor:
    """脚离地高度跟踪步态目标（作者 §6.27 的 feet_clearance_cmd_linear）。"""
    rd = _rd(env)
    # 作者用 foot_indices 推 φ_f = 1 - |1 - 2·clip(2c-1,0,1)|
    c = rd.foot_indices
    phi = 1.0 - torch.abs(1.0 - 2.0 * torch.clamp(2.0 * c - 1.0, 0.0, 1.0))
    h_tar = 0.04 * phi + 0.02
    foot_pos_w, _ = _foot_points(env)
    h = foot_pos_w[..., 2]
    d = rd.desired_contact_states
    return torch.sum(torch.square(h_tar - h) * (1.0 - d), dim=1)


def tracking_contacts_shaped_force(env) -> torch.Tensor:
    sensor = env.scene["contact_forces"]
    ids = env.rd_feet_sensor_ids
    d = _rd(env).desired_contact_states
    forces = torch.sum(torch.square(sensor.data.net_forces_w.torch[:, ids]), dim=-1)
    return -torch.sum((1.0 - d) * (1.0 - torch.exp(-forces / 100.0)), dim=1) / 4.0


def tracking_contacts_shaped_vel(env) -> torch.Tensor:
    robot = _robot(env)
    ids = env.rd_feet_body_ids
    d = _rd(env).desired_contact_states
    vel = torch.sum(torch.square(robot.data.body_lin_vel_w.torch[:, ids]), dim=-1)
    return -torch.sum(d * (1.0 - torch.exp(-vel / 10.0)), dim=1) / 4.0


# ----------------------------------------------------------------------
# 记录用（权重 0，仅出现在作者的表里）
# ----------------------------------------------------------------------
def jump(env) -> torch.Tensor:
    """作者表第 1 项，scale = 0（永远不生效）。保留占位以便对照。"""
    return torch.square(_robot(env).data.root_pos_w.torch[:, 2] - 0.30)
