"""末端（机械臂）几何：RoboDuet 的 lpy / abg 与工具点定义。

★作者的定义（逐行照抄 legged_robot.py:634-683）：
    末端刚体   = 刚体下标 23 → piper_link6（腕法兰）        ← 已在 robots.py 里核对过
    姿态偏置   = ee_rot_offset (xyzw) = (0, 0.7071, 0, -0.7071)
                 q_new = q_raw ⊗ q_off  （绕末端本体系 +Y 转 -90°，让 X 轴朝外）
    工具点     = p_tip = p_ee + R(q_new) · [0.12, 0, 0]
    lpy        = 工具点在"机身偏航系"里的球坐标 (l, p, yaw)
    abg        = 末端姿态的"轴投影角" (α, β, γ)（不是 RPY！）
    z 的基准   = 0.38（标称机身高度），地面高度 plane 模式下恒为 0

注意 Isaac Lab 的四元数约定是 wxyz，作者/Isaac Gym 是 xyzw —— 这里全部转换好再用。
"""

from __future__ import annotations

import torch

# ---- 常量（作者代码里的硬编码值）----
EE_BODY_NAME = "piper_link6"
EE_ROT_OFFSET_XYZW = (0.0, 0.7071, 0.0, -0.7071)
EE_ROT_OFFSET_WXYZ = (-0.7071, 0.0, 0.7071, 0.0)  # 同一个旋转，wxyz 顺序
GRASPER_MOVE = (0.12, 0.0, 0.0)
NOMINAL_BASE_HEIGHT = 0.38  # 作者 lpy 里 z 的基准高度

# 指令采样范围（legged_robot_config.py:51-90）
LPY_L_RANGE = (0.3, 0.7)
LPY_P_RANGE = (-0.45 * torch.pi, 0.45 * torch.pi)
LPY_Y_RANGE = (-0.5 * torch.pi, 0.5 * torch.pi)
EE_ROLL_RANGE = (-0.45 * torch.pi, 0.45 * torch.pi)
EE_PITCH_RANGE = (-60.0 / 180 * torch.pi, 60.0 / 180 * torch.pi)
EE_YAW_RANGE = (-75.0 / 180 * torch.pi, 75.0 / 180 * torch.pi)

# 奖励归一化用的 range = 采样上下界之差（legged_robot.py:1521-1534）
LPY_RANGE = (
    LPY_L_RANGE[1] - LPY_L_RANGE[0],
    LPY_P_RANGE[1] - LPY_P_RANGE[0],
    LPY_Y_RANGE[1] - LPY_Y_RANGE[0],
)
RPY_RANGE = (
    EE_ROLL_RANGE[1] - EE_ROLL_RANGE[0],
    EE_PITCH_RANGE[1] - EE_PITCH_RANGE[0],
    EE_YAW_RANGE[1] - EE_YAW_RANGE[0],
)

# 目标采样的安全过滤盒（asset_config.py ARM_COMMAND_SAFETY）
COLLISION_LOWER = (-0.38, -0.16, -0.30)
COLLISION_UPPER = (0.30, 0.16, 0.10)
UNDERGROUND_LIMIT = -0.38


def _ee_index(env) -> int:
    """末端刚体在 articulation 里的下标（运行时按名字查，不硬编码 23）。"""
    idx = getattr(env, "_rd_ee_idx", None)
    if idx is None:
        robot = env.scene["robot"]
        idx = robot.body_names.index(EE_BODY_NAME)
        env._rd_ee_idx = idx
    return idx


def quat_to_angle(q: torch.Tensor) -> torch.Tensor:
    """作者 `quat_to_angle`：终端姿态 → (α, β, γ) 轴投影角。输入/输出 wxyz。"""
    from isaaclab.utils.math import quat_apply

    ex = torch.zeros_like(q[:, :3])
    ex[:, 0] = 1.0
    ey = torch.zeros_like(q[:, :3])
    ey[:, 1] = 1.0
    ez = torch.zeros_like(q[:, :3])
    ez[:, 2] = 1.0
    x_axis = quat_apply(q, ex)
    y_axis = quat_apply(q, ey)
    z_axis = quat_apply(q, ez)
    alpha = torch.atan2(y_axis[:, 2], y_axis[:, 1])
    beta = torch.atan2(z_axis[:, 0], z_axis[:, 2])
    gamma = torch.atan2(x_axis[:, 1], x_axis[:, 0])
    return torch.stack([alpha, beta, gamma], dim=-1)


def ee_tip_pose(env):
    """返回 (p_tip_w, q_ee_new) —— p_tip 是工具点世界坐标，q_ee_new 是加了偏置的末端姿态。"""
    from isaaclab.utils.math import quat_apply, quat_mul

    robot = env.scene["robot"]
    i = _ee_index(env)
    p = robot.data.body_pos_w.torch[:, i]
    q_raw = robot.data.body_quat_w.torch[:, i]  # wxyz
    q_off = torch.tensor(EE_ROT_OFFSET_WXYZ, device=p.device).unsqueeze(0).repeat(p.shape[0], 1)
    q_new = quat_mul(q_raw, q_off)
    offset = torch.tensor(GRASPER_MOVE, device=p.device).unsqueeze(0).repeat(p.shape[0], 1)
    p_tip = p + quat_apply(q_new, offset)
    return p_tip, q_new


def base_yaw(env) -> torch.Tensor:
    """机身偏航角 ψ（作者：atan2(F_y, F_x)，F = R(q_base)·[1,0,0]）。"""
    from isaaclab.utils.math import quat_apply

    robot = env.scene["robot"]
    q = robot.data.root_quat_w.torch
    ex = torch.zeros_like(q[:, :3])
    ex[:, 0] = 1.0
    f = quat_apply(q, ex)
    return torch.atan2(f[:, 1], f[:, 0])


def lpy_in_base(env) -> torch.Tensor:
    """当前工具点在机身偏航系里的 (l, p, yaw)（作者 get_lpy_in_base_coord）。"""
    robot = env.scene["robot"]
    p_tip, _ = ee_tip_pose(env)
    root = robot.data.root_pos_w.torch
    psi = base_yaw(env)
    dx = p_tip[:, 0] - root[:, 0]
    dy = p_tip[:, 1] - root[:, 1]
    x = torch.cos(psi) * dx + torch.sin(psi) * dy
    y = -torch.sin(psi) * dx + torch.cos(psi) * dy
    z = p_tip[:, 2] - NOMINAL_BASE_HEIGHT  # plane 模式：measured_heights ≡ 0
    l = torch.sqrt(x**2 + y**2 + z**2)
    p = torch.atan2(z, torch.sqrt(x**2 + y**2))
    yaw = torch.atan2(y, x)
    return torch.stack([l, p, yaw], dim=-1)


def abg_in_base(env) -> torch.Tensor:
    """当前末端姿态相对机身偏航系的 (α, β, γ)（作者 get_alpha_beta_gamma_in_base_coord）。"""
    from isaaclab.utils.math import quat_from_euler_xyz, quat_mul

    _, q_new = ee_tip_pose(env)
    psi = base_yaw(env)
    zeros = torch.zeros_like(psi)
    q_yaw = quat_from_euler_xyz(zeros, zeros, psi)  # Rz(ψ)
    q_yaw_inv = torch.cat([q_yaw[:, :1], -q_yaw[:, 1:]], dim=-1)  # 共轭
    q_rel = quat_mul(q_yaw_inv, q_new)
    return quat_to_angle(q_rel)


def ee_quat_in_base_yaw(env) -> torch.Tensor:
    """臂策略的 privileged obs：末端【原始】四元数转到偏航对齐的世界系（wxyz）。"""
    from isaaclab.utils.math import quat_from_euler_xyz, quat_mul

    robot = env.scene["robot"]
    q_raw = robot.data.body_quat_w.torch[:, _ee_index(env)]
    psi = base_yaw(env)
    zeros = torch.zeros_like(psi)
    q_yaw = quat_from_euler_xyz(zeros, zeros, psi)
    return quat_mul(q_yaw, q_raw)


def lpy_to_world_xyz(env, lpy: torch.Tensor) -> torch.Tensor:
    """lpy → 世界坐标（作者 _lpy_to_world_xyz；可视化用）。"""
    robot = env.scene["robot"]
    root = robot.data.root_pos_w.torch
    psi = base_yaw(env)
    l, p, y = lpy[:, 0], lpy[:, 1], lpy[:, 2]
    X = l * torch.cos(p) * torch.cos(y)
    Y = l * torch.cos(p) * torch.sin(y)
    Z = l * torch.sin(p)
    x = X * torch.cos(psi) - Y * torch.sin(psi) + root[:, 0]
    yw = X * torch.sin(psi) + Y * torch.cos(psi) + root[:, 1]
    zw = Z + NOMINAL_BASE_HEIGHT
    return torch.stack([x, yw, zw], dim=-1)


def lpy_to_local_xyz(lpy: torch.Tensor) -> torch.Tensor:
    """lpy → 机身偏航系下的 xyz（作者 _arm_lpy_to_local_xyz；用于碰撞盒检查）。"""
    l, p, y = lpy[:, 0], lpy[:, 1], lpy[:, 2]
    return torch.stack([l * torch.cos(p) * torch.cos(y), l * torch.cos(p) * torch.sin(y), l * torch.sin(p)], dim=-1)


def target_abg_from_euler(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """目标姿态 Rz(yaw)·Ry(pitch)·Rx(roll) → abg（作者 §4.4）。"""
    from isaaclab.utils.math import quat_from_euler_xyz, quat_mul

    zeros = torch.zeros_like(roll)
    q1 = quat_from_euler_xyz(zeros, zeros, yaw)
    q2 = quat_from_euler_xyz(zeros, pitch, zeros)
    q3 = quat_from_euler_xyz(roll, zeros, zeros)
    q = quat_mul(q1, quat_mul(q2, q3))
    return quat_to_angle(q)
