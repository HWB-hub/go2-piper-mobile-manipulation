# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""本任务自己写的 MDP 项（奖励函数等）。

约定（和框架一致）：
    函数签名 = func(env, **params)  →  返回形状 (num_envs,) 的 torch.Tensor

提供两项：
    arm_pose_tracking   —— 机械臂保持"标称姿态"（相对默认关节角的偏差越小越好）
    arm_joint_vel_l2    —— 机械臂关节速度惩罚（让手臂动作平滑，别乱甩）
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse, wrap_to_pi, yaw_quat

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv


def arm_pose_tracking(
    env: ManagerBasedRLEnv,
    std: float = 0.25,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["piper_joint.*"]),
) -> torch.Tensor:
    """机械臂姿态跟踪奖励：越接近标称姿态分越高，值域 (0, 1]。

    用「相对默认关节角」计算，所以标称姿态天然就是 init_state 里写的那组角度：
        q_err = 当前关节角 - 默认(标称)关节角
        reward = exp( - ||q_err||² / std² )

    Args:
        std: 允许的偏差尺度（rad）。std 越小，"保持姿势"的要求越严格。
             0.25 rad ≈ 14°，是个常用的起点。
        asset_cfg: 指向机械臂那 6 个关节。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # 相对默认关节角（default 来自配置里的 init_state.joint_pos）
    q_rel = asset.data.joint_pos.torch[:, asset_cfg.joint_ids] - asset.data.default_joint_pos.torch[:, asset_cfg.joint_ids]
    # 沿关节维求平方和 → 每个环境一个标量
    err_sq = torch.sum(torch.square(q_rel), dim=1)
    return torch.exp(-err_sq / (std**2))


def arm_joint_vel_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["piper_joint.*"])
) -> torch.Tensor:
    """机械臂关节速度平方和（惩罚项，配合负权重使用）。

    作用：抑制手臂高频抖动。四足+机械臂的系统里，手臂乱甩会通过反作用力矩干扰机身，
    是 locomanipulation 里非常常见的抖动来源。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel.torch[:, asset_cfg.joint_ids]), dim=1)


def gait_phase_observation(env: ManagerBasedRLEnv, period: float = 0.65) -> torch.Tensor:
    """显式提供步态时钟，避免相位奖励对策略变成不可观测噪声。"""
    t = _t(env.episode_length_buf).float() * float(env.step_dt)
    phase = 2.0 * torch.pi * torch.remainder(t / period, 1.0)
    return torch.stack((torch.sin(phase), torch.cos(phase)), dim=-1)


def base_height_exp(
    env: ManagerBasedRLEnv,
    target_height: float,
    std: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """机身高度奖励（指数核，值域 (0,1]，配合正权重使用）。

    ★为什么不用内置的 base_height_l2（平方误差）：
        L2 形式在偏差小时"几乎没有梯度/量级"。
        实测：目标 0.30 m、实际 0.216 m（趴低了 8.4 cm）时，
              (0.30-0.216)² × weight(-5) × dt(0.02) ≈ -0.0006/步 —— 基本等于没惩罚，
              于是策略学会了"低姿态爬行"这种退化解。
        指数核：偏差 8.4 cm 时核值 exp(-0.084²/0.05²) = 0.060（不是 1.0），
              即每步少拿约 0.94×weight×dt ≈ 0.019 —— 量级和速度跟踪项可比，才真正约束姿态。
              （★这里原写 0.0002 是算错了，2026-10-04 代码审查时核实修正。）

    Args:
        target_height: 目标机身高度（本案例 = 0.30 m，由 fk_check.py 用 URDF 正运动学算出）
        std: 允许的高度波动（m）。越小要求越严；0.05 是个起点。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    height = asset.data.root_pos_w.torch[:, 2]
    return torch.exp(-torch.square(height - target_height) / (std**2))


def _ee_body_pos_w(env, asset_cfg) -> torch.Tensor:
    """取末端刚体的世界位置，返回 (num_envs, 3)。"""
    asset: Articulation = env.scene[asset_cfg.name]
    pos = asset.data.body_pos_w.torch[:, asset_cfg.body_ids]
    if pos.dim() == 3:  # 选中多个刚体时是 (N, B, 3)，这里只取第一个
        pos = pos[:, 0]
    return pos


def ee_goal_tracking_exp(
    env: ManagerBasedRLEnv,
    command_name: str = "ee_goal",
    std: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="piper_link6"),
) -> torch.Tensor:
    """末端跟踪指令目标点的奖励（指数核，值域 (0,1]，配合正权重使用）。

    移植自 visual_wholebody::ManipLoco_rewards._reward_tracking_ee_world：
        ee_pos_error = Σ|x_ee - x_goal|        （L1，三轴绝对值之和）
        reward       = exp(-ee_pos_error / (sigma * 2))

    ★本实现把 (sigma*2) 合并成了一个 std 参数，默认 0.5：
        原仓库 sigma=1 -> exp(-err/2)，err=0.5 m 时还有 0.78，梯度非常平，
        只有在惩罚项都很小、且主要靠 Metrics 观察时才够用。
        这里默认 0.5（err=0.2 -> 0.67，err=0.5 -> 0.37），学"够得准"更快；
        想完全对齐原仓库就传 std=2.0。

    Args:
        command_name: 目标点指令器的名字（EeGoalCommand）。
        std: 误差尺度（m）。越小要求越严。
        asset_cfg: 指向末端执行器刚体（本资产 = piper_link6，夹爪已合并进来）。
    """
    cmd = env.command_manager.get_term(command_name)
    ee_pos_w = _ee_body_pos_w(env, asset_cfg)
    err = torch.sum(torch.abs(ee_pos_w - cmd.goal_pos_w), dim=1)
    return torch.exp(-err / std)


def ee_goal_distance(
    env: ManagerBasedRLEnv,
    command_name: str = "ee_goal",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="piper_link6"),
) -> torch.Tensor:
    """末端到目标点的 L1 距离（纯度量项，配合 weight=0 只用于记录，不影响训练）。

    有了它，TensorBoard 上的 Episode_Reward/ee_goal_distance 就是"够得准不准"的直接读数 ——
    判断移植是否真的work，看这条曲线比看总奖励可靠。
    """
    cmd = env.command_manager.get_term(command_name)
    ee_pos_w = _ee_body_pos_w(env, asset_cfg)
    return torch.sum(torch.abs(ee_pos_w - cmd.goal_pos_w), dim=1)


def ee_pos_in_base_frame(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="piper_link6"),
) -> torch.Tensor:
    """末端位置在 base 系下的表达（x/y 随偏航旋转，z 为世界竖直），shape (N, 3)。

    与 EeGoalCommand.command 用的是同一套坐标约定，所以
    ``command - ee_pos_in_base_frame`` 就是当前的目标误差向量。
    **把这个直接喂给策略**能显著加快学习：策略不必从关节角自己反推末端在哪。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    rel_w = _ee_body_pos_w(env, asset_cfg) - asset.data.root_pos_w.torch
    return quat_apply_inverse(yaw_quat(asset.data.root_quat_w.torch), rel_w)


def ee_pos_tracking_error(
    env: ManagerBasedRLEnv,
    command_name: str = "ee_goal",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="piper_link6"),
) -> torch.Tensor:
    """目标误差向量（base 系，shape (N,3)）：目标点 - 当前末端位置。

    比给策略"目标位置"更直接 —— 相当于把残差喂进去，是最常用的做法。
    """
    cmd = env.command_manager.get_term(command_name)
    return cmd.command - ee_pos_in_base_frame(env, asset_cfg)


# =============================================================================
# 拾取（pick）专用：围绕【场景里的可抓取目标物】而非指令点来算
# =============================================================================
# 设计思路（接触密集型任务，必须做奖励塑形，否则几乎学不出来）：
#   ① 靠近：夹取中心 → 球心的指数核距离奖励（把手臂引到球上）
#   ② 闭合时机：离球近时奖励"手指闭合"，离球远时惩罚闭合（避免空夹、避免张着手乱晃）
#   ③ 抬起：球离地高度线性奖励（这是"真的抓起来了"的核心信号；球被撞跑在地上滚 → 不给分）
#   三者配合：靠近拿分 → 到位才敢闭合 → 夹住了抬起来才开始拿大分
# -----------------------------------------------------------------------------


def _object_pos_w(env, object_cfg: SceneEntityCfg) -> torch.Tensor:
    """场景里目标物的世界坐标中心，shape (N,3)。"""
    return env.scene[object_cfg.name].data.root_pos_w.torch


def pinch_object_distance(
    env: ManagerBasedRLEnv,
    z_offset: float = 0.0,
    object_cfg: SceneEntityCfg = SceneEntityCfg("pick_object"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="piper_link7"),
    second_body_name: str = "piper_link8",
) -> torch.Tensor:
    """夹取中心到目标物中心的距离 [m]，shape (N,)。

    ★夹取中心 = **两个手指参考点的中点**（link7 与 link8 的平均位置），不是 link7 本身。
      这份资产的手指关节定义是：两个手指的关节原点重合、轴向相反，
        闭合 (joint7=joint8=0)：两指重合在中心线上 → 中点 == link7
        张开 (各 ±0.03)：两指各偏中心线 3 cm → 此时 **link7 偏在一边**，
                        真正的夹取中心仍是两指中点
      实测教训：用 link7 单点作参考，张开时瞄准点偏 3 cm，球会被手指“合在空处”
      （地面抓取测试 1/125，且夹爪角闭合到 0 说明球根本不在指间）。
      用中点后，无论开合，参考点都落在中心线上。

    `z_offset`：把目标点从物体中心**抬高** z_offset 再算距离。
    ★为什么需要（实测教训）：手指尖比两指中点低约 5 cm。按物体中心对齐时，
      指尖正好落在台面顶（0.14）上 —— "手指被挡住"里混进了"指尖蹭台面"这种假夹持，
      `pick_grip_state` 涨到 0.69 但 `pick_lift` 始终 ≈ 0。
      把瞄准点抬到物体中心上方 4 cm（= 夹物体的上半段），指尖离台面才有 4 cm 余量。
      距离的参考点必须跟着一起抬，否则奖励会把"正确瞄准"判成误差。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    ids = asset_cfg.body_ids
    if isinstance(ids, slice):  # 兜底：万一没解析出具体下标
        ids = [asset.data.body_names.index("piper_link7")]
    i7 = ids[0]
    if second_body_name in asset.data.body_names:
        i8 = asset.data.body_names.index(second_body_name)
        pinch_w = 0.5 * (asset.data.body_pos_w.torch[:, i7] + asset.data.body_pos_w.torch[:, i8])
    else:
        pinch_w = asset.data.body_pos_w.torch[:, i7]
    target_w = _object_pos_w(env, object_cfg).clone()
    if z_offset != 0.0:
        target_w[:, 2] = target_w[:, 2] + z_offset
    return torch.norm(target_w - pinch_w, dim=1)


def object_distance_exp(
    env: ManagerBasedRLEnv,
    std: float = 0.06,
    z_offset: float = 0.0,
    object_cfg: SceneEntityCfg = SceneEntityCfg("pick_object"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="piper_link7"),
) -> torch.Tensor:
    """① 靠近奖励：exp(-d/std)，d = 夹取中心到目标点的距离。值域 (0, 1]，越近越接近 1。"""
    d = pinch_object_distance(env, z_offset=z_offset, object_cfg=object_cfg, asset_cfg=asset_cfg)
    return torch.exp(-d / std)


def gripper_close_shaped(
    env: ManagerBasedRLEnv,
    near_dist: float = 0.05,
    far_dist: float = 0.10,
    very_far: float = 0.20,
    far_penalty: float = 0.3,
    stroke: float = 0.035,
    z_offset: float = 0.0,
    object_cfg: SceneEntityCfg = SceneEntityCfg("pick_object"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="piper_link7"),
    gripper_joint_names: tuple[str, ...] = ("piper_joint7", "piper_joint8"),
) -> torch.Tensor:
    """② 闭合时机奖励：**近处闭合给满、只有很远时轻罚**（不对称）。

        close = 1 - (|q7| + |q8|) / (2·stroke)        # 1 = 完全闭合，0 = 全开
        near  = clamp((far_dist - d)/(far_dist - near_dist), 0, 1)   # d<5cm→1，d>10cm→0
        vfar  = clamp((d - very_far)/(very_far*0.5), 0, 1)           # d>20cm 才起来
        reward = close · (near − far_penalty · vfar)

    ★为什么改成不对称（实测教训）：
      最初写的是 `close·(2·near − 1)`，即"近处 +1、远处 −1"。这有**自我抵消**的缺陷：
      若手臂只有一半时间在物体附近，"永远闭合"和"永不闭合"的期望收益都是 0 分，
      策略没有压力去学"恰好在位时才闭合"—— 实测 900 轮里 `pick_grip_close` 一直在
      −0.03~+0.1 徘徊，`pick_hold` 恒为 0（物体从没被真正夹住）。
      改成"近处 +1、只有 20 cm 之外才 −0.3"之后，闭合本身不再被总体惩罚，
      而在位闭合是唯一的正收益来源 ⇒ 策略有明确的动机学"到位再合"。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    d = pinch_object_distance(env, z_offset=z_offset, object_cfg=object_cfg, asset_cfg=asset_cfg)

    ids = [asset.data.joint_names.index(n) for n in gripper_joint_names]
    q = asset.data.joint_pos.torch[:, ids]
    close = 1.0 - q.abs().sum(dim=1) / (2.0 * stroke)
    close = torch.clamp(close, 0.0, 1.0)

    near = torch.clamp((far_dist - d) / max(far_dist - near_dist, 1e-6), 0.0, 1.0)
    vfar = torch.clamp((d - very_far) / (very_far * 0.5), 0.0, 1.0)
    return close * (near - far_penalty * vfar)


def object_lift_reward(
    env: ManagerBasedRLEnv,
    rest_height: float = 0.03,
    target_lift: float = 0.08,
    object_cfg: SceneEntityCfg = SceneEntityCfg("pick_object"),
) -> torch.Tensor:
    """③ 抬起奖励，值域 [0, 1]：球比静止高度高 target_lift 时拿满。

    单调递增（不是核），所以"举得越高越好"，不会像高斯核那样在中间高度形成局部最优。
    ★球被撞跑、在地上滚 → 高度没变 → 这一项恒为 0，策略从"推球"上拿不到任何分。
    """
    z = _object_pos_w(env, object_cfg)[:, 2]
    return torch.clamp((z - rest_height) / target_lift, 0.0, 1.0)


def gripper_grip_reward(
    env: ManagerBasedRLEnv,
    near_dist: float = 0.05,
    far_dist: float = 0.10,
    stroke: float = 0.035,
    blocked_lo: float = 0.20,
    blocked_hi: float = 0.97,
    z_offset: float = 0.0,
    object_cfg: SceneEntityCfg = SceneEntityCfg("pick_object"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="piper_link7"),
    gripper_joint_names: tuple[str, ...] = ("piper_joint7", "piper_joint8"),
) -> torch.Tensor:
    """②b **奖励"真的夹住了"这个状态**（而不是"合爪这个动作"）。

    判据：手指张开度落在中间区间 + 离物体够近
        qsum = |q7| + |q8|        （0 = 完全闭合=指间是空的；2·stroke = 全开）
        blocked = (blocked_lo·2·stroke < qsum < blocked_hi·2·stroke)
        near    = clamp((far_dist - d)/(far_dist - near_dist), 0, 1)
        reward  = blocked × near

    ★为什么必须加这一项（实测教训）：
      只奖励"合爪动作"时，策略可以在物体**旁边**合爪照样拿分，根本不用真夹住 ——
      前 6 轮训练全部卡在这个局部最优：`pick_grip_close` 涨到 1.06（看起来学得很好），
      但 `pick_lift` 恒为 0、`pick_hold` 恒为 0（物体从未被真正夹持）。
      而"手指张开度被卡在中间"是个**物理状态**：只有物体真的在指间才会出现
      （指间是空的 → 一路合到 0；物体太宽 → 一直全开）。
      `probe_grasp_trace.py` 实测：真夹住时 |q7| = 0.0214（= 0.61·stroke）✓ 落在区间内。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    d = pinch_object_distance(env, z_offset=z_offset, object_cfg=object_cfg, asset_cfg=asset_cfg)

    ids = [asset.data.joint_names.index(n) for n in gripper_joint_names]
    qsum = asset.data.joint_pos.torch[:, ids].abs().sum(dim=1)
    lo, hi = blocked_lo * 2.0 * stroke, blocked_hi * 2.0 * stroke
    blocked = ((qsum > lo) & (qsum < hi)).float()

    near = torch.clamp((far_dist - d) / max(far_dist - near_dist, 1e-6), 0.0, 1.0)
    return blocked * near


def object_lift_vel_reward(
    env: ManagerBasedRLEnv,
    max_dist: float = 0.08,
    vmax: float = 0.15,
    z_offset: float = 0.0,
    object_cfg: SceneEntityCfg = SceneEntityCfg("pick_object"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="piper_link7"),
) -> torch.Tensor:
    """③b **抬起过程的引导**：奖励"物体在夹取中心附近时向上运动"。

        near = clamp((max_dist - d)/max_dist, 0, 1)
        reward = near × clamp(v_z / vmax, 0, 1)

    ★为什么需要（前 10 轮的共同瓶颈）：
      `object_lift_reward` 奖励的是**结果高度**（物体被抬起来的高度）。
      这是典型的"稀疏结果奖励"：策略必须先偶然做对（夹住 + 往上抬），
      才能拿到第一个非零梯度。实测 10 轮训练里 `pick_lift` 一直在 0.02~0.06 徘徊，
      而"靠近"与"合爪"都学得很好 —— 典型的后段动作（抬起）探索不到。
      本项奖励**过程**（握住时物体往上走），把一个"有方向"的梯度直接递给策略：
      只要夹着往上抬一点，立刻有分。为了避免"把物体打飞"的作弊路线，
      用 `near` 把奖励绑在"物体还在夹取中心附近"这个条件上。
    """
    obj = env.scene[object_cfg.name]
    d = pinch_object_distance(env, z_offset=z_offset, object_cfg=object_cfg, asset_cfg=asset_cfg)
    vz = obj.data.root_lin_vel_w.torch[:, 2]
    near = torch.clamp((max_dist - d) / max_dist, 0.0, 1.0)
    return near * torch.clamp(vz / vmax, 0.0, 1.0)


def object_pos_in_base_frame(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("pick_object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """目标物在机器人 base 系下的位置 (x, y, z)，shape (N, 3)。

    ★为什么需要：指令点只记录了"球被摆在哪"，球被手臂碰到或滚动之后就不准了。
      把**球的真实位置**直接喂给策略，它才能"看着球去够"（尤其是之后"走过去抓"的阶段）。
      坐标约定与 EeGoalCommand.command / ee_pos_in_base_frame 一致，可直接相减得误差。
    """
    obj = env.scene[object_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    rel_w = obj.data.root_pos_w.torch - robot.data.root_pos_w.torch
    return quat_apply_inverse(yaw_quat(robot.data.root_quat_w.torch), rel_w)


def object_held_bonus(
    env: ManagerBasedRLEnv,
    rest_height: float = 0.03,
    min_lift: float = 0.04,
    max_dist: float = 0.07,
    z_offset: float = 0.0,
    object_cfg: SceneEntityCfg = SceneEntityCfg("pick_object"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="piper_link7"),
) -> torch.Tensor:
    """④ 抓住保持奖励（0/1）：**球既被抬起、又确实在夹取中心附近**才给 1。

    ★为什么要加 max_dist（"在手上"）这一条：
      只判"球的高度"会被作弊——策略完全可能学会**把球打飞**（一巴掌把球拍上天），
      球飞起来的高度同样能给分，但那不是抓取。加"球心必须离夹取中心 <7 cm"
      之后，"打飞"这种路径拿不到这一项（求飞出去的球立刻离开夹取中心）。
      球半径 3 cm，夹住时球心离夹取中心约 3~4 cm，所以 7 cm 的门槛不会误伤真抓取。
      本项与 Metrics 里的 success 判据保持一致。
    """
    z = _object_pos_w(env, object_cfg)[:, 2]
    d = pinch_object_distance(env, z_offset=z_offset, object_cfg=object_cfg, asset_cfg=asset_cfg)
    return ((z - rest_height > min_lift) & (d < max_dist)).float()


# ============================================================================
# S2【追踪任务】用到的奖励项：目标在世界系、必须走过去
# ============================================================================

def _t(x):
    """兼容 Isaac Lab 3.0 的 data 代理写法（有的字段要 .torch）。"""
    return x.torch if hasattr(x, "torch") else x


def _goal_w(env, command_name: str) -> torch.Tensor:
    return env.command_manager.get_term(command_name).goal_pos_w


def base_goal_standoff_error(
    env: ManagerBasedRLEnv, command_name: str = "ee_goal", target_distance: float = 0.35
) -> torch.Tensor:
    """基座到目标的【站位误差】= |水平距离 − 理想站位|，返回 (N,)。

    为什么不是"越近越好"：机械臂只有约 35 cm 的作业半径，机身怼到目标上反而够不着。
    理想站位就是 target_distance（默认 0.35 m）。
    """
    goal = _goal_w(env, command_name)
    root = _t(env.scene["robot"].data.root_pos_w)
    d = torch.norm((goal - root)[:, :2], dim=1)
    return (d - target_distance).abs()


def base_goal_distance_exp(
    env: ManagerBasedRLEnv, command_name: str = "ee_goal", target_distance: float = 0.35, std: float = 0.6
) -> torch.Tensor:
    """站位指数核（值域 (0,1]，配合正权重）：站得越准分越高。"""
    return torch.exp(-base_goal_standoff_error(env, command_name, target_distance) / std)


def base_goal_progress(
    env: ManagerBasedRLEnv, command_name: str = "ee_goal", target_distance: float = 0.35
) -> torch.Tensor:
    """站位【进步量】（势函数差分）= 上一步误差 − 这一步误差。

    ★为什么用差分而不是直接给距离：直接给距离，站着不动也能白拿一个常数分，
      策略最划算的做法就是站着不动（Pick 任务实测过这个局部最优）。
      差分项在"不动"时恒为 0，只有真的朝站位挪动才有正奖励 ⇒ **腿必须动**。

    环境刚重置那一步，"上一步距离"来自上一个回合，会造出假的巨大进步；
    所以对 episode_length_buf <= 1 的环境把上一步设成当前值（进步 = 0）。
    """
    goal = _goal_w(env, command_name)
    root = _t(env.scene["robot"].data.root_pos_w)
    e_now = (torch.norm((goal - root)[:, :2], dim=1) - target_distance).abs()
    buf = _t(env.episode_length_buf)
    prev = getattr(env, "_base_goal_e_prev", None)
    if prev is None or prev.shape != e_now.shape:
        env._base_goal_e_prev = e_now.detach().clone()
        return torch.zeros_like(e_now)
    e_prev = torch.where(buf <= 1, e_now, prev)
    env._base_goal_e_prev = e_now.detach().clone()
    return (e_prev - e_now).clamp(min=-0.3, max=0.3)


def base_goal_heading_exp(env: ManagerBasedRLEnv, command_name: str = "ee_goal", std: float = 0.6) -> torch.Tensor:
    """机身朝向目标方向的指数核：走到附近还得转过去，胳膊才够得到。"""
    goal = _goal_w(env, command_name)
    robot = env.scene["robot"]
    root_xy = _t(robot.data.root_pos_w)[:, :2]
    d = goal[:, :2] - root_xy
    d = d / (torch.norm(d, dim=1, keepdim=True) + 1e-6)
    yaw = _t(robot.data.heading_w)
    fwd = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=1)
    cos_err = torch.sum(fwd * d, dim=1).clamp(-1.0, 1.0)
    return torch.exp(-torch.acos(cos_err) / std)

# ============================================================================
# 四足支撑约束（2026-10-04 代码审查后补——"四条腿稳定支撑"的正主）
# ============================================================================


def foot_clearance(
    env: ManagerBasedRLEnv,
    target: float = 0.08,
    std: float = 0.04,
    ground_z: float = 0.0,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_calf"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*_calf"),
) -> torch.Tensor:
    """摆动腿离地高度奖励（**只对腾空的脚**计算，返回四腿【平均】值，值域 [0,1]）。

    ★为什么需要它：`feet_air_time` 只奖励"滞空时间长"，完全不约束"抬多高"。
      没有这一项时，策略可以用"抬很高 + 步频很慢 + 支撑腿在地上蹭"来白拿滞空奖励
      —— 这正是"看着不会用腿提供稳定支撑"的成因之一。

    ★ground_z 必须实测、不能填 0：本资产把脚端球体 merge 进了 `*_calf`，
      所以 `body_pos_w` 给的是**小腿刚体原点**的高度，不是脚底。
      站立时它近似为常数，因此判据是"站立高度 + target"。

    ★in_air 掩码来自接触力历史（而不是用高度判断），避免"脚在地面划过去"被误判为腾空。

    ★返回四腿【平均】而不是【求和】（2026-10-04 自查修正）：
      求和时值域是 [0,4]，配 0.5 权重等效 2.0 ⇒ 会奖励"脚在空中待得越久越多越好"，
      与"稳定支撑/摆动腿该落地就落地"直接冲突（退化解：四条腿尽量都腾空）。
      取平均后这一项只衡量"正在摆动的那条腿抬的高度对不对"，不奖励腾空时长
      （腾空时长由 feet_air_time 负责，承重由 stance_duty/trot_symmetry 负责，各司其职）。
    """
    cs = env.scene.sensors[sensor_cfg.name]
    f = _t(cs.data.net_forces_w_history)[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0]
    in_air = f < 1.0
    z = _t(env.scene[asset_cfg.name].data.body_pos_w)[:, asset_cfg.body_ids, 2]
    h = (z - ground_z).clamp(min=0.0)
    per_leg = torch.exp(-torch.square(h - target) / std**2) * in_air
    return per_leg.sum(dim=1) / per_leg.shape[1]  # 四腿平均（÷腿数，值域 [0,1]）


def stance_duty(
    env: ManagerBasedRLEnv,
    min_stance: int = 2,
    threshold: float = 1.0,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_calf"),
) -> torch.Tensor:
    """承重腿条数不足 min_stance 时返回 1（配合**负权重**做惩罚），返回 (N,)。

    ★这是"四条腿得撑住"的显式约束，是 `feet_air_time` 的反向平衡项：
      air_time 奖励"腿抬起来"，本项惩罚"脚不够用"。

    ⚠ min_stance **不要设 3**：四足标准 trot 的稳态本来就只有 **2 条腿承重**，
      设 3 会把正确步态一起罚掉（这是代码审查给的初值里唯一需要修正的地方）。
      min_stance=2 的含义是"少于两条腿在承重 = 跳 / 摔 / 腾空"。
    """
    cs = env.scene.sensors[sensor_cfg.name]
    f = _t(cs.data.net_forces_w_history)[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0]
    return (f > threshold).sum(dim=1).float().lt(min_stance).float()

def trot_symmetry(
    env: ManagerBasedRLEnv,
    threshold: float = 1.0,
    period: float = 0.8,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_calf"),
    order: tuple = ("FL", "FR", "RL", "RR"),
) -> torch.Tensor:
    """对角步态（trot）相位约束，返回 (N,) 值域 [0,1]。

    ============================ 为什么需要它 ============================
    第 2 轮步态实测（probe_gait_quality.py，指令 vx=0.8）：
        FL_calf 占空比 1.00、抬脚 0 次  ⇒ 一条腿**从头到尾贴地拖着**
        RL_calf 占空比 0.00、触地 0 次  ⇒ 一条腿**从头到尾悬在空中**
        FR/RR 5~6.5 Hz 迈步             ⇒ 实际只有两条腿工作，步幅只有正常值 15~38%
    而 feet_air_time（只奖励"腾空后落地"的脚）和 stance_duty（只看承重腿总数）都拦不住它。

    ================= 第一版被静态破解（2026-10-04 实测教训）=================
    第一版只用接触状态判相位："FL 与 RR 同相 且 FR 与 RL 同相 且 两组反相"。
    策略随即摆出**固定姿势**：FL+RR 永久踩地、FR+RL 永久抬起 —— 永久满足、一次都不迈步。
    实测证据（第 4 轮 5850 轮权重，12 秒采样）：四条腿触地切换次数全为 0，
    "同时着地腿数分布：2 腿 = 100%"。而速度跟踪 0.06、姿态全过，指标看似"变好"实为退化
    （靠两条腿在地上滑行）。
    ★教训：相位类奖励必须**同时**约束"相位对不对"和"到底有没有在动"，缺一不可。

    ============================ 现在两道锁 ============================
    锁 1【相位时钟】用环境自身时间做周期 period 的时钟（0.8 s ⇒ 1.25 Hz trot）：
        前半周期要求 FL+RR 踩地、FR+RL 抬起，后半周期反过来。
        静态姿势最多满足一半时间（上限 0.5 分），真正交替迈步才能拿满。
        顺带把步频锁在 1/period 附近 —— 正好治"步频 5~7 Hz、步幅过小"。
    锁 2【活跃度门控】维护每环境的"最近是否换过接触状态"衰减量 active ∈ [0,1]
        （本步有任一条腿换状态 ⇒ 1，之后每步按 step_dt/period 线性衰减到 0）。
        奖励 = 相位匹配 × active ⇒ **从不换状态的静态姿势直接得 0**，连那 0.5 分也拿不到。
    两锁叠加后，只有"真的在对角交替迈步"才能持续得分。

    Args:
        threshold: 接触判定阈值（N），超过记为着地。
        period: trot 周期（秒）。0.8 s ⇒ 1.25 Hz；想更慢更优雅可调到 1.0~1.2。
        order: 四条小腿命名顺序（FL/FR/RL/RR），按本资产调整。
    """
    cs = env.scene.sensors[sensor_cfg.name]
    names = list(cs.body_names) if hasattr(cs, "body_names") else list(cs.data.body_names)
    ids = [names.index(f"{k}_calf") for k in order]
    f = _t(cs.data.net_forces_w_history)
    body_axis = next(ax for ax in range(1, f.dim()) if f.shape[ax] == len(names))
    f = f.movedim(body_axis, 2).norm(dim=-1).max(dim=1)[0]  # (N,B) 历史窗口内最大合力
    c = f[:, ids] > threshold  # (N,4) bool: FL FR RL RR
    c_fl, c_fr, c_rl, c_rr = c[:, 0], c[:, 1], c[:, 2], c[:, 3]

    step_dt = float(env.step_dt)

    # ---- 锁 2：活跃度（最近是否换过接触状态）----
    prev = getattr(env, "_trot_prev_c", None)
    active = getattr(env, "_trot_active", None)
    if prev is None or prev.shape != c.shape or active is None or active.shape[0] != c.shape[0]:
        prev = c.detach().clone()
        active = torch.zeros(c.shape[0], device=c.device)
    changed = (c != prev).any(dim=1)
    active = torch.where(changed, torch.ones_like(active), (active - step_dt / period).clamp(min=0.0))
    env._trot_prev_c = c.detach().clone()
    env._trot_active = active

    # ---- 锁 1：相位时钟 ----
    t = _t(env.episode_length_buf).float() * step_dt
    phase = (t % period) / period  # [0,1)
    want_a = phase < 0.5  # True：对角组 A（FL+RR）应踩地
    ok = (c_fl == c_rr) & (c_fl == want_a) & (c_fr == c_rl) & (c_fr != want_a)

    return ok.float() * active

# ============================================================================
# 移植自参考仓库 DeepRoboticsLab/rl_training（四足 Lite3 任务）
#   该仓库 velocity/config/quadruped/deeprobotics_lite3/rough_env_cfg.py 里有注释
#   "add the following rewards to improve the gait"，本文件的两项即来自那里。
# ============================================================================


def joint_mirror(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    mirror_joints: list = None,
) -> torch.Tensor:
    """对角腿"关节镜像对称"惩罚（配合负权重）：要求成对的两条腿关节角相同，返回 (N,)。

    ★为什么需要它（参考仓库的做法，直接对应我们的病态步态）：
      实测第 2 轮步态里 ΔFL 拖地、RL 悬空、只有 FR/RR 在迈步 —— 四条腿严重不对称。
      真正的对角步态（trot）里，**对角两条腿是同相的**（FL 与 RR 一起前摆/后蹬），
      因此它们的关节角应当近似相等。本项就是把这条"结构先验"写进奖励：
          惩罚 = Σ_pairs || q_腿A − q_腿B ||²
      腿不对称 ⇒ 该项很大 ⇒ 策略被迫让对角两腿协调一致。

    ★只在"机身基本竖直"时生效（照抄参考仓库的写法）：
      乘以 clamp(-projected_gravity_z, 0, 0.7)/0.7 —— 倒地/翻滚时不惩罚（那种情况由
      姿态终止和 flat_orientation 负责），避免在摔跤瞬间给出混乱梯度。

    参考实现：DeepRoboticsLab/rl_training  velocity/mdp/rewards.py::joint_mirror

    Args:
        mirror_joints: 成对关节的正则列表，例如
            [["FL_(hip|thigh|calf)_joint", "RR_(hip|thigh|calf)_joint"],
             ["FR_(hip|thigh|calf)_joint", "RL_(hip|thigh|calf)_joint"]]
    """
    asset = env.scene[asset_cfg.name]
    if mirror_joints is None:
        mirror_joints = [
            ["FL_(hip|thigh|calf)_joint", "RR_(hip|thigh|calf)_joint"],
            ["FR_(hip|thigh|calf)_joint", "RL_(hip|thigh|calf)_joint"],
        ]
    cache = getattr(env, "_joint_mirror_cache", None)
    if cache is None:
        cache = [[asset.find_joints(name)[0] for name in pair] for pair in mirror_joints]
        env._joint_mirror_cache = cache

    reward = torch.zeros(env.num_envs, device=env.device)
    for pair in cache:
        a, b = pair[0], pair[1]  # 各自的关节索引列表
        n = min(len(a), len(b))
        diff = torch.square(_t(asset.data.joint_pos)[:, a[:n]] - _t(asset.data.joint_pos)[:, b[:n]])
        reward += diff.sum(dim=-1)
    reward /= max(len(cache), 1)
    # 只在机身竖直时生效
    grav_z = _t(asset.data.projected_gravity_b)[:, 2]
    reward = reward * (torch.clamp(-grav_z, 0.0, 0.7) / 0.7)
    return reward


def foot_impact_velocity(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_calf"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*_calf"),
    speed_threshold: float = 0.10,
) -> torch.Tensor:
    """落地冲击速度惩罚（配合负权重）：脚"砸"地越快罚越重，返回 (N,)。

    ★作用：让脚轻柔落地。参考仓库用它抑制"跺脚式"步态；对我们还能顺带压制
      "整条腿硬砸地面借力滑行"这类退化解（那种步态落地垂速很大）。

    只在**首次接触那一帧**计分：penalty = Σ (clamp(-v_z,0) − threshold)²  （仅触地脚）
    参考实现：DeepRoboticsLab/rl_training  velocity/mdp/rewards.py::foot_impact_velocity
    """
    cs = env.scene.sensors[sensor_cfg.name]
    asset = env.scene[asset_cfg.name]
    first_contact = cs.compute_first_contact(env.step_dt)
    first_contact = _t(first_contact)[:, sensor_cfg.body_ids].float()
    v_z = _t(asset.data.body_lin_vel_w)[:, asset_cfg.body_ids, 2]
    downward = torch.clamp(-v_z, min=0.0) - speed_threshold
    downward = torch.clamp(downward, min=0.0)
    return torch.sum(first_contact * torch.square(downward), dim=1)

def phase_gait_soft(
    env: ManagerBasedRLEnv,
    period: float = 0.8,
    threshold: float = 1.0,
    phase_offsets: tuple = (0.0, 0.5, 0.5, 0.0),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_calf"),
    order: tuple = ("FL", "FR", "RL", "RR"),
) -> torch.Tensor:
    """【逐腿软核】相位步态奖励，返回 (N,) 值域 [0,1]。四条腿各自打分再平均。

    ==================== 为什么替换掉原来的硬 0/1 版本 ====================
    第 5 轮实测（参考仓库奖励配置、8249 轮后）：向前走速度误差已到 0.0107（近乎完美），
    但四腿占空比仍是 FL 0.89 / FR 0.45 / **RL 0.00（全程悬空）** / **RR 1.00（全程踩地）**。
    根因不是权重不够，而是**判据形式错了**：
        旧的 trot_symmetry 要求"(FL==RR) 且 (FR==RL) 且 两组反相"三条同时成立才给 1 分，
        不成立就是 **0 分**（而且还有活跃度门控）。策略既然做不到就**拿不到任何梯度**，
        而速度跟踪项是软的（处处有梯度，权重 4.0）—— 理性选择就是放弃步态、专注速度。
    ⇒ 改成本函数：**每条腿单独打分**，按"它当前的接触状态与相位要求差多远"给 0~1，
       四条腿取平均。好处：
         * 每条腿都有独立梯度（软核，不是硬门）；
         * "某条腿全程不动"只能拿到 ~0.5 分，"四条腿都按相位交替"才拿满 1.0；
         * 不需要活跃度门控：静态腿自然只能拿一半分。

    相位定义（trot）：FL 与 RR 同相（offset 0.0），FR 与 RL 同相（offset 0.5）。
    每只脚的期望状态：phase < 0.5 → 踩地，否则抬起。

    参考：DeepRoboticsLab/rl_training 的 `feet_gait`（官方新版 Isaac Lab）与本函数意图一致，
    但官方是硬判据 + 相位时钟；本函数把硬判据换成逐腿软核以提供梯度。
    """
    cs = env.scene.sensors[sensor_cfg.name]
    names = list(cs.body_names) if hasattr(cs, "body_names") else list(cs.data.body_names)
    ids = [names.index(f"{k}_calf") for k in order]
    f = _t(cs.data.net_forces_w_history)
    body_axis = next(ax for ax in range(1, f.dim()) if f.shape[ax] == len(names))
    f = f.movedim(body_axis, 2).norm(dim=-1).max(dim=1)[0]
    contact = (f[:, ids] > threshold).float()  # (N,4) 1=着地 0=腾空

    # 每只脚自己的相位（含 offset）
    t = _t(env.episode_length_buf).float() * float(env.step_dt)
    base_phase = torch.remainder(t / period, 1.0).unsqueeze(1)  # (N,1)
    offs = torch.tensor(phase_offsets, device=env.device, dtype=base_phase.dtype).unsqueeze(0)
    phase = torch.remainder(base_phase + offs, 1.0)  # (N,4)
    desired_contact = (phase < 0.5).float()  # (N,4) 1=期望踩地

    # 逐腿得分：接触状态与期望一致 = 1，否则 0（软化：用 (1 - |差|) 的加权平滑）
    #   为保证处处有梯度，用 |contact - desired| 的平滑版本：
    #     完全一致 → 1；完全相反 → 0。这里不做"硬 0/1"，而是按接近程度线性给分。
    per_leg = 1.0 - torch.abs(contact - desired_contact)
    return per_leg.mean(dim=1)


def foot_placement_and_turning(
    env: ManagerBasedRLEnv,
    period: float = 0.65,
    ground_z: float = 0.2401,
    target_stride: float = 0.24,
    target_clearance: float = 0.045,
    std_x: float = 0.075,
    # ★2026-10-07 方案A：σ_y 0.06→0.03。原来 exp 核太宽，横向保持的梯度弱到
    #   策略可以不理（这就是髋关节乱摆的直接原因）。
    std_y: float = 0.03,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_calf"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*_calf"),
    command_name: str = "base_velocity",
    order: tuple = ("FL", "FR", "RL", "RR"),
) -> torch.Tensor:
    """约束足端前后摆幅，并按角速度产生内外侧落脚差。"""
    asset = env.scene[asset_cfg.name]
    # ★按刚体【名字】查索引，不用 asset_cfg.body_ids：
    #   实测 body_ids 未按正则解析成 4 条 calf，导致 pos_w 变成全部 21 个刚体，
    #   与 (N,4) 的相位张量相加时报 "size of tensor a (4) must match b (21)"。
    _bn = list(asset.data.body_names)
    _leg_ids = [_bn.index(f"{k}_calf") for k in order]
    pos_w = _t(asset.data.body_pos_w)[:, _leg_ids, :]
    pos_b = quat_apply_inverse(
        yaw_quat(_t(asset.data.root_quat_w)).unsqueeze(1).expand(-1, pos_w.shape[1], -1),
        pos_w - _t(asset.data.root_pos_w).unsqueeze(1),
    )
    # asset_cfg 的正则表达式只匹配四个 calf，Isaac Lab 会按资产刚体顺序返回四条腿。
    cmd = env.command_manager.get_term(command_name).command
    vx, wz = cmd[:, 0], cmd[:, 2]
    # vx≈0 的原地转向不能保留直行用的最小前后步幅。
    stride = torch.clamp(torch.abs(vx) * period * 0.95, min=0.0, max=target_stride)
    phase = torch.remainder((_t(env.episode_length_buf).float() * env.step_dt) / period, 1.0)
    offsets = torch.tensor((0.0, 0.5, 0.5, 0.0), device=pos_b.device, dtype=pos_b.dtype)
    leg_phase = torch.remainder(phase.unsqueeze(1) + offsets.unsqueeze(0), 1.0)
    # 用连续余弦轨迹代替支撑/摆动切换处的目标跳变。
    # phase=0 时脚在前方，phase=0.5 时脚在后方，整个周期内位置和速度都连续。
    gait_wave = torch.cos(2.0 * torch.pi * leg_phase)
    x_nom = pos_b[:, :, 0].detach()
    y_nom = pos_b[:, :, 1].detach()
    x_translate = 0.5 * stride[:, None] * torch.sign(vx)[:, None] * gait_wave
    # 原地转向采用绕机身的切向轨迹，而不是继续执行直行前后摆腿。
    # ★助手修正 1（方向）：相位 0 是【落地瞬间】，脚应落在行进方向前方
    #   ——与直行"脚落在 +x 前方"同一约定。机身以 +ωz 绕中心旋转时，
    #   位置 r=(x,y) 处的切向方向是 (-y,+x)；波峰在相位 0，故目标取 (-y,+x)，
    #   相对原来的 (y,-x) 要取负号。数值例子：x=0,y=+0.2,ωz=+1 ⇒ ω×r=(-0.2,0)，
    #   脚相对机身向 +x 滑，落地时应在 -x 侧。
    # ★助手修正 2（幅值）：支撑相 0.325 s 内机身转过 Δθ=ω·T/2=0.325 rad，
    #   脚相对机身总行程 = r·Δθ = 0.065 m；余弦轨迹是 ±A 对称摆动 ⇒
    #   A = 0.25·ω·T·r。原来 0.5·ω·T·r 正好是 2 倍（对照：x_translate 幅值
    #   是 stride/2，stride 是总行程——同一约定）。
    # ★2026-10-07 方案A：撤销旋转门控。A/B 实测证明门控使直走髋外展更差
    #   （对角髋角差 0.335 → 0.825），且第 13 轮无门控时 FL_hip 均值 +0.087
    #   本就接近名义值 +0.1 ⇒ "髋外展"的主因不是旋转轨迹。
    turn_arc = -0.25 * wz[:, None] * period
    x_rotate = turn_arc * y_nom * gait_wave
    y_rotate = -turn_arc * x_nom * gait_wave
    x_target = x_translate + x_rotate
    y_target = y_nom + y_rotate
    xy_err = torch.square((pos_b[:, :, 0] - x_target) / std_x) + torch.square((pos_b[:, :, 1] - y_target) / std_y)
    xy_reward = torch.exp(-0.5 * xy_err).mean(dim=1)
    _cs = env.scene.sensors[sensor_cfg.name]
    _sn = list(_cs.data.body_names) if hasattr(_cs.data, "body_names") else list(_cs.body_names)
    _sids = [_sn.index(f"{k}_calf") for k in order]
    force = _t(_cs.data.net_forces_w_history).norm(dim=-1).max(dim=1)[0][:, _sids]
    swing = leg_phase >= 0.5
    # ★高度基准从硬编码 0.240 改为参数：该值=小腿刚体原点在地面接触时的高度，
    #   会随站立姿态变化（实测 0.2401 / 0.2693 都出现过）。硬编码会让"离地高度"
    #   系统性偏 3 cm，进而让 7 cm 抬脚目标失准。
    height = pos_w[:, :, 2] - ground_z
    clearance = torch.exp(-torch.square(height - target_clearance) / 0.04**2)
    return xy_reward + 0.35 * (clearance * swing.float() * (force < 1.0).float()).mean(dim=1)


def rear_leg_support_and_push(
    env: ManagerBasedRLEnv,
    min_rear_force: float = 8.0,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_calf"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*_calf"),
    order: tuple = ("FL", "FR", "RL", "RR"),
) -> torch.Tensor:
    """保证左右后腿都参与承重，并减少后腿悬空/拖步。"""
    cs = env.scene.sensors[sensor_cfg.name]
    force = _t(cs.data.net_forces_w_history).norm(dim=-1).max(dim=1)[0]
    names = list(cs.body_names)
    ids = [names.index(f"{k}_calf") for k in order]
    rear = force[:, [ids[2], ids[3]]]
    rear_support = (rear > min_rear_force).float().mean(dim=1)
    asset = env.scene[asset_cfg.name]
    foot_v = _t(asset.data.body_lin_vel_w)[:, asset_cfg.body_ids, :2]
    rear_stable = torch.exp(-torch.sum(torch.square(foot_v[:, 2:4]), dim=-1) / 0.08**2).mean(dim=1)
    return 0.65 * rear_support + 0.35 * rear_stable


def contact_load_balance(
    env: ManagerBasedRLEnv,
    threshold: float = 8.0,
    std: float = 0.04,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_calf"),
    order: tuple = ("FL", "FR", "RL", "RR"),
) -> torch.Tensor:
    """均衡四足承重，避免机械臂负载诱发单侧支撑和明显 roll/pitch 补偿。

    ★修正记录（原版会反噬）：原来用 `torch.var(share * active)` 在【全部四条腿】上算方差，
      而正确的 trot 同一时刻只有两条腿承重（另两条的 share≈0），方差必然大 ⇒
      实测：完美 trot 只得 0.044 分、而"四条腿站着不动"得 1.000 分 —— 变成了
      【奖励站着不动、惩罚正确步态】，与"减少单侧过度承重"的本意正好相反（权重 0.8 不小）。
    现在改为：**只在实际承重（力 > threshold）的腿之间**比较受力占比的均衡度。
      * 正确 trot：对角支撑的两条腿受力接近 ⇒ 高分（对角线的载荷均衡 = 机身投影落在支撑对角线上）
      * 四条腿均摊：同样高分（不排斥，但不再是唯一最优）
      * 当前坏步态（FL 独大、RL 几乎不承重）：3 条腿互不均衡 ⇒ 中低分
    """
    cs = env.scene.sensors[sensor_cfg.name]
    _names = list(cs.data.body_names) if hasattr(cs.data, "body_names") else list(cs.body_names)
    _ids = [_names.index(f"{k}_calf") for k in order]
    f = _t(cs.data.net_forces_w_history).norm(dim=-1).max(dim=1)[0][:, _ids]
    active = (f > threshold).float()
    n_active = active.sum(dim=1, keepdim=True).clamp(min=1.0)
    share = f / f.sum(dim=1, keepdim=True).clamp(min=1.0)
    # 承重腿之间的平均占比（非承重腿不参与，避免"腿不落地"被算成不均衡）
    mean_active = (share * active).sum(dim=1, keepdim=True) / n_active
    var_active = (((share - mean_active) ** 2) * active).sum(dim=1) / n_active.squeeze(1)
    return torch.exp(-var_active / std) * (n_active.squeeze(1) >= 2).float()

def action_acc_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """动作的二阶差分 L2（急动/jerk），抑制动作抖动。

    ★移植自 ZoreAnuj/go2-piper-quiet-goto（其 action_acc 权重 -0.01）。
    与已有的 action_rate_l2（一阶差分，权重 -0.1）互补：
      action_rate 管"动作变化多快"，action_acc 管"变化率本身抖不抖"。
    """
    a = env.action_manager.action
    if (not hasattr(env, "_act_prev")) or env._act_prev.shape != a.shape:
        env._act_prev = torch.zeros_like(a)
        env._act_prev2 = torch.zeros_like(a)
    acc = a - 2.0 * env._act_prev + env._act_prev2
    env._act_prev2 = env._act_prev.clone()
    env._act_prev = a.clone()
    return torch.sum(torch.square(acc), dim=1)


def hip_abduction_l2(
    env: ManagerBasedRLEnv,
    nominal: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    order: tuple = ("FL", "FR", "RL", "RR"),
) -> torch.Tensor:
    """髋关节外展惩罚：偏离名义值的平方和，返回 (N,)。

    名义值来自机器人初始姿态：FL/RL = +0.1，FR/RR = -0.1 rad。
    正常四足行走时"前后摆腿"由 thigh 关节负责，hip 只做很小的横向调整；
    若 hip 长时间大幅偏离名义值，就等于"腿往外拐着走"（本轮要治的病）。

    ★为什么需要它（2026-10-07 方案A）：
      原先只靠足端横向目标 `y_target = y_nom` 间接约束髋外展，那个 exp 核太宽
      （σ_y=0.06），梯度弱到策略可以不理；实测直走时对角髋角差达 0.335 rad、
      后左腿髋关节在 0.19~1.00 rad 之间摆动（47°）。
      这是一个**直接作用在被控关节上**的罚项，不依赖轨迹目标的间接传导。
    """
    asset = env.scene[asset_cfg.name]
    ids = [asset.joint_names.index(f"{k}_hip_joint") for k in order]
    q = _t(asset.data.joint_pos)[:, ids]
    target = torch.tensor([nominal, -nominal, nominal, -nominal],
                          device=q.device, dtype=q.dtype)
    return torch.sum(torch.square(q - target), dim=1)
