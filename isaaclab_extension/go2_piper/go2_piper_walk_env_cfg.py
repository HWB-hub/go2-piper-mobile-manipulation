# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""S1【步态任务】：只学走路（正常向前走 + 转圈），手臂固定在标称姿态。

## 为什么要单独有这一步（用户从视频里看出来的真问题）

用户看 S2 追踪视频时指出：机器狗"像是还没学会走路，没能自动调整腿部姿态来提供稳定支撑，
原本应该趴着操作机械臂的动作看着像倒立"。

查证后发现这是**结构性缺口**，不是调参问题：
    <historical-local-path>/2026-10-03_18-23-21  → 只有 2 个权重，
    最新是 model_1.pt  ⇒ **平地运动任务从来没真正训练过**（跑 1 轮就停的空壳）。

也就是说：此前的 Reach / ReachGrip / Pick / Track 全部是在一个**根本不会走路**的机器人上
直接训"够物 / 抓取"。策略为了拿奖励，只能用最省事的方式（趴、爬、拿身体当支点），
于是出现了"倒立"这种姿态。

⇒ 按用户提出的分期方案重建：
    第 1 步（本文件）  只学步态：向前走 / 转圈，手臂保持标称姿态
    第 2 步            加入手臂运动：边走 / 边转圈边动胳膊（在此步态权重上继续）
    第 3 步            追目标 + 抓取

## 与基座任务（Go2Piper-Flat）的关系

基座 `Isaac-Go2Piper-Flat-v0` 用的是**自建 18 自由度资产**（无夹爪），
而下游 Reach/ReachGrip/Pick/Track 用的是**作者版 20 自由度带夹爪资产**。
两者观测维度不同（75 vs 79），权重无法直接迁移 ⇒ 本任务直接在**带夹爪资产**上训，
保证第 2、3 步能无缝续训（观测 81 维、动作 18 维完全一致；新增 2 维步态时钟）。

## 奖励设计（本任务最关键的一点）

* `ee_goal_tracking = None` —— **关掉够物奖励**。这一步只学走路，手臂伸出去够东西
  会和"保持稳定支撑的步态"直接冲突（训练初期策略会用伸胳膊/压身体来骗分）。
* `arm_pose_tracking` 权重从 0.2 提到 **2.0** —— 反过来**强制手臂待在标称姿态**，
  让它成为走路时的一个"固定负载"，这样第 2 步加手臂动作时才有干净的基础。
* 其余沿用基座的 legged_gym 标准奖励集（速度跟踪 + 高度 + 姿态 + 步频 + 力矩/加速度/动作率）。
"""

from __future__ import annotations

import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as velocity_mdp

from . import mdp
from .mdp.commands import SlewRateVelocityCommand  # 限速率速度指令（移植自参考项目）
# 站立时"小腿刚体原点"的高度（本资产脚端并进 calf，body_pos_w 给的是刚体原点而非脚底）。
# ★实测值（probe_calf_height.py，64 环境 × 200 步）：0.2401 m
#   同一探针同时量得：各腿接触力 FL=144.6N FR=45.1N RL=11.5N RR=80.6N（总 282N ≈ 28.7kg，
#   与"机+臂"重量相符），承重腿条数(>1N)均值 2.72~2.94 ⇒ 该策略用约 3 条腿支撑。
#   ⚠ 探针坑：接触力历史数组按【传感器自己的刚体表】排列（21 个，calf 索引 [3,6,9,12]），
#     不是机器人刚体表（calf 索引 [11,12,13,14]）。用错索引会读出"只有一条腿受力"的假象。
GO2_CALF_GROUND_Z = 0.2715  # ★实测校正：站姿小腿刚体原点高度
#   历史值 0.2401（第5轮权重）/0.2693（第6轮）/0.2674（第7轮 model_999）——
#   该值随腿部折叠姿态变化，硬编码会让奖励里的"离地高度"系统性偏 2~3 cm，
#   直接导致抬脚目标失真（目标 0.045 实际只有 1.8 cm，会蹭地）。
#   每次换权重或改站姿后应重新用 probe_calf_height.py 实测。

from .go2_piper_reach_env_cfg import (
    ReachCommandsCfg,
    ReachEventsCfg,
    ReachRewardsCfg,
    ReachTerminationsCfg,
)
from .go2_piper_reach_grip_env_cfg import Go2PiperReachGripEnvCfg


@configclass
class WalkCommandsCfg(ReachCommandsCfg):
    """速度指令按用户要求侧重"向前走 + 转圈"。

    与基座的差别：
      * lin_vel_x 改成 **(-0.4, 1.0)**：前向范围更大、后退范围收小（用户要"正常向前走"）；
      * ang_vel_z **(-1.0, 1.0)**：把"转圈"练足；
      * lin_vel_y 收窄到 (-0.3, 0.3)：侧移对四足很难，先别让它拖累主任务；
      * 站住的环境从 2% 提到 10%：站姿也是"稳定支撑"的一部分。
    """

    def __post_init__(self):
        super().__post_init__()
        # 更频繁切换命令，避免策略只优化直行段。
        self.base_velocity.resampling_time_range = (5.0, 7.0)
        self.base_velocity.rel_standing_envs = 0.10
        self.base_velocity.rel_heading_envs = 1.0
        self.base_velocity.heading_command = True
        self.base_velocity.heading_control_stiffness = 0.5
        self.base_velocity.ranges.lin_vel_x = (-0.25, 0.85)
        self.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        self.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.base_velocity.ranges.heading = (-math.pi, math.pi)
        # ★移植自 ZoreAnuj/go2-piper-quiet-goto：把速度指令换成【限速率】版本。
        #   原版重采样是阶跃（目标瞬间从 A 跳到 B），策略被迫学急反应 ⇒ 转弯猛扭头。
        #   这里给指令加速度上限：线 1.0 m/s²、角 1.5 rad/s²，约 1 秒平滑到位。
        self.base_velocity.class_type = SlewRateVelocityCommand
        self.base_velocity.slew_rate_lin = 1.0
        self.base_velocity.slew_rate_ang = 1.5
        # 目标点指令保留（只为让观测维度与下游任务一致 = 79 维），
        # 但奖励里已把它关掉 ⇒ 策略学不到也无需理会它。
        self.ee_goal.debug_vis = False


@configclass
class WalkEventsCfg(ReachEventsCfg):
    """域随机化：在 Reach 的基础上补【机械臂自身质量】随机化。

    为什么必须补这一项（用户特别点名要 event 部分做鲁棒性）：
        机械臂是**后装**在机器狗背上的负载，它的实际质量与 CAD/URDF 标称值必然有偏差；
        而机身质心已经随机了（±5 cm）但**质量没有** ⇒ 两者的组合才是真实的失配。
        训练时把四条臂段的"质量"整体缩放 ±15%，策略就不会依赖"精确的臂重"这个假设，
        真机换臂、加装线束/夹爪后仍然稳定 —— 这是 sim2real 里最值钱的一类随机化。

    注意：整机的其他随机化已在基座/R each 里覆盖（摩擦、机身质量、机身质心、
          驱动器增益、重置位姿、周期性推动），这里只补这一项，不重复设置。
    """

    arm_payload_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            # piper_link1~6 = 六条臂段（7/8 是夹爪手指，单独留着不动）
            "asset_cfg": SceneEntityCfg("robot", body_names=["piper_link[1-6]"]),
            "mass_distribution_params": (1.0 / 1.15, 1.15),  # ±15%
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )


@configclass
class WalkRewardsCfg(ReachRewardsCfg):
    """步态奖励：关掉够物，强化"手臂保持标称姿态"。"""

    # ★关掉够物：这一步不学够物（它是 Reaceh 系列的核心项，权重 2.0）
    ee_goal_tracking = None

    # ★★转向：**放宽指数核**，而不是继续加权重（2026-10-04 代码审查纠正了我上一轮的错判）
    #
    #   我上一轮把权重从 0.75 加到 1.5，归因于"权重只有前进项一半"。审查指出这个归因是错的：
    #   真正的问题是**核宽**。数值核实（已用工具算过）：
    #       转向误差 1.5 rad/s 时 exp(-1.5²/0.5²) = 1.2e-4
    #       ⇒ 权重 0.75→1.5 只是把 2.4e-4 变成 4.8e-4，梯度仍≈0，白加。
    #       而前进方向好使，是因为指令 0.8 / 实测 0.7，误差 0.1 ≪ std=0.25。
    #   ⇒ 正确做法：把 std 从 √0.25=0.5 放宽到 √1.0=1.0（误差 1.5 时核值从 1.2e-4 升到 0.105，有梯度了）。
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        # ★1.5→2.5：参考项目 track_linear_velocity=2.0 与 track_angular_velocity=2.0
        #   【等权】；我们原来是 4.0 : 1.5（2.7 倍），角速度相对被压低 ⇒ 转弯跟不上、
        #   靠身体扭转凑。提到 2.5 让转向跟踪与直行同量级。
        weight=2.5,  # ★提到 2.5（参考项目线/角等权 2.0:2.0）
        params={"command_name": "base_velocity", "std": math.sqrt(1.0)},
    )

    # ★手臂当成固定负载：权重 0.2 → 2.0
    #   理由：走路时手臂乱甩会破坏支撑；而且第 2 步要在此基础上"主动动手臂"，
    #   所以第 1 步必须先把"手臂不动"变成一个稳定的默认行为。
    # ★权重 2.0 → 1.0（2026-10-04 审查）：这一项是 6 个关节的平方和，
    #   单关节等效 σ = 0.25/√6 = 0.102 rad ≈ 5.8°（不是我原注释写的 14°），
    #   6 个关节各偏 0.25 rad 时奖励只剩 exp(-6)=0.25%。
    #   配上 2.0（本来就已是全场最大权重）确实会主导策略、压制走路的自然度。
    #   降到 1.0：仍足以让手臂保持"固定负载"，但不再压过步态本身。
    arm_pose_tracking = RewTerm(
        func=mdp.arm_pose_tracking,
        weight=1.0,
    )

    # ★★补上官方速度任务的"非足部触地"惩罚（下一轮生效）
    #   这一项直接针对"趴地 / 爬行 / 拿胳膊或膝盖当支点"这类退化解：
    #   除足端以外任何刚体以 >1 N 的力触地就扣分。
    #
    #   ⚠ 正则不能照抄官方！官方 Go2 写的是 "#(?!.*FOOT)(.*)"，
    #     因为它的脚是独立刚体 *_foot；而本资产转换时用了 --merge-joints，
    #     **脚端球体已并进 *_calf**（已验证：21 个刚体里没有 *_foot）。
    #     所以这里必须用 (?!.*_calf)(.*)：除四条小腿（=真正的足端）之外的刚体
    #     （base / hip / thigh / piper_link1-8）触地都要罚。
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="(?!.*_calf)(.*)"),
            "threshold": 1.0,
        },
    )

    # ============================================================================
    # ★★按参考仓库 DeepRoboticsLab/rl_training（四足 Lite3）重设权重
    #    来源: tasks/.../velocity/config/quadruped/deeprobotics_lite3/rough_env_cfg.py
    #    该文件里有注释 "add the following rewards to improve the gait"，
    #    它把我们这套"两条腿走、两条腿悬空"的病态步态作为重点压制对象。
    #
    #    核心差异（我原值 → 参考值）：姿态类惩罚全体弱了一个数量级，
    #    所以"贴着地滑行/摆固定姿势"这类退化解的成本太低、压不住：
    #        flat_orientation_l2   -2.0  →  -20.0
    #        lin_vel_z_l2          -2.0  →  -20.0
    #        ang_vel_xy_l2        -0.05  →  -0.25
    #        action_rate_l2       -0.01  →  -0.1
    #        track_lin_vel_xy_exp   1.5  →   4.0   （速度跟踪给足，别让别项盖过主任务）
    #        track_ang_vel_z_exp    1.0  →   1.5
    #        feet_air_time         0.05  →   0.5   （我原值几乎等于没给，参考仓库是 5.0）
    # ============================================================================
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=4.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )

    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-20.0)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-20.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.25)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.1)
    joint_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1e-8)
    joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-2.5e-4)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="(?!.*_calf)(.*)"),
            "threshold": 1.0,
        },
    )
    feet_air_time = RewTerm(
        func=velocity_mdp.feet_air_time,
        weight=0.5,  # ★0.05 → 0.5：原值小到可忽略，(空)时长几乎没有塑形作用
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_calf"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )

    # ---- 移植自参考仓库的两项（见 mdp/rewards.py 里的实现与注释）----
    joint_mirror = RewTerm(
        func=mdp.joint_mirror,
        weight=-0.3,  # ★从参考仓库的 -0.05 提到 6 倍：第 5 轮实测 RL 恒悬空/RR 恒踩地，
                      #   即对角腿完全不对称，而参考仓库的 -0.05 是配合它自己的 feet_gait
                      #   （官方硬相位判据，能直接约束住四腿）用的；我们没有那个约束，
                      #   等效地放大了本该由相位项施加的压力。
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mirror_joints": [
                ["FL_(hip|thigh|calf)_joint", "RR_(hip|thigh|calf)_joint"],
                ["FR_(hip|thigh|calf)_joint", "RL_(hip|thigh|calf)_joint"],
            ],
        },
    )
    foot_impact_velocity = RewTerm(
        func=mdp.foot_impact_velocity,
        weight=-2.0,  # 与参考仓库一致
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_calf"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_calf"),
            "speed_threshold": 0.10,
        },
    )

    # ================= 足端与支撑约束（2026-10-04 补齐，这是最大的缺口）=================
    # 审查结论：与"脚"有关的原来只有 feet_air_time，而它**只奖励摆动腿滞空**，
    # 不含任何"支撑腿踩住 / 抬脚高度 / 承重腿条数"的约束。缺了下面三项，
    # 最省事的解就是"抬得高、步频慢、支撑腿在地上蹭"——正是"看着不会用腿稳定支撑"。
    #
    # ① 支撑腿打滑惩罚（官方自带，直接挂）
    #    body_names 用 .*_calf：本资产 merge-joints 后没有 *_foot 刚体（脚并进 calf）
    feet_slide = RewTerm(
        func=velocity_mdp.feet_slide,
        # 参考仓库为 -0.05；我一度调到 -0.2 去压"FL 拖地"，但那治标不治本
        # （根因是姿态类惩罚太弱），现改回中间值 -0.1，配合 joint_mirror / 相位约束一起用。
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_calf"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_calf"),
        },
    )

    # ② 摆动腿离地高度（自己写：feet_air_time 只管"多久"，不管"多高"）
    #    ground_z = 站立时小腿刚体原点高度，实测值见 GO2_CALF_GROUND_Z
    foot_clearance = RewTerm(
        func=mdp.foot_clearance,
        weight=0.25,
        params={"target": 0.040, "std": 0.035, "ground_z": GO2_CALF_GROUND_Z},   # ★0.05→0.04：足端略高
    )

    # ③ 对角步态相位（trot）—— 治"一条腿拖地、一条腿悬空"的正主
    #    ⚠ 第 5 轮实测证明旧版 trot_symmetry（硬 0/1 判据）无效：速度误差已到 0.0107，
    #      但 RL 占空比仍为 0.00（全程悬空）、RR 1.00（全程踩地）。原因是硬判据不满足=0 分，
    #      策略拿不到梯度，而速度项处处有梯度 ⇒ 策略理性地放弃步态。
    #      现改为【逐腿软核】phase_gait_soft（每条腿独立打分，四腿取平均），
    #      静态腿最多拿 0.5 分、正确交替才拿满 1.0 ⇒ 每条腿都有独立梯度。
    #      period=0.65 s ⇒ 单腿步频约 1.54 Hz，给后腿足够时间完成后蹬，避免原地碎步。
    phase_gait_soft = RewTerm(
        func=mdp.phase_gait_soft,
        weight=3.0,  # 高于速度跟踪项（4.0）的同一量级，确保相位不被速度项压掉
        params={"period": 0.65, "threshold": 1.0},
    )

    # ④ 承重腿条数：少于 min_stance 条腿在承重就罚（feet_air_time 的反向平衡项）
    #    ⚠ min_stance=2 而不是审查稿里的 3：四足标准 trot 稳态本来就只有 2 条腿承重，
    #      设 3 会把正确步态一起罚掉。
    stance_duty = RewTerm(
        func=mdp.stance_duty,
        weight=-2.0,
        params={"min_stance": 2},
    )

    # 手臂关节速度惩罚加强一点（别乱甩）
    # ★移植自参考项目：动作急动（二阶差分）惩罚，权重 -0.01
    action_acc_l2 = RewTerm(func=mdp.action_acc_l2, weight=-0.01)

    # ★2026-10-07 方案A：髋关节外展惩罚（治"直走时腿往外拐着走"）
    #   名义值 FL/RL=+0.1、FR/RR=-0.1 rad；直走髋角偏离 0.3 rad ⇒ 该项约 -0.72。
    hip_abduction_l2 = RewTerm(func=mdp.hip_abduction_l2, weight=-2.0, params={"nominal": 0.1})

    arm_joint_vel_l2 = RewTerm(
        func=mdp.arm_joint_vel_l2,
        weight=-5.0e-3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["piper_joint.*"])},
    )

    foot_placement_and_turning = RewTerm(
        func=mdp.foot_placement_and_turning,
        weight=2.5,
        params={"period": 0.65, "target_stride": 0.24, "target_clearance": 0.035
                , "ground_z": GO2_CALF_GROUND_Z},
    )
    rear_leg_support_and_push = RewTerm(
        func=mdp.rear_leg_support_and_push,
        weight=1.5,
    )
    contact_load_balance = RewTerm(
        func=mdp.contact_load_balance,
        weight=0.8,
    )


##
# Environment configuration
##


@configclass
class Go2PiperWalkEnvCfg(Go2PiperReachGripEnvCfg):
    """S1 步态任务（带夹爪资产，观测 81 维 / 动作 18 维，与下游任务完全一致）。"""

    commands: WalkCommandsCfg = WalkCommandsCfg()
    rewards: WalkRewardsCfg = WalkRewardsCfg()
    events: WalkEventsCfg = WalkEventsCfg()
    terminations: ReachTerminationsCfg = ReachTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 20.0
        self.commands.ee_goal.debug_vis = False
        # 参考项目使用 0.25；配合较柔和 PD 可明显降低每个策略周期的硬跳变。
        self.actions.joint_pos.scale = 0.25


@configclass
class Go2PiperWalkEnvCfg_PLAY(Go2PiperWalkEnvCfg):
    """回放：少环境、去掉域随机化、固定一组指令（前进 + 左转 + 站住各 1 个）。

    这样录像里能同时看到三种行为：向前走 / 原地转 / 站住。
    """

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 3
        self.observations.policy.enable_corruption = False
        # ★千万别把 reset_base 置 None！
        #   排查记录：为了让回放"干净"，曾经把它设成 None，结果回放里机器人是翻倒的
        #   （机身高度 0.157 m、俯仰 ≈π），而**同一权重在训练配置下高度 0.283 m、
        #   重力投影 -0.994（完全竖直）** —— 训练好好的，只是回放配置自己坏了。
        #   reset_base = reset_root_state_uniform，它负责在每次重置时写入机身位姿/速度；
        #   去掉它之后重置行为不再完整，成了"看着像策略不行"的假象。
        #   这里只保留"不推机器人"（那是纯扰动），并把位置随机收成 0。
        self.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
        self.events.push_robot = None
        # ranges 设成同值 ⇒ 所有环境拿到同一个指令；靠 rel_standing_envs 再制造"站住"
        self.commands.base_velocity.ranges.lin_vel_x = (0.7, 0.7)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        # ★修正自相矛盾（2026-10-04 审查）：docstring 说回放要看到
        #   "向前走 / 原地转 / 站住"三种行为，但原来把 rel_standing_envs 和
        #   rel_heading_envs 都设成 0、ang_vel_z=(0,0) ⇒ 3 个环境全是同一个直行指令，
        #   既没转身也没站住。而且 heading_command=True 时 ang_vel_z 区间是死配置
        #   （ωz 由 heading 误差经 P 控制器生成），所以"站住"只能靠 rel_standing_envs 制造。
        #
        #   注意：录像脚本 record_walk.py 会自己覆盖这些字段（它要精确控制单一指令），
        #   这里修的是"直接 play 这个配置"时的行为。
        self.commands.base_velocity.rel_standing_envs = 1.0 / 3.0  # 3 个环境里 1 个站住
        self.commands.base_velocity.rel_heading_envs = 0.0  # 让 ωz 真从区间采样（可指定"原地转"）
