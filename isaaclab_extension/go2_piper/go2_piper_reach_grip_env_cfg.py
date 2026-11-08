# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reach 任务 · **作者版 Go2+Piper（带夹爪）资产**。

与 `Isaac-Go2Piper-Reach-v0` 的差别只有资产与随之而来的标定，MDP 设计完全一致：

    ① 机器人资产换成 Sivan666666/roboduet_go2piper 作者提供的整机
       （21 刚体 / 20 DOF，比自建版多两个夹爪平移关节 piper_joint7/8），
       配置直接复用 RoboDuet 任务里那份 ArticulationCfg（作者的增益/默认姿态/力矩上限）。
    ② 动作关节显式列出 18 个（腿 12 + 臂 piper_joint[1-6]），**排除夹爪**。
    ③ 目标点采样盒按新资产重新标定（见文件末尾的实测数据）。

★为什么必须显式排除夹爪关节（否则会踩坑）：
    基类动作项写的是 `joint_names=[".*"]`。换成 20 DOF 的资产后，`.*` 会把两个夹爪
    平移关节也拉进动作空间（动作维度变成 20）。而夹爪关节的限位只有 ±0.035 m
    （行程 3.5 cm），却会按**默认尺度**接到指令 —— 目标位 = 默认位 ± 尺度，
    直接顶死限位、PD 会把手指往死里压。作者原仓库的处理是把动作 pad 成 0
    （即夹爪不参与动作，靠隐式 PD 保持在默认的闭合位），这里与之保持一致。

★另一处几何提醒：夹爪两指尖（piper_link7/8）比末端参考点 piper_link6 **还往前 13.6 cm**
    （实测 base 系 link6 = (+0.113, -0.013, +0.394)，link7/8 = (+0.249, -0.015, +0.384)）。
    本任务仍用 piper_link6 作末端参考（与前三轮同口径、可直接对比），
    所以"够到目标点"= link6 到位，夹爪会略微越过目标点。真要做"夹住"时，
    应把参考点改到两指尖中点（那时才需要把夹爪关节接进动作）。
"""

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

import isaaclab_tasks.manager_based.locomanipulation.go2_piper.mdp as mdp
from isaaclab_tasks.manager_based.locomanipulation.roboduet_go2piper.robots import ROBODUET_GO2PIPER_CFG

from .go2_piper_env_cfg import Go2PiperSceneCfg
from .go2_piper_reach_env_cfg import (
    Go2PiperReachEnvCfg,
    Go2PiperReachEnvCfg_PLAY,
    ReachActionsCfg,
    ReachCommandsCfg,
    ReachPlaySceneCfg,
)

##
# 被驱动的 18 个关节（显式列出，排除夹爪 7/8）
##
DRIVEN_JOINT_NAMES = [".*_hip_joint", ".*_thigh_joint", ".*_calf_joint", "piper_joint[1-6]"]


##
# 机器人资产：作者版几何 + 【本任务自己的驱动器增益】
##
# ★为什么不直接用作者原配增益（kp 40 / kd 1）：
#   那套值是作者给他自己的 Isaac Gym 环境调的；而本任务的奖励（base_height 目标 0.30、
#   速度跟踪、能耗/平滑惩罚）是按"带 3.5 kg 负载、kp 100 / kd 4"实测调出来的。
#   直接用软腿增益跑，策略会钻空子：实测机身高度被压到 0.10~0.21 m、
#   `Episode_Reward/base_height` 从 0.068 塌到 0.005、回合长度只有 3~9 步（0.1 s 就被重置）、
#   24 s 内重置 15 次 —— 也就是"趴着走"，走路项也只学到 0.587（旧资产 1.11）。
#   → 换成本任务原配增益（与 go2_piper/robots.py 一致），只让"资产几何"成为唯一变量。
#   夹爪仍用作者的隐式 PD（它不参与动作，只需保持在默认闭合位）。
##
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg

GRIP_ROBOT_CFG = ROBODUET_GO2PIPER_CFG.replace(
    prim_path="{ENV_REGEX_NS}/Robot",
    actuators={
        # ---- 髋/大腿：★2026-10-06 按参考项目 go2-piper-quiet-goto 降刚度 ----
        #   100/4 → 45/1.8。高刚度会让 PD 强硬执行每次动作更新，表现为"腿部像机械弹簧、
        #   小腿突然折叠、落脚冲击大、转弯时身体先扭腿再补偿"。
        #   参考项目用 20/1（纯 Go2，无机械臂），本机背负 Piper ⇒ 取带负载的折中值。
        #   ⚠ 若训练后出现后腿下沉，先把这里调到 55，不要直接回到 100。
        "hip_thigh": DCMotorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint"],
            effort_limit=23.7,
            saturation_effort=23.7,
            velocity_limit=30.1,
            stiffness=45.0,
            damping=1.8,
            armature=0.01,
            friction=0.0,
        ),
        # ---- 小腿：力矩上限 45.4（Go2 小腿电机更大）----
        # ---- 小腿：★100/4 → 55/2.4（同上；小腿力矩上限 45.4 不变）----
        "calf": DCMotorCfg(
            joint_names_expr=[".*_calf_joint"],
            effort_limit=45.4,
            saturation_effort=45.4,
            velocity_limit=15.7,
            stiffness=55.0,
            damping=2.4,
            armature=0.01,
            friction=0.0,
        ),
        # ---- 臂：只覆盖 piper_joint1..6（夹爪单独一组）----
        # ★★刚度 60 → 400、阻尼 3 → 15（今晚最后的关键修正）
        #   实测：零动作下手臂 8 秒内自己沉降 **6 cm**（重力稳态误差，Isaac Lab 的
        #   隐式驱动器不做重力补偿）。这意味着"手根本停不住"——策略要把手伸到物体
        #   旁边并保持 1~2 秒才谈得上合爪，而手在这期间已经飘走 6 cm。
        #   这也解释了为什么"对位误差稳定在 2 cm 就不动了"：不是策略学不动，
        #   而是物理上到不了更准（把物体收窄到 2.2 cm 放宽容差也验证了这点，无改善）。
        #   刚度提到 400（≈7 倍）后稳态误差按比例缩小；阻尼 15 保证不振荡。
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["piper_joint[1-6]"],
            effort_limit_sim=80.0,
            velocity_limit_sim=5.0,
            stiffness=400.0,
            damping=15.0,
            armature=0.005,
        ),
        # ---- 夹爪：不参与动作（Reach 任务）/ 参与动作（Pick 任务），用隐式 PD 保持在默认位 ----
        # ★刚度必须够大！实测教训：
        #   手指压住 3 cm 宽的物体时，位置误差 = 1.5 cm。
        #     刚度 20 N/m  → 单指法向力 20×0.015 = 0.3 N
        #                    → 两指摩擦 2×0.3×1.5 = 0.9 N
        #                    → **小于物体重量 0.15 kg×9.81 = 1.47 N** ⇒ 一抬就滑
        #     这正是"策略学会了靠近(0.50)和合爪(0.84)、`pick_lift` 却恒为 0"的物理原因。
        #   刚度 300 N/m → 单指可达 4.5 N（被 effort_limit 限住）→ 摩擦 13 N ≫ 1.47 N ✓
        #   速度限制留小一点（0.5 m/s）：闭合太快容易把物体撞飞。
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["piper_joint[7-8]"],
            effort_limit_sim=20.0,
            velocity_limit_sim=0.5,
            stiffness=300.0,
            damping=10.0,
        ),
    },
)


@configclass
class GripSceneCfg(Go2PiperSceneCfg):
    """场景：只把机器人换成作者版带夹爪的整机，其余（地面/天空光/接触传感器）继承。"""

    # 接触传感器仍然覆盖机器人所有刚体；足端仍用 ".*_calf"
    # （这份资产转换时也 merge_fixed_joints=True，虽然 URDF 里标了 dont_collapse，
    #   实际生成后 21 个刚体里没有 *_foot —— 体检脚本已确认）
    robot = GRIP_ROBOT_CFG


@configclass
class GripCommandsCfg(ReachCommandsCfg):
    """指令：速度指令（继承）+ 末端目标点，采样盒按新资产重新标定。"""

    ee_goal = mdp.EeGoalCommandCfg(
        asset_name="robot",
        body_name="piper_link6",  # 这份资产里同样叫 piper_link6（作者 ee_idx=23 就是它）
        resampling_time_range=(5.0, 9.0),
        traj_time=(1.5, 3.0),
        world_frame=False,
        debug_vis=True,
        # ★采样盒：新资产实测可达点云的分位数区间（probe_ee_actual.py 实测，
        #   见文件末尾"实测数据"一节）。四舍五入到 0.01 并留了一点余量。
        ranges=mdp.EeGoalCommandCfg.Ranges(
            pos_x=(-0.05, 0.36),
            pos_y=(-0.24, 0.24),
            pos_z=(0.14, 0.52),
        ),
    )


@configclass
class GripActionsCfg(ReachActionsCfg):
    """动作：18 个关节的位置控制（腿 12 + 臂 6），**显式排除夹爪的两个平移关节**。

    per-joint 尺度的取值与旧资产版一致（腿 0.25、臂 2.0/1.2/1.2/0.6/0.6/1.0）——
    用同样的尺度再看新资产的可达空间，标定才有可比性。尺度的"够不够用"由
    probe_ee_actual.py 检验（它会打印"策略真实可用范围 vs 关节限位"）。
    """

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=DRIVEN_JOINT_NAMES,
        scale={
            # 腿：精细（和旧资产一致）
            ".*_hip_joint": 0.25,
            ".*_thigh_joint": 0.25,
            ".*_calf_joint": 0.25,
            # 臂：放大（joint1 底座回转决定左右伸展范围，给最大）
            "piper_joint1": 2.0,
            "piper_joint2": 1.2,
            "piper_joint3": 1.2,
            "piper_joint4": 0.6,
            "piper_joint5": 0.6,
            "piper_joint6": 1.0,
        },
        use_default_offset=True,
    )

    # ★夹爪关节不接动作，但也不要被 reset_arm_joints 随机化 —— 让它们稳稳待在默认（闭合）位
    #   （随机化夹爪没有意义，还会让"闭/开"状态在回合间乱跳）


##
# Environment configuration
##


@configclass
class Go2PiperReachGripEnvCfg(Go2PiperReachEnvCfg):
    """训练用环境（作者版 Go2+Piper 带夹爪资产）。"""

    scene: GripSceneCfg = GripSceneCfg(num_envs=4096, env_spacing=2.5, clone_in_fabric=True)
    commands: GripCommandsCfg = GripCommandsCfg()
    actions: GripActionsCfg = GripActionsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        # ★域随机化里的"手臂关节"表达式收窄到 piper_joint[1-6]：
        #   基类里写的是 "piper_joint.*"，对这份资产会把夹爪 7/8 也算进去
        #   （随机化夹爪位置/速度没有意义，还会引入无关扰动）。
        self.events.reset_arm_joints.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=["piper_joint[1-6]"]
        )


@configclass
class GripPlaySceneCfg(ReachPlaySceneCfg):
    """回放场景：继承"目标点小球"，同样把机器人换成作者版资产。"""

    robot = GRIP_ROBOT_CFG


@configclass
class Go2PiperReachGripEnvCfg_PLAY(Go2PiperReachGripEnvCfg):
    """回放用环境：少量环境 + 关随机化 + 目标点可视小球（继承 Reach 版的全套设置）。"""

    scene: GripPlaySceneCfg = GripPlaySceneCfg(num_envs=16, env_spacing=2.5, clone_in_fabric=True)

    def __post_init__(self) -> None:
        super().__post_init__()
        # 与 Reach 的 Play 配置保持一致（那里是给 ReachPlaySceneCfg 设的，这里要重设一遍）
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.base_external_force_torque = None
        self.commands.ee_goal.marker_asset_name = "ee_goal_marker"
        self.commands.ee_goal.debug_vis = False
        self.commands.base_velocity.debug_vis = False
        self.viewer.eye = (3.5, 3.5, 2.5)
        self.viewer.lookat = (0.0, 0.0, 0.5)


##
# 实测数据（inspect_grip_asset.py / probe_ee_actual.py）
##
# 资产：21 刚体 / 20 关节（18 被驱动）
#   关节顺序（Isaac Lab 导入序，腿与臂交错）：
#     0 FL_hip 1 FR_hip 2 RL_hip 3 RR_hip 4 piper_joint1 5 FL_thigh 6 FR_thigh
#     7 RL_thigh 8 RR_thigh 9 piper_joint2 10-13 FL/FR/RL/RR_calf 14 piper_joint3
#     15 piper_joint4 16 piper_joint5 17 piper_joint6 18 piper_joint7 19 piper_joint8
#   夹爪：piper_joint7 [+0.000,+0.035] / piper_joint8 [-0.035,+0.000]（平移，均默认 0 = 闭合）
#   刚体：base / FL|FR|RL|RR_hip|thigh|calf / piper_link1..8（无 *_foot，脚并进 *_calf）
#   零动作稳定后机身高度 0.3008 m（基类的 base_height 目标 0.30 依然合适）
#   零动作稳定后 piper_link6 base 系 = (+0.113, -0.013, +0.394)
