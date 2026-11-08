# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Go2 + Piper —— 【Loco-Manipulation 版】：边走边把末端伸到指令目标点。

## 这是什么

把 visual_wholebody（CoRL 2024, B1+Z1 全身控制）的**低层全身控制器设计**移植到
本机已有的 Isaac Lab Go2+Piper 任务上。原来的任务只要求"手臂保持标称姿态"，
本任务要求"末端跟踪一个不断重采样的目标点，同时按速度指令行走" ——
这才是 locomanipulation 的核心：机体运动与手臂操作互相耦合。

## 与原任务（Isaac-Go2Piper-Flat-v0）的差异一览

| 项 | 原任务 | 本任务（Reach） |
|---|---|---|
| 手臂目标 | 保持标称关节角 | **跟踪 base 系的末端目标点**（每 5~9 s 重采样） |
| 手臂动作 scale | 0.25（与腿相同） | **per-joint：joint1=2.0 / 2,3=1.2 / 4,5=0.6 / 6=1.0** |
| 观测 66 维 | 无末端信息 | **+9 维**：目标点(3) + 当前末端(3) + 误差(3) |
| 域随机化 | 摩擦/质量/质心 | **再加驱动器增益 ±15%**，摩擦范围放宽 |
| 奖励 | arm_pose_tracking（保持姿势） | **ee_goal_tracking（够得准）**，权重 0.8 |

保留不变的部分：速度指令、base 高度、姿态/能耗/平滑类惩罚、终止条件、接触传感器。

## 关于目标点采样范围（有依据，不是拍脑袋）

来自 asset_import/go2_piper/probe_ee.py 的实测（3000+ 次随机关节姿态）：
    可达空间（base 系）  x[-0.69,+0.53]  y[-0.61,+0.62]  z[-0.16,+0.81]
    标称末端位置         (+0.084, 0.000, +0.505)
本任务取 x[-0.05,0.32] y[-0.22,0.22] z[0.22,0.58] —— 即以标称点为中心、
各轴约 ±0.25 m 的区域，"距标称 ≤0.30 m"的姿态占随机采样的 28%，可达性有保证。

## 运行

    isaaclab.bat train --rl_library rsl_rl --task Isaac-Go2Piper-Reach-v0 --headless --num_envs 512
    isaaclab.bat play  --rl_library rsl_rl --task Isaac-Go2Piper-Reach-Play-v0 --num_envs 16
"""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

import isaaclab_tasks.manager_based.locomanipulation.go2_piper.mdp as mdp

from .go2_piper_env_cfg import (
    ActionsCfg,
    CommandsCfg,
    EventsCfg,
    Go2PiperEnvCfg,
    Go2PiperSceneCfg,
    ObservationsCfg,
    RewardsCfg,
    TerminationsCfg,
)

##
# MDP settings：只覆盖与"够物"有关的部分，其余继承原任务
##


@configclass
class ReachCommandsCfg(CommandsCfg):
    """速度指令（继承）+ 末端目标点指令（新增）。"""

    ee_goal = mdp.EeGoalCommandCfg(
        asset_name="robot",
        body_name="piper_link6",
        resampling_time_range=(5.0, 9.0),  # 每 5~9 秒换一个新目标点
        traj_time=(1.5, 3.0),  # 目标点用 1.5~3 s 平滑移过去
        world_frame=False,  # False=目标跟随机身（同原仓库低层）；True=固定在世界里（更难）
        debug_vis=True,  # 回放时能看到绿色小球标记目标
        ranges=mdp.EeGoalCommandCfg.Ranges(
            pos_x=(-0.05, 0.32),
            pos_y=(-0.22, 0.22),
            pos_z=(0.22, 0.58),
        ),
    )


@configclass
class ReachActionsCfg(ActionsCfg):
    """动作：腿部维持 0.25 的精细粒度，**手臂放大**到够得到目标。

    ★为什么必须改：原任务 18 个关节统一 scale=0.25，
      意味着目标角 = 默认角 ± 0.25 rad —— 手臂只能在标称姿态附近微调，
      根本伸不到别处（实测标称末端 (0.084,0,0.505)，目标点可在 ±0.25 m 外）。
      这与原仓库的做法一致：它的 arm action_scale = [2.1, 0.6, 0.6, 0, 0, 0]，
      只给真正决定末端位置的前 3 个关节大尺度。
    """

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale={
            # 腿：保持原值（精细、稳定）
            ".*_hip_joint": 0.25,
            ".*_thigh_joint": 0.25,
            ".*_calf_joint": 0.25,
            # 臂：放大（joint1 是底座回转，决定左右伸展范围，所以给最大）
            "piper_joint1": 2.0,
            "piper_joint2": 1.2,
            "piper_joint3": 1.2,
            "piper_joint4": 0.6,
            "piper_joint5": 0.6,
            "piper_joint6": 1.0,
        },
        use_default_offset=True,
    )


@configclass
class ReachObservationsCfg(ObservationsCfg):
    """策略观测：原 66 维 + 9 维末端信息。

    ★把"目标点"和"当前末端位置/误差"直接喂给策略，而不是让它从关节角反推，
      这是 reach 类任务能不能学起来的关键（策略一眼就知道该往哪伸）。
    """

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        # observation terms (order preserved)
        ee_goal_b = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "ee_goal"},
        )
        ee_pos_b = ObsTerm(
            func=mdp.ee_pos_in_base_frame,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="piper_link6")},
        )
        ee_error_b = ObsTerm(
            func=mdp.ee_pos_tracking_error,
            params={
                "command_name": "ee_goal",
                "asset_cfg": SceneEntityCfg("robot", body_names="piper_link6"),
            },
        )

        def __post_init__(self):
            super().__post_init__()

    policy: PolicyCfg = PolicyCfg()


@configclass
class ReachRewardsCfg(RewardsCfg):
    """奖励：加入末端目标跟踪，去掉"手臂保持标称姿态"（两者互相冲突）。"""

    # -- 主任务⑤（新增，本任务的核心）：末端够到指令目标点 --
    ee_goal_tracking = RewTerm(
        func=mdp.ee_goal_tracking_exp,
        # 权重演进：0.8（第 1、2 轮）-> 2.0（第 3 轮）
        #   原因：把核从 0.5 收到 0.15 后，这一项【奖励量级】被压小了，等于顺手削弱了权重。
        #   第 2 轮 TensorBoard 回合累计值：track_lin_vel 1.116、base_height 0.865、
        #   ee_goal_tracking 仅 0.242 —— 够物项只占约 11% 的信号，
        #   策略最划算的做法就是"走好路 + 站好高，够物随缘"（实测静止后误差 0.210 m、
        #   静止状态下进入 5 cm 的比例 = 0.0%）。提到 2.0 让它和走稳/站高同量级。
        #   ⚠ 复查时必须同时看 Episode_Reward/track_lin_vel_xy_exp 有没有塌掉
        #   —— 权重过大可能让机器人"站着不动去够"。
        weight=2.0,
        params={
            "command_name": "ee_goal",
            # ★误差尺度(m)。原仓库等价于 2.0（核很平：误差 0.2 m 仍得 0.90，梯度几乎为零）。
            #   第一轮用 0.5 时停在 0.19~0.28 m 就不再改善 ——
            #   因为误差 0.27 m 还能拿到 exp(-0.27/0.5)=0.58，没有梯度去啃最后几厘米。
            #   收到 0.15：误差 0.2/0.1/0.05 m 分别得 0.26/0.51/0.72，梯度才够。
            "std": 0.15,
            "asset_cfg": SceneEntityCfg("robot", body_names="piper_link6"),
        },
    )

    # ★关掉"保持标称姿态"：目标点离标称姿态有距离，两者会互相抵消。
    #   （原任务里它的权重已从 0.75 降到 0.2，就是因为会和运动任务打架）
    arm_pose_tracking = None


@configclass
class ReachEventsCfg(EventsCfg):
    """域随机化：在原任务基础上补上驱动器增益，摩擦范围放宽（对齐原仓库思路）。"""

    # 摩擦：原仓库 [0.3, 3.0]（很激进）；这里取中间值，保证 8GB 显卡上还能训得动
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.5, 1.2),
            "dynamic_friction_range": (0.3, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    # 驱动器增益随机化（原任务没有）：电机实际输出和仿真有偏差，这是 sim2real 的关键一项
    actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "stiffness_distribution_params": (0.85, 1.15),
            "damping_distribution_params": (0.9, 1.1),
            "operation": "scale",
            "distribution": "uniform",
        },
    )


@configclass
class ReachTerminationsCfg(TerminationsCfg):
    """终止条件：原任务的"超时 + 躯干接触"，**再补上 visual_wholebody 的"姿态 + 高度"判定**。

    ★为什么必须补：原任务是靠"躯干刚体接触力 > 1 N"判倒地，但机器人侧翻/前扑时
      往往是压在腿或机械臂上、躯干没碰地 → 永远不终止，只能等 20 s 回合超时，
      期间它一直躺着。这既让录像难看，**也让训练数据被污染**（倒地后仍在采样，
      而对倒地状态来说"够物奖励"几乎恒为 0，等于纯灌噪声/惩罚）。

    原仓库 legged_gym/envs/manip_loco/manip_loco.py::check_termination 的做法是：
        r_term = |roll|  > 0.8      （横滚超 0.8 rad ≈ 46°）
        p_term = |pitch| > 0.8      （俯仰超 0.8 rad）
        z_term = base_z  < 0.1      （机身掉到 0.1 m 以下）
        reset_buf = 接触 | 超时 | r_term | p_term | z_term
      且它的 terminate_after_contacts_on = []（根本不靠接触判）。

    本任务用 Isaac Lab 现成的两项等效实现：
        root_height_below_minimum  ↔ z_term
        bad_orientation            ↔ r_term | p_term
      （注意差异：bad_orientation 判的是【总倾角】roll/pitch 合成后的角度，
        原设计是分轴判；对"倒地"这种大角度场景两者等价，但会更早一点点触发。）
    """

    # 原设计 z_term：机身高度 < 0.1 m → 倒地，终止
    base_height_low = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.10, "asset_cfg": SceneEntityCfg("robot")},
    )

    # 原设计 r_term | p_term：机身倾斜超过 0.8 rad → 翻了，终止
    base_attitude = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": 0.8, "asset_cfg": SceneEntityCfg("robot")},
    )


##
# Environment configuration
##


@configclass
class Go2PiperReachEnvCfg(Go2PiperEnvCfg):
    """训练用环境（Loco-Manipulation Reach）。"""

    commands: ReachCommandsCfg = ReachCommandsCfg()
    actions: ReachActionsCfg = ReachActionsCfg()
    observations: ReachObservationsCfg = ReachObservationsCfg()
    rewards: ReachRewardsCfg = ReachRewardsCfg()
    events: ReachEventsCfg = ReachEventsCfg()
    terminations: ReachTerminationsCfg = ReachTerminationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        # 观测维度变大（66 -> 75），但网络不变，显存/速度影响可忽略
        self.episode_length_s = 20.0


@configclass
class ReachPlaySceneCfg(Go2PiperSceneCfg):
    """回放专用场景：多一个"目标点小球"。

    它是一个真实场景物体（kinematic 刚体、无碰撞、无重力），每步被写到当前目标点位置上，
    所以在录像里一定看得见（而 debug_vis 的 VisualizationMarkers 在 headless 录像里不渲染）。
    """

    ee_goal_marker = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/EeGoalMarker",
        spawn=sim_utils.SphereCfg(
            radius=0.045,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.05, 0.85, 0.15),
                emissive_color=(0.0, 0.35, 0.05),  # 自发光，保证在暗背景下也看得清
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.6)),
    )


@configclass
class Go2PiperReachEnvCfg_PLAY(Go2PiperReachEnvCfg):
    """回放用环境：少量环境 + 关随机化 + 打开目标点可视小球。"""

    scene: ReachPlaySceneCfg = ReachPlaySceneCfg(num_envs=16, env_spacing=2.5, clone_in_fabric=True)

    def __post_init__(self) -> None:
        super().__post_init__()

        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.base_external_force_torque = None
        # ★把目标点接到真实小球上（录像/肉眼检查必需）
        self.commands.ee_goal.marker_asset_name = "ee_goal_marker"
        # ★关掉所有 debug_vis 标记：它们在 headless 录像里不会跟着指令走，
        #   而是以"原型 prim"的形式堆在原点，画面里表现为机身下方一个蓝圆台+绿半圆柱，
        #   非常容易被误认成目标物。既然已有真实小球，就全部关掉。
        self.commands.ee_goal.debug_vis = False
        self.commands.base_velocity.debug_vis = False

        self.viewer.eye = (3.5, 3.5, 2.5)
        self.viewer.lookat = (0.0, 0.0, 0.5)
