# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""机器人资产：Sivan666666/roboduet_go2piper 仓库里作者提供的 Go2+Piper 整机。

★这份资产和我之前建的 go2_piper 任务【不是同一份 URDF】：
    之前的：用 unitree 官方 go2_description.urdf + agilex 官方 piper URDF 自己合并（我选了安装位姿）
    这一份：原样使用作者仓库里的 resources/robots/go2_piper/go2/urdf/go2piper.urdf
            （安装位姿 piper_mount xyz = -0.01 0 0.085 是作者定的，质量/惯量/limits 全部照抄）
    → 只有网格路径被改写成绝对路径（package://piper_description/meshes/x.STL → 本地路径），见
      <historical-local-path>/prepare_repo_asset.py

整机 21 个刚体 / 20 个自由度（12 腿 + 6 臂转动 + 2 夹爪移动），
其中【被驱动的是 18 个】：12 腿 + piper_joint1..6；两个夹爪移动关节（piper_joint7/8）
在作者代码里不参与动作（动作被 pad 成 0 → 保持默认位置），这里同样只做位置保持。

★作者训练时用的增益（逐字抄自他的 asset_config.py 与 auto_train.py）：
    腿（12 个）：kp=40, kd=1        ← asset_config.py: Cnfg.dog.control.stiffness_leg/damping_leg = {'joint': 40.}/{ 'joint': 1.}
    臂（8 个） ：ARP_STIFFNESS_ARM / ARM_DAMPING_ARM
                  joint1 40/3, joint2 80/6, joint3 80/6, joint4 40/3,
                  joint5 80/6, joint6 40/3, joint7 40/3, joint8 40/3
    力矩上限   ：直接取 URDF 里的 effort（腿 23.7，臂 100，夹爪 10）
    action_scale = 0.5，hip_scale_reduction = 1.0（等于不缩放）
    初始机身高度 = 0.34 m（作者 config_go1 里的 init_state.pos，他没有为 Go2+Piper 改过）
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

##
# USD 路径（URDF 转换器输出的是"资产包"目录，主文件是 包目录/名字/名字.usda）
##
ROBODUET_GO2PIPER_USD_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data",
    "go2piper.usd",
    "go2piper",
    "go2piper.usda",
)

LEG_JOINT_NAMES = [".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"]
ARM_JOINT_NAMES = ["piper_joint[1-6]"]  # 6 个转动关节（夹爪 7/8 不参与动作）
GRIPPER_JOINT_NAMES = ["piper_joint7", "piper_joint8"]
EE_BODY_NAME = "piper_link6"  # ★作者的 ee_idx = 23 → 在 Isaac Gym 的刚体顺序里就是 piper_link6

ROBODUET_GO2PIPER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=ROBODUET_GO2PIPER_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,  # 作者设的 self_collisions=0 是"开启"位掩码写法；这里关掉更稳、更快
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=1,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.34),  # ★作者 config_go1 的 init_state.pos
        # ★作者的默认关节角（asset_config.py 的 default_joint_angles，逐字照抄）
        joint_pos={
            "FL_hip_joint": 0.1,
            "RL_hip_joint": 0.1,
            "FR_hip_joint": -0.1,
            "RR_hip_joint": -0.1,
            "FL_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0,
            "FR_thigh_joint": 0.8,
            "RR_thigh_joint": 1.0,
            "FL_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
            "piper_joint1": 0.0,
            "piper_joint2": 0.6,
            "piper_joint3": -0.5,
            "piper_joint4": 0.0,
            "piper_joint5": 0.0,
            "piper_joint6": 0.0,
            "piper_joint7": 0.0,
            "piper_joint8": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=1.0,  # ★作者 soft_dof_pos_limit=0.9 只用在【惩罚项】里，物理限位不软化
    actuators={
        # ---- 12 个腿关节：kp=40 / kd=1（作者 dog.control 的值），力矩上限取 URDF 的 23.7 ----
        "legs": DCMotorCfg(
            joint_names_expr=LEG_JOINT_NAMES,
            effort_limit=23.7,
            saturation_effort=23.7,
            velocity_limit=30.1,
            stiffness=40.0,
            damping=1.0,
            friction=0.0,
        ),
        # ---- 臂的 6 个转动关节：按作者的 kw/kd 分两组 ----
        "arm_hi": ImplicitActuatorCfg(
            joint_names_expr=["piper_joint2", "piper_joint3", "piper_joint5"],
            effort_limit_sim=100.0,
            velocity_limit_sim=5.0,
            stiffness=80.0,
            damping=6.0,
        ),
        "arm_mid": ImplicitActuatorCfg(
            joint_names_expr=["piper_joint1", "piper_joint4", "piper_joint6"],
            effort_limit_sim=100.0,
            velocity_limit_sim=5.0,
            stiffness=40.0,
            damping=3.0,
        ),
        # ---- 两个夹爪移动关节：不参与动作，用隐式 PD 保持在默认位置 ----
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=GRIPPER_JOINT_NAMES,
            effort_limit_sim=10.0,
            velocity_limit_sim=1.0,
            stiffness=40.0,
            damping=3.0,
        ),
    },
)
"""Go2+Piper（作者 URDF）：21 刚体 / 20 DOF，驱动 18 DOF。"""
