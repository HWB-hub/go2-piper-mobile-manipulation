# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""机器人资产配置：Unitree Go2（12 DOF）+ AgileX Piper 机械臂（6 DOF）。

这个 USD 是本地生成的（不是从云端下载的官方资产），生成流程见：
    <historical-local-path>/merge_urdf.py       —— 合并两个 URDF
    isaaclab.bat -p scripts\\tools\\convert_urdf.py --merge-joints --headless <urdf> data/go2_piper.usd
网格来源：
    Go2 : unitreerobotics/unitree_ros  → robots/go2_description（URDF + dae 网格）
    Piper: agilexrobotics/agx_arm_urdf → piper/（URDF + dae/stl 网格）
两套模型的官方链接都写在 00_教程.md 里，方便复现。

★关节命名（18 个可动关节，顺序就是仿真里的关节顺序）：
    腿（12）：FL/FR/RL/RR + _hip_joint / _thigh_joint / _calf_joint
    臂（6） ：piper_joint1 ... piper_joint6   （原 URDF 里叫 joint1..6，已重命名）
注意：转换时用了 --merge-joints，所以 URDF 里那些固定关节链接（FL_foot、FL_hip_rotor、
      imu、front_camera、piper 的 base_link 等）都被合并进父链接了。
      → 因此本资产里【没有 *_foot 这个刚体】，足端接触检测要用 ".*_calf"。
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

##
# 本地 USD 路径（相对于本文件 → 便于整个任务包拷贝/迁移）
##
# 注意：URDF 转换器输出的是 USD "资产包"（一个以 .usd 结尾的目录）：
#   data/go2_piper.usd/                        ← 包目录
#       ├── config.yaml                        ← 转换时的参数（可追溯）
#       ├── .asset_hash
#       └── go2_piper/go2_piper.usda           ← ★真正要加载的主文件
# 所以下面指向的是 "包目录/名字/名字.usda"，而不是包目录本身。
GO2_PIPER_USD_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "go2_piper.usd", "go2_piper", "go2_piper.usda"
)


GO2_PIPER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=GO2_PIPER_USD_PATH,
        activate_contact_sensors=True,  # 打开接触传感器（足端着地检测/air_time 需要）
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
            enabled_self_collisions=False,  # 自碰撞关掉：机械臂+机身贴得近，开了容易抖动且更慢
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=1,
        ),
    ),
    # 初始状态：机身位姿 + 18 个关节的初始角度
    init_state=ArticulationCfg.InitialStateCfg(
        # ★机身初始高度 = 该姿态的【真实站立高度】，由 URDF 正运动学算出（fk_check.py）：
        #   默认腿姿态下脚底在 base 原点下方 0.30 m（前腿）/ 0.291 m（后腿），
        #   所以机身原点离地约 0.30 m。给高了机器人会"砸地"，给低了会被地面顶起来。
        pos=(0.0, 0.0, 0.30),
        joint_pos={
            # ---- Go2 官方推荐站立姿态（和 isaaclab_assets 里的 UNITREE_GO2_CFG 一致）----
            ".*L_hip_joint": 0.1,
            ".*R_hip_joint": -0.1,
            "F[L,R]_thigh_joint": 0.8,  # 前腿大腿
            "R[L,R]_thigh_joint": 1.0,  # 后腿大腿（后腿更弯，重心偏后更稳）
            ".*_calf_joint": -1.5,  # 小腿
            # ---- Piper 机械臂的"搬运"姿态（前折，压低重心）----
            # ★实测结论（diag_stand.py 反复试出来的三种姿态）：
            #   ① 竖直向上 (0.0, -0.5)   → 高倒立摆，零动作就倒
            #   ② 大幅前折 (0.9, -1.2)   → 稳定但前倾 32°（质心被手臂拉到前脚之外，靠后移安装点解决）
            #   ③ 后折   (0.3, -1.6)     → 前臂会垂到机身下方/后腿之间，反而更容易翻
            #   → 采用 ②，并把安装点后移 8 cm（见 merge_urdf.py 的 MOUNT_XYZ）
            "piper_joint1": 0.0,  # 底座回转
            "piper_joint2": 0.9,  # 肩俯仰：前折 0.9 rad（约 52°）
            "piper_joint3": -1.2,  # 肘：向后折回
            "piper_joint4": 0.0,  # 腕回转
            "piper_joint5": 0.7,  # 腕俯仰
            "piper_joint6": 0.0,  # 腕滚转
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,  # 软限位 = 硬限位 × 0.9（留余量，防止顶死）
    actuators={
        # ---- 髋/大腿：DC 电机模型（力矩上限取自 Go2 URDF：23.7 N·m）----
        # ★平滑步态采用比原先 100/4.0 更柔和的 PD。
        #   参考 go2-piper-quiet-goto 使用 20/1；这里考虑 Piper 负载和当前
        #   机身高度，取 45/1.8，避免高刚度 PD 把每次策略更新变成硬碰硬。
        "hip_thigh": DCMotorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint"],
            effort_limit=23.7,
            saturation_effort=23.7,
            velocity_limit=30.1,
            stiffness=45.0,
            damping=1.8,
            # ★armature（转子折算惯量）：直接加到关节的等效惯量上。
            #   作用：抑制高刚度 PD 在 200 Hz 下的高频抖振。
            #   实测不加它时关节加速度 RMS 高达 ~6500 rad/s²，
            #   奖励项 joint_acc_l2 直接爆到 −4000 量级，把训练带崩。
            armature=0.01,
            friction=0.0,
        ),
        # ---- 小腿：力矩上限 45.4 N·m（Go2 小腿电机扭矩更大，URDF 里写得很清楚）----
        # ⚠ 官方示例把三条腿的关节合并成一组、统一用 23.5 N·m —— 带负载时这是不够的，
        #   一定要按 URDF 的真实上限分组，否则蹲下时力矩饱和、扶不起来。
        "calf": DCMotorCfg(
            joint_names_expr=[".*_calf_joint"],
            effort_limit=45.4,
            saturation_effort=45.4,
            velocity_limit=15.7,
            # 小腿略高于参考项目的 40/2，保留机械臂负载下的承重余量。
            stiffness=55.0,
            damping=2.4,
            armature=0.01,
            friction=0.0,
        ),
        # ---- 机械臂：隐式 PD 位置控制 ----
        #★effort_limit_sim 是仿真里的力矩上限。Piper 真机各关节额定力矩约 30-40 N·m 量级，
        #   这里取 40；piper URDF 里写的是 100（模型上限），想更保守可以调小。
        #   velocity_limit_sim=5 rad/s 来自 piper URDF 的 velocity 限制。
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["piper_joint.*"],
            effort_limit_sim=40.0,
            velocity_limit_sim=5.0,
            stiffness=60.0,
            damping=3.0,
            armature=0.005,
        ),
    },
)
"""Go2 + Piper 整机配置（18 个可动关节：12 腿 + 6 臂）。"""
