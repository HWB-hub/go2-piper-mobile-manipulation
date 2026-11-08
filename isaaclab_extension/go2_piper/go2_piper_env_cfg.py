# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Go2 + Piper 机械臂 的 locomanipulation（运动+操作）环境配置。

【任务定义】机器狗按速度指令行走，同时机械臂保持标称（收纳）姿态。
    机械臂在每次 reset 时被随机偏移 ±0.4 rad，于是策略必须学会
    "一边走、一边把手臂收回标称姿态、并保持住" —— 这就是最基础的
    locomanipulation：机身运动与手臂动力学强耦合（手臂的反作用力矩会干扰机身）。

【MDP 一览（维度都是实测值）】
    动作 a ：18 维 = 12 腿关节 + 6 臂关节，位置控制（目标角 = 默认角 + 0.5 × a_net）
    观测 s ：66 维 = 机身线速度3 + 角速度3 + 重力投影3 + 速度指令3 + 关节角18 + 关节速度18 + 上一步动作18
    奖励 r ：见 RewardsCfg（速度跟踪 + 手臂姿态跟踪 + 姿态/能耗/平滑惩罚）
    终止   ：超时(20 s) 或 机身(base)撞地
    初始   ：底座位姿/速度 + 全身关节角 随机；手臂额外随机偏移

【继承体系（重要）】
    ManagerBasedRLEnvCfg（框架）
        └── Go2PiperEnvCfg（本文件，训练用）
                └── Go2PiperEnvCfg_PLAY（回放用：环境更少、关掉随机化，方便肉眼看）

【运行】
    isaaclab.bat train --rl_library rsl_rl --task Isaac-Go2Piper-Flat-v0 --headless --num_envs 512
    isaaclab.bat play  --rl_library rsl_rl --task Isaac-Go2Piper-Flat-Play-v0 --num_envs 16 --viz kit
"""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

# 本任务自己的 mdp（内含 isaaclab.envs.mdp 的全部通用项 + 我们自写的两项）
import isaaclab_tasks.manager_based.locomanipulation.go2_piper.mdp as mdp

# 四足专用的 mdp（这里只借用 feet_air_time 这一个"抬脚"塑形奖励）
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as velocity_mdp
from isaaclab_tasks.manager_based.locomanipulation.go2_piper.robots import GO2_PIPER_CFG

##
# Scene definition
##


@configclass
class Go2PiperSceneCfg(InteractiveSceneCfg):
    """场景：平地 + 天空光 + 机器人本体 + 接触传感器。"""

    # 平地（和 Go2 官方 flat 任务一样，不生成地形网格，最快）
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # 天空光：只是为了好看/录像；训练开销可忽略
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.8, 1.0), intensity=1000.0),
    )

    # 机器人（USD 由 Go2 + Piper 两个 URDF 合并而来，见 robots.py 顶部说明）
    robot = GO2_PIPER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # 接触传感器：覆盖机器人所有刚体
    #   history_length=3  : 保存最近 3 个物理步的接触力（用来算足端接触持续时间）
    #   track_air_time=True: 记录"离地时长"，feet_air_time 奖励需要它
    # ★注意：本资产用 --merge-joints 合并了固定关节，脚部刚体被并进了 *_calf，
    #   所以下面所有涉及"足端"的地方都用 ".*_calf" 而不是 *_foot。
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """指令：机身目标速度 (vx, vy, ωz)。

    heading_command=True 时，策略在转角速度上其实是在跟踪"朝向"，
    相当于给了一个目标方向，比直接跟踪角速度更容易学（Isaac Lab 的标准做法）。
    """

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),  # 每 10 秒换一次指令
        rel_standing_envs=0.02,  # 2% 的环境被要求"站住"（vx=vy=ωz=0）
        rel_heading_envs=1.0,  # 100% 的环境使用朝向指令
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            # 平地起步阶段用较小范围；学会之后再放开（rough 任务里是 ±1.0）
            lin_vel_x=(-0.8, 0.8),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-0.8, 0.8),
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class ActionsCfg:
    """动作：18 个关节的位置控制（增量式）。

    JointPositionActionCfg(use_default_offset=True) 的含义：
        目标关节角 = 默认关节角(init_state 里的姿态) + scale × 网络输出
    网络输出被 clip 在 [-1,1]，所以每次决策关节角最多变化 ±scale rad。

    ★scale 用 0.25（官方 Go2 rough 也是 0.25，我第一版用了 0.5）：
      scale 太大时单步动作过猛、容易"抽搐式"移动，学出来的步态生硬；
      小一点的控制粒度更细，配上 50 Hz 控制频率完全够用。
    """

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True)


@configclass
class ObservationsCfg:
    """观测：66 维（见 ObsTerm 注释）。"""

    @configclass
    class PolicyCfg(ObsGroup):
        """策略网络的输入。"""

        # observation terms (order preserved) —— ★顺序 = 拼接后张量的列顺序
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        # 四足步态相位（sin/cos）；所有 Go2+Piper 子任务共用，保证观测维度一致。
        gait_phase = ObsTerm(func=mdp.gait_phase_observation, params={"period": 0.65})

        def __post_init__(self):
            # 打开观测加噪（模拟真机传感器噪声 → sim2real 必需）。
            # 每个 ObsTerm 自己带 noise 参数，这里的开关是总闸。
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventsCfg:
    """事件：启动时的域随机化 + 每回合重置 + 周期性扰动。

    域随机化（domain randomization）是 sim2real 的核心：训练时把"物理参数"随机化，
    逼策略对参数误差不敏感。机械臂是后装的，它的质量和质心肯定和真机有偏差，
    所以 base 的质量/质心随机化尤其重要。
    """

    # ---- startup：只在仿真启动时执行一次 ----
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.6, 1.0),  # 摩擦系数随机化（地面材质未知）
            "dynamic_friction_range": (0.4, 0.8),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (1 / 1.2, 1.2),  # ±20% 缩放
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            # ★机械臂装在背上 → 质心必然偏移。这里给它 ±5 cm 的随机范围。
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.02, 0.02)},
        },
    )

    # ---- reset：每次环境重置时执行 ----
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (0.0, 0.0),  # 需要施加初始外力时改这里
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.5, 0.5),
                               "roll": (-0.5, 0.5), "pitch": (-0.5, 0.5), "yaw": (-0.5, 0.5)},
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (0.5, 1.5), "velocity_range": (0.0, 0.0)},
    )

    # ★本任务的关键事件：把机械臂随机偏一下，让策略学会"边走边把手收回来"
    #   随机范围 ±0.4 rad（≈23°）；想更难就加大到 ±0.8
    reset_arm_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["piper_joint.*"]),
            "position_range": (-0.4, 0.4),
            "velocity_range": (0.0, 0.0),
        },
    )

    # ---- interval：训练过程中周期性执行（提高鲁棒性）----
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class RewardsCfg:
    """奖励：★你以后调得最多的就是这里。

    机制提醒：每步奖励 = Σ(项值 × weight × dt)，本任务 dt = decimation × sim.dt = 0.02 s。

    ★★ 权重是按【官方 Go2 的成熟配方】对齐的（见 locomotion/velocity/config/go2/rough_env_cfg.py），
       自己凭空想的权重很容易出问题。第一版的血泪教训记录在下面每一项的注释里。
    """

    # -- 主任务①：跟踪机身水平速度（vx, vy）--
    # 官方 Go2 用 1.5（不是我第一版的 1.0）：跟踪项必须足够大，才能压过各种"别动"的惩罚。
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    # -- 主任务②：跟踪转向（官方 0.75）--
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.75, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    # -- 主任务③：机身保持站立高度 --
    # ★用自写的指数核（mdp/base_height_exp），不用内置的 base_height_l2。
    #   原因：L2 平方误差在 8 cm 偏差时几乎为 0，策略会学会"低姿态爬行"这种退化解（实测中招过）。
    #   指数核：站在 0.30 m 得 1.0/步，趴到 0.22 m 只剩 0.0002/步。
    base_height = RewTerm(func=mdp.base_height_exp, weight=1.0, params={"target_height": 0.30, "std": 0.05})
    # -- 主任务④：机械臂保持标称姿态（本任务特有的"操作"目标，自写函数见 mdp/rewards.py）--
    # ★权重从第一版的 0.75 降到 0.2、std 放宽到 0.4：
    #   原来这一项在偷偷"惩罚行走"——手臂在行走时必然被机身晃动带偏，
    #   于是"站着不动"能拿满这 0.75，而走路只能拿 0.5 左右，策略就选择了站着不动。
    #   教训：任何"奖励静止状态"的项，都在和运动任务作对。
    arm_pose_tracking = RewTerm(
        func=mdp.arm_pose_tracking,
        weight=0.2,
        params={"std": 0.4, "asset_cfg": SceneEntityCfg("robot", joint_names=["piper_joint.*"])},
    )

    # -- 惩罚项（权重对齐官方 Go2）--
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)  # 别上下颠
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)  # 别乱扭
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0)  # 机身保持水平（官方 flat 用 -2.5）
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-2.0e-4)  # 省力（官方 -2e-4，比我第一版强 20 倍）
    joint_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)  # 关节别猛加速（加了 armature 后可恢复官方量级）
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)  # 动作平滑（防抖）
    arm_joint_vel_l2 = RewTerm(
        func=mdp.arm_joint_vel_l2, weight=-5.0e-4, params={"asset_cfg": SceneEntityCfg("robot", joint_names=["piper_joint.*"])}
    )  # 手臂别乱甩（自写函数；第一版 -0.002 偏大，又是一项"惩罚运动"）
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)  # 别顶到关节限位
    # ★undesired_contacts 官方 Go2 直接删掉了（对小型四足容易误触发）；
    #   第一版我给 -1.0，它会在正常步态里乱扣分。这里保持关闭。
    undesired_contacts = None

    # -- 步态塑形 --
    # ★官方 Go2 的 feet_air_time 权重是 0.01（很小，只是"临门一脚"的提示），
    #   第一版我给了 0.25（过大）→ 会鼓励"蹦跳"而不是走路。这里取 0.05。
    feet_air_time = RewTerm(
        func=velocity_mdp.feet_air_time,
        weight=0.05,
        params={
            # ★本资产没有 *_foot 刚体（固定关节被合并了），足端用 *_calf
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_calf"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )

    # -- 终止惩罚 --
    terminating = RewTerm(func=mdp.is_terminated, weight=-200.0)  # 摔倒 = 大幅扣分（legged_gym 惯例）


@configclass
class TerminationsCfg:
    """终止条件。"""

    # 超时（正常"下班"，不算失败）——time_out=True 这个标记必须留着
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # 机身撞地 → 摔倒，提前结束
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )


##
# Environment configuration
##


@configclass
class Go2PiperEnvCfg(ManagerBasedRLEnvCfg):
    """训练用环境（平地版）。"""

    # Scene settings
    # ★num_envs=512 是按 8 GB 显存给的保守值；>=16 GB 可以调到 2048~4096，训练快 4~8 倍。
    #   （本资产合并后有 19 个刚体、18 个自由度，比纯 Go2 更吃显存/算力）
    scene: Go2PiperSceneCfg = Go2PiperSceneCfg(num_envs=512, env_spacing=2.5, clone_in_fabric=True)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    # no curriculum: 平地任务不需要课程学习
    curriculum = None

    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 4  # 策略 50 Hz，物理 200 Hz
        self.episode_length_s = 20.0  # 每回合 20 秒 = 1000 个策略步
        # simulation settings
        self.sim.dt = 0.005  # 物理步长 5 ms（四足用 200 Hz 比倒立摆的 120 Hz 更稳）
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.ground.spawn.physics_material
        # 传感器更新周期：接触传感器每个物理步都更新（air_time 计算依赖它）
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt


@configclass
class Go2PiperEnvCfg_PLAY(Go2PiperEnvCfg):
    """回放/评估用环境：少量环境 + 关掉随机化，方便肉眼观察和录视频。"""

    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # 少量环境（可视化时不需要那么多）
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        # 关掉观测噪声
        self.observations.policy.enable_corruption = False
        # 关掉部分随机化（保留初始位姿随机，这样每回合起点不同，方便观察泛化）
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        # 让相机对准机器人（可视化机位）
        self.viewer.eye = (3.5, 3.5, 2.5)
        self.viewer.lookat = (0.0, 0.0, 0.5)
