# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RoboDuet (Go2 + Piper) 环境的 Isaac Lab 配置。

数值全部来自作者仓库 Sivan666666/roboduet_go2piper（spec 见
<historical-local-path>/01..04）：
    sim.dt=0.005, decimation=4 → 50 Hz 策略 / 200 Hz 物理, episode 20 s（1000 步）
    动作 18 维 = 12 腿 + 6 臂，位置目标 = 默认角 + 0.25×动作（髋再 ×0.5），夹爪 7/8 不参与动作
    观测：dog 56 / arm 20（各自 30 帧历史）；critic 2 / 9
    指令：速度(±0.6/±0.6/±1，课程放宽) + 机身材姿(±0.4，阶段 2 由臂策略给) + 臂 6D（2~3 s 重采样）
    奖励：作者 33 项表（权重见 RewardsCfg，两阶段权重见 STAGE_* 表）
    终止：20 s 超时、机身高度 < 0.28 m、阶段 2 的"机身姿态与臂目标矛盾"
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

import isaaclab_tasks.manager_based.locomanipulation.roboduet_go2piper.mdp as mdp

from .robots import ROBODUET_GO2PIPER_CFG

LEG_JOINTS = [".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"]
ARM_JOINTS = ["piper_joint[1-6]"]
FEET_BODIES = ".*_calf"  # 足端刚体在合并固定关节后并进了 calf
PENALISED_BODIES = ["base", ".*_thigh", ".*_calf", "piper_link.*"]


##
# Scene
##
@configclass
class RoboDuetSceneCfg(InteractiveSceneCfg):
    """平地 + 机器人 + 接触传感器（作者 terrain.mesh_type = "plane"）。"""

    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)))
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight", spawn=sim_utils.DomeLightCfg(color=(0.75, 0.8, 1.0), intensity=1000.0)
    )
    robot = ROBODUET_GO2PIPER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=False
    )


##
# Actions
##
@configclass
class ActionsCfg:
    """18 维动作：腿 12（髋 scale 减半）+ 臂 6，位置目标 + 电机零位偏置。"""

    legs = mdp.JointPositionActionWithOffsetCfg(
        asset_name="robot",
        joint_names=LEG_JOINTS,
        scale={".*_hip_joint": 0.125, ".*_thigh_joint": 0.25, ".*_calf_joint": 0.25},  # 0.25×hip_scale_reduction(0.5)
        use_default_offset=True,
    )
    arm = mdp.JointPositionActionWithOffsetCfg(
        asset_name="robot", joint_names=ARM_JOINTS, scale=0.25, use_default_offset=True
    )


##
# Commands
##
@configclass
class CommandsCfg:
    velocity: mdp.VelocityCommandCfg = mdp.VelocityCommandCfg()
    body_pose: mdp.BodyPoseCommandCfg = mdp.BodyPoseCommandCfg()
    arm_pose: mdp.ArmPoseCommandCfg = mdp.ArmPoseCommandCfg()


##
# Observations
##
@configclass
class ObservationsCfg:
    @configclass
    class DogPolicyCfg(ObsGroup):
        """犬策略 56 维（顺序 = 作者 get_dog_observations）。"""

        projected_gravity = ObsTerm(func=mdp.projected_gravity)  # 3
        leg_dof_pos = ObsTerm(func=mdp.leg_dof_pos_rel)  # 12
        leg_dof_vel = ObsTerm(func=mdp.leg_dof_vel)  # 12
        leg_last_action = ObsTerm(func=mdp.last_action_split, params={"start": 0, "end": 12})  # 12
        commands = ObsTerm(func=mdp.dog_commands)  # 5
        arm_commands = ObsTerm(func=mdp.arm_commands_obs)  # 6（阶段 1 为 0）
        roll = ObsTerm(func=mdp.roll_obs)  # 1
        pitch = ObsTerm(func=mdp.pitch_obs)  # 1
        clock = ObsTerm(func=mdp.clock_inputs)  # 4

        def __post_init__(self):
            self.enable_corruption = False  # 作者把加噪代码全部注释掉了 → 无观测噪声
            self.concatenate_terms = True
            self.history_length = 30  # dog_num_observation_history
            self.flatten_history_dim = True

    @configclass
    class ArmPolicyCfg(ObsGroup):
        """臂策略 20 维（顺序 = 作者 get_arm_observations；注意没有关节速度）。"""

        arm_dof_pos = ObsTerm(func=mdp.arm_dof_pos_rel)  # 6
        arm_last_action = ObsTerm(func=mdp.last_action_split, params={"start": 12, "end": 18})  # 6
        arm_commands = ObsTerm(func=mdp.arm_commands_obs)  # 6
        roll = ObsTerm(func=mdp.roll_obs)  # 1
        pitch = ObsTerm(func=mdp.pitch_obs)  # 1

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 30
            self.flatten_history_dim = True

    @configclass
    class DogCriticCfg(ObsGroup):
        """犬 critic 的 privileged 观测：摩擦、弹性（2 维）。"""

        friction_restitution = ObsTerm(func=mdp.friction_restitution)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class ArmCriticCfg(ObsGroup):
        """臂 critic 的 privileged 观测：摩擦、弹性、当前 lpy(3)、末端姿态(4) = 9 维。"""

        arm_privileged = ObsTerm(func=mdp.arm_privileged)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    dog_policy: DogPolicyCfg = DogPolicyCfg()
    arm_policy: ArmPolicyCfg = ArmPolicyCfg()
    dog_critic: DogCriticCfg = DogCriticCfg()
    arm_critic: ArmCriticCfg = ArmCriticCfg()


##
# Rewards（权重 = 作者的 cfg scale；Isaac Lab 会再乘 dt=0.02，与作者一致）
##
@configclass
class RewardsCfg:
    # ---- 机械臂 ----
    arm_manip_commands_tracking_combine = RewTerm(func=mdp.arm_manip_commands_tracking_combine, weight=1.0)
    vis_manip_commands_tracking_lpy = RewTerm(func=mdp.vis_manip_commands_tracking_lpy, weight=1.0)
    vis_manip_commands_tracking_rpy = RewTerm(func=mdp.vis_manip_commands_tracking_rpy, weight=1.0)
    arm_control_smoothness_1 = RewTerm(func=mdp.arm_control_smoothness_1, weight=-0.1)
    arm_control_limits = RewTerm(func=mdp.arm_control_limits, weight=-5.0)
    arm_energy = RewTerm(func=mdp.arm_energy, weight=-4.0e-5)
    arm_dof_vel = RewTerm(func=mdp.arm_dof_vel, weight=-1.0e-3)
    arm_dof_acc = RewTerm(func=mdp.arm_dof_acc, weight=-2.5e-6)
    arm_action_rate = RewTerm(func=mdp.arm_action_rate, weight=-0.1)
    arm_action_smoothness_1 = RewTerm(func=mdp.arm_action_smoothness_1, weight=-0.5)
    arm_action_smoothness_2 = RewTerm(func=mdp.arm_action_smoothness_2, weight=-0.5)

    # ---- 机身姿态 ----
    orientation_control = RewTerm(func=mdp.orientation_control, weight=-10.0)
    orientation_heuristic = RewTerm(func=mdp.orientation_heuristic, weight=-2.0)

    # ---- 主任务 ----
    tracking_lin_vel = RewTerm(func=mdp.tracking_lin_vel, weight=1.0)  # 阶段 2 → 0.7
    tracking_ang_vel = RewTerm(func=mdp.tracking_ang_vel, weight=0.5)  # 阶段 2 → 0.25

    # ---- 腿 / 机身惩罚 ----
    lin_vel_z = RewTerm(func=mdp.lin_vel_z, weight=-0.02)
    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy, weight=-0.001)
    torques = RewTerm(func=mdp.torques, weight=-1.0e-4)
    dof_acc = RewTerm(func=mdp.dof_acc, weight=-2.5e-7)
    dof_vel = RewTerm(func=mdp.dof_vel, weight=-1.0e-4)
    action_rate = RewTerm(func=mdp.action_rate, weight=-0.01)
    hip_action_l2 = RewTerm(func=mdp.hip_action_l2, weight=-0.05)
    action_smoothness_1 = RewTerm(func=mdp.action_smoothness_1, weight=-0.1)
    action_smoothness_2 = RewTerm(func=mdp.action_smoothness_2, weight=-0.1)
    dof_pos_limits = RewTerm(func=mdp.dof_pos_limits, weight=-10.0)
    loco_energy = RewTerm(func=mdp.loco_energy, weight=-4.0e-5)
    collision = RewTerm(func=mdp.collision, weight=-5.0)
    feet_slip = RewTerm(func=mdp.feet_slip, weight=-0.04)

    # ---- 步态塑形（walk-these-ways 的一套）----
    feet_clearance_cmd_linear = RewTerm(func=mdp.feet_clearance_cmd_linear, weight=-30.0)
    tracking_contacts_shaped_force = RewTerm(func=mdp.tracking_contacts_shaped_force, weight=4.0)
    tracking_contacts_shaped_vel = RewTerm(func=mdp.tracking_contacts_shaped_vel, weight=4.0)

    # 作者表里权重为 0 的两项：jump / raibert_heuristic
    jump = RewTerm(func=mdp.jump, weight=0.0)


##
# Terminations
##
@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    body_height = DoneTerm(func=mdp.body_height_too_low)
    arm_inconsistent = DoneTerm(func=mdp.arm_pose_inconsistent)


##
# Events（域随机化）
##
@configclass
class EventsCfg:
    # 摩擦/弹性：只在启动时一次（作者 randomize_rigids_after_start=False）
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.1, 3.0),  # wtw_config friction_range
            "dynamic_friction_range": (0.1, 3.0),
            "restitution_range": (0.0, 0.4),  # wtw_config restitution_range
            "num_buckets": 64,
            "make_consistent": False,
        },
    )
    cache_material = EventTerm(func=mdp.cache_material_properties, mode="startup", params={})

    # 每次 reset 的随机化
    # ★作者的时序（legged_robot.py:1207-1220）：每 200 策略步（rand_interval=4 s）只随机化【电机】
    #   强度/零位；摩擦/弹性/质量【只在启动时】采样一次（_create_envs 里 line 1811 无条件调
    #   _randomize_rigid_body_props，之后被 randomize_rigids_after_start=False 挡住）。
    #   → 所以这里把质量改成 startup，电机改成 interval(4 s)，与他的时序一致。
    base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-2.0, 2.0),  # auto_train added_mass_range
            "operation": "add",
            "distribution": "uniform",
        },
    )
    motor_strength = EventTerm(
        func=mdp.randomize_motor_strength,
        mode="interval",
        interval_range_s=(4.0, 4.0),  # 作者 rand_interval_s = 4 s
        params={"strength_range": (0.9, 1.1), "actuator_names": ("legs",), "asset_cfg": SceneEntityCfg("robot")},
    )
    motor_offset = EventTerm(
        func=mdp.randomize_motor_offset, mode="interval", interval_range_s=(4.0, 4.0),
        params={"offset_range": (-0.02, 0.02)}
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )
    reset_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (0.5, 1.5), "velocity_range": (0.0, 0.0)},
    )


##
# Env
##
@configclass
class RoboDuetEnvCfg(ManagerBasedRLEnvCfg):
    scene: RoboDuetSceneCfg = RoboDuetSceneCfg(num_envs=512, env_spacing=2.5, clone_in_fabric=True)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    curriculum = None

    def __post_init__(self) -> None:
        self.decimation = 4  # 策略 50 Hz
        self.episode_length_s = 20.0  # 1000 策略步
        self.sim.dt = 0.005  # 物理 200 Hz
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.ground.spawn.physics_material
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt


@configclass
class RoboDuetEnvCfg_PLAY(RoboDuetEnvCfg):
    """回放/录像用：少环境 + 关掉随机化。"""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 16
        self.events.base_mass = None
        self.events.motor_strength = None
        self.events.motor_offset = None
        self.viewer.eye = (3.5, 3.5, 2.5)
        self.viewer.lookat = (0.0, 0.0, 0.5)


##
# 两阶段奖励权重（作者 hybrid 字典：阶段 1 里"仅阶段 2"的项权重为 0）
##
HYBRID_ONLY_TERMS = (
    "arm_manip_commands_tracking_combine",
    "vis_manip_commands_tracking_lpy",
    "vis_manip_commands_tracking_rpy",
    "orientation_control",
    "orientation_heuristic",
    "hip_action_l2",
    "arm_control_limits",
    "arm_control_smoothness_1",
    "arm_energy",
    "arm_dof_vel",
    "arm_dof_acc",
    "arm_action_rate",
    "arm_action_smoothness_1",
    "arm_action_smoothness_2",
)
STAGE1_WEIGHTS = {name: 0.0 for name in HYBRID_ONLY_TERMS}
STAGE1_WEIGHTS.update({"tracking_lin_vel": 1.0, "tracking_ang_vel": 0.5})
STAGE2_WEIGHTS = {
    "arm_manip_commands_tracking_combine": 1.0,
    "vis_manip_commands_tracking_lpy": 1.0,
    "vis_manip_commands_tracking_rpy": 1.0,
    "orientation_control": -10.0,
    "orientation_heuristic": -2.0,
    "hip_action_l2": -0.05,
    "arm_control_limits": -5.0,
    "arm_control_smoothness_1": -0.1,
    "arm_energy": -4.0e-5,
    "arm_dof_vel": -1.0e-3,
    "arm_dof_acc": -2.5e-6,
    "arm_action_rate": -0.1,
    "arm_action_smoothness_1": -0.5,
    "arm_action_smoothness_2": -0.5,
    "tracking_lin_vel": 0.7,
    "tracking_ang_vel": 0.25,
}
