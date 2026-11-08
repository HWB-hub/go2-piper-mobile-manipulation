# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""拾取（Pick）任务：站在球前 → 伸臂夹住地面上的小球 → 抬起来。

与 Reach/ReachGrip 的本质区别（三个"从虚到实"的改动）：

    ① **目标物是真实刚体**（`pick_object`：有质量、有碰撞、受重力、有摩擦），
       不再是 Reach 任务里那个 kinematic 的绿色标记球 —— 标记球每步被写位姿，夹爪永远夹不住。
       目标物只在**指令重采样时**瞬移（"球出现在新的地方"），其余时间自由。
    ② **夹爪接进动作空间**：动作 18 → 20 维，夹爪两个平移关节尺度 ±0.02 m
       （行程只有 ±0.035 m；joint8 的开启方向是负的，所以它的 scale 给负值，
        这样"动作 +1 = 张开"对两个手指是一致的）。
    ③ **末端参考点换成 `piper_link7`**（两指闭合时的重合点 = 夹取中心）。
       ⚠ 与 Reach 任务不可直接比误差：那里用 piper_link6（手腕法兰），在它后面 13.6 cm。

目标点坐标约定：`world_frame=True` + `object_rest_z=0.03`（球半径）
    → 球永远落在**世界系地面**上（z = 球心 = 半径），x/y 仍按机身系盒子采样（"出现在身前"）。
      这样机器人走路/蹲下都不会让球跟着漂 —— 为下一步"走过去抓"打基础。

本配置默认是 **S1 阶段（站定抓取）**：`rel_standing_envs = 1.0`，机器人原地不动，
球落在手臂够得着的范围内（x 0.16~0.36 m）。S2（走过去抓）只要把
① 采样盒放大到 x 0.6~2.2 m ② `rel_standing_envs` 降到 0.05 即可。
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

import isaaclab_tasks.manager_based.locomanipulation.go2_piper.mdp as mdp

from .go2_piper_env_cfg import Go2PiperSceneCfg
from .go2_piper_reach_env_cfg import (
    ReachActionsCfg,
    ReachCommandsCfg,
    ReachObservationsCfg,
    ReachRewardsCfg,
)
from .go2_piper_reach_grip_env_cfg import (
    GRIP_ROBOT_CFG,
    GripActionsCfg,
    Go2PiperReachGripEnvCfg,
    Go2PiperReachGripEnvCfg_PLAY,
)

##
# 目标物尺寸（这些数在命令项、奖励项里要一致，集中放这里）
##
# ★形状怎么选出来的（三轮实测）：
#   球 直径 6 cm → 只有"精确放中点"才 40%、偏 1.5 cm 掉到 20%；直径 4 cm 球 0/125（一碰就滚）
#   方块 4.8 cm → 不滚了、能面接触，但横向余量只有 (7−4.8)/2 = 1.1 cm，
#                  策略的精细对位停在 ~3.5 cm（`pick_approach` 0.37，手指总在侧面撞到物体把它推开）
#   ★细高"瓶子" 3×3×8 cm → 横向余量 (7−3)/2 = 2 cm，而且**竖直方向容差极大**
#                  （沿 8 cm 高的柱面任意位置都能夹），这是让抓取真正学得出来的关键
#   ★尺寸的第二条约束（实测，别删）：**手指比"两指中点"还要低约 5 cm**。
#     所以中点的世界高度必须 ≥ 台面顶 + 5 cm，否则手指会戳进台面，
#     台面把机器人和手臂一起顶开 —— 实测机器人被顶退 45 cm，
#     `|两指中点−物体|` 变成 0.49 m，`pick_approach_far` 从 0.82 掉到 0.03，训练直接学不动。
#     12 cm 高 → 中心在台面顶上方 6 cm，中点正好落在 0.20 附近，手指离台面 2.5 cm ✓
OBJECT_SIZE = (0.022, 0.022, 0.070)
"""目标物三边尺寸 [m]（细高柱状，像个小瓶子/细罐）。

★四个约束一起决定这个尺寸（都是实测出来的，别随手改）：
  ⓿ 最新一轮：2.5 → **2.2 cm 宽**。自动进阶课程实测把"自由刚体档"的成功率边界
     压在 alpha≈0.03~0.06（物体离手仅 0.3~1 cm），说明**策略定位误差 ~2 cm 已经吃光了
     抓取容差 ±2.25 cm**。把物体收窄到 2.2 cm → 容差放宽到 ±2.4 cm，给定位误差留余量。
  ① 横向：必须小于夹爪开口（半开 4 cm / 全开 7 cm）。
     ★宽度直接决定抓取容差 = (开口 − 宽度)/2：
       4.0 cm 宽 → 容差只有 ±1.5 cm，而策略实测对位误差 ~3 cm ⇒ 夹不住
       2.5 cm 宽 → 容差 ±2.25 cm，落在策略能力范围内 ✓
  ② 高度下限：**手指比"两指中点"还要低约 5 cm**，中点的世界高度须 ≥ 台面顶 + 5 cm；
     8 cm 高 → 中心在台面顶上方 4 cm（配合 GRASP_Z_OFFSET=4 cm 后指尖离台面 3 cm）✓
  ③ 稳定性：12 cm 高、4:1 细长比实测会被手指碰倒；8 cm 高、3.2:1 更稳
  ④ 竖直容差要大（柱面任意高度都能夹）—— 这是柱状优于球/方块的核心原因
"""

OBJECT_HALF = OBJECT_SIZE[0] / 2
"""横向半边长 [m]（用于判定间距/容差）。"""

BALL_MASS = 0.15
"""目标物质量 [kg]。★不能太轻：手指闭合时会撞到它，轻物容易被"弹出去"；
   细高物体还要够重才不会被碰倒（150 g 比 80 g 稳）。"""

TABLE_HEIGHT = 0.14
"""低位台面高度 [m]。★这是让"抓取"可学的关键改动之一。

为什么需要台面（实测教训）：
    直接在**地面上**抓（前两轮训练）：手臂必须折到 base 系 z≈−0.30 的极端位形，
    那里关节耦合最强、手指离地只有 2.4 cm，精细对位极难学。
    实测这两轮都卡在"碰到物体、把它推来推去，但夹不住"的局部最优
    （`pick_approach` 0.18~0.34、`pick_lift` 0.19~0.29，但 `pick_hold` ≈0.000、`pick_success` =0）。
    把物体抬到 14 cm 台面上之后：
      · 手臂不用折到底，工作点回到关节空间中较"舒展"的区域，精细对位容易得多
      · 手指两侧有充足空间开合，不再贴着地面
    台面在机器人正前方（x≈0.33），把目标物的采样盒整个盖住。
"""

TABLE_SIZE = (0.32, 0.40, TABLE_HEIGHT)
"""台面尺寸 [m]。★近边必须在**前腿之外**：Go2 前脚在 base 系 x≈+0.19 m、摆动 ±0.1 m，
   台面若从 x=0.11 开始会把前腿埋进台面里（仿真直接崩）。所以台面从 x=0.29 起。"""

TABLE_POS = (0.45, 0.0, TABLE_HEIGHT / 2)
"""台面中心位置（env 系）。x 跨度 0.29~0.61，y 跨度 ±0.20。"""

OBJECT_REST_Z = TABLE_HEIGHT + OBJECT_SIZE[2] / 2
"""目标物静止时的中心高度 [m]（台面顶 + 物高的一半）。

★注意：命令项 `object_rest_z`、奖励里的 `rest_height`、评估脚本必须**全部**用这个值，
  否则"抬起"的基准错了，success 判据就废了。
"""

PICK_OBJECT_INIT_POS = (0.33, 0.0, OBJECT_REST_Z)
"""目标物初始位置（台面上；之后由命令项每次重采样时摆位）。"""

GRASP_Z_OFFSET = 0.04
"""夹爪瞄准点相对物体中心的高度偏移 [m]（见 EeGoalCommandCfg.object_grasp_z_offset 的说明）。

手指尖比两指中点低约 5 cm；瞄准物体中心时指尖会蹭到台面顶（假夹持）。
抬到中心上方 4 cm → 指尖离台面 4 cm，夹的是物体上半段（物体 10 cm 高，足够）。
命令项、奖励参考点必须**用同一个值**，否则奖励会把"正确瞄准"判成误差。
"""

GRIPPER_HALF_OPEN = 0.02
"""夹爪默认开度 [m]（每个手指偏离夹取中心 2 cm，合计 4 cm 开口）。"""

##
# 拾取专用机器人配置：几何/增益沿用 GRIP_ROBOT_CFG，只把**夹爪默认位改成半开**
##
# ★为什么必须改：这份资产的关节定义是
#     piper_joint7  origin (0,0,0.1358) rpy(90°,0,0)   axis (0,0, 1)  限位 [0,   +0.035]
#     piper_joint8  origin (0,0,0.1358) rpy(90°,0,180°) axis (0,0,-1)  限位 [-0.035, 0]
#   两个手指的关节原点重合，轴向相反 —— 所以 **joint7=joint8=0 表示"完全闭合"**，
#   两边各走 ±0.035 才是张开（最大开口 7 cm）。
#   而 `use_default_offset=True` 下"中性动作"就是默认位：默认位=0 会变成"一上来就握拳"，
#   策略只能从闭合单向张开。把默认位放在半开（±0.02）后，动作 ±1 分别对应
#   "再张开"和"闭上"，双向可控，接触密集型任务学起来容易得多。
from isaaclab.assets import ArticulationCfg as _ArticulationCfg

PICK_ROBOT_CFG = GRIP_ROBOT_CFG.replace(
    init_state=_ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.34),  # 与作者资产一致（实测零动作稳定在 0.31）
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
            # ★初始臂姿改成"半折下探"，不是作者的直立姿态：
            #   训练实测 —— 从直立姿态（joint2=0.6，末端在世界 0.71 m）要学到
            #   "折到地面"（末端 0.024 m）是个 2.4 rad 的大动作，纯靠随机探索太慢：
            #   前 400 轮 `pick_approach_far` 只从 0.013 涨到 0.039，`pick_approach` 恒为 0。
            #   把起点放在半折姿态（joint2=2.2，末端更靠近地面）后，
            #   策略只需要学"再往下一点 + 对准 + 合爪"，探索难度大幅下降。
            #   （这也是抓取类任务常见的做法：初始状态就摆在"准备抓"的姿态上）
            # ★臂：默认位形 = "台面上方的准备姿态"（probe_ready_pose.py 实测选出）
            #   (j2,j3,j5)=(2.60,-1.70,+0.80) → 两指中点落在 base 系 (+0.414,+0.004,-0.096)，
            #   世界高度 ≈ 0.215（物体中心 0.20 附近，手指下探到位时离台面还有 ~2.5 cm）。
            #   |开合轴z|=0.003（水平 ✓）、接近轴z=-0.996（竖直下探 ✓）。
            #   ★为什么不选那个 |z| 更低的位形 (2.60,-1.40,0.00)：
            #     它把中点放到世界 0.148，比台面顶(0.14)只高 8 mm —— 手指会戳进台面，
            #     机器人和手臂被台面顶退 45 cm（实测）。**手指比中点低约 5 cm** 是硬约束。
            "piper_joint2": 2.60,
            "piper_joint3": -1.70,
            "piper_joint4": 0.0,
            "piper_joint5": 0.80,
            "piper_joint6": 0.0,
            "piper_joint7": GRIPPER_HALF_OPEN,  # ★半开
            "piper_joint8": -GRIPPER_HALF_OPEN,  # ★半开（反向）
        },
        joint_vel={".*": 0.0},
    ),
)


@configclass
class PickSceneCfg(Go2PiperSceneCfg):
    """场景：带夹爪的 Go2+Piper + **一个可抓取的小球**。"""

    robot = PICK_ROBOT_CFG

    pick_object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/PickObject",
        spawn=sim_utils.CuboidCfg(
            size=OBJECT_SIZE,
            # 绿色自发光，录像里一眼能看见（和 Reach 任务的标记球同色系）
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.05, 0.85, 0.15),
                emissive_color=(0.0, 0.25, 0.03),
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,  # ★真实刚体：受重力、能被推动/抓取
                disable_gravity=False,
                max_depenetration_velocity=1.0,
                retain_accelerations=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=BALL_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            # 高摩擦：夹住之后不容易从指间滑掉（这是能不能"抬手不丢"的关键之一）
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.5, dynamic_friction=1.2, restitution=0.0
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=PICK_OBJECT_INIT_POS),
    )

    # ★低位台面：把目标物抬到手更好操作的 14 cm 高度（见 TABLE_HEIGHT 的说明）。
    #   静态碰撞体（不给 rigid_props → 不参与动力学，只当障碍物），固定在机器人正前方。
    pick_table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/PickTable",
        spawn=sim_utils.CuboidCfg(
            size=TABLE_SIZE,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.28, 0.28, 0.34)),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0, dynamic_friction=0.9, restitution=0.0
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=TABLE_POS),
    )


@configclass
class PickActionsCfg(GripActionsCfg):
    """动作：18 个关节 + **2 个夹爪平移关节** = 20 维。

    ★夹爪尺度为什么是 ±0.02 且 joint8 取负：
      两个夹爪关节的限位是 joint7 [0, +0.035]、joint8 [-0.035, 0]（方向相反），
      默认都是 0（闭合）。若沿用默认尺度 0.25，目标位会算到 ±0.25 m —— 直接顶死限位。
      取 0.02 并把 joint8 的尺度写成负值后，"动作 = +1"对两个手指都表示"张开 2 cm"，
      策略面对的两个通道就是同号的，学起来更容易。
    """

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            ".*_hip_joint",
            ".*_thigh_joint",
            ".*_calf_joint",
            "piper_joint[1-6]",
            "piper_joint7",
            "piper_joint8",
        ],
        scale={
            # 腿：精细（同 Reach/ReachGrip）
            ".*_hip_joint": 0.25,
            ".*_thigh_joint": 0.25,
            ".*_calf_joint": 0.25,
            # ★臂：这里只用 ±1.2 左右，**不是** ReachGrip 那套 2.8/2.0 的大尺度。
            #   原因：抓取容差只有 ±2 cm，需要**精细**对位；而动作尺度就是"每单位动作让关节转多少"，
            #   尺度 2.8 意味着探索噪声（init_std=1.0）会把肩关节甩 ±2.8 rad —— 完全淹没精细信号。
            #   实测对比：
            #     尺度 2.8/2.0（为了够地面而放大）→ `pick_approach` 卡在 0.12（中点离物体 7 cm 就不动了）
            #     尺度 1.2/1.2（Reach 任务用的值，那里做到 3 cm 精度）→ 见本轮结果
            #   大范围的需求已经由"初始臂姿就摆在半折工作位形"满足了，不需要靠放大尺度来覆盖。
            "piper_joint1": 1.2,  # 底座回转
            "piper_joint2": 1.2,  # 肩俯仰（默认 2.2 → 覆盖 [1.0, 3.14]）
            "piper_joint3": 1.2,  # 肘（默认 -1.0 → 覆盖 [-2.2, 0]）
            "piper_joint4": 0.5,  # 腕回转
            "piper_joint5": 0.8,  # 腕俯仰
            "piper_joint6": 0.8,  # 腕滚转
            # 夹爪：张开方向统一为正（joint8 的开向是负的，所以尺度取负）
            "piper_joint7": 0.02,
            "piper_joint8": -0.02,
        },
        use_default_offset=True,
    )


@configclass
class PickCommandsCfg(ReachCommandsCfg):
    """指令：目标物摆位点（世界系地面）。"""

    ee_goal = mdp.EeGoalCommandCfg(
        asset_name="robot",
        # ★夹取中心：闭合时 joint7/joint8 的参考点重合处
        body_name="piper_link7",
        resampling_time_range=(6.0, 10.0),
        # 目标物是"瞬移"过去的，所以命令也不做平滑插值（1 步到位），
        # 否则观测里的指令点会慢慢地飘，而球早就在新位置了，两者不一致。
        traj_time=(0.02, 0.02),
        world_frame=True,  # ★球固定在世界里（走路/蹲下都不会带着球跑）
        sample_in_env_frame=True,  # ★采样盒解释为环境坐标系 → 物体永远落在台面上
        debug_vis=False,
        marker_asset_name=None,  # 不需要标记球：目标物本身就是可见的实体
        object_asset_name="pick_object",  # ★重采样时把方块摆到新位置
        object_rest_z=OBJECT_REST_Z,  # ★方块中心固定在这个世界高度（= 台面顶 + 半边长）
        object_success_lift=0.05,  # 抬起 5 cm 才算"抓起来了"（写进 Metrics）
        object_grasp_z_offset=GRASP_Z_OFFSET,  # ★瞄准物体上半段（指尖不蹭台面）
        # ★★课程学习（针对"抬起学不出来"这个 13 轮的瓶颈）：
        #   alpha=0 → 物体直接摆到两指之间，策略只需学"合爪 + 抬起"这两步，
        #   先把"合爪→抬起→成功"这条因果链体验出来；等它稳了再把物体挪远。
        #   阶段二把 curriculum_alpha 调到 1.0（或设 curriculum_ramp_steps 让它自动升）。
        # ★阶段二（课程退火）：alpha 从 0 线性升到 1，跨 60000 环境步（≈2500 轮）。
        #   含义：物体从"握在手里"逐步挪到"台面上的采样点" ——
        #   抬起技能已经学会（阶段一 140 轮、成功率 100%），现在把"对准"这一环慢慢加回去。
        #   为什么要这么长的退火：挪远一点，策略就在新位置上重新练一次"对准+合爪+抬起"，
        #   步子太大（直接 alpha=1）会把它阶段一学的东西冲掉。
        # ★阶梯式课程（比线性退火稳）：每一档把 alpha 固定住跑一段，策略适应后再加深。
        #   实测线性退火 60000 步太快：alpha 到 21% 时成功率就从 0.46 掉到 0.17，
        #   策略一直在追难度、没有一站是练稳的。
        #   档位：0.0（握在手里）→ 0.3 → 0.6 → 1.0（自由放在台面采样点）
        # ★★改成【自动进阶课程】：从"物体握在手里"(alpha=0) 开始，
        #   成功率 EMA > 0.6 才升一档、< 0.3 就退回来（见 EeGoalCommandCfg.curriculum_auto）。
        #   为什么必须这样：手工把 alpha 从 0 提到 0.4 时，策略连第 1 档学会的"抬起"
        #   都丢了（录 alpha=0.15 的像时 0/1500 全败）—— 在被丢到做不到的难度上
        #   持续训练，会把失败动作的概率压掉。
        # ★★固定在【自由刚体档】长训（今晚结论的直接应用）：
        #   自动进阶课程证明——黏连档成功率 93%、自由档立刻掉到 8%，
        #   说明"夹住自由刚体"这件事策略从未学会。那就把 alpha 钉在自由档里
        #   （0.06 > held_below=0.05，物体不再跟手、真的会被手指推走），
        #   让策略在这个"物体真的会动"的环境里反复练，而不是一升档就退回去。
        curriculum_auto=False,
        curriculum_alpha=0.06,  # 自由刚体档：物体落在手前 0.6~2.4 cm 的台面上
        # ★步长放到 0.03：实测 0.1 时课程会在 0.1↔0.2 之间反复振荡
        #   （0.1 档成功率 0.93、0.2 档只有 0.09），太粗的步长让策略没有
        #   "刚好够难"的档位可练。细步长能让它停在边界附近持续学习。
        curriculum_step=0.03,
        # ★★held_below 从默认 0.2 降到 0.05（今晚最关键的发现之一）：
        #   实测自动进阶课程的振荡边界**恰好压在 0.2**——因为 0.2 就是我原来设的
        #   "物体还黏在手上"的阈值。alpha≥0.2 时物体变成台面上真正的自由刚体，
        #   成功率立刻从 93% 掉到 8%（哪怕物体只离手 1~3 cm）。
        #   ⇒ 策略之前学到的只是"物体在手上时合爪+抬起"，**抓自由刚体这件事还没学会**。
        #   把黏连阈值压到 0.05，让"自由刚体"从"基本贴手"的位置就开始练，
        #   再由自动课程慢慢推远 —— 这样才真的在练"抓"。
        curriculum_ramp_steps=0,
        ranges=mdp.EeGoalCommandCfg.Ranges(
            # ★采样盒前移，离开手臂的"静止落点"（实测末端自然停在 base 系 x≈0.37）：
            #   物体若摆在那里，手指一落地就把它撞倒/推走（实测物体被推 8~15 cm 并翻倒）。
            #   放在 0.44~0.54 → 需要策略主动前伸 7~17 cm 才够得到，既避免初始碰撞，
            #   又是"真在够"而不是"站在原地就有"。
            pos_x=(0.44, 0.54),
            pos_y=(-0.06, 0.06),
            pos_z=(0.0, 0.0),  # 被 object_rest_z 覆盖，这里只占位
        ),
    )


@configclass
class PickObservationsCfg(ReachObservationsCfg):
    """观测：在 Reach 版基础上**加上目标物的真实位置**。

    ★为什么必须加：Reach 版观测里的 `ee_goal_b` 是指令点（球被摆在哪），
      球被手臂碰到/滚走之后就不准了；`object_pos_b` 给的是球的**实际位置**，
      策略才有"看着球去够"的依据（也是下一步"走过去抓"的必要观测）。
    """

    @configclass
    class PolicyCfg(ReachObservationsCfg.PolicyCfg):
        object_pos_b = ObsTerm(
            func=mdp.object_pos_in_base_frame,
            params={
                "object_cfg": SceneEntityCfg("pick_object"),
                "robot_cfg": SceneEntityCfg("robot"),
            },
        )

        def __post_init__(self):
            super().__post_init__()

    policy: PolicyCfg = PolicyCfg()


@configclass
class PickRewardsCfg(ReachRewardsCfg):
    """奖励：把"够物"换成"抓取+抬起"的塑形。

    取舍说明：
      - 保留 `ee_goal_tracking`（权重降到 1.0）：指令点就是球的摆位点，仍是个有用的先验，
        但球被撞跑之后指令点不再等于球的位置，所以主导信号必须是**以球为准**的下面四项。
      - 三项塑形的分工：靠近（拿到球上方） → 闭合时机（到位才闭，远处别握拳） → 抬起（真抓起来）。
    """

    ee_goal_tracking = RewTerm(
        func=mdp.ee_goal_tracking_exp,
        weight=1.0,
        params={
            "command_name": "ee_goal",
            "std": 0.15,
            "asset_cfg": SceneEntityCfg("robot", body_names="piper_link7"),
        },
    )

    # ① 靠近：夹取中心（两指中点）→ 物体中心
    #    ★两级核（粗 + 细），这是"远处没梯度"问题的标准解法：
    #      只用细核（std 0.035）时，手臂离物体 30 cm → exp(-0.3/0.035) ≈ 0，
    #      奖励恒为 0、学不动（实测前 90 轮 `pick_approach` 一直是 0.0000）。
    #      粗核 (std 0.15) 在远处仍有梯度，负责"把手臂引过去"；
    #      细核 (std 0.035) 在近处拉开区分度，负责"压到最后两厘米"。
    pick_approach_far = RewTerm(
        func=mdp.object_distance_exp,
        weight=3.0,  # ★权重提高：拾取是任务本身，走路/站高只是约束
        params={
            "std": 0.15,
            "z_offset": GRASP_Z_OFFSET,
            "object_cfg": SceneEntityCfg("pick_object"),
            "asset_cfg": SceneEntityCfg("robot", body_names="piper_link7"),
        },
    )

    pick_approach = RewTerm(
        func=mdp.object_distance_exp,
        weight=8.0,  # ★再加权：抓取容差只有 ±2.25 cm，而策略停在 3.4 cm —— 就差最后 1 cm
        params={
            # ★核再收紧到 0.012（0.035→0.02→0.012）：实测策略对位稳定在 3.4 cm 就不动了，
            #   说明这个距离上的梯度还不足以把它推过抓取容差（±2.25 cm）。
            "std": 0.012,
            "z_offset": GRASP_Z_OFFSET,  # 抓取容差只有 ±2 cm，近处必须够陡
            "object_cfg": SceneEntityCfg("pick_object"),
            "asset_cfg": SceneEntityCfg("robot", body_names="piper_link7"),
        },
    )

    # ② 闭合时机：近处闭给分、远处闭扣分
    #    ★权重从 1.5 提到 4.0：这是"合爪"这个关键动作的**唯一**信号，
    #      实测给 1.5 时前 900 轮 `pick_grip_close` 一直在 −0.03~+0.1 徘徊，
    #      策略学会了"把物体推来推去"却从不真正合爪（`pick_hold` ≈ 0.000）。
    pick_grip_close = RewTerm(
        func=mdp.gripper_close_shaped,
        weight=4.0,
        params={
            "near_dist": 0.05,  # d<5 cm 视为"在位"
            "z_offset": GRASP_Z_OFFSET,
            "far_dist": 0.10,  # d>10 cm 视为"不在位"
            "very_far": 0.20,  # 只有 d>20 cm 才罚（不对称设计，见函数 docstring）
            "far_penalty": 0.3,
            "stroke": 0.035,
            "object_cfg": SceneEntityCfg("pick_object"),
            "asset_cfg": SceneEntityCfg("robot", body_names="piper_link7"),
        },
    )

    # ②b 真实夹持状态：手指被物体挡住（= 指间真夹住了）———— 打破"在旁边合爪"的局部最优
    #   ★权重只能给"塑形"量级（1.5），不能给大：给到 5.0 时实测出了**新的局部最优**——
    #     策略光"夹着不动"就拿到 2.17 的大分，而"抬起来"有风险还要额外探索，
    #     于是 `pick_grip_state` 一路涨而 `pick_lift` 原地不动（第 394 轮实测）。
    #     夹持只是中间状态，真正要赚的分必须来自抬起（pick_lift）与保持（pick_hold）。
    pick_grip_state = RewTerm(
        func=mdp.gripper_grip_reward,
        weight=1.5,
        params={
            "near_dist": 0.05,
            "far_dist": 0.10,
            "stroke": 0.035,
            "z_offset": GRASP_Z_OFFSET,
            "object_cfg": SceneEntityCfg("pick_object"),
            "asset_cfg": SceneEntityCfg("robot", body_names="piper_link7"),
        },
    )

    # ③b 抬起过程引导：握住时物体向上运动就给分（把"抬起"这个动作的方向递给策略）
    #    ★为什么必须加：`pick_lift` 是**结果**奖励（高度），属于稀疏信号 ——
    #      策略得先偶然做对"夹住+抬起"才能拿到第一份梯度。前 10 轮实测都卡在这里：
    #      靠近 0.99、合爪 0.7~2.5（学得很好），`pick_lift` 却始终 0.02~0.06。
    #      本项直接奖励"物体在夹取中心附近且往上走"，给出有方向的梯度。
    pick_lift_vel = RewTerm(
        func=mdp.object_lift_vel_reward,
        weight=2.0,
        params={
            "max_dist": 0.08,  # 物体必须还在夹取中心附近（防"把物体打飞"的作弊路线）
            "vmax": 0.15,  # 物体上升 0.15 m/s 就给满
            "z_offset": GRASP_Z_OFFSET,
            "object_cfg": SceneEntityCfg("pick_object"),
            "asset_cfg": SceneEntityCfg("robot", body_names="piper_link7"),
        },
    )

    # ③ 抬起：物体离台面越高分越多（核心信号）
    pick_lift = RewTerm(
        func=mdp.object_lift_reward,
        weight=6.0,  # ★核心信号，权重给到最大
        params={
            "rest_height": OBJECT_REST_Z,  # ★基准 = 台面上静止时的高度
            "target_lift": 0.015,  # ★梯度 3 倍：抬 1 cm 给 0.67 分，先把手抬起来的动作 bootstrap 出来
            "object_cfg": SceneEntityCfg("pick_object"),
        },
    )

    # ④ 保持：抓起来且不丢（与 success 指标同判据）
    pick_hold = RewTerm(
        func=mdp.object_held_bonus,
        weight=4.0,
        params={
            "rest_height": OBJECT_REST_Z,  # ★同上
            "min_lift": 0.05,
            "z_offset": GRASP_Z_OFFSET,
            "object_cfg": SceneEntityCfg("pick_object"),
            # ★必须显式传：函数签名里的默认 SceneEntityCfg 不会被 manager 解析，
            #   body_ids 会停在初始的 slice(None) 上 → 报 "'slice' object is not subscriptable"。
            "asset_cfg": SceneEntityCfg("robot", body_names="piper_link7"),
        },
    )


##
# Environment configuration
##


@configclass
class Go2PiperPickEnvCfg(Go2PiperReachGripEnvCfg):
    """训练用环境（S1：站定抓取）。"""

    scene: PickSceneCfg = PickSceneCfg(num_envs=4096, env_spacing=2.5, clone_in_fabric=True)
    commands: PickCommandsCfg = PickCommandsCfg()
    actions: PickActionsCfg = PickActionsCfg()
    observations: PickObservationsCfg = PickObservationsCfg()
    rewards: PickRewardsCfg = PickRewardsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        # ★S1：让机器人原地站住（速度指令恒为 0），专心学"抓"
        self.commands.base_velocity.rel_standing_envs = 1.0
        # 目标物也在域随机化里给一点质量/摩擦扰动，别让策略只适应一个球
        self.events.physics_material.params["static_friction_range"] = (0.6, 1.4)
        self.events.physics_material.params["dynamic_friction_range"] = (0.5, 1.2)

        # ★手臂初始随机化收窄 ±0.4 → ±0.15 rad，并把夹爪排除在外。
        #   默认位形刚从"半折"改成"台面上方的准备姿态"（base 系 x≈0.45），
        #   ±0.4 rad 的随机化会把末端甩出工作区（实测能甩到机器人头顶附近），
        #   策略每回合先得把手臂"找回来"，探索预算全浪费在回到工作区上。
        #   夹爪随机化更有害 —— 开合状态在回合间乱跳，而"何时闭合"正是本任务的核心难点。
        # ★随机化收到 ±0.05 rad：±0.15 时末端在**竖直方向**会甩 ±4 cm，
        #   而"指尖比两指中点低 5 cm"是硬约束 —— 一旦中点被甩到低于台面顶 +5 cm，
        #   指尖就插进台面（实测中点落到 z=0.172、物体被压到 0.152，物理已经不干净）。
        #   台面任务对"臂姿落点"的容差很窄，所以这里必须收窄。
        self.events.reset_arm_joints.params["position_range"] = (-0.05, 0.05)
        self.events.reset_arm_joints.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=["piper_joint[1-6]"]
        )

        # ★台面拾取阶段先关掉"推动"扰动：
        #   台面固定在环境坐标系，机器人被推走就会"人台分离"（物体虽在台面上，
        #   但末端得跨更远的距离才够得着），对"学会抓"这件事只有坏处没有好处。
        #   ★注意 force_range/torque_range 必须给 (0,0)：这个事件是在 reset 时把力
        #   **写进持续缓冲**的，之后每一步都在施加 —— 实测 ±2 N 持续作用 20 s，
        #   机器人会漂到离台面 0.8 m 之外（`pick_approach_far` 从 0.82 掉到 0.005）。
        #   等抓取稳定后再打开以增强鲁棒性（或改用课程学习逐步加）。
        self.events.push_robot = None
        self.events.base_external_force_torque.params["force_range"] = (0.0, 0.0)
        self.events.base_external_force_torque.params["torque_range"] = (0.0, 0.0)

        # ★★出生位姿必须固定（这是台面任务能否成立的前提）：
        #   基类把 pose 随化成 x,y ±0.5 m、**yaw ±π**，速度 ±0.5。
        #   后果实测：机器人朝向随机（手臂朝的不是台面方向）+ 位置偏离台面 ±0.5 m，
        #   于是"放在台面上的物体"离两指中点 0.84 m —— 训练从第一步就跑偏了。
        #   台面固定在环境坐标系，机器人就必须"站在台前、面朝台面"。
        self.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
        self.events.reset_base.params["velocity_range"] = {
            "x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (0.0, 0.0),
            "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (-0.1, 0.1),
        }
        # 腿部关节缩放也收一下（0.5~1.5 会把站姿拉得很散，末端跟着晃）
        self.events.reset_robot_joints.params["position_range"] = (0.9, 1.1)


@configclass
class PickPlaySceneCfg(PickSceneCfg):
    """回放场景：目标物本身就是可见实体，不需要额外的标记球。"""

    pass


@configclass
class Go2PiperPickEnvCfg_PLAY(Go2PiperPickEnvCfg):
    """回放用环境：少量环境 + 关随机化 + 相机对准。"""

    scene: PickPlaySceneCfg = PickPlaySceneCfg(num_envs=8, env_spacing=2.5, clone_in_fabric=True)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.base_external_force_torque = None
        self.commands.ee_goal.debug_vis = False
        self.commands.base_velocity.debug_vis = False
        # 默认站定（S1）；想录"走过去抓"的版本把下面这行注释掉，
        # 并把 commands.ee_goal.ranges.pos_x 改成 (0.6, 2.2)、rel_standing_envs 改 0.05
        self.commands.base_velocity.rel_standing_envs = 1.0
        self.viewer.eye = (2.6, 2.6, 1.9)
        self.viewer.lookat = (0.0, 0.0, 0.25)
