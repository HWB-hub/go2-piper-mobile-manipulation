# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""S2【追踪任务】：目标点在世界系 1~2 m 外，机器狗必须【走过去】再用机械臂够到。

## 为什么要有这个任务（用户的判断，也是本项目的真实缺口）

之前的 Reach / ReachGrip / Pick 三个任务都有一个共同的隐藏前提：
    `commands.ee_goal.ranges` 的采样盒只有 ±0.32 m，而且 `world_frame=False`
    （目标跟随机身）⇒ **机械臂自己就够得到，腿根本不用动**。
    实测证据：`go2_piper_pick_env_cfg.py:525` 里
        self.commands.base_velocity.rel_standing_envs = 1.0
    意思是"所有环境的速度指令恒为 0"，腿从来没被要求走过路。
    ⇒ 录像里"腿不动、只有胳膊在动"不是巧合，而是**整个任务被跳过了走路这一步**。

本任务把这一步补上：
1. **目标点改到世界系 x ∈ [0.9, 1.9] m**（机械臂作业半径只有 ~0.35 m ⇒ 必须走过去）；
2. **走路由目标驱动**（而不是速度指令）：速度指令仍然全为 0，唯一的"得分来源"是
   *靠近目标站位* ⇒ 想拿分就得动起来（见 rewards.py::base_goal_progress 的注释：
   用势函数差分而不是直接给距离，否则站着不动白拿常数分是局部最优）；
3. **成功后立即结束回合**（治"20 s 里只有 3~5 s 在干活、剩下 15+ s 空站"的浪费）。

## 与其他任务的关系

* `Isaac-Go2Piper-Flat-v0`  —— 原始平地运动（未改动）
* `Isaac-Go2Piper-Reach-v0` —— 站定 + 手臂够到近处目标（未改动）
* `Isaac-Go2Piper-ReachGrip-v0` —— 上者 + 带夹爪资产（未改动）
* `Isaac-Go2Piper-Pick-v0` —— 站定 + 抓手 + 抓取抬起（未改动）
* **`Isaac-Go2Piper-Track-v0`（本文件）—— 走过去 + 追目标（S2，新增）**

S3（在本任务上接抓取）才是最终目标形态。
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

from . import mdp
from .go2_piper_reach_env_cfg import (
    ReachEventsCfg,
    ReachRewardsCfg,
    ReachTerminationsCfg,
)
# ★必须建在 ReachGrip 之上（而不是 Reach）：ReachGrip 用的是【作者版带夹爪资产】，
#   观测里 joint_pos/joint_vel 用 ".*" 匹配，在带夹爪资产上是 20 个关节而不是 18 个
#   ⇒ 观测 75+4 = 79 维。用 Reach（75 维）会导致权重加载失败：
#       RuntimeError: Error(s) in loading state_dict for MLPModel
#   同时这也保证 S3（接抓取）不用再换资产。
from .go2_piper_reach_grip_env_cfg import (
    GripActionsCfg,
    GripCommandsCfg,
    GripSceneCfg,
    Go2PiperReachGripEnvCfg,
)

# ---------------------------------------------------------------------------
# 关键几何：机械臂的作业半径
# ---------------------------------------------------------------------------
# 实测（见 07_拾取任务_实测记录与结论.md §七）：
#   扶正姿态下末端能到 base 系 (+0.312, -0.026, -0.325)
#   ⇒ 水平前伸约 0.31 m、向下 0.33 m
# 所以"理想站位"取 0.35 m —— 站到这个距离上，目标正好落在手臂最舒服的作业区间里。
ARM_REACH = 0.35

# 目标点采样盒（世界系）。机器人出生在 (0,0) ⇒ 目标在正前方 0.9~1.9 m。
GOAL_BOX = {
    "pos_x": (0.90, 1.90),
    "pos_y": (-0.70, 0.70),
    "pos_z": (0.26, 0.34),  # 与机身同高：胳膊抬平就能碰到
}


@configclass
class TrackCommandsCfg(GripCommandsCfg):
    """目标点指令：改成世界系 + 远距离采样盒（这是本任务与 Reach 的核心差别）。"""

    ee_goal = mdp.EeGoalCommandCfg(
        asset_name="robot",
        body_name="piper_link6",
        resampling_time_range=(4.0, 6.0),  # 4~6 s 换一个新目标（比 Reach 快，多练几次"走过去"）
        traj_time=(0.8, 1.2),  # 目标点快速移过去（不需要慢慢挪，反正要走过去）
        # ★核心：True ⇒ 目标固定在世界里。机器人走开后目标不会跟着跑，
        #   所以"走到目标旁边"这件事第一次成为**必须完成**的任务。
        world_frame=True,
        debug_vis=True,  # 回放/录像里能看到绿色目标球
        ranges=mdp.EeGoalCommandCfg.Ranges(**GOAL_BOX),
    )


@configclass
class TrackRewardsCfg(ReachRewardsCfg):
    """奖励：在 Reach 的"够到目标"基础上，加入**驱动走路**的三项。

    ⚠ 权重设计的关键约束（踩过的坑，别随手改回"直接给距离"或"越近越好"）：
      1. 站位奖励必须是 |距离 − ARM_REACH|（不是"越近越好"）——
         机身怼到目标上，末端反而够不着（手臂只有 0.35 m）。
      2. 驱动走路的主项必须是**势函数差分**（base_goal_progress），不能是距离本身——
         距离本身站着不动就有常数分，策略会站住不动（Pick 任务实测过这个局部最优）。
      3. 距离本身只作为"最后一米的定位核"（base_goal_distance_exp，核宽 0.5 m）。
    """

    # -- 主任务①：朝目标"靠近"的进步量（腿动的唯一动力来源）--
    base_goal_progress = RewTerm(
        func=mdp.base_goal_progress,
        weight=8.0,  # 与够物项（2.0）同量级偏大：现阶段"走过去"比"够得准"更欠训练
        params={"command_name": "ee_goal", "target_distance": ARM_REACH},
    )

    # -- 主任务②：站位定位核（离"理想站位"还有多远）--
    base_goal_distance_exp = RewTerm(
        func=mdp.base_goal_distance_exp,
        weight=4.0,
        params={"command_name": "ee_goal", "target_distance": ARM_REACH, "std": 0.5},
    )

    # -- 主任务③：机身朝向目标（走到旁边还得转过去，胳膊才够得到）--
    base_goal_heading_exp = RewTerm(
        func=mdp.base_goal_heading_exp,
        weight=1.5,
        params={"command_name": "ee_goal", "std": 0.6},
    )

    # ee_goal_tracking（继承自 Reach，权重 2.0）保持不变：
    #   核 std=0.15 ⇒ 目标在 1~2 m 外时该项≈0（没有梯度、也没有害处），
    #   等机器人走到目标附近（<0.5 m）它才开始起作用，正好充当"最后几厘米"的精度项。


@configclass
class TrackTerminationsCfg(ReachTerminationsCfg):
    """终止条件：在"超时 + 倒地"之外，加一条**够到就结束**。"""

    goal_reached = DoneTerm(
        func=mdp.goal_reached,
        params={
            "command_name": "ee_goal",
            "threshold": 0.08,  # 8 cm 以内算够到（与 Reach 任务里"达标"的判据一致）
            "asset_cfg": SceneEntityCfg("robot", body_names="piper_link6"),
        },
    )


@configclass
class TrackEventsCfg(ReachEventsCfg):
    """域随机化：在 Reach 的基础上，让机器人**出生朝向随机**。

    ★为什么必要：目标固定在世界系，如果机器人永远朝 +x 出生，它只需要学会
      "往前走"；朝向随机 ±0.6 rad 之后，它必须学会"先转向目标再走/边走边转"，
      这才是真正的"追踪"。
    """

    def __post_init__(self):
        # 出生位置固定在原点附近，但朝向随机
        self.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (-0.6, 0.6)}


##
# Environment configuration
##


@configclass
class Go2PiperTrackEnvCfg(Go2PiperReachGripEnvCfg):
    """S2 追踪任务：走过去 + 够到目标。"""

    commands: TrackCommandsCfg = TrackCommandsCfg()
    rewards: TrackRewardsCfg = TrackRewardsCfg()
    terminations: TrackTerminationsCfg = TrackTerminationsCfg()
    events: TrackEventsCfg = TrackEventsCfg()
    # observations / actions 不覆盖：直接继承 Go2PiperReachGripEnvCfg 的
    # （ReachGrip 的观测 79 维、动作 18 维 —— 与迁移用的权重完全对齐）

    def __post_init__(self):
        super().__post_init__()
        # 目标在世界系 1~2 m 外 ⇒ 回合要够长（走过去 ~2 s + 对准 ~2 s）
        # 注意：成功后 goal_reached 会立刻结束回合，所以这里给 20 s 不会浪费。
        self.episode_length_s = 20.0
        # 速度指令仍然全为 0：本任务里"走路"由目标驱动，不由速度指令驱动。
        # （若以后要做"边按指令走边够物"，把它调小即可，但要先解决两套目标冲突的问题。）
        self.commands.base_velocity.rel_standing_envs = 1.0
        self.commands.ee_goal.debug_vis = True


@configclass
class Go2PiperTrackEnvCfg_PLAY(Go2PiperTrackEnvCfg):
    """回放：少环境、无随机化。"""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.commands.ee_goal.debug_vis = True
        # 固定一个明确的目标（回放时看得清"走过去"这件事）
        self.commands.ee_goal.ranges.pos_x = (1.4, 1.4)
        self.commands.ee_goal.ranges.pos_y = (0.0, 0.0)
        self.commands.ee_goal.ranges.pos_z = (0.30, 0.30)
