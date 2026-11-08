# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""注册 Go2 + Piper 的 locomanipulation 任务。

任务名 → 配置类：
    Isaac-Go2Piper-Flat-v0        训练用（512 环境、含域随机化）
    Isaac-Go2Piper-Flat-Play-v0   回放用（16 环境、关随机化、相机对准机器人）

注册机制：本文件所在包会被 isaaclab_tasks 的 import_packages() 自动扫描并 import，
所以【不需要】手动去改上层 __init__.py。（这一点和"把任务加到别的仓库"不同，注意区分。）
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Go2Piper-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_piper_env_cfg:Go2PiperEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2PiperPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Go2Piper-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_piper_env_cfg:Go2PiperEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2PiperPPORunnerCfg",
    },
)

##
# Loco-Manipulation（边走边够物）—— 移植自 visual_wholebody 的低层全身控制器设计
# 详见 go2_piper_reach_env_cfg.py 顶部的"差异一览"
##

gym.register(
    id="Isaac-Go2Piper-Reach-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_piper_reach_env_cfg:Go2PiperReachEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2PiperReachPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Go2Piper-Reach-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_piper_reach_env_cfg:Go2PiperReachEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2PiperReachPPORunnerCfg",
    },
)

##
# Reach · 作者版带夹爪资产（Sivan666666/roboduet_go2piper 的整机）
#   与上面的差别只有资产本身：21 刚体 / 20 DOF（多两个夹爪平移关节），
#   动作仍 18 维（夹爪不参与动作）。详见 go2_piper_reach_grip_env_cfg.py
##

gym.register(
    id="Isaac-Go2Piper-ReachGrip-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_piper_reach_grip_env_cfg:Go2PiperReachGripEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2PiperReachGripPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Go2Piper-ReachGrip-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_piper_reach_grip_env_cfg:Go2PiperReachGripEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2PiperReachGripPPORunnerCfg",
    },
)

##
# Pick · 拾取任务：夹住地面上的真实小球并抬起来（作者版带夹爪资产）
#   与 Reach 的本质差别：目标物是**真实刚体**（可夹住）、夹爪进入动作空间（20 维动作）、
#   末端参考点用 piper_link7（夹取中心）。详见 go2_piper_pick_env_cfg.py 顶部说明。
##

gym.register(
    id="Isaac-Go2Piper-Pick-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_piper_pick_env_cfg:Go2PiperPickEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2PiperPickPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Go2Piper-Pick-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_piper_pick_env_cfg:Go2PiperPickEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2PiperPickPPORunnerCfg",
    },
)

##
# Track · 追踪任务（S2）：目标点在世界系 1~2 m 外，机器狗必须【走过去】再用机械臂够到
#   与 Reach 的本质差别：ee_goal.world_frame=True + 采样盒 0.9~1.9 m ⇒ 机械臂
#   （作业半径 0.35 m）够不到，**腿必须动**；走路由"靠近目标"的势函数差分奖励驱动，
#   而不是速度指令（速度指令仍恒为 0）。另加"够到就结束回合"以消除空站浪费。
#   详见 go2_piper_track_env_cfg.py 顶部说明。
##

gym.register(
    id="Isaac-Go2Piper-Track-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_piper_track_env_cfg:Go2PiperTrackEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2PiperTrackPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Go2Piper-Track-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_piper_track_env_cfg:Go2PiperTrackEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2PiperTrackPPORunnerCfg",
    },
)

##
# Walk · 步态任务（S1）：只学走路（向前走 / 转圈），手臂固定在标称姿态
#   与基座 Flat 任务的关键差别：建在【带夹爪资产】上（观测 79 维 / 动作 18 维，
#   与 ReachGrip / Pick / Track 完全一致）⇒ 第 2、3 步可直接续训。
#   奖励里关掉 ee_goal_tracking（不学够物）、把 arm_pose_tracking 提到 2.0（手臂当固定负载）。
#   详见 go2_piper_walk_env_cfg.py 顶部说明。
##

gym.register(
    id="Isaac-Go2Piper-Walk-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_piper_walk_env_cfg:Go2PiperWalkEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2PiperWalkPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Go2Piper-Walk-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_piper_walk_env_cfg:Go2PiperWalkEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2PiperWalkPPORunnerCfg",
    },
)
