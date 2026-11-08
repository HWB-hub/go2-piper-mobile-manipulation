# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL 的 PPO 超参配置（Go2 + Piper 平地 locomanipulation）。

对照参考：
    locomotion/velocity/config/go2/agents/rsl_rl_ppo_cfg.py （纯四足 Go2）
本任务网络略大一点，因为观测 66 维、动作 18 维，且要同时处理"走路"和"手臂姿态"。
"""

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class Go2PiperPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # 每环境每次迭代采 24 步 × 512 环境 = 12288 条经验/轮
    num_steps_per_env = 24
    # 训练轮数。★8 GB 显卡上先跑 300 轮看趋势；有更多显存/时间再往上加（官方 Go2 常用 1500）
    max_iterations = 300
    save_interval = 50
    experiment_name = "go2_piper_flat"  # → logs/rsl_rl/go2_piper_flat/<时间戳>/
    actor = RslRlMLPModelCfg(
        hidden_dims=[128, 128, 128],  # 比纯四足（512,256,128）小：动作空间小、任务简单
        activation="elu",
        obs_normalization=False,  # 观测已做噪声模拟、量纲接近，暂不开
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[128, 128, 128],
        activation="elu",
        obs_normalization=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,  # 比纯四足略小：动作维度 18，探索需求介于倒立摆和四足之间
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",  # 自适应学习率（按 KL 调 lr），rsl_rl 的招牌功能
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class Go2PiperReachPPORunnerCfg(Go2PiperPPORunnerCfg):
    """Loco-Manipulation Reach 任务的 PPO 超参。

    与原任务（保持手臂姿态）的区别：
        1. experiment_name 换成 go2_piper_reach —— 日志/checkpoint 分开，便于 A/B 对比；
        2. actor 第一层放大到 256：观测 75 维（多了目标点/末端位置/误差），
           而且"够物"比"保持姿势"需要更细的空间映射；
        3. entropy_coef 略微调大，动作空间现在包含大幅度的手臂运动，需要更多探索。
    """

    experiment_name = "go2_piper_reach"
    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 128],
        activation="elu",
        obs_normalization=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class Go2PiperReachGripPPORunnerCfg(Go2PiperReachPPORunnerCfg):
    """Reach 任务 + **作者版带夹爪资产** 的 PPO 超参。

    与 Go2PiperReachPPORunnerCfg 只差 experiment_name：
        "go2_piper_reach_grip" —— 日志/checkpoint 与前三轮（go2_piper_reach）分开，
        否则 get_checkpoint_path(".*") 取"最新目录"时两边会互相串。
    其余超参（actor 256/128/128、entropy 0.01、adaptive lr）与 Reach 版一致 ——
    本轮换的只是资产，超参不跟着变，这样"新旧资产"的对比才干净。
    """

    # ★必须带类型标注！
    #   踩过：在"第二层"子类里裸写 `experiment_name = "..."`（不带 `: str`）会被
    #   Isaac Lab 的 configclass 忽略，静默沿用父类的值 —— 本轮训练就这样落进了
    #   旧任务的日志目录 logs/rsl_rl/go2_piper_reach/，两个任务的曲线混在一起。
    experiment_name: str = "go2_piper_reach_grip"


@configclass
class Go2PiperPickPPORunnerCfg(Go2PiperReachGripPPORunnerCfg):
    """拾取（Pick）任务的 PPO 超参。

    与 Reach 版差别（都是为了让"精细对位"能学出来）：
        experiment_name -> "go2_piper_pick"（日志分开）
        actor init_std 1.0 -> 0.5：动作尺度小、又要精细对位，初始探索噪声不能太大
        entropy_coef 0.01 -> 0.004：熵奖励会让策略迟迟不肯收窄动作分布，
            而抓取需要把分布收到 1~2 cm 的量级
    """

    experiment_name: str = "go2_piper_pick"

    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.5),
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.004,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class Go2PiperTrackPPORunnerCfg(Go2PiperReachGripPPORunnerCfg):
    """S2 追踪任务（走过去 + 够到目标）。

    与 ReachGrip 只差 experiment_name（日志目录分开，便于 A/B 对比）：
        Go2PiperReachGripPPORunnerCfg 的其余超参（网络、PPO、save_interval）
        对本任务都适用 —— 观测维度一样（Reach 的 75 维，不含夹爪），
        动作维度一样（18 维，不含夹爪）。

    ⚠ 踩过的坑（别再犯）：二级 configclass 子类**必须带 @configclass**，
      否则不是 dataclass，这里的 experiment_name 覆盖会被静默忽略，
      日志会落进父类的目录（go2_piper_reach_grip），A/B 对比就废了。
    """

    experiment_name: str = "go2_piper_track"
    max_iterations: int = 4000


@configclass
class Go2PiperWalkPPORunnerCfg(Go2PiperTrackPPORunnerCfg):
    """S1 步态任务（只学走路）。

    ⚠ 观测 79 维 / 动作 18 维，与 ReachGrip / Pick / Track 完全一致
      （都建在作者版带夹爪资产上），所以后续任务可以直接 --resume 本任务的权重。
    """

    experiment_name: str = "go2_piper_walk"
    max_iterations: int = 3000
