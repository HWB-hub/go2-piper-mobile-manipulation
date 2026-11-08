# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO 超参（与作者 go1_gym_learn/ppo_cse_automatic 的 PPO_Args 对齐，供自写 trainer 使用）。

作者的 PPO（go1_gym_learn/ppo_cse_automatic/ppo.py）关键超参：
    clip_param 0.2、value loss coef 1.0、entropy 0.0、num_learning_epochs 5、num_mini_batches 4、
    lr 1e-3（自适应 KL，desired_kl 0.01）、gamma 0.99、lam 0.95、max_grad_norm 1.0、
    num_steps_per_env 24、use_clipped_value_loss True
（详细数值见 spec/03_training_and_DR.md，本文件只列 trainer 真正读取的项。）
"""

from isaaclab.utils.configclass import configclass


@configclass
class RoboDuetPPOCfg:
    # ---- rollout ----
    num_steps_per_env: int = 24
    # ---- PPO ----
    clip_param: float = 0.2
    entropy_coef: float = 0.0
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    learning_rate: float = 1.0e-3
    schedule: str = "adaptive"  # adaptive / fixed
    desired_kl: float = 0.01
    gamma: float = 0.99
    lam: float = 0.95
    max_grad_norm: float = 1.0
    # ---- 网络 ----
    actor_hidden_dims: tuple[int, ...] = (512, 256, 128)
    critic_hidden_dims: tuple[int, ...] = (512, 256, 128)
    activation: str = "elu"
    # ---- 训练规模 ----
    max_iterations: int = 10000
    save_interval: int = 100


@configclass
class RoboDuetPPORunnerCfg(RoboDuetPPOCfg):
    """占位：让 gym.register 的 kwargs 有东西可指。"""

    experiment_name: str = "roboduet_go2piper"
