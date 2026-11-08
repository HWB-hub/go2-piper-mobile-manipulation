# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""注册 RoboDuet (Go2+Piper) 的两个任务。

    Isaac-RoboDuet-Go2Piper-v0       训练用（512 环境，作者配置）
    Isaac-RoboDuet-Go2Piper-Play-v0  回放/录像用（16 环境，关掉随机化）

★注意 entry_point：本任务的 env 类是我们自己的 RoboDuetEnv（不是默认的 ManagerBasedRLEnv），
  因为它要额外挂步态时钟/历史状态、plan() 接口和 dog/arm 奖励拆分。
"""

import gymnasium as gym

from . import agents  # noqa: F401

##
# Register Gym environments.
##

gym.register(
    id="Isaac-RoboDuet-Go2Piper-v0",
    entry_point=f"{__name__}.roboduet_env:RoboDuetEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.roboduet_env_cfg:RoboDuetEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RoboDuetPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-RoboDuet-Go2Piper-Play-v0",
    entry_point=f"{__name__}.roboduet_env:RoboDuetEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.roboduet_env_cfg:RoboDuetEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RoboDuetPPORunnerCfg",
    },
)
