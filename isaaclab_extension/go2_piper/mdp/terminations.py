# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""本任务自己写的终止条件（DoneTerm 需要一个可调用对象，放在 mdp 包里最省事）。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def goal_reached(
    env: ManagerBasedRLEnv,
    command_name: str = "ee_goal",
    threshold: float = 0.08,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="piper_link6"),
) -> torch.Tensor:
    """末端到达目标点（距离 < threshold）→ 本回合立刻结束。

    ★加这一条是为了治"效率太低"：
      原来 episode 固定 20 s，而任务只要 3~5 s 就做完了，剩下 15+ s 机器人站着空转
      （既浪费时间，也让训练数据里塞满"已经做完却还在采样"的无效步）。
      成功后立刻重置 ⇒ 单位时间能采到的有效回合数成倍增加。
    """
    cmd = env.command_manager.get_term(command_name)
    asset = env.scene[asset_cfg.name]
    pos = asset.data.body_pos_w
    pos = pos.torch if hasattr(pos, "torch") else pos
    pos = pos[:, asset_cfg.body_ids]
    if pos.dim() == 3:
        pos = pos[:, 0]
    d = torch.norm(pos - cmd.goal_pos_w, dim=1)
    return d < threshold
