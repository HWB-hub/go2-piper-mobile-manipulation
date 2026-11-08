"""终止条件（作者 check_termination()，L336-371）。

    ① 超时          : episode_length_buf > 1000 步（20 s）        → Isaac Lab 的 mdp.time_out
    ② 机身太低      : 机身高度 < 0.28 m （use_terminal_body_height=True）
    ③ 阶段 2 才启用 : 机身 pitch 与"末端目标高度"矛盾时判负（use_terminal_pitch=True）
                      roll 那条（use_terminal_roll）作者默认关着，这里也不开

注意作者【没有】接触类终止（terminate_after_contacts_on 建了索引但从未使用）。
"""

from __future__ import annotations

import torch

from . import arm_kinematics as ak
from .observations import base_roll_pitch

TERMINAL_BODY_HEIGHT = 0.28
TERMINAL_BODY_PITCH = 0.2
HEADUPDOWN_THRES = 0.1


def body_height_too_low(env) -> torch.Tensor:
    robot = env.scene["robot"]
    return robot.data.root_pos_w.torch[:, 2] < TERMINAL_BODY_HEIGHT


def arm_pose_inconsistent(env) -> torch.Tensor:
    """阶段 2：机身 pitch 与臂的目标伸展方向矛盾 → 判定失败（作者 reverse 规则）。"""
    if not env.arm_enabled:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    cmd = env.command_manager.get_term("arm_pose").command
    robot = env.scene["robot"]
    _, pitch = base_roll_pitch(env)
    # delta_z = l·sin(p) + 0.38 - base_z
    delta_z = cmd[:, 0] * torch.sin(cmd[:, 1]) + ak.NOMINAL_BASE_HEIGHT - robot.data.root_pos_w.torch[:, 2]
    rule = ((pitch < -TERMINAL_BODY_PITCH) & (delta_z < -HEADUPDOWN_THRES)) | (
        (pitch > TERMINAL_BODY_PITCH) & (delta_z > HEADUPDOWN_THRES)
    )
    # 作者的额外门条件：目标轨迹走完 60% 之后才允许判负
    term = env.command_manager.get_term("arm_pose")
    elapsed_frac = 1.0 - term.time_left / 2.5  # 采样区间 [2,3] s 取中值 2.5 作参考
    return rule & (elapsed_frac > 0.6)
