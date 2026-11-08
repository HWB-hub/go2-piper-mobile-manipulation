"""自定义动作项：在关节位置目标上加"电机零位偏置"（作者的 motor_offset 随机化）。

作者 `_randomize_dof_props`：`motor_offsets[:, :] ~ U(-0.02, 0.02)`，
在 `_compute_torques` / 位置目标里都是 `target + motor_offset`。
Isaac Lab 没有现成的等价事件，所以在动作项里加一个每环境固定的偏置张量。
"""

from __future__ import annotations

import torch

from isaaclab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from isaaclab.utils.configclass import configclass


class JointPositionActionWithOffset(JointPositionAction):
    """target = default + scale·action + offset（offset 由 rd.motor_offset 提供，按关节切片）。"""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        # 本动作项管的是哪些关节（相对整个 articulation 的下标）
        self._joint_ids = self._joint_ids if self._joint_ids != slice(None) else list(range(self.action_dim))

    def apply_actions(self):
        target = self.processed_actions
        offset = getattr(self._env.rd, "motor_offset", None)
        if offset is not None:
            target = target + offset[:, self._joint_ids]
        self._asset.set_joint_position_target(target, joint_ids=self._joint_ids)


@configclass
class JointPositionActionWithOffsetCfg(JointPositionActionCfg):
    class_type: type = JointPositionActionWithOffset
