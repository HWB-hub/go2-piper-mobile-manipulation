"""RoboDuet 移植版需要的"环境级状态"：步态时钟 + 各类历史缓冲。

为什么要单独一个类：
    作者的 legged_robot.py 里有一堆 self.last_actions / last_joint_pos_target /
    last_dof_vel / desired_contact_states / gait_indices ... 这些"跨步记忆"。
    Isaac Lab 的 manager 体系里没有统一的存放处，所以集中放在这里，
    由 trainer 在每个策略步【进入 env.step 之前】调用 pre_step() 更新。

pre_step() 的语义 = "把上一帧结束时的量记为 last_*"，正好对应作者代码里
    compute_reward() 用的 last_actions / last_dof_vel（都是上一帧的值）。
"""

from __future__ import annotations

import torch

# 作者 _step_contact_targets() 里的常量（逐字照抄）
GAIT_FREQUENCY = 3.0
GAIT_NAME = "trotting"
GAIT_PHASES, GAIT_OFFSETS, GAIT_BOUNDS = 0.5, 0.0, 0.0
GAIT_DURATIONS = 0.5
KAPPA_GAIT_PROBS = 0.07


class RoboDuetState:
    """挂在 env 上的跨步状态（env.rd）。"""

    def __init__(self, num_envs: int, device: str, dt: float):
        self.num_envs = num_envs
        self.device = device
        self.dt = dt

        # ---- 历史缓冲（作者 compute_reward 里用到的）----
        self.last_actions = torch.zeros(num_envs, 18, device=device)
        self.last_last_actions = torch.zeros(num_envs, 18, device=device)
        self.last_dof_vel = torch.zeros(num_envs, 20, device=device)
        self.last_joint_pos_target = torch.zeros(num_envs, 20, device=device)
        self.last_last_joint_pos_target = torch.zeros(num_envs, 20, device=device)
        self.last_contacts = torch.zeros(num_envs, 4, dtype=torch.bool, device=device)
        self.last_plan_actions = torch.zeros(num_envs, 2, device=device)

        # ---- 步态时钟 ----
        self.gait_indices = torch.zeros(num_envs, device=device)
        self.foot_indices = torch.zeros(num_envs, 4, device=device)
        self.clock_inputs = torch.zeros(num_envs, 4, device=device)
        self.desired_contact_states = torch.zeros(num_envs, 4, device=device)

        # ---- 机械臂目标轨迹计时 ----
        self.arm_time_buf = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.T_trajs = torch.full((num_envs,), 2.0, device=device)

    # ------------------------------------------------------------------
    # 每步更新
    # ------------------------------------------------------------------
    def pre_step(self, env) -> None:
        """在 env.step() 之前调用：滚动历史 + 推进步态相位。"""
        robot = env.scene["robot"]
        am = env.action_manager
        # 上一帧动作（与作者 self.actions 对应：已被 clip 的动作）
        prev = am.action if getattr(am, "action", None) is not None else torch.zeros(self.num_envs, 18, device=self.device)
        self.last_last_actions.copy_(self.last_actions)
        self.last_actions.copy_(prev)
        self.last_last_joint_pos_target.copy_(self.last_joint_pos_target)
        self.last_joint_pos_target.copy_(robot.data.joint_pos_target.torch)
        self.last_dof_vel.copy_(robot.data.joint_vel.torch)

        # 步态时钟：作者的 _step_contact_targets()（trot 步态，3 Hz）
        self.step_gait(env.commands_xy_norm())

    def step_gait(self, cmd_xy_norm: torch.Tensor) -> None:
        """作者 `_step_contact_targets()` 的逐行移植（trot 分支）。"""
        self.gait_indices = torch.remainder(self.gait_indices + self.dt * GAIT_FREQUENCY, 1.0)
        if GAIT_OFFSETS != 0.0:  # pacing_offset=True 时才走这一支；trot 下 offsets=bounds=0
            foot_indices = [
                self.gait_indices + GAIT_PHASES + GAIT_OFFSETS + GAIT_BOUNDS,
                self.gait_indices + GAIT_BOUNDS,
                self.gait_indices + GAIT_OFFSETS,
                self.gait_indices + GAIT_PHASES,
            ]
        else:
            foot_indices = [
                self.gait_indices + GAIT_PHASES + GAIT_OFFSETS + GAIT_BOUNDS,
                self.gait_indices + GAIT_OFFSETS,
                self.gait_indices + GAIT_BOUNDS,
                self.gait_indices + GAIT_PHASES,
            ]
        # 站着不动时把相位钉在 0.25（作者的 "mark stand"）
        for idx in foot_indices:
            idx[cmd_xy_norm < 0.1] = 0.25
        # 摆动/支撑相分别重映射到 [0,0.5]/[0.5,1]
        for idx in foot_indices:
            rem = torch.remainder(idx, 1.0)
            stance = rem < GAIT_DURATIONS
            swing = ~stance
            idx[stance] = rem[stance] * (0.5 / GAIT_DURATIONS)
            idx[swing] = 0.5 + (rem[swing] - GAIT_DURATIONS) * (0.5 / (1.0 - GAIT_DURATIONS))

        self.foot_indices = torch.remainder(
            torch.cat([foot_indices[i].unsqueeze(1) for i in range(4)], dim=1), 1.0
        )
        for i in range(4):
            self.clock_inputs[:, i] = torch.sin(2 * torch.pi * foot_indices[i])

        # von Mises 分布的平滑接触目标（作者的 smoothing_multiplier_*）
        kappa = KAPPA_GAIT_PROBS
        cdf = torch.distributions.Normal(0.0, kappa).cdf
        for i in range(4):
            f = torch.remainder(foot_indices[i], 1.0)
            mult = cdf(f) * (1 - cdf(f - 0.5)) + cdf(f - 1) * (1 - cdf(f - 0.5 - 1))
            self.desired_contact_states[:, i] = mult

    def on_reset(self, env_ids: torch.Tensor) -> None:
        """对应作者 reset_idx 里对历史/相位的清零。"""
        if len(env_ids) == 0:
            return
        self.gait_indices[env_ids] = 0.0
        self.foot_indices[env_ids] = 0.0
        self.clock_inputs[env_ids] = 0.0
        self.desired_contact_states[env_ids] = 0.0
        self.last_contacts[env_ids] = False
        self.arm_time_buf[env_ids] = 0
