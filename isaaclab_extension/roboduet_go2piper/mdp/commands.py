"""三路指令：速度指令 / 机身材姿指令 / 机械臂末端 6D 指令。

对应作者 legged_robot.py 里的 commands_dog(5 维) + commands_arm_obs(6 维)：
    0) velocity  : (vx, vy, ω_yaw)                              —— 10 s 重采样 + 课程
    1) body_pose : (pitch_cmd, roll_cmd)                        —— 阶段 1 课程采样，阶段 2 由臂策略 plan() 写
    2) arm_pose  : (l, p, yaw, α, β, γ)                          —— 2~3 s 重采样，带碰撞盒过滤

★阶段划分（对应作者的 global_switch.switch_open）：
    env.arm_enabled == False（阶段 1）: arm_pose 恒为 0；body_pose 由课程采样
    env.arm_enabled == True （阶段 2）: arm_pose 正常采样；body_pose 由臂策略输出写（plan）
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils.configclass import configclass

from . import arm_kinematics as ak


class VelocityCommandTerm(CommandTerm):
    """速度指令 (vx, vy, ω_z)，带"越跑越好就放范围"的课程。"""

    def __init__(self, cfg: CommandTermCfg, env):
        super().__init__(cfg, env)
        self.robot = env.scene["robot"]
        self.command_buf = torch.zeros(self.num_envs, 3, device=self.device)
        # 课程：当前采样范围（作者用分箱权重表示，这里用区间宽度等价实现）
        self.cur_lin_x = list(cfg.ranges.lin_vel_x)
        self.cur_lin_y = list(cfg.ranges.lin_vel_y)
        self.cur_ang_z = list(cfg.ranges.ang_vel_z)
        self.limit_lin_x = [-5.0, 5.0]
        self.limit_lin_y = list(cfg.ranges.lin_vel_y)
        self.limit_ang_z = [-5.0, 5.0]
        # 课程用：每个 env 上一段指令的跟踪得分
        self.guide_sum_lin = torch.zeros(self.num_envs, device=self.device)
        self.guide_sum_ang = torch.zeros(self.num_envs, device=self.device)
        self.guide_steps = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.command_buf

    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        n = len(env_ids)
        if n == 0:
            return
        # ---- 课程：用上一段的跟踪得分决定是否放宽范围 ----
        steps = self.guide_steps[env_ids].clamp(min=1.0)
        score_lin = (self.guide_sum_lin[env_ids] / steps).mean().item()
        score_ang = (self.guide_sum_ang[env_ids] / steps).mean().item()
        if score_lin > 0.8:  # curriculum_thresholds.tracking_lin_vel × pretrained scale(1.0)
            self.cur_lin_x = self._widen(self.cur_lin_x, self.limit_lin_x, 0.1)
            self.cur_lin_y = self._widen(self.cur_lin_y, self.limit_lin_y, 0.1)
        if score_ang > 0.5:  # curriculum_thresholds.tracking_ang_vel × pretrained scale(1.0)
            self.cur_ang_z = self._widen(self.cur_ang_z, self.limit_ang_z, 0.1)
        self.guide_sum_lin[env_ids] = 0.0
        self.guide_sum_ang[env_ids] = 0.0
        self.guide_steps[env_ids] = 0.0

        # ---- 采样 ----
        cmd = torch.empty(n, 3, device=self.device).uniform_(-1.0, 1.0)
        cmd[:, 0] = cmd[:, 0] * (self.cur_lin_x[1] - self.cur_lin_x[0]) / 2 + (self.cur_lin_x[0] + self.cur_lin_x[1]) / 2
        cmd[:, 1] = cmd[:, 1] * (self.cur_lin_y[1] - self.cur_lin_y[0]) / 2 + (self.cur_lin_y[0] + self.cur_lin_y[1]) / 2
        cmd[:, 2] = cmd[:, 2] * (self.cur_ang_z[1] - self.cur_ang_z[0]) / 2 + (self.cur_ang_z[0] + self.cur_ang_z[1]) / 2
        # 10% 的 env 速度指令清零（作者 zero_mask）
        zero_mask = torch.rand(n, device=self.device) < 0.1
        cmd[zero_mask, :] = 0.0
        # 死区（作者死区：|vx|>0.07, |vy|>0.07, |ω|>0.1）
        cmd[:, 0] *= (cmd[:, 0].abs() > 0.07).float()
        cmd[:, 1] *= (cmd[:, 1].abs() > 0.07).float()
        cmd[:, 2] *= (cmd[:, 2].abs() > 0.1).float()
        self.command_buf[env_ids] = cmd

    @staticmethod
    def _widen(cur, limit, frac):
        """把区间按 limit 的方向放宽 frac 倍。"""
        out = list(cur)
        out[0] = max(limit[0], out[0] - abs(limit[0] - out[0]) * frac)
        out[1] = min(limit[1], out[1] + abs(limit[1] - out[1]) * frac)
        return out

    def _update_command(self):
        pass

    def record_tracking(self, lin_reward: torch.Tensor, ang_reward: torch.Tensor, dt: float):
        """由 trainer 每步喂入跟踪奖励，用于课程判断（作者用 command_sums 的近似）。"""
        self.guide_sum_lin += lin_reward / max(dt, 1e-6)
        self.guide_sum_ang += ang_reward / max(dt, 1e-6)
        self.guide_steps += 1.0


class BodyPoseCommandTerm(CommandTerm):
    """机身姿态指令 (pitch, roll)。

    阶段 1：课程采样 U(-0.4, 0.4)（作者 `_resample_commands` 里 new_commands[:,3:5]）
    阶段 2：每个策略步由臂策略最后两维经 plan() 写入（作者 `VelocityTrackingEasyEnv.plan`）
    """

    def __init__(self, cfg: CommandTermCfg, env):
        super().__init__(cfg, env)
        self.command_buf = torch.zeros(self.num_envs, 2, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.command_buf

    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        # 阶段 2 由 plan() 每步写，重采样不再覆盖
        if getattr(self._env, "arm_enabled", False):
            return
        n = len(env_ids)
        self.command_buf[env_ids, 0] = torch.empty(n, device=self.device).uniform_(
            self.cfg.pitch_range[0], self.cfg.pitch_range[1]
        )
        self.command_buf[env_ids, 1] = torch.empty(n, device=self.device).uniform_(
            self.cfg.roll_range[0], self.cfg.roll_range[1]
        )

    def _update_command(self):
        pass

    def set_from_plan(self, plan_actions: torch.Tensor):
        """作者 VelocityTrackingEasyEnv.plan()：x0.4 再截断（pitch 上限被特意收窄到 0.3）。"""
        rescaled = plan_actions * 0.4
        self.command_buf[:, 0] = rescaled[:, 0].clamp(self.cfg.pitch_range[0], self.cfg.pitch_range[1] / 4 * 3)
        self.command_buf[:, 1] = rescaled[:, 1].clamp(self.cfg.roll_range[0], self.cfg.roll_range[1])


class ArmPoseCommandTerm(CommandTerm):
    """机械臂末端 6D 指令：lpy(3) + 目标姿态 abg(3)。"""

    def __init__(self, cfg: CommandTermCfg, env):
        super().__init__(cfg, env)
        self.command_buf = torch.zeros(self.num_envs, 6, device=self.device)
        # 先给一个合法的初值（否则 reset 前是 0 → l=0 会在 exp 里出现极端误差）
        self.command_buf[:, 0] = 0.5
        self.target_abg = torch.zeros(self.num_envs, 3, device=self.device)
        self.lpy = self.command_buf[:, :3]

    @property
    def command(self) -> torch.Tensor:
        return self.command_buf

    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        # 阶段 1：臂指令保持 0（作者 `if not switch_open: return`）
        if not getattr(self._env, "arm_enabled", False):
            return
        n = len(env_ids)
        dev = self.device

        def uni(rng):
            return torch.empty(n, device=dev).uniform_(rng[0], rng[1])

        l = uni(ak.LPY_L_RANGE)
        p = uni(ak.LPY_P_RANGE)
        y = uni(ak.LPY_Y_RANGE)
        roll = uni(ak.EE_ROLL_RANGE)
        pitch = uni(ak.EE_PITCH_RANGE)
        yaw = uni(ak.EE_YAW_RANGE)

        lpy = torch.stack([l, p, y], dim=-1)
        # 碰撞盒 / 插地过滤：非法就沿用上一条 lpy（作者 _arm_target_collision_mask）
        xyz = ak.lpy_to_local_xyz(lpy)
        lower = torch.tensor(ak.COLLISION_LOWER, device=dev)
        upper = torch.tensor(ak.COLLISION_UPPER, device=dev)
        invalid = ((xyz > lower) & (xyz < upper)).all(dim=-1) | (xyz[:, 2] < ak.UNDERGROUND_LIMIT)
        prev_lpy = self.command_buf[env_ids, :3]
        lpy = torch.where(invalid.unsqueeze(-1), prev_lpy, lpy)

        abg = ak.target_abg_from_euler(roll, pitch, yaw)
        self.command_buf[env_ids, :3] = lpy
        self.command_buf[env_ids, 3:6] = abg
        self.target_abg[env_ids] = abg

    def _update_command(self):
        pass

    def zero(self):
        """阶段 1 用：臂指令归零（作者在 switch 关时保持 0）。"""
        self.command_buf[:] = 0.0
        self.command_buf[:, 0] = 0.5


# ----------------------------------------------------------------------
# cfg
# ----------------------------------------------------------------------
@configclass
class VelocityCommandCfg(CommandTermCfg):
    class_type: type = VelocityCommandTerm
    resampling_time_range: tuple[float, float] = (10.0, 10.0)  # 作者 resampling_time = 10 s

    @configclass
    class Ranges:
        lin_vel_x: tuple[float, float] = (-0.6, 0.6)
        lin_vel_y: tuple[float, float] = (-0.6, 0.6)
        ang_vel_z: tuple[float, float] = (-1.0, 1.0)

    ranges: Ranges = Ranges()


@configclass
class BodyPoseCommandCfg(CommandTermCfg):
    class_type: type = BodyPoseCommandTerm
    resampling_time_range: tuple[float, float] = (10.0, 10.0)
    pitch_range: tuple[float, float] = (-0.4, 0.4)  # auto_train: body_pitch_range
    roll_range: tuple[float, float] = (-0.4, 0.4)  # auto_train: body_roll_range


@configclass
class ArmPoseCommandCfg(CommandTermCfg):
    class_type: type = ArmPoseCommandTerm
    resampling_time_range: tuple[float, float] = (2.0, 3.0)  # 作者 T_traj = [2, 3] s
