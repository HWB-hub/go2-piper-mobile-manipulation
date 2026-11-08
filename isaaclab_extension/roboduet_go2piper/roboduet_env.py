"""RoboDuet 环境子类：把作者 env 里"跨管理器"的东西集中起来。

需要它做 4 件 Isaac Lab manager 体系本身不提供的事：
    1. 挂一个跨步状态对象 env.rd（步态时钟、历史缓冲、随机化记录）
    2. 每步进入仿真前调 rd.pre_step()（滚动历史 + 推进相位）—— 对应作者 _post_physics_step_callback
    3. 暴露 plan()（臂策略最后 2 维 → 机身材姿指令）与"臂是否启用"开关（对应 global_switch）
    4. 把 Isaac Lab 的单一奖励缓冲按作者的规则拆成 dog / arm 两份
"""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv

from .mdp import events as rd_events
from .mdp.roboduet_state import RoboDuetState

VIS_TERMS = ("vis_manip_commands_tracking_lpy", "vis_manip_commands_tracking_rpy")
DOG_ONLY_TERMS = ("tracking_lin_vel", "tracking_ang_vel")


class RoboDuetEnv(ManagerBasedRLEnv):
    """与作者 VelocityTrackingEasyEnv 对应的 Isaac Lab 环境。"""

    def __init__(self, cfg, render_mode=None, **kwargs):
        # ★rd 必须在 super().__init__ 之前建好：startup/reset 事件与 _reset_idx 都会用到它
        self.rd = RoboDuetState(cfg.scene.num_envs, cfg.sim.device, cfg.decimation * cfg.sim.dt)
        self.rd.raw_actions = torch.zeros(cfg.scene.num_envs, 18, device=cfg.sim.device)
        self.arm_enabled = False
        self.rd_iteration = 0
        self.rd_switch_iter = 0
        self.rd_plan_actions = torch.zeros(cfg.scene.num_envs, 2, device=cfg.sim.device)

        super().__init__(cfg, render_mode)

        # ---- 关节 / 刚体索引（按名字解析，不硬编码）----
        robot = self.scene["robot"]
        self.rd_leg_joint_ids = torch.tensor(
            [i for i, n in enumerate(robot.joint_names) if n.endswith(("_hip_joint", "_thigh_joint", "_calf_joint"))],
            device=self.device,
        )
        self.rd_arm_joint_ids = torch.tensor(
            [i for i, n in enumerate(robot.joint_names) if n.startswith("piper_joint") and n[-1] in "123456"],
            device=self.device,
        )
        # 作者用 [12:] 覆盖的 8 个 piper 关节（6 转动 + 2 夹爪 prismatic）
        self.rd_piper_joint_ids = torch.tensor(
            [i for i, n in enumerate(robot.joint_names) if n.startswith("piper_joint")], device=self.device
        )
        self.rd_driven_joint_ids = torch.cat([self.rd_leg_joint_ids, self.rd_arm_joint_ids])
        feet = [i for i, n in enumerate(robot.body_names) if n.endswith("_calf")]  # 足端合并进了 calf
        self.rd_feet_body_ids = feet  # ★articulation 侧索引（取 body_pos_w / body_lin_vel_w 用）
        # 惩罚接触用的刚体：作者的 penalize_contacts_on 模糊匹配到 {base, thigh, calf, piper 系列}，
        # ★但本 USD 里"足端球体"已并进 calf 刚体（Isaac Gym 那边 foot 是独立刚体），
        #   若把 calf 也算接触惩罚，正常走路时每只脚着地都会被当成"碰撞"扣分（实测每步 -0.1×4）。
        #   所以这里去掉 calf，只保留 base / thigh / piper 系列。
        penal_names = [
            n for n in robot.body_names if n == "base" or n.endswith("_thigh") or n.startswith("piper_link")
        ]
        self.rd_penalised_body_ids = penal_names  # 先存名字，下面统一转成传感器索引

        # ★关键：ContactSensor 的刚体顺序【不等于】articulation 的刚体顺序，
        #   接触力张量必须用"传感器侧"的索引（否则会取到别的刚体 —— 实测把小腿的接触算成了 base/大腿/臂）。
        sensor = self.scene["contact_forces"]
        sensor_names = list(sensor.body_names)
        self.rd_feet_sensor_ids = [sensor_names.index(robot.body_names[i]) for i in feet if robot.body_names[i] in sensor_names]
        self.rd_penalised_sensor_ids = [sensor_names.index(n) for n in penal_names if n in sensor_names]
        self.rd_penalised_body_ids = [
            robot.body_names.index(n) for n in penal_names
        ]  # 保留 articulation 索引（调试打印用）

        # ---- 两阶段开关（作者 global_switch）----
        # 主跟踪奖励的权重课程（auto_train.py 的值）
        self.rd_manip_w = {
            "lpy_start": 4.0,
            "lpy_end": 3.0,
            "rpy_start": 0.0,
            "rpy_end": 1.0,
            "keep_sum_constant": True,
            "transition_iters": 5000,
            "transition_power": 1.0,
        }
        # 电机零位偏置（动作项使用）；reset 事件可能已经写过，缺失才建
        if not hasattr(self.rd, "motor_offset"):
            self.rd.motor_offset = torch.zeros(self.num_envs, self.rd_driven_joint_ids.numel(), device=self.device)
        if getattr(self.rd, "friction", None) is None:
            self.rd.friction = torch.full((self.num_envs,), 0.5, device=self.device)
            self.rd.restitution = torch.zeros(self.num_envs, device=self.device)

        # 启动时随机化后的材质（privileged obs 用）
        rd_events.cache_material_properties(self)
        # 启动时也做一次电机强度/零位随机化（作者在 _create_envs / reset_idx 都会做）
        rd_events.randomize_motor_strength(self, torch.arange(self.num_envs, device=self.device), (0.9, 1.1))
        rd_events.randomize_motor_offset(self, torch.arange(self.num_envs, device=self.device), (-0.02, 0.02))
        self.rd.raw_actions = torch.zeros(self.num_envs, 18, device=self.device)

    # ------------------------------------------------------------------
    # 与作者 env 对应的接口
    # ------------------------------------------------------------------
    @property
    def plan_actions(self) -> torch.Tensor:
        return self.rd_plan_actions

    def pre_step(self, action: torch.Tensor) -> None:
        """进入仿真前：滚动历史 + 推进步态相位。action = 原始 18 维网络输出（未缩放）。"""
        self.rd.pre_step(self)
        self.rd.raw_actions = action.detach().clone()

    def plan(self, plan_actions: torch.Tensor) -> None:
        """作者 VelocityTrackingEasyEnv.plan()：臂策略最后 2 维 → 机身材姿指令。"""
        term = self.command_manager.get_term("body_pose")
        term.set_from_plan(plan_actions)
        self.rd.last_plan_actions.copy_(self.rd_plan_actions)
        self.rd_plan_actions.copy_(plan_actions)

    def set_arm_enabled(self, enabled: bool, iteration: int | None = None) -> None:
        """打开/关闭二阶段（作者 global_switch.open_switch()）。"""
        if enabled and not self.arm_enabled:
            self.rd_switch_iter = iteration if iteration is not None else self.rd_iteration
        self.arm_enabled = enabled
        if not enabled:
            # 臂指令清零（作者 switch 关时 commands_arm 保持 0）
            self.command_manager.get_term("arm_pose").zero()

    def commands_xy_norm(self) -> torch.Tensor:
        """作者 `_step_contact_targets` 里的 torch.norm(commands_dog[:, :3])。"""
        return torch.norm(self.command_manager.get_term("velocity").command, dim=1)

    # ------------------------------------------------------------------
    # 奖励拆分（作者 compute_reward 的规则）
    # ------------------------------------------------------------------
    def split_rewards(self) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 (rew_buf_dog, rew_buf_arm)，单位与 Isaac Lab 一致（已含 weight×dt）。

            dog = 全部项 − 两个 vis_*（它们只记录，不进任何 buffer）
            arm = dog − tracking_lin_vel − tracking_ang_vel
        """
        names = self.reward_manager.active_terms
        step = self.reward_manager._step_reward  # (N, T) 已含 weight×dt
        dog = torch.zeros(self.num_envs, device=self.device)
        arm = torch.zeros(self.num_envs, device=self.device)
        for i, name in enumerate(names):
            if name in VIS_TERMS:
                continue
            dog += step[:, i]
            if name not in DOG_ONLY_TERMS:
                arm += step[:, i]
        # ★Isaac Lab 的 _step_reward 存的是 weight×原始值（它把 dt 除掉又乘回来了），
        #   作者那边是 weight×dt×原始值，所以这里补回 dt，保证两边数值一致
        return dog * self.step_dt, arm * self.step_dt

    def reward_term_dict(self) -> dict[str, torch.Tensor]:
        """当前步每项奖励（名字 → (N,)），用于日志。"""
        step = self.reward_manager._step_reward
        return {n: step[:, i] for i, n in enumerate(self.reward_manager.active_terms)}

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if hasattr(self, "rd"):
            self.rd.on_reset(env_ids)
