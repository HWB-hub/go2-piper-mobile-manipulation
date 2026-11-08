# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""末端执行器目标指令生成器 —— 移植自 visual_wholebody 的低层全身控制器。

## 来源

对应源码（Isaac Gym / legged_gym 版）：
    low-level/legged_gym/envs/manip_loco/manip_loco.py
        :: _resample_ee_goal_sphere_once / _resample_ee_goal / _update_curr_ee_goal
        :: _get_ee_goal_spherical_center / _collision_check
    low-level/legged_gym/envs/manip_loco/b1z1_config.py :: class goal_ee

## 原设计要点（照搬的部分）

1. **目标是"机身相对"的**：世界坐标 = 机身位置 + R_yaw(机身偏航) @ 机身系偏移。
   机器人走动时目标跟着一起平移，手臂必须持续补偿机身运动 —— 这是"全身协同"的来源。
2. **目标是平滑移动的轨迹**：每次重采样后，用 traj_time 从【上一个目标】插值到
   【新目标】，而不是瞬间跳变，奖励不会突然掉一大截。
3. 原仓库还有落地/碰撞检查（_collision_check）；本实现的采样盒已按实测可达空间缩过，
   所以省掉检查（见下）。

## 本实现的取舍

- 采样方式：原仓库用球坐标(l, pitch, yaw)；这里改为 base 系下的**盒式均匀采样**，
  便于按 Go2+Piper 的实际可达空间标定。标定依据：asset_import/go2_piper/probe_ee.py
  实测可达 x[-0.69,+0.53] y[-0.61,+0.62] z[-0.16,+0.81]，
  标称末端在 base 系 (0.084, 0, 0.505)。
- 坐标系约定：x/y 随偏航旋转，**z 取世界竖直**（不随机身俯仰/滚转倾斜），
  等价于"目标像放在桌上"。原仓库的 _get_ee_goal_spherical_center 只取机身 xy + 地形 z，同思路。
- 加了 world_frame 开关：True 时目标固定在世界里（更难，要求"走过去再够"）。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import quat_apply, quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv

# ★注意：不要在本模块顶层 import isaaclab.markers / isaaclab.sim 里的可视化相关类。
#   `isaaclab train` 这类 CLI 会【先解析配置、后启动仿真】，配置解析时会 import 本模块；
#   如果在此时就把 pxr(USD) 拉进来，仿真启动会报
#   "Modules: ['pxr', ...] were loaded before SimulationApp was started" 然后崩溃。
#   所以可视化相关 import 一律放到 _set_debug_vis_impl() 函数体内（延迟导入）。


class EeGoalCommand(CommandTerm):
    """生成末端执行器（gripper）的目标位置指令。

    ``command`` 张量 shape = (num_envs, 3)，含义是 **base 系下的目标点**
    （x/y 随偏航旋转，z 为世界竖直）。世界系目标同步维护在 :attr:`goal_pos_w`，奖励用它算误差。
    """

    cfg: EeGoalCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: EeGoalCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.ee_body_idx: int = self.robot.data.body_names.index(cfg.body_name)
        # 自动进阶课程的状态：成功率 EMA + 当前档位（见 cfg.curriculum_auto 的说明）
        self._succ_ema: float = 0.5
        self._alpha_cur: float = float(cfg.curriculum_alpha)
        self._auto_tick: int = 0
        # 可选的第二个刚体（如另一根手指），与 ee_body_idx 一起取中点作为夹取中心
        self._second_body_idx: int | None = None
        if getattr(cfg, "second_body_name", None) and cfg.second_body_name in self.robot.data.body_names:
            self._second_body_idx = self.robot.data.body_names.index(cfg.second_body_name)

        # 指令缓冲（base 系）与对应的世界系目标
        self.goal_pos_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.goal_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

        # 轨迹插值：起点/终点（两个坐标系各存一份）+ 计时
        self._start_pos_b = torch.zeros(self.num_envs, 3, device=self.device)
        self._target_pos_b = torch.zeros(self.num_envs, 3, device=self.device)
        self._start_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._target_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

        self._sampled_w = torch.zeros(self.num_envs, 3, device=self.device)  # 课程用：原始采样点
        # ★单位必须一致：_elapsed 累加的是【秒】(step_dt)，所以轨迹时长也用【秒】。
        #   曾经写成 (dur / step_dt) 存"步数"，却拿秒去除 → 插值慢了 50 倍
        #   （1.5~3 s 的轨迹要 77 s 才走完），表现为"目标点几乎不动"。
        self._traj_time = torch.ones(self.num_envs, device=self.device)
        self._elapsed = torch.zeros(self.num_envs, device=self.device)
        self._initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # 指标：末端到目标的位置误差（自动记录到 TensorBoard 的 Metrics/ 下）
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)

        # 可选的目标点小球：在这里【一次性解析】，解析不到就降级为不显示，而不是每步崩。
        # （踩过：探测脚本把该场景物体去掉后，命令仍引用它 → KeyError 直接崩掉整个仿真）
        self._marker = None
        if cfg.marker_asset_name is not None:
            try:
                self._marker = env.scene[cfg.marker_asset_name]
            except KeyError:
                print(f"[WARN] EeGoalCommand: 场景里没有 '{cfg.marker_asset_name}'，"
                      f"跳过目标点小球（功能不受影响）")

        # 可选的目标物（RigidObject）：同样一次性解析，允许场景里没有
        self._object = None
        if cfg.object_asset_name is not None:
            try:
                self._object = env.scene[cfg.object_asset_name]
            except KeyError:
                print(f"[WARN] EeGoalCommand: 场景里没有 '{cfg.object_asset_name}'，"
                      f"退化为纯够物模式（不会瞬移目标物）")

    def __str__(self) -> str:
        msg = "EeGoalCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tTrajectory time range: {self.cfg.traj_time}\n"
        msg += f"\tWorld frame (fixed goal): {self.cfg.world_frame}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """base 系下的目标点，shape (num_envs, 3)。"""
        return self.goal_pos_b

    """
    Implementation specific functions
    """

    def _resample_command(self, env_ids: Sequence[int]):
        env_ids = torch.as_tensor(env_ids, device=self.device)

        # 轨迹起点 = 当前（已插值的）目标
        self._start_pos_b[env_ids] = self.goal_pos_b[env_ids]
        self._start_pos_w[env_ids] = self.goal_pos_w[env_ids]

        # 采样新的 base 系目标点
        r = torch.empty(len(env_ids), device=self.device)
        self._target_pos_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.pos_x)
        self._target_pos_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.pos_y)
        self._target_pos_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.pos_z)

        # 世界系目标：以【采样时刻】的机身位姿换算一次
        bp = self.robot.data.root_pos_w.torch[env_ids]
        bq = yaw_quat(self.robot.data.root_quat_w.torch[env_ids])

        # ★目标物：强制落在世界系固定高度（台面顶）
        if self.cfg.object_rest_z is not None:
            self._target_pos_b[env_ids, 2] = self.cfg.object_rest_z - bp[:, 2]
        # ★夹爪瞄准点再抬高一点（见 object_grasp_z_offset 的说明）：
        #   物体本体仍在 object_rest_z 高度，只有"指令点/奖励参考点"抬高。
        grasp_z_w = None
        if self.cfg.object_rest_z is not None:
            grasp_z_w = self.cfg.object_rest_z + self.cfg.object_grasp_z_offset

        if self.cfg.sample_in_env_frame:
            # ★把采样值当【环境坐标系】偏移用：世界点 = 环境原点 + 采样值。
            #   台面是固定在环境坐标系的，这样物体永远落在台面上，
            #   不受机器人被推走的影响（详见 cfg.sample_in_env_frame 的说明）。
            #   之后再从世界点反算 base 系（供观测/奖励用）。
            self._target_pos_w[env_ids] = (
                self._env.scene.env_origins[env_ids] + self._target_pos_b[env_ids]
            )
            if grasp_z_w is not None:
                self._target_pos_w[env_ids, 2] = grasp_z_w
            else:
                self._target_pos_w[env_ids, 2] = (
                    self._env.scene.env_origins[env_ids][:, 2] + self._target_pos_b[env_ids][:, 2]
                )
            self._target_pos_b[env_ids] = quat_apply_inverse(bq, self._target_pos_w[env_ids] - bp)
        else:
            if grasp_z_w is not None:
                self._target_pos_b[env_ids, 2] = grasp_z_w - bp[:, 2]
            self._target_pos_w[env_ids] = bp + quat_apply(bq, self._target_pos_b[env_ids])

        # 首次采样（环境刚建好）不做插值，直接落到目标点
        first = ~self._initialized[env_ids]

        # ★课程需要"原始采样点"（课程插值前的目标），每步跟随用它做插值另一端
        self._sampled_w[env_ids] = self._target_pos_w[env_ids].clone()

        # ★★课程学习：把物体的摆位从"夹取中心"平滑插值到"采样点"。
        #   alpha=0 → 直接摆在当前两指中点（物体已在指间，只需学"合爪+抬起"）
        #   alpha=1 → 正常难度
        #   注意 z 分量始终用 object_rest_z（物体必须稳稳放在台面上，不能悬在空中）。
        alpha = self.cfg.curriculum_alpha
        if self.cfg.curriculum_ramp_steps > 0:
            alpha = min(
                float(self.cfg.curriculum_alpha),
                float(self._env.common_step_counter) / float(self.cfg.curriculum_ramp_steps),
            )
        if alpha < 1.0:
            pinch_w = self.robot.data.body_pos_w.torch[env_ids, self.ee_body_idx]
            if self._second_body_idx is not None:
                pinch_w = 0.5 * (pinch_w + self.robot.data.body_pos_w.torch[env_ids, self._second_body_idx])
            placed = pinch_w + alpha * (self._target_pos_w[env_ids] - pinch_w)
            # ★钳制到台面范围内（环境系）：重采样那一刻手臂可能还在就位中，
            #   手的位置未必在台面正上方，插值点会落到台面外 → 物体直接掉地上
            #   （实测 alpha=0.4 档 `pick_approach` 归零到 0.0009，物体离手 17 cm）。
            if self.cfg.object_rest_z is not None:
                placed[:, 2] = self.cfg.object_rest_z
            lo = self._env.scene.env_origins[env_ids]
            placed[:, 0] = torch.clamp(
                placed[:, 0], lo[:, 0] + self.cfg.curriculum_x_range[0], lo[:, 0] + self.cfg.curriculum_x_range[1]
            )
            placed[:, 1] = torch.clamp(
                placed[:, 1], lo[:, 1] + self.cfg.curriculum_y_range[0], lo[:, 1] + self.cfg.curriculum_y_range[1]
            )
            self._target_pos_w[env_ids] = placed
            self._target_pos_b[env_ids] = quat_apply_inverse(bq, placed - bp)

        if first.any():
            idx = env_ids[first]
            self._start_pos_b[idx] = self._target_pos_b[idx]
            self._start_pos_w[idx] = self._target_pos_w[idx]
            self.goal_pos_b[idx] = self._target_pos_b[idx]
            self.goal_pos_w[idx] = self._target_pos_w[idx]
            self._initialized[idx] = True

        # 轨迹时长（秒）——与 _elapsed 同单位
        dur = torch.empty(len(env_ids), device=self.device).uniform_(*self.cfg.traj_time)
        self._traj_time[env_ids] = torch.clamp(dur, min=self._env.step_dt)
        self._elapsed[env_ids] = 0.0

        # ★把可抓取目标物瞬移到新目标点（并清零速度），之后它是自由刚体。
        #   用"每次重采样才瞬移"而不是"每步写位姿"，否则它永远动不了、也夹不住。
        if self._object is not None:
            pose = torch.zeros(len(env_ids), 7, device=self.device)
            pose[:, :3] = self._target_pos_w[env_ids]
            # ★物体本体落在**静态高度**上（不带 grasp_z_offset）：
            #   指令点可以抬高去瞄准物体上半段，但物体自己必须稳稳放在台面上。
            if self.cfg.object_rest_z is not None:
                pose[:, 2] = self.cfg.object_rest_z
            pose[:, 3] = 1.0  # 单位四元数
            # ★用 *_index 版（新 API 是关键字参数形式）；旧版 write_root_pose_to_sim
            #   在 Isaac Lab 3.0 已标记弃用，会刷 DeprecationWarning。
            self._object.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
            self._object.write_root_velocity_to_sim_index(
                root_velocity=torch.zeros(len(env_ids), 6, device=self.device), env_ids=env_ids
            )

    def _curriculum_alpha(self) -> float:
        """当前课程系数（0 = 物体握在手里，1 = 正常难度：物体在台面采样点上自由放置）。

        三种模式（优先级从高到低）：
          自动进阶（cfg.curriculum_auto）→ 用 self._alpha_cur，由成功率 EMA 驱动
          线性退火（cfg.curriculum_ramp_steps > 0）→ 按环境步数升
          固定档位 → 直接用 cfg.curriculum_alpha
        """
        if self.cfg.curriculum_auto:
            return float(self._alpha_cur)
        a = self.cfg.curriculum_alpha
        if self.cfg.curriculum_ramp_steps > 0:
            a = min(a, float(self._env.common_step_counter) / float(self.cfg.curriculum_ramp_steps))
        return a

    def _update_command(self):
        t = torch.clamp(self._elapsed / self._traj_time, 0.0, 1.0).unsqueeze(-1)
        bp = self.robot.data.root_pos_w.torch
        bq = yaw_quat(self.robot.data.root_quat_w.torch)

        if self.cfg.world_frame:
            # 目标固定在世界里：在世界系插值，再换算回 base 系给观测用
            self.goal_pos_w[:] = torch.lerp(self._start_pos_w, self._target_pos_w, t)
            self.goal_pos_b[:] = quat_apply_inverse(bq, self.goal_pos_w - bp)
        else:
            # 目标跟随机身：在 base 系插值，世界坐标每步重算（同原仓库）
            self.goal_pos_b[:] = torch.lerp(self._start_pos_b, self._target_pos_b, t)
            self.goal_pos_w[:] = bp + quat_apply(bq, self.goal_pos_b)

        self._elapsed += self._env.step_dt

        # ★★课程阶段 (alpha≈0)："物体握在手里" —— 每步都把它摆到当前夹取中心。
        #   为什么必须每步重摆：零动作下手臂自己会缓慢沉降/后移（实测 8 s 内末端后退 6 cm），
        #   只在重采样时摆一次的话，物体立刻被"落"在原地，任务又变回"够物"了（实测 2.6 cm → 8.6 cm）。
        #   握在手里之后，策略面对的就纯粹是"合爪 + 抬起"这两步 —— 正是要 bootstrap 的子技能。
        #   ★用 alpha<1 而不是"alpha 恰好为 0"：阶段二退火一开始 alpha 就是 0.004，
        #     若用严格等于 0 判定，物体立刻脱离手、被手臂沉降落下（实测成功率瞬间归零）。
        #     alpha<1 期间每步把物体摆到"手心 → 采样点"的插值位置（沿插值轴 z 始终贴台面高度）。
        if self._object is not None and self._curriculum_alpha() < self.cfg.held_below:
            a = self._curriculum_alpha()
            pinch_w = self.robot.data.body_pos_w.torch[:, self.ee_body_idx]
            if self._second_body_idx is not None:
                pinch_w = 0.5 * (pinch_w + self.robot.data.body_pos_w.torch[:, self._second_body_idx])
            placed = pinch_w + a * (self._sampled_w - pinch_w)
            if self.cfg.object_rest_z is not None:
                # z 从"手心高度"平滑落到"台面静止高度"
                placed[:, 2] = (1.0 - a) * pinch_w[:, 2] + a * self.cfg.object_rest_z
            lo2 = self._env.scene.env_origins
            placed[:, 0] = torch.clamp(
                placed[:, 0], lo2[:, 0] + self.cfg.curriculum_x_range[0], lo2[:, 0] + self.cfg.curriculum_x_range[1]
            )
            placed[:, 1] = torch.clamp(
                placed[:, 1], lo2[:, 1] + self.cfg.curriculum_y_range[0], lo2[:, 1] + self.cfg.curriculum_y_range[1]
            )
            pose = torch.zeros(self.num_envs, 7, device=self.device)
            pose[:, :3] = placed
            pose[:, 3] = 1.0
            self._object.write_root_pose_to_sim_index(root_pose=pose)
            self._object.write_root_velocity_to_sim_index(
                root_velocity=torch.zeros(self.num_envs, 6, device=self.device)
            )

        # ★★救援：物体掉出台面（或被撞飞）就重新摆位。
        #   实测反复出现的失败模式：物体被手臂/手指碰下台面后，一回合剩下的时间都在
        #   "追一个够不到的目标"（`pick_approach` 从 0.2 掉到 0.005，物体离手 15 cm），
        #   训练信号全废。台面任务里物体必须可用，所以掉了就摆回"课程落点"。
        if self._object is not None and self.cfg.object_rest_z is not None:
            obj_z = self._object.data.root_pos_w.torch[:, 2]
            lost = obj_z < (self.cfg.object_rest_z - 0.06)
            if bool(lost.any()):
                ids = lost.nonzero(as_tuple=False).flatten()
                a = self._curriculum_alpha()
                pinch_w = self.robot.data.body_pos_w.torch[ids, self.ee_body_idx]
                if self._second_body_idx is not None:
                    pinch_w = 0.5 * (
                        pinch_w + self.robot.data.body_pos_w.torch[ids, self._second_body_idx]
                    )
                if a < self.cfg.held_below:
                    placed = pinch_w + a * (self._sampled_w[ids] - pinch_w)
                    placed[:, 2] = (1.0 - a) * pinch_w[:, 2] + a * self.cfg.object_rest_z
                else:
                    placed = pinch_w + a * (self._sampled_w[ids] - pinch_w)
                    placed[:, 2] = self.cfg.object_rest_z
                lo = self._env.scene.env_origins[ids]
                placed[:, 0] = torch.clamp(
                    placed[:, 0], lo[:, 0] + self.cfg.curriculum_x_range[0], lo[:, 0] + self.cfg.curriculum_x_range[1]
                )
                placed[:, 1] = torch.clamp(
                    placed[:, 1], lo[:, 1] + self.cfg.curriculum_y_range[0], lo[:, 1] + self.cfg.curriculum_y_range[1]
                )
                pose = torch.zeros(len(ids), 7, device=self.device)
                pose[:, :3] = placed
                pose[:, 3] = 1.0
                self._object.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
                self._object.write_root_velocity_to_sim_index(
                    root_velocity=torch.zeros(len(ids), 6, device=self.device), env_ids=ids
                )

        # 可选：把目标点位置写到一个真实场景小球上（录像/肉眼检查用）。
        # 用 RigidObject 的批量位姿写入，避免逐 prim 操作 USD（那样 512 环境会很慢）。
        if self._marker is not None:
            pose = torch.zeros(self.num_envs, 7, device=self.device)
            pose[:, :3] = self.goal_pos_w
            pose[:, 3] = 1.0  # 单位四元数
            self._marker.write_root_pose_to_sim(pose)

    def _update_metrics(self):
        # ★夹取中心 = 两指中点（见 cfg.second_body_name 的说明）
        ee_pos_w = self.robot.data.body_pos_w.torch[:, self.ee_body_idx]
        if self._second_body_idx is not None:
            ee_pos_w = 0.5 * (ee_pos_w + self.robot.data.body_pos_w.torch[:, self._second_body_idx])
        # 与原仓库一致：L1 误差（三轴绝对值之和），单位 m
        self.metrics["position_error"] = torch.sum(torch.abs(ee_pos_w - self.goal_pos_w), dim=1)

        # ★拾取任务的指标（TensorBoard 里看 Metrics/ee_goal/object_lift / pick_success）：
        #   object_lift  = 目标物比静止高度高了多少（m）
        #   pick_success = 是否超过判定阈值（0/1，回合均值就是成功率）
        #   ★这是"自己写的指标"，比训练日志里的奖励值可信 —— 但仍然要用独立评估脚本复核。
        if self._object is not None:
            obj_pos_w = self._object.data.root_pos_w.torch
            rest = self.cfg.object_rest_z if self.cfg.object_rest_z is not None else 0.0
            lift = torch.clamp(obj_pos_w[:, 2] - rest, min=0.0)
            # ★"在手上"判据：球心必须离夹取中心足够近，否则"把球打飞"也会被算成成功
            pinch_w = self.robot.data.body_pos_w.torch[:, self.ee_body_idx]
            if self._second_body_idx is not None:
                pinch_w = 0.5 * (pinch_w + self.robot.data.body_pos_w.torch[:, self._second_body_idx])
            d_pinch = torch.norm(obj_pos_w - pinch_w, dim=1)
            self.metrics["object_lift"] = lift
            self.metrics["pick_success"] = (
                (lift > self.cfg.object_success_lift) & (d_pinch < self.cfg.object_success_max_dist)
            ).float()

        # ★★自动进阶课程：用"最近成功率"驱动 alpha（见 cfg.curriculum_auto 的说明）
        #   成功率 > 0.6 → 升一档；< 0.3 → 退一档。这样策略永远不会被丢到
        #   "自己做不到的难度"上持续训练 —— 那会把已学会的技能反向抹掉
        #   （实测：第 1 档 100% 学会后无条件升到 0.4，结果连第 1 档的技能都丢了）。
        if self.cfg.curriculum_auto and "object_lift" in self.metrics:
            # ★用 object_lift（物体离台面高度）而不是 pick_success 当进阶判据：
            #   pick_success 还要求"物体同时在夹取中心 7 cm 内"，而快速抬升时这个距离
            #   会被"指标计算早于本步重摆位"的 1 步延迟污染（实测抬着 0.82 m 时读数只有 0.04），
            #   结果 EMA 永远到不了 0.6 → 课程永远不升档。用高度判据稳得多。
            m = float((self.metrics["object_lift"] > 0.05).float().mean())
            # ★EMA 系数 0.99（≈100 步时间常数）。踩过的坑：原来用 0.999 + "每次评估都重置回 0.5"，
            #   而 200 步内 EMA 从 0.5 最多只能爬到 ~0.59 ⇒ **永远够不到 0.6、课程永远不升档**。
            self._succ_ema = 0.99 * self._succ_ema + 0.01 * m
            self._auto_tick += 1
            if self._auto_tick >= 200:
                self._auto_tick = 0
                if self._succ_ema > 0.6:
                    self._alpha_cur = min(1.0, self._alpha_cur + self.cfg.curriculum_step)
                    print(f"[CURRICULUM] 成功率EMA={self._succ_ema:.2f}>0.6 -> alpha 升到 {self._alpha_cur:.2f}", flush=True)
                    self._succ_ema = 0.5  # 换档后重新统计（只在真的换档时重置）
                elif self._succ_ema < 0.3:
                    self._alpha_cur = max(0.0, self._alpha_cur - self.cfg.curriculum_step)
                    print(f"[CURRICULUM] 成功率EMA={self._succ_ema:.2f}<0.3 -> alpha 退到 {self._alpha_cur:.2f}", flush=True)
                    self._succ_ema = 0.5

    def _set_debug_vis_impl(self, debug_vis: bool):
        # 延迟导入：见模块顶部注释（避免 pxr 在仿真启动前被加载）
        import isaaclab.sim as sim_utils
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

        if debug_vis:
            if not hasattr(self, "goal_visualizer"):
                self.goal_visualizer = VisualizationMarkers(
                    VisualizationMarkersCfg(
                        prim_path="/Visuals/Command/ee_goal",
                        markers={
                            "goal": sim_utils.SphereCfg(
                                radius=0.035,
                                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.9, 0.2)),
                            )
                        },
                    )
                )
            self.goal_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_visualizer"):
                self.goal_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        self.goal_visualizer.visualize(self.goal_pos_w)


@configclass
class EeGoalCommandCfg(CommandTermCfg):
    """末端目标指令的配置。

    ranges 的数值来自 asset_import/go2_piper/probe_ee.py 的实测可达空间标定
    （3000+ 次随机姿态采样）。要改之前建议先重跑那个脚本，别凭感觉填。
    """

    class_type: type = EeGoalCommand

    asset_name: str = "robot"
    """机器人在场景里的名字。"""

    body_name: str = "piper_link6"
    """末端执行器刚体名（Piper 最后一段；本资产用 --merge-joints 转换，夹爪已并入此刚体）。"""

    traj_time: tuple[float, float] = (1.5, 3.0)
    """从上一目标插值到新目标所用的时间范围 [s]。"""

    world_frame: bool = False
    """True = 目标固定在世界里（更难，要求"走过去再够"）；False = 目标跟随机身（同原仓库低层）。"""

    marker_asset_name: str | None = None
    """场景里用于显示目标点的小球资产名（可选）。

    ★为什么需要它：`debug_vis` 用的 VisualizationMarkers 在 headless + 相机录像时
      **不会出现在渲染结果里**（实测画面里找不到任何绿色像素），所以录像看不出目标在哪。
      把目标做成真实场景物体（RigidObject）后，它一定会出现在画面里。
      仅在回放配置里启用（16 个环境，开销可忽略），训练配置保持 None。
    """

    object_grasp_z_offset: float = 0.0
    """★夹爪瞄准点相对"目标物静止中心"的高度偏移 [m]。

    ★为什么需要它（实测教训）：
      手指尖大约比"两指中点"低 5 cm。若让中点瞄准物体**中心**（台面顶上方 5 cm），
      指尖就正好落在台面顶上 —— 于是"手指被挡住"里混进了"指尖蹭到台面"这种假夹持：
      实测 `pick_grip_state` 涨到 0.69（看起来夹得很好），`pick_lift` 却始终 ≈ 0。
      把瞄准点抬到物体中心上方 4 cm，指尖离台面就有 4 cm 余量，
      手指真正夹住的是物体的**上半段**（物体 10 cm 高，中心上方 2~7 cm 都是实的）。
    """

    curriculum_alpha: float = 1.0
    """★课程学习：目标物摆位的插值系数。

    0.0 = 物体直接摆到**当前夹取中心**（= 已经在两指之间，策略只需学"合爪 + 抬起"）
    1.0 = 正常难度（摆到采样盒位置，策略必须自己"对准 → 合爪 → 抬起"）

    ★为什么需要它（13 轮训练的教训）：
      "抬起"是**依存于前两步成功才生效**的结果奖励，属于稀疏信号 ——
      策略得先偶然做对"夹住+抬起"才能拿到第一份梯度。实测 13 轮里
      「靠近」(0.99 学得很好)、「合爪」(0.7~2.5 学得很好)，而 `pick_lift` 始终 0.02 上下。
      把物体（在课程阶段）先摆到指间，等于**把任务拆开**：先让策略体验"合爪→抬起→成功"
      这条因果链，再逐步把物体挪远、把"对准"这一环加回去。
    """

    curriculum_auto: bool = False
    """★自动进阶课程：按"最近成功率"自动调 alpha，而不是无条件加深。

    ★为什么必须这么做（今晚最关键的教训）：
      第 1 档（alpha=0）141 轮就学会"合爪+抬起"（成功率 100%）；
      接着无条件把 alpha 提到 0.4，结果**连第 1 档的技能都丢了**
      （录 alpha=0.15 的像时 0/1500 全败）—— RL 在"自己做不到的难度"上持续训练，
      会把失败动作的概率压掉，等于把学会的技能反向抹掉。
      正确做法：**当前档练稳了才升，做不到就退回去**（成功率 >0.6 升、<0.3 降）。
    """

    curriculum_step: float = 0.1
    """每次自动调整 alpha 的步长。"""

    curriculum_x_range: tuple[float, float] = (0.32, 0.58)
    """课程插值落点允许的 x 范围（环境系）—— 必须落在台面上，否则物体掉地上。"""

    curriculum_y_range: tuple[float, float] = (-0.18, 0.18)
    """课程插值落点允许的 y 范围（环境系）。"""

    held_below: float = 0.05
    """alpha 低于这个值时启用"握在手里"（每步把物体摆到夹取中心附近）。

    ★默认值定为 0.05（原 0.2）—— 今晚最关键的发现：
      自动进阶课程的振荡边界**恰好压在 0.2**，因为 0.2 就是"物体还黏在手上"的阈值。
      alpha≥0.2 时物体变成台面上真正的自由刚体，成功率立刻 93% → 8%
      （哪怕它只离手 1~3 cm）⇒ 策略学到的只是"物体在手上时合爪+抬起"，
      **抓自由刚体这件事还没学会**。阈值压到 0.05 才能从"几乎贴手"就开始练"抓"。

    ★为什么不能一直跟手（实测教训）：`placed = 手心 + alpha·(采样点 − 手心)` 如果**每步**重算，
    物体就随手的移动按比例后退 —— 变成"追不上的移动靶"，手越靠近它跑得越远。
    实测 alpha=0.6 档成功率直接归零（比 alpha=0 和 alpha=1 都难）。
    所以只在 alpha 很小（物体基本就在手上）时才逐步跟随；alpha 稍大就按
    "重采样时一次性摆到 手心 + alpha·(采样点 − 手心)" 放成台面上的自由刚体。
    """

    curriculum_ramp_steps: int = 0
    """>0 时 alpha 从 0 线性升到 curriculum_alpha，跨这么多【环境步】（0 = 不自动升，用固定值）。

    参考量级：512 环境 × 24 步/回合，训练 1000 轮 ≈ 24000 环境步。
    想让"合爪+抬起"先练稳再上难度，可设 40000~60000。
    """

    object_asset_name: str | None = None
    """可选：**可抓取的目标物**（场景里的 RigidObject）名字。

    设置后，每次目标点被重采样时会把该物体瞬移到新目标点（并清零速度），
    之后它是自由刚体（受重力、有碰撞），可以被夹爪夹住并抬起。
    ★这是"拾取"任务与"够物"任务的分界：够物任务里的小球是 kinematic 标记（夹不住），
      拾取任务里的小球必须是真实刚体。
    """

    object_rest_z: float | None = None
    """若设置：目标物总是落在**这个世界系高度**上（= 球半径，即"放在地面上"）。

    ★为什么需要它：默认目标 z 是"相对机身原点"的偏移，机身离地 0.30 m 时，
      想让球落在地面上就得填 z=-0.30 —— 而机器人一走路/一蹲下，机身高度变了，
      球的世界高度就跟着漂了。设了这个字段后，x/y 仍按 base 系盒子采样（即"出现在身前"），
      z 强制为世界系固定值，球就稳稳在地面上。
    """

    object_success_lift: float = 0.04
    """判定"抓起来了"的高度阈值 [m]：目标物升到静止高度以上这么多就算成功（写进 Metrics）。"""

    sample_in_env_frame: bool = False
    """★是否把采样盒解释为【环境坐标系】而不是机身 base 系。

    ★为什么需要这个开关（实测教训）：
      台面是**固定在环境坐标系**里的静态几何体（sim 里没法让它跟着机器人走），
      而"相对 base 系摆物体"意味着：机器人一旦被 push_robot / 外力随机化推走，
      物体就会跟着摆到台面外面、掉到地上 —— 而奖励里"抬起的基准"仍是台面高度，
      于是 `pick_lift` 物理上永远不可能为正（物体在台面下方 14 cm）。
      实测（probe_grasp_trace.py）：两指中点落在世界 x≈0，而机器人自身被推到 x≈−0.455，
      物体 z=0.015（地上）、`pick_lift` 恒为 0 —— 正是这个机制。
      打开后采样值被当成"相对环境原点"的偏移，物体永远落在台面上。
    """

    object_success_max_dist: float = 0.07
    """判定"抓起来了"的**距离**阈值 [m]：球心必须离末端夹取中心这么近。

    ★两个条件必须同时满足（高度 + 在手上），否则"把球打飞"也会被算成成功：
      球飞上天高度够了，但它立刻离开夹取中心 → 不算成功。
      球半径 3 cm，真正夹住时球心离夹取中心约 3~4 cm，7 cm 的门槛不会误伤真抓取。
    """

    second_body_name: str | None = "piper_link8"
    """可选的第二刚体名：与 body_name 一起取**中点**作为夹取中心。

    ★为什么需要：夹爪的两个手指（piper_link7 / piper_link8）在张开时各偏中心线 3 cm，
      单用 link7 作参考点会让瞄准点偏 3 cm（实测地面抓取 1/125 就是这么来的）。
      取中点后无论开合，参考点都在两指中心线上。
    """

    @configclass
    class Ranges:
        """目标点在 base 系下的均匀采样范围（单位 m）。

        z 是【相对机身原点的竖直高度】：机身原点默认离地 0.30 m，
        所以 z=0.50 约等于离地 0.80 m（标称姿态末端所在高度）。
        """

        pos_x: tuple[float, float] = (-0.05, 0.32)
        """前后（前方为正）。"""

        pos_y: tuple[float, float] = (-0.22, 0.22)
        """左右。"""

        pos_z: tuple[float, float] = (0.22, 0.58)
        """竖直（世界竖直，不随机身俯仰倾斜）。"""

    ranges: Ranges = Ranges()

    debug_vis: bool = True

# =============================================================================
# ★移植自 ZoreAnuj/go2-piper-quiet-goto 的"丝滑"手段 1：slew-limited 速度指令
#   该项目 README 明确写：Smoothness = action-rate/jerk penalties + **slew-limited
#   velocity commands** + softened goal P-law。
#   原理：Isaac Lab 原生命令在重采样瞬间把目标从 A 直接跳到 B（阶跃），策略被迫
#   学"急反应"，转弯就表现为猛扭头/猛蹬地。这里给指令加一个变化速率上限，
#   重采样后指令在约 1~2 秒内平滑过渡，策略学到的是"跟着指令匀速加力"。
# =============================================================================
class SlewRateVelocityCommand(UniformVelocityCommand):
    """在基类算完目标指令后，限制每步的变化量（默认：线 1.0 m/s²，角 1.5 rad/s²）。

    ★设计要点（踩过的坑，务必保留）：
      基类的 `_update_command` 只重写【偏航】分量（heading 模式）和【站立环境】，
      **不会重写线速度**。因此若直接读 `self.vel_command_b` 当作"本轮目标"，
      读到的其实是"我上一步限速后的输出" ⇒ 每步增量 = 目标 − 自己 = 0，
      指令会**永远卡在第一步的增量**上（实测卡在 0.02 m/s 不动），
      策略于是学成"跟踪一个恒为零的指令"：训练里速度跟踪奖励满分，实机站着不动。
      ⇒ 必须**单独保存原始目标** `_raw_target`，限速率只影响输出、不回写目标。
    """

    SLEW_LIN_DEFAULT = 1.0   # m/s²
    SLEW_ANG_DEFAULT = 1.5   # rad/s²

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._raw_target = self.vel_command_b.clone()      # 原始目标（不受限速影响）
        self._prev_command = self.vel_command_b.clone()    # 上一步实际输出
        rate_lin = float(getattr(cfg, "slew_rate_lin", self.SLEW_LIN_DEFAULT))
        rate_ang = float(getattr(cfg, "slew_rate_ang", self.SLEW_ANG_DEFAULT))
        self._max_delta = torch.tensor(
            [rate_lin * env.step_dt, rate_lin * env.step_dt, rate_ang * env.step_dt],
            device=self.device, dtype=self.vel_command_b.dtype)

    def _resample_command(self, env_ids):
        """新采样到指令时，更新【原始目标】。"""
        super()._resample_command(env_ids)
        self._raw_target[env_ids] = self.vel_command_b[env_ids]

    def _update_command(self):
        super()._update_command()   # 基类更新 heading 偏航与站立环境
        # 把基类刚写入的分量（偏航 / 站立归零）吸收为新的原始目标分量
        if self.cfg.heading_command:
            ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
            if len(ids) > 0:
                self._raw_target[ids, 2] = self.vel_command_b[ids, 2]
        stand = self.is_standing_env.nonzero(as_tuple=False).flatten()
        if len(stand) > 0:
            self._raw_target[stand, :] = 0.0
        # 限速率：输出朝原始目标靠拢，每步最多动 _max_delta
        delta = torch.clamp(self._raw_target - self._prev_command, -self._max_delta, self._max_delta)
        self._prev_command = self._prev_command + delta
        self.vel_command_b = self._prev_command.clone()

@configclass
class SlewRateVelocityCommandCfg(UniformVelocityCommandCfg):
    """限速率速度指令的配置（与原生 UniformVelocityCommandCfg 同字段，多两个速率）。"""

    class_type: type = SlewRateVelocityCommand
    slew_rate_lin: float = 1.0   # m/s² 线速度指令最大变化率
    slew_rate_ang: float = 1.5   # rad/s² 角速度指令最大变化率
