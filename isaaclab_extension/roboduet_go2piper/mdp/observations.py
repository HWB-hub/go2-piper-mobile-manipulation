"""观测项：把作者 get_dog_observations()(56) / get_arm_observations()(20) 精确搬过来。

拼接顺序（必须与作者一致，历史窗口也是按这个顺序展平的）：
    dog(56) = projected_gravity(3)
              + 腿关节角偏差(12) + 腿关节速度×0.05(12) + 上一帧腿动作(12)
              + 指令 5 维(×[2,2,0.25,1,1]) + 臂指令 6 维(阶段 1 时为 0)
              + roll(1) + pitch(1) + 步态时钟(4)
    arm(20) = 臂关节角偏差(6) + 上一帧臂动作(6) + 臂 6D 指令(6) + roll(1) + pitch(1)
    dog_critic(2)  = 摩擦、弹性（归一化）
    arm_critic(9)  = 摩擦、弹性 + 当前末端 lpy(3) + 末端原始姿态(偏航对齐系, 4)
"""

from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat

from . import arm_kinematics as ak

OBS_SCALES = {"dof_pos": 1.0, "dof_vel": 0.05}
CMD_SCALE_DOG = (2.0, 2.0, 0.25, 1.0, 1.0)
CLIP_ACTIONS = 10.0


def _robot(env):
    return env.scene["robot"]


def _joint_ids(env, kind: str) -> torch.Tensor:
    """惰性解析关节下标（观测管理器的"探维度"调用发生在 env 子类初始化之前，不能用 env.rd_*）。"""
    attr = f"rd_{kind}_joint_ids"
    ids = getattr(env, attr, None)
    if ids is None:
        robot = env.scene["robot"]
        if kind == "leg":
            ids = [i for i, n in enumerate(robot.joint_names) if n.endswith(("_hip_joint", "_thigh_joint", "_calf_joint"))]
        else:
            ids = [i for i, n in enumerate(robot.joint_names) if n.startswith("piper_joint") and n[-1] in "123456"]
        ids = torch.tensor(ids, device=env.device)
        setattr(env, attr, ids)
    return ids


def base_roll_pitch(env):
    """机身 roll / pitch（标准 ZYX 欧拉角，和作者 quaternion_to_rpy 一致）。"""
    q = _robot(env).data.root_quat_w.torch
    roll, pitch, _ = euler_xyz_from_quat(q)
    return roll, pitch


def leg_dof_pos_rel(env, asset_cfg: SceneEntityCfg = None) -> torch.Tensor:
    robot = _robot(env)
    ids = _joint_ids(env, "leg")
    return (robot.data.joint_pos.torch[:, ids] - robot.data.default_joint_pos.torch[:, ids]) * OBS_SCALES["dof_pos"]


def leg_dof_vel(env, asset_cfg: SceneEntityCfg = None) -> torch.Tensor:
    robot = _robot(env)
    return robot.data.joint_vel.torch[:, _joint_ids(env, "leg")] * OBS_SCALES["dof_vel"]


def arm_dof_pos_rel(env, asset_cfg: SceneEntityCfg = None) -> torch.Tensor:
    robot = _robot(env)
    ids = _joint_ids(env, "arm")
    return (robot.data.joint_pos.torch[:, ids] - robot.data.default_joint_pos.torch[:, ids]) * OBS_SCALES["dof_pos"]


def last_action_split(env, start: int, end: int) -> torch.Tensor:
    """上一帧【网络原始输出】（已被 clip 到 ±10），按 dog/arm 切片。"""
    a = getattr(env, "rd", None)
    if a is None:
        return torch.zeros(env.num_envs, end - start, device=env.device)
    return env.rd.raw_actions[:, start:end]


def dog_commands(env) -> torch.Tensor:
    vel = env.command_manager.get_term("velocity").command
    bp = env.command_manager.get_term("body_pose").command
    scale = torch.tensor(CMD_SCALE_DOG, device=vel.device)
    return torch.cat([vel * scale[:3], bp * scale[3:]], dim=-1)


def arm_commands_obs(env) -> torch.Tensor:
    """臂 6D 指令；阶段 1（臂未启用）时为全 0（作者在 switch 关闭时置 0）。"""
    if not getattr(env, "arm_enabled", False):
        return torch.zeros(env.num_envs, 6, device=env.device)
    return env.command_manager.get_term("arm_pose").command


def roll_obs(env) -> torch.Tensor:
    roll, _ = base_roll_pitch(env)
    return roll.unsqueeze(-1)


def pitch_obs(env) -> torch.Tensor:
    _, pitch = base_roll_pitch(env)
    return pitch.unsqueeze(-1)


def clock_inputs(env) -> torch.Tensor:
    rd = getattr(env, "rd", None)
    if rd is None:
        return torch.zeros(env.num_envs, 4, device=env.device)
    return rd.clock_inputs


# ---- privileged（critic）----
def friction_restitution(env) -> torch.Tensor:
    """作者的 get_scale_shift([0,1])：value*2 - 1。"""
    rd = getattr(env, "rd", None)
    if rd is None or getattr(rd, "friction", None) is None:
        return torch.zeros(env.num_envs, 2, device=env.device)
    return torch.stack([rd.friction * 2.0 - 1.0, rd.restitution * 2.0 - 1.0], dim=-1)


def arm_privileged(env) -> torch.Tensor:
    if getattr(env, "rd", None) is None:
        return torch.zeros(env.num_envs, 9, device=env.device)
    lpy = ak.lpy_in_base(env)
    q_ee = ak.ee_quat_in_base_yaw(env)
    return torch.cat([friction_restitution(env), lpy, q_ee], dim=-1)
