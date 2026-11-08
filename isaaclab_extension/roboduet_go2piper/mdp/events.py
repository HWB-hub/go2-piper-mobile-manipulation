"""事件（域随机化）：把作者 `_randomize_dof_props` / `_randomize_rigid_body_props` 落地。

作者实际生效的域随机化（auto_train + wtw_config 之后的最终值）：
    摩擦系数  randomize_friction=True,  friction_range=[0.1, 3.0]      —— 启动时一次（randomize_rigids_after_start=False）
    弹性系数  randomize_restitution=True, restitution_range=[0.0, 0.4]  —— 同上
    机身质量  randomize_base_mass=True,  added_mass_range=[-2.0, 2.0]   —— 每次 reset 加一个质量
    电机强度  randomize_motor_strength=True, [0.9, 1.1]                 —— 每次 reset（力矩上限缩放）
    电机零位  randomize_motor_offset=True, [-0.02, 0.02]                —— 每次 reset（动作项里加偏置）
    重力      randomize_gravity=True, gravity_range=[-1,1]（且只用 env0 的向量当全局重力）
              → 本移植版【未实现】（Isaac Lab 里重力是全局参数、逐环境改不了，且幅值只有 ±1 m/s² ≈ ±6°）
    质心偏移  wtw_config 关掉了（randomize_com_displacement=False）→ 不做
    Kp/Kd 因子、推力扰动、地面摩擦：作者都关了 → 不做
"""

from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg


def randomize_motor_strength(env, env_ids: torch.Tensor, strength_range: tuple[float, float],
                             actuator_names: tuple[str, ...] = ("legs",), asset_cfg: SceneEntityCfg = None):
    """把电机的力矩上限按 U(0.9,1.1) 缩放（作者 motor_strengths 的作用）。"""
    if env_ids is None or len(env_ids) == 0:
        return
    asset = env.scene[asset_cfg.name if asset_cfg is not None else "robot"]
    lo, hi = strength_range
    for name in actuator_names:
        act = asset.actuators.get(name)
        if act is None:
            continue
        base = getattr(env, f"_rd_base_effort_{name}", None)
        if base is None:
            base = act.effort_limit.clone()
            setattr(env, f"_rd_base_effort_{name}", base)
        n = act.effort_limit.shape[-1]
        if act.effort_limit.shape[0] != env.num_envs:
            # 不支持逐环境：扩展成 (N, J)（第一次调用时做）
            act.effort_limit = base.expand(env.num_envs, n).clone()
        s = torch.empty(len(env_ids), 1, device=env.device).uniform_(lo, hi)
        act.effort_limit[env_ids] = base[0, :n] * s
    # 记录给 critic / 日志（作者 priv_observe_motor_strength 是关的，这里只存不用）
    if not hasattr(env.rd, "motor_strength"):
        env.rd.motor_strength = torch.ones(env.num_envs, 1, device=env.device)
    env.rd.motor_strength[env_ids] = s


def randomize_motor_offset(env, env_ids: torch.Tensor, offset_range: tuple[float, float]):
    """每环境每关节一个零位偏置（作用于被驱动的 18 个关节）。"""
    if env_ids is None or len(env_ids) == 0:
        return
    if not hasattr(env.rd, "motor_offset"):
        env.rd.motor_offset = torch.zeros(env.num_envs, env.rd_driven_joint_ids.numel(), device=env.device)
    n = env.rd.motor_offset.shape[1]
    env.rd.motor_offset[env_ids] = torch.empty(len(env_ids), n, device=env.device).uniform_(*offset_range)


def cache_material_properties(env, env_ids=None):
    """把仿真里实际的摩擦/弹性读回来，喂给 critic 的 privileged obs。

    Isaac Lab 的 randomize_rigid_body_material 是按"材质桶"随机分配的，
    直接读 physics view 才是真实值（作者是逐环境写 shape props）。
    """
    try:
        import warp as wp

        asset = env.scene["robot"]
        props = wp.to_torch(asset.root_view.get_material_properties())  # (N, shapes, 3)
        static = props[..., 0].mean(dim=1).to(env.device)
        rest = props[..., 2].mean(dim=1).to(env.device)
        env.rd.friction = static
        env.rd.restitution = rest
    except Exception as exc:  # noqa: BLE001
        if not hasattr(env.rd, "friction"):
            env.rd.friction = torch.full((env.num_envs,), 0.5, device=env.device)
            env.rd.restitution = torch.full((env.num_envs,), 0.0, device=env.device)
        if not getattr(env, "_rd_mat_warned", False):
            print(f"[roboduet] 材质属性读取失败（privileged obs 用默认值）: {exc}")
            env._rd_mat_warned = True
