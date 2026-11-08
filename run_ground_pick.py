"""Go2 v8 approach with continuous crouch and feedback-controlled physical grasp."""
import argparse
import json
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--run-name", required=True)
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=861)
parser.add_argument("--distance", type=float, default=1.25)
parser.add_argument("--lateral", type=float, default=0.)
parser.add_argument("--approach-time", type=float, default=4.0)
parser.add_argument("--stance-distance", type=float, default=.43)
parser.add_argument("--grasp-height", type=float, default=.057)
parser.add_argument("--video", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = args.video
app = AppLauncher(args).app

import gymnasium as gym
import torch
import isaaclab_tasks
from isaaclab.utils.math import quat_apply_inverse, yaw_quat
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.manager_based.locomanipulation.go2_piper.go2_piper_pick_env_cfg import PickSceneCfg, OBJECT_SIZE

from cooperative_task import tensor
from natural_task import natural_config
from spin_task import SpinGoal


class ApproachGoal(SpinGoal):
    def update_targets(self):
        if not hasattr(self, "anchor"):
            return
        self.spin_mask.fill_(True)
        self.anchor[:] = tensor(self._env.scene.env_origins)
        elapsed = tensor(self._env.episode_length_buf).float() * self._env.step_dt
        u = (elapsed / (args.approach_time - 1.)).clamp(0., 1.)
        blend = u.pow(3) * (10. + u * (-15. + 6. * u))
        self.anchor[:, 0] += .20 + (args.distance - args.stance_distance - .20) * blend
        self.anchor[:, 1] += args.lateral * blend
        self.goal_pos_w[:] = self.anchor
        self.goal_pos_w[:, 0] += .35
        self.goal_pos_w[:, 2] += .64
        robot = self.robot
        self.goal_pos_b[:] = quat_apply_inverse(
            yaw_quat(tensor(robot.data.root_quat_w)), self.goal_pos_w - tensor(robot.data.root_pos_w)
        )


def ease(x):
    x = min(1., max(0., x))
    return x**3 * (10. + x * (-15. + 6. * x))


def main():
    root = Path(__file__).resolve().parent
    out = root / "runs" / args.run_name
    out.mkdir(parents=True, exist_ok=True)
    cfg = natural_config(args.num_envs, args.device or "cuda:0", args.seed, "adaptive", "spin", False, True)
    cfg.commands.ee_goal.class_type = ApproachGoal
    cfg.commands.ee_goal.debug_vis = False
    cfg.scene.pick_object = PickSceneCfg().pick_object.copy()
    cfg.scene.pick_object.init_state.pos = (args.distance, args.lateral, OBJECT_SIZE[2] / 2)
    cfg.episode_length_s = 16.
    raw = gym.make("Isaac-Go2Piper-Walk-v0", cfg=cfg, render_mode="rgb_array" if args.video else None)
    sim = raw.unwrapped
    origin = tensor(sim.scene.env_origins)
    if args.video:
        for recorder in (sim.video_recorder.cfg, sim.video_recorder._capture.cfg):
            recorder.eye = tuple((origin[0] + origin.new_tensor([2.3, -2.2, 1.35])).tolist())
            recorder.lookat = tuple((origin[0] + origin.new_tensor([.72, 0., .28])).tolist())
        raw = gym.wrappers.RecordVideo(raw, str(out / "video"), step_trigger=lambda step: step == 0,
                                      video_length=749, name_prefix="ground_pick", disable_logger=True)
    env = RslRlVecEnvWrapper(raw, clip_actions=None)
    robot = sim.scene["robot"]
    obj = sim.scene["pick_object"]
    obs, _ = env.reset()
    with (root / "runs/stability_v8/policy_jit.pt").open("rb") as stream:
        policy = torch.jit.load(stream, map_location=sim.device).eval()
    term = sim.action_manager.get_term("joint_pos")
    ids = list(term._joint_ids)
    names = list(robot.data.joint_names)
    arm_ids = [names.index(f"piper_joint{i}") for i in range(1, 7)]
    fingers = [names.index(f"piper_joint{i}") for i in (7, 8)]
    finger_links = [robot.data.body_names.index(f"piper_link{i}") for i in (7, 8)]
    arm_cols = [ids.index(i) for i in arm_ids]
    leg_cols = [i for i, joint in enumerate(ids) if not names[joint].startswith("piper")]
    standing = tensor(robot.data.default_joint_pos)[:, ids].clone()
    scale = torch.broadcast_to(torch.as_tensor(term._scale, device=sim.device), standing.shape)
    crouched = standing.clone()
    for column in leg_cols:
        name = names[ids[column]]
        if "thigh" in name:
            crouched[:, column] += .38
        if "calf" in name:
            crouched[:, column] -= .76
    limits = tensor(robot.data.soft_joint_pos_limits)[:, arm_ids]
    eye = torch.eye(6, device=sim.device)
    columns = [robot.num_base_dofs + joint for joint in arm_ids]
    start_obj = tensor(obj.data.root_pos_w).clone()
    held = torch.zeros(args.num_envs, device=sim.device)
    alive = torch.ones(args.num_envs, dtype=torch.bool, device=sim.device)
    trace = []
    arm_target = standing[:, arm_cols].clone()
    approach_steps = round(args.approach_time / sim.step_dt)
    for step in range(750):
        elapsed = step * sim.step_dt
        t = (step - approach_steps) * sim.step_dt + 4.4
        q = tensor(robot.data.joint_pos)
        links = tensor(robot.data.body_link_pos_w)[:, finger_links]
        midpoint = links.mean(1)
        object_pos = tensor(obj.data.root_pos_w)
        if step < approach_steps:
            action = policy(obs["policy"])
        else:
            if step == approach_steps:
                arm_target = q[:, arm_ids].clone()
                arm_start = midpoint.clone()
                handover_pose = q[:, ids].clone()
            legs = torch.lerp(standing, crouched, ease((t - 5.3) / 1.45))
            targets = torch.lerp(handover_pose, legs, ease((t - 4.4) / .9))
            action = (targets - term._offset) / scale
            pregrasp = start_obj.clone()
            pregrasp[:, 2] = origin[:, 2] + .18
            grasp = start_obj.clone()
            grasp[:, 2] = origin[:, 2] + args.grasp_height
            goal = torch.lerp(arm_start, pregrasp, ease((t - 5.15) / 1.75))
            goal = torch.lerp(goal, grasp, ease((t - 6.9) / 2.0))
            goal[:, 2] += .19 * ease((t - 10.) / 2.4)
            jacobian = tensor(robot.data.body_link_jacobian_w)[:, finger_links][:, :, :, columns].mean(1)
            span = torch.nn.functional.normalize(links[:, 0] - links[:, 1], dim=-1)
            lateral = torch.zeros_like(span)
            lateral[:, 1] = torch.where(span[:, 1] < 0., -1., 1.)
            omega = torch.linalg.cross(span, lateral)
            # Ignore rotation about the finger separation axis.
            angular = jacobian[:, 3:]
            angular = angular - span[:, :, None] * (span[:, None, :] @ angular)
            jacobian = torch.cat((jacobian[:, :3], .12 * angular), dim=1)
            error = torch.cat(((goal - midpoint).clamp(-.08, .08), .12 * omega), dim=1)
            delta = (jacobian.transpose(1, 2) @ torch.linalg.solve(
                jacobian @ jacobian.transpose(1, 2) + .0025 * eye, error[..., None]
            )).squeeze(-1)
            proposed = q[:, arm_ids] + delta.clamp(-.18, .18)
            arm_target += (proposed - arm_target).clamp(-.022, .022)
            arm_target = torch.maximum(limits[:, :, 0], torch.minimum(limits[:, :, 1], arm_target))
            action[:, arm_cols] = (arm_target - term._offset[:, arm_cols]) / scale[:, arm_cols]
        opening = .029 * (1. - ease((t - 8.95) / 1.))
        finger_target = q.new_tensor([opening, -opening]).expand(args.num_envs, -1).contiguous()
        robot.set_joint_position_target_index(target=finger_target, joint_ids=fingers)
        if step % 10 == 0:
            trace.append({"time": elapsed, "base": (tensor(robot.data.root_pos_w)-origin).tolist(),
                          "object": (object_pos-origin).tolist(), "pinch": (midpoint-origin).tolist(),
                          "fingers": q[:, fingers].tolist(), "arm": q[:, arm_ids].tolist()})
        lifted = (object_pos[:, 2] - start_obj[:, 2] > .10) & ((object_pos - midpoint).norm(dim=-1) < .07)
        held = torch.where(lifted & alive, held + sim.step_dt, 0.)
        obs, _, done, _ = env.step(action)
        alive &= ~done.bool()
    result = {"episodes": args.num_envs, "distance": args.distance, "lateral": args.lateral,
              "seed": args.seed, "success": (held >= 1.).tolist(),
              "final_hold_s": held.tolist(), "alive": alive.tolist(), "trace": trace}
    (out / "evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("GROUND_PICK", result["success"], result["final_hold_s"], flush=True)
    env.close()


try:
    with torch.inference_mode():
        main()
except Exception:
    traceback.print_exc()
    raise
finally:
    app.close()
