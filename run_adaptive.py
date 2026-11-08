"""Train/evaluate learned Go2+Piper curved locomotion and adaptive posture."""
from __future__ import annotations
import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["train", "eval"], default="train")
parser.add_argument("--profile", choices=["curves", "adaptive", "dynamic"], default="adaptive")
parser.add_argument("--pattern", choices=["combined", "spin", "transition", "curves", "serpentine", "walk_run"], default="combined")
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--iterations", type=int, default=400)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--run_name", required=True)
parser.add_argument("--seed", type=int, default=81)
parser.add_argument("--video", action="store_true")
parser.add_argument("--legacy", action="store_true")
parser.add_argument("--gait_recipe", choices=["natural", "baseline", "stability", "stability_transition", "stability_balanced", "stability_recovery", "stability_guard", "stability_support", "stability_capped"], default="natural")
parser.add_argument("--learning_rate", type=float, default=1.0e-4)
parser.add_argument("--duration", type=float, default=38.)
parser.add_argument("--trace_env", type=int, default=0, help="Environment index for diagnostic telemetry; videos use index 0.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
assert 0 <= args.trace_env < args.num_envs
assert not args.video or args.trace_env == 0
if args.video:
    args.enable_cameras = True
app = AppLauncher(args).app

import gymnasium as gym
import numpy as np
import torch
import isaaclab_tasks
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry
from isaaclab.utils.math import euler_xyz_from_quat
from rsl_rl.models import MLPModel
from rsl_rl.runners import OnPolicyRunner
from cooperative_task import tcp_world, tensor
from adaptive_task import adaptive_config, feet_state
from natural_task import natural_config
from gait_reference import equivalent_speed

torch.set_num_threads(4)
root = Path(__file__).resolve().parent
out = root / "runs" / args.run_name
out.mkdir(parents=True, exist_ok=True)
(out / "invocation.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
cfg = natural_config(args.num_envs, args.device or "cuda:0", args.seed, args.profile,
                      args.pattern, args.video, args.mode == "eval", args.legacy, args.gait_recipe)
cfg.episode_length_s = args.duration
(out / "effective_config.json").write_text(json.dumps(cfg.to_dict(), indent=2, default=str), encoding="utf-8")
snapshot = out / "source_snapshot"
snapshot.mkdir(exist_ok=True)
for name in ("run_adaptive.py", "adaptive_task.py", "moving_task.py", "cooperative_task.py", "demo_trails.py", "spin_task.py", "natural_task.py", "gait_reference.py"):
    (snapshot / name).write_bytes((root / name).read_bytes())
raw = gym.make("Isaac-Go2Piper-Walk-v0", cfg=cfg, render_mode="rgb_array" if args.video else None)
if args.video:
    recording_env = raw.unwrapped
    origin = tensor(recording_env.scene.env_origins)[0].tolist()
    eye_world = tuple(origin[i] + (3.0, -4.0, 2.2)[i] for i in range(3))
    target_world = tuple(origin[i] + (.8, 0., .5)[i] for i in range(3))
    for rc in (recording_env.video_recorder.cfg, recording_env.video_recorder._capture.cfg):
        rc.eye, rc.lookat = eye_world, target_world
    import omni.usd
    from pxr import UsdGeom
    stage = omni.usd.get_context().get_stage()
    for index in range(1, args.num_envs):
        UsdGeom.Imageable(stage.GetPrimAtPath(f"/World/envs/env_{index}")).MakeInvisible()
    UsdGeom.Imageable(stage.GetPrimAtPath("/World/envs/env_0")).MakeVisible()
    raw = gym.wrappers.RecordVideo(raw, str(out / "video"), step_trigger=lambda step: step == 0,
                                  video_length=int(args.duration / .02), name_prefix="adaptive_gait", disable_logger=True)
env = RslRlVecEnvWrapper(raw, clip_actions=None)
agent = load_cfg_from_registry("Isaac-Go2Piper-Walk-v0", "rsl_rl_cfg_entry_point").to_dict()
allowed = set(inspect.signature(MLPModel.__init__).parameters) | {"class_name"}
for key in ("actor", "critic"):
    agent[key] = {k:v for k,v in agent[key].items() if k in allowed}
agent["algorithm"]["learning_rate"] = args.learning_rate
agent["algorithm"]["schedule"] = "fixed"
agent["algorithm"]["entropy_coef"] = .003
agent["save_interval"] = 100
agent["logger"] = "tensorboard"
agent["obs_groups"] = {"actor": ["policy"], "critic": ["policy"]}
runner = OnPolicyRunner(env, agent, log_dir=str(out) if args.mode == "train" else None, device=cfg.sim.device)
loaded = torch.load(args.checkpoint, map_location=cfg.sim.device, weights_only=False)
expected = 84 if args.legacy else 92
for kind in ("actor", "critic"):
    state = loaded[f"{kind}_state_dict"]
    w = state["mlp.0.weight"]
    if w.shape[1] < expected:
        state["mlp.0.weight"] = torch.cat((w, torch.zeros(w.shape[0], expected-w.shape[1], device=w.device)), dim=1)
    elif w.shape[1] != expected:
        raise ValueError(f"Unsupported checkpoint observation dimension {w.shape[1]} vs {expected}")
runner.alg.load(loaded, {"actor":True,"critic":True,"optimizer":False,"iteration":False}, strict=True)
print(f"[ADAPTIVE] {args.mode} {args.profile} {args.pattern} {args.num_envs} environments", flush=True)

if args.mode == "train":
    runner.learn(args.iterations, init_at_random_ep_len=False)
    runner.save(str(out / "model_final.pt"))
    exported = torch.jit.script(runner.alg.get_policy().as_jit().cpu())
    with (out / "policy_jit.pt").open("wb") as stream:
        torch.jit.save(exported, stream)
    print(f"[TRAINING_COMPLETE] {out}", flush=True)
else:
    policy = runner.get_inference_policy(device=cfg.sim.device)
    # Network width changes consume different RNG counts at construction. Reseed here
    # so baseline and new model receive identical reset states and target trajectories.
    torch.manual_seed(args.seed + 100000)
    np.random.seed(args.seed + 100000)
    obs, _ = env.reset()
    u = env.unwrapped
    robot, goal = u.scene["robot"], u.command_manager.get_term("ee_goal")
    cmd = u.command_manager.get_term("base_velocity")
    spin_flags=goal.spin_mask.clone()
    spin_directions=goal.direction.clone()
    reset_fingerprints = {}
    for field in ("root_pos_w","root_quat_w","root_lin_vel_w","root_ang_vel_w","joint_pos","joint_vel"):
        values = tensor(getattr(robot.data,field)).detach().cpu().numpy()
        reset_fingerprints[field] = hashlib.sha256(values.tobytes()).hexdigest()
    n, dt = args.num_envs, u.step_dt
    active = torch.ones(n, dtype=torch.bool, device=u.device)
    falls = torch.zeros_like(active)
    previous_base = tensor(robot.data.root_pos_w).clone()
    previous_action = torch.zeros(n, 18, device=u.device)
    previous_delta = previous_action.clone()
    previous_command = cmd.command.clone()
    max_command_change = torch.zeros(3,device=u.device)
    max_posture_change = torch.zeros(3,device=u.device)
    max_phase_step = 0.
    previous_phase = cmd.phase.clone() if hasattr(cmd,"phase") else None
    previous_posture = cmd.posture.clone() if hasattr(cmd,"posture") else None
    in_band = torch.zeros(n, device=u.device)
    path = torch.zeros_like(in_band)
    samples = torch.zeros_like(in_band)
    records, trace, gait_records = [], [], []
    spin_distance_max=torch.zeros(n,device=u.device)
    spin_distance_sq=torch.zeros(n,device=u.device)
    spin_samples=torch.zeros(n,device=u.device)
    rotation=torch.zeros(n,device=u.device)
    previous_yaw=None
    spin_yaw_error_sq=torch.zeros(n,device=u.device)
    camera_focus = tensor(robot.data.root_pos_w)[0].clone()
    params = [{"start_w":goal.start_w[i].tolist(),"amplitude":goal.amplitude[i].tolist(),
               "phase":goal.phase[i].tolist(),"omega":goal.omega[i,0].item(),"kind":goal.kind[i].item(),
               "speed":goal.speed[i].item(),"modulation":goal.modulation[i].item()} for i in range(n)]
    for i,item in enumerate(params):
        item.update(spin=spin_flags[i].item(),transition=goal.transition_mask[i].item(),direction=spin_directions[i].item(),
                    origin=tensor(u.scene.env_origins)[i].tolist())
    if args.video:
        trails = None
    with torch.inference_mode():
        for step in range(u.max_episode_length):
            t = step * dt
            base = tensor(robot.data.root_pos_w)
            tcp = tcp_world(u)
            error = (tcp - goal.goal_pos_w).norm(dim=-1)
            r, p, yaw = euler_xyz_from_quat(tensor(robot.data.root_quat_w))
            rpy = torch.stack((r, p, yaw), -1)
            rpy = torch.atan2(torch.sin(rpy), torch.cos(rpy))
            spin_window=(t>=12.) and (t<26.)
            if spin_window:
                selected=active & spin_flags
                drift=(base-goal.anchor)[:,:2].norm(dim=-1)
                spin_distance_max=torch.maximum(spin_distance_max,torch.where(selected,drift,0.))
                spin_distance_sq+=torch.where(selected,drift.square(),0.)
                spin_samples+=selected.float()
                yaw_error=torch.atan2(torch.sin(yaw-goal.heading),torch.cos(yaw-goal.heading))
                spin_yaw_error_sq+=torch.where(selected,yaw_error.square(),0.)
                if previous_yaw is not None:
                    delta_yaw=torch.atan2(torch.sin(yaw-previous_yaw),torch.cos(yaw-previous_yaw))
                    rotation+=torch.where(selected,delta_yaw*spin_directions,0.)
            previous_yaw=yaw.clone()
            feet, feet_vel, contacts = feet_state(u)
            actions = policy(obs)
            if active.any():
                max_command_change = torch.maximum(max_command_change,(cmd.command-previous_command)[active].abs().max(dim=0).values)
                if previous_phase is not None and step>0:
                    max_phase_step = max(max_phase_step,torch.remainder(cmd.phase-previous_phase,1.)[active].max().item())
                    max_posture_change = torch.maximum(max_posture_change,(cmd.posture-previous_posture)[active].abs().max(dim=0).values)
            previous_command.copy_(cmd.command)
            if previous_phase is not None:
                previous_phase.copy_(cmd.phase)
                previous_posture.copy_(cmd.posture)
            delta = actions - previous_action
            jerk = delta - previous_delta
            height = base[:,2] - tensor(u.scene.env_origins)[:,2]
            velocity = tensor(robot.data.root_lin_vel_w)
            slip = (feet_vel[:, :, :2].norm(dim=-1) * contacts).sum(-1) / contacts.sum(-1).clamp(min=1)
            row = torch.cat((error[:,None],height[:,None],rpy,velocity[:,:2],
                              delta.square().mean(-1,keepdim=True),jerk.square().mean(-1,keepdim=True),
                              slip[:,None],contacts.float(),goal.goal_pos_w[:,2:3]-tensor(u.scene.env_origins)[:,2:3]),-1)
            if t >= 4.:
                in_band += (active & (error < .08)).float()
                samples += active.float()
                path += torch.where(active, (base-previous_base)[:,:2].norm(dim=-1), 0.)
                records.append(torch.where(active[:,None], row, float("nan")).clone())
                sensor = u.scene.sensors['contact_forces'].data
                gait_row = torch.cat((feet[:,:,2]-tensor(u.scene.env_origins)[:,None,2]-.022,
                    contacts.float(), tensor(sensor.current_contact_time)[:,u._adaptive_sensor_ids],
                    tensor(sensor.current_air_time)[:,u._adaptive_sensor_ids], feet_vel.norm(dim=-1),
                    tensor(robot.data.root_ang_vel_w)[:,2:3], cmd.command[:,2:3], cmd.phase[:,None],
                    cmd.frequency[:,None], equivalent_speed(cmd.command)[:,None]),dim=-1)
                if args.pattern == "walk_run":
                    instant_contact = tensor(sensor.net_forces_w)[:, u._adaptive_sensor_ids].norm(dim=-1) > 3.
                    gait_row = torch.cat((gait_row, instant_contact.float()), dim=-1)
                gait_records.append(torch.where(active[:,None], gait_row, float('nan')).clone())
            if step % 5 == 0:
                j = args.trace_env
                trace.append({"time_s":t,"active":active[j].item(),"target_world":goal.goal_pos_w[j].tolist(),
                              "tcp_world":tcp[j].tolist(),"base_world":base[j].tolist(),"rpy_rad":rpy[j].tolist(),
                              "anchor_world":goal.anchor[j].tolist(),"heading_reference_rad":goal.heading[j].item(),
                              "spin_task":goal.spin_mask[j].item(),"transition_task":goal.transition_mask[j].item(),
                              "base_velocity_world":velocity[j].tolist(),"velocity_command":cmd.command[j].tolist(),
                              "error_m":error[j].item(),"feet_world":feet[j].tolist(),"foot_contacts":contacts[j].tolist(),
                              "posture_preference":cmd.posture[j].tolist() if hasattr(cmd,"posture") else None,
                              "gait_frequency_hz":cmd.frequency[j].item() if hasattr(cmd,"frequency") else 1/.65})
            if args.video:
                if trails is not None and step % 5 == 0:
                    trails.update(base[0].tolist())
                # Smooth translation-only tracking camera leaves all actual turning visible.
                camera_focus += .05 * (base[0] - camera_focus)
                focus = camera_focus.tolist()
                for rc in (u.video_recorder.cfg, u.video_recorder._capture.cfg):
                    rc.eye = (focus[0]+2.3,focus[1]-3.6,2.0)
                    rc.lookat = (focus[0]+.25,focus[1],.5)
                from isaacsim.core.rendering_manager import ViewportManager
                rc = u.video_recorder._capture.cfg
                ViewportManager.set_camera_view(rc.camera_prim_path, eye=list(rc.eye), target=list(rc.lookat))
            previous_base.copy_(base)
            previous_action.copy_(actions)
            previous_delta.copy_(delta)
            obs, _, done, _ = env.step(actions)
            falls |= active & tensor(u.termination_manager.terminated).bool()
            active &= ~done.bool()
            if args.video and step in (50, 500, 1000):
                from PIL import Image
                Image.fromarray(raw.render()).save(out / f"preview_{step}.png")
            if step % 250 == 0:
                print(f"[EVAL] t={t:.1f}s active={active.sum().item()} error={error[active].mean().item() if active.any() else -1:.4f}", flush=True)
            if not active.any():
                break
    data = torch.stack(records).cpu().numpy() if records else np.full((1,n,15),np.nan)
    np.savez_compressed(out / "samples.npz", values=data,
                        columns=np.array(["error","height","roll","pitch","yaw","vx","vy","action_delta_sq","action_acc_sq","slip","FL","FR","RL","RR","target_height"]))
    gait_columns = [f'{kind}_{foot}' for kind in ('height','contact','stance_s','air_s','speed') for foot in ('FL','FR','RL','RR')]
    gait_columns += ['yaw_rate','command_yaw','phase','frequency','equivalent_speed']
    if args.pattern == "walk_run":
        gait_columns += [f'instant_contact_{foot}' for foot in ('FL','FR','RL','RR')]
    gait_data = torch.stack(gait_records).cpu().numpy() if gait_records else np.empty((0,n,len(gait_columns)))
    np.savez_compressed(out / 'gait_samples.npz', values=gait_data,
        columns=np.array(gait_columns), dt=dt, start_time=4.)
    expected_samples = int(round((args.duration-4.)/dt))
    fraction = in_band / expected_samples
    spin_valid=(spin_distance_max<.20)&(rotation>=2*math.pi)&(spin_samples>=699)
    success = (fraction >= .9) & ~falls & torch.where(spin_flags,spin_valid,path>=.4)
    def finite(value):
        value = float(value)
        return value if math.isfinite(value) else None
    def quant(index, q):
        return finite(np.nanquantile(data[:,:,index], q)) if np.isfinite(data[:,:,index]).any() else None
    result = {"checkpoint":str(Path(args.checkpoint).resolve()),"seed":args.seed,"profile":args.profile,
              "travel_tcp_assistance_gain":getattr(cmd,"tcp_assistance_gain",0.),
              "reset_state_sha256":reset_fingerprints,
              "command_continuity":{"max_velocity_command_step":[float(x) for x in max_command_change],
                                    "max_posture_preference_step":[float(x) for x in max_posture_change],
                                    "max_gait_phase_step":max_phase_step},
              "pattern":args.pattern,"legacy":args.legacy,"gait_recipe":args.gait_recipe,"episodes":n,"duration_s":args.duration,
              "success_definition":"First episode: >=90% time within 8cm, no fall; spin/transition requires >=2pi rad rotation during t12..26 with max anchor drift <20cm; regular paths require >=0.4m travel.",
              "spin_window_s":[12,26],"spin_episode_count":spin_flags.sum().item(),
              "spin_max_drift_mean_m":finite(spin_distance_max[spin_flags].mean()) if spin_flags.any() else None,
              "spin_rotation_mean_rad":finite(rotation[spin_flags].mean()) if spin_flags.any() else None,
              "successful_episodes":success.sum().item(),"fall_count":falls.sum().item(),
              "mean_tracking_fraction":fraction.mean().item(),"mean_error_m":finite(np.nanmean(data[:,:,0])),
              "p95_error_m":quant(0,.95),"base_height_p05_p95_m":[quant(1,.05),quant(1,.95)],
              "roll_p05_p95_rad":[quant(2,.05),quant(2,.95)],"pitch_p05_p95_rad":[quant(3,.05),quant(3,.95)],
              "base_speed_p50_p95_m_s":[finite(np.nanquantile(np.linalg.norm(data[:,:,5:7],axis=-1),q)) for q in (.5,.95)],
              "joint_target_delta_rms_rad":finite(.25*np.sqrt(np.nanmean(data[:,:,7]))),
              "joint_target_second_difference_rms_rad":finite(.25*np.sqrt(np.nanmean(data[:,:,8]))),
              "mean_stance_slip_m_s":finite(np.nanmean(data[:,:,9])),"mean_base_path_m":path.mean().item(),
              "contact_duty_factors":[finite(np.nanmean(data[:,:,i])) for i in range(10,14)],
              "trajectory_parameters":params,"trace_env0":trace,"trace_env_id":args.trace_env,
              "per_episode":[{"success":success[i].item(),"fall":falls[i].item(),"fraction":fraction[i].item(),
                               "samples":samples[i].item(),"base_path_m":path[i].item(),
                               "spin":spin_flags[i].item(),"spin_max_drift_m":spin_distance_max[i].item(),
                               "spin_rotation_rad":rotation[i].item(),"spin_samples":spin_samples[i].item()} for i in range(n)]}
    (out / "evaluation.json").write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")
    print("[EVALUATION] "+json.dumps({k:v for k,v in result.items() if k not in ("trajectory_parameters","trace_env0","per_episode")}),flush=True)
env.close()
app.close()
