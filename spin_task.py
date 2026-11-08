"""Stationary yaw turns and smooth travel/turn transitions with world-space arm goals."""
import math
import torch
from isaaclab.managers import RewardTermCfg
from isaaclab.utils.math import quat_apply_inverse, yaw_quat
from adaptive_task import CurvedGoalCommand, adaptive_config, feet_state, command
from cooperative_task import tensor


def ramp_integral(t, start, width=2.):
    u=(t-start).clamp(min=0.)
    s=(u/width).clamp(max=1.)
    value=torch.where(u<width,width*(s**3-.5*s**4),u-width/2)
    rate=3*s*s-2*s**3
    acc=torch.where(u<width,(6*s-6*s*s)/width,torch.zeros_like(t))
    return value,rate,acc


class SpinGoal(CurvedGoalCommand):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        self.anchor=torch.zeros_like(self.goal_pos_w)
        self.anchor_velocity=torch.zeros_like(self.goal_pos_w)
        self.heading=torch.zeros(self.num_envs,device=self.device)
        self.heading_rate=torch.zeros_like(self.heading)
        self.direction=torch.ones_like(self.heading)
        self.spin_mask=torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)
        self.transition_mask=torch.zeros_like(self.spin_mask)

    def _resample_command(self,env_ids):
        # Parent samples regular trajectories; a subset receives the new skills.
        super()._resample_command(env_ids)
        ids=torch.as_tensor(env_ids,device=self.device)
        pattern=self.cfg.spin_pattern
        random=torch.rand(len(ids),device=self.device)
        self.spin_mask[ids]=random<getattr(self.cfg,"spin_probability",.70) if pattern=="combined" else pattern in ("spin","transition")
        self.transition_mask[ids]=(random<getattr(self.cfg,"transition_probability",.35)) if pattern=="combined" else pattern=="transition"
        self.direction[ids]=torch.where(torch.rand(len(ids),device=self.device)<.5,-1.,1.)
        self.update_targets()

    def update_targets(self):
        super().update_targets()
        if not hasattr(self,"spin_mask"):
            return
        t=tensor(self._env.episode_length_buf).float()*self._env.step_dt
        if self.cfg.spin_pattern == "walk_run":
            rise = ramp_integral(t, 7., width=5.)
            fall = ramp_integral(t, 22., width=5.)
            start = ramp_integral(t, 2., width=2.)
            increase = .95 + 1.8 * self.speed
            x, vx, ax = [
                .28 * start[i] + increase * (rise[i] - fall[i]) for i in range(3)
            ]
            self.anchor[:] = self.start_w
            self.anchor[:, 0] += x - .35
            self.anchor[:, 2] = tensor(self._env.scene.env_origins)[:, 2]
            self.goal_pos_w[:] = self.start_w
            self.goal_pos_w[:, 0] += x
            self.goal_vel_w.zero_()
            self.goal_acc_w.zero_()
            self.goal_vel_w[:, 0] = vx
            self.goal_acc_w[:, 0] = ax
            self.goal_pos_b[:] = quat_apply_inverse(
                yaw_quat(tensor(self.robot.data.root_quat_w)),
                self.goal_pos_w - tensor(self.robot.data.root_pos_w),
            )
            if self._marker is not None:
                pose = torch.zeros(self.num_envs, 7, device=self.device)
                pose[:, :3] = self.goal_pos_w
                pose[:, 3] = 1.
                self._marker.write_root_pose_to_sim(pose)
            return
        a=[ramp_integral(t,start) for start in (2.,8.,26.)]
        travel=[(a[0][i]-a[1][i]+a[2][i])*self.transition_mask for i in range(3)]
        s,sd,sdd=travel
        turn=ramp_integral(t,10.)
        stop=ramp_integral(t,26.)
        pure=ramp_integral(t,4.)
        angle,rate,acc=[torch.where(self.transition_mask,turn[i]-stop[i],pure[i])*.48*self.direction for i in range(3)]
        angle+=.30*torch.sin(.5*s)
        rate+=.15*torch.cos(.5*s)*sd
        acc+=.15*torch.cos(.5*s)*sdd-.075*torch.sin(.5*s)*sd.square()
        origins=tensor(self._env.scene.env_origins)
        self.anchor[:]=origins
        self.anchor[:,0]+=.65+.23*s
        self.anchor[:,1]+=.30*torch.sin(.55*s)
        self.anchor_velocity[:,0]=.23*sd
        self.anchor_velocity[:,1]=.165*torch.cos(.55*s)*sd
        self.anchor_velocity[:,2]=0.
        anchor_acc=torch.zeros_like(self.anchor)
        anchor_acc[:,0]=.23*sdd
        anchor_acc[:,1]=.165*torch.cos(.55*s)*sdd-.09075*torch.sin(.55*s)*sd.square()
        self.heading[:]=angle
        self.heading_rate[:]=rate
        position=self.anchor.clone()
        position[:,0]+=.35*torch.cos(angle)
        position[:,1]+=.35*torch.sin(angle)
        position[:,2]+=.64+.10*torch.sin(.25*t+self.phase[:,2])
        velocity=self.anchor_velocity.clone()
        velocity[:,0]-=.35*torch.sin(angle)*rate
        velocity[:,1]+=.35*torch.cos(angle)*rate
        velocity[:,2]=.025*torch.cos(.25*t+self.phase[:,2])
        acceleration=anchor_acc
        acceleration[:,0]-=.35*(torch.cos(angle)*rate.square()+torch.sin(angle)*acc)
        acceleration[:,1]+=.35*(-torch.sin(angle)*rate.square()+torch.cos(angle)*acc)
        acceleration[:,2]=-.00625*torch.sin(.25*t+self.phase[:,2])
        ids=self.spin_mask
        self.goal_pos_w[ids]=position[ids]
        self.goal_vel_w[ids]=velocity[ids]
        self.goal_acc_w[ids]=acceleration[ids]
        self.goal_pos_b[:]=quat_apply_inverse(yaw_quat(tensor(self.robot.data.root_quat_w)),self.goal_pos_w-tensor(self.robot.data.root_pos_w))
        if self._marker is not None:
            pose=torch.zeros(self.num_envs,7,device=self.device)
            pose[:,:3]=self.goal_pos_w
            pose[:,3]=1.
            self._marker.write_root_pose_to_sim(pose)


def anchor_reward(env):
    goal=env.command_manager.get_term("ee_goal")
    err=(tensor(env.scene["robot"].data.root_pos_w)-goal.anchor)[:,:2].square().sum(-1)
    return torch.exp(-err/.10**2)*goal.spin_mask


def smooth_contact_reward(env):
    _,_,contact=feet_state(env)
    c=command(env)
    speed=c.vel_command_b[:,:2].norm(dim=-1)+.30*c.vel_command_b[:,2].abs()
    phase=c.phase[:,None]+torch.tensor([0.,.5,.5,0.],device=env.device)
    # Continuous transition around touchdown/liftoff replaces a binary clock score.
    desired=torch.sigmoid(5.*(torch.cos(2*math.pi*(phase-.28))+.20))
    return torch.where(speed>.05,1.-(contact.float()-desired).square().mean(-1),contact.float().mean(-1))


def spin_config(num_envs,device,seed,profile,pattern,video=False,evaluate=False,legacy=False):
    motion_pattern = "serpentine" if pattern == "serpentine" else "mixed"
    cfg=adaptive_config(num_envs,device,seed,profile,motion_pattern,video,evaluate,legacy)
    cfg.commands.ee_goal.class_type=SpinGoal
    cfg.commands.ee_goal.spin_pattern=pattern
    cfg.commands.ee_goal.resampling_time_range=(1000.,1000.)
    cfg.episode_length_s=38.
    cfg.rewards.anchor_tracking=RewardTermCfg(func=anchor_reward,weight=4.)
    cfg.rewards.phase_gait_soft=RewardTermCfg(func=smooth_contact_reward,weight=2.)
    cfg.rewards.hip_abduction_l2.weight=-.15
    cfg.rewards.joint_mirror.weight=-.02
    cfg.rewards.action_rate_l2.weight=-.18
    cfg.rewards.action_acc_l2.weight=-.04
    cfg.rewards.foot_impact_velocity.weight=-3.
    cfg.rewards.feet_slide.weight=-1.5
    return cfg
