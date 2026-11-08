"""Measure actual foot cycles and yaw-rate variation, not just small joint changes."""
import argparse
import json
from pathlib import Path
import numpy as np

parser=argparse.ArgumentParser()
parser.add_argument('run_names',nargs='+')
args=parser.parse_args()
root=Path(__file__).resolve().parent
for name in args.run_names:
    folder=root/'runs'/name
    result=json.loads((folder/'evaluation.json').read_text(encoding='utf-8'))
    raw=np.load(folder/'gait_samples.npz')
    values=raw['values']
    dt=float(raw['dt'])
    times=float(raw['start_time'])+np.arange(len(values))*dt
    windows={'tracking':times>=4.,'spin_window':(times>=12.)&(times<26.)}
    report={}
    for window,mask in windows.items():
        d=values[mask]
        eligible=np.isfinite(d[:,:,0])
        if window=='spin_window':
            eligible &= np.array([p['spin'] for p in result['trajectory_parameters']])[None]
        else:
            eligible &= d[:,:,24]>.10
        contact=d[:,:,4:8]>.5
        valid=eligible[:,:,None]
        peaks=[[] for _ in range(4)]
        air_durations=[[] for _ in range(4)]
        touchdown_speeds=[[] for _ in range(4)]
        cycle_counts=np.zeros((d.shape[1],4),int)
        for env in range(d.shape[1]):
            for foot in range(4):
                airborne=False
                peak=0.
                length=0
                for i in range(len(d)):
                    if not eligible[i,env]:
                        airborne=False
                        continue
                    if not contact[i,env,foot]:
                        if not airborne:
                            # Ignore incomplete air segment at window start.
                            airborne=i>0 and eligible[i-1,env] and contact[i-1,env,foot]
                            peak=0.;length=0
                        if airborne:
                            peak=max(peak,float(d[i,env,foot]));length+=1
                    elif airborne:
                        if length*dt>=.06 and peak>=.012:
                            peaks[foot].append(peak)
                            air_durations[foot].append(length*dt)
                            touchdown_speeds[foot].append(float(d[i-1,env,16+foot]))
                            cycle_counts[env,foot]+=1
                        airborne=False
        def avg(x):return float(np.mean(x)) if len(x) else None
        valid_time=eligible.sum(axis=0)*dt
        sufficient=valid_time>8.
        rates=cycle_counts[sufficient]/valid_time[sufficient,None]
        active_count=int(eligible.sum())
        if not active_count:
            continue
        yaw_error=d[:,:,20]-d[:,:,21]
        yaw_delta=np.diff(d[:,:,20],axis=0)/dt
        pairs=eligible[1:]&eligible[:-1]
        report[window]={
            'moving_samples':active_count,
            'four_feet_contact_fraction':float(np.mean(contact.sum(-1)[eligible]==4)),
            'contact_duty_per_foot':contact[eligible].mean(0).tolist(),
            'support_over_0_7s_fraction':float(np.mean((d[:,:,8:12]>.7)[eligible])),
            'support_time_p95_s':float(np.nanquantile(d[:,:,8:12][eligible],.95)),
            'completed_lifts_per_foot':cycle_counts.sum(0).tolist(),
            'lift_frequency_per_foot_hz':[avg(cycle_counts[sufficient,i]/valid_time[sufficient]) for i in range(4)],
            'episodes_with_8s_valid_motion':int(sufficient.sum()),
            'all_feet_over_1hz_fraction':float(np.mean(np.min(rates,axis=1)>=1.)) if len(rates) else None,
            'weakest_foot_lift_rate_p10_hz':float(np.quantile(np.min(rates,axis=1),.1)) if len(rates) else None,
            'mean_lift_peak_per_foot_m':[avg(x) for x in peaks],
            'mean_swing_duration_per_foot_s':[avg(x) for x in air_durations],
            'mean_pre_touchdown_foot_speed_m_s':[avg(x) for x in touchdown_speeds],
            'yaw_tracking_rmse_rad_s':float(np.sqrt(np.mean(yaw_error[eligible]**2))),
            'yaw_acceleration_rms_rad_s2':float(np.sqrt(np.mean(yaw_delta[pairs]**2))),
        }
        if window=='spin_window':
            reference=.48*np.array([p['direction'] for p in result['trajectory_parameters']])
            report[window]['world_yaw_reference_rmse_rad_s']=float(np.sqrt(np.mean((d[:,:,20]-reference[None])[eligible]**2)))
            direction_details={}
            for direction,label in [(-1,'clockwise'),(1,'counterclockwise')]:
                selected=sufficient & (reference*direction>0)
                if not selected.any():continue
                rate_by_foot=cycle_counts[selected]/valid_time[selected,None]
                direction_details[label]={'episodes':int(selected.sum()),
                    'lift_frequency_per_foot_hz':rate_by_foot.mean(0).tolist(),
                    'all_feet_over_1hz_fraction':float(np.mean(rate_by_foot.min(1)>=1.))}
            report[window]['by_direction']=direction_details
    (folder/'gait_quality.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'run':name,**report},ensure_ascii=False))
