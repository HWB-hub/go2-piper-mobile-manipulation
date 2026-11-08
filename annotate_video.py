"""Overlay measured simulation telemetry on the original simulation recording."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import cv2
import imageio_ffmpeg

parser = argparse.ArgumentParser()
parser.add_argument("run_name")
parser.add_argument("--title",default="Go2 + Piper：复杂步态与机械臂协同")
args = parser.parse_args()
root = Path(__file__).resolve().parent
folder = root/"runs"/args.run_name
data = json.loads((folder/"evaluation.json").read_text(encoding="utf-8"))
source = folder/"video/adaptive_gait-step-0.mp4"
cap = cv2.VideoCapture(str(source))
ok, first = cap.read()
fps = cap.get(cv2.CAP_PROP_FPS)
frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width,height = int(cap.get(3)),int(cap.get(4))
cap.release()
assert ok and frames>=int((data["duration_s"]-.2)*50) and 49<fps<51
drop = int(first.mean()<1.)

def stamp(t):
    h = int(t//3600)
    m = int(t//60)%60
    s = t%60
    return f"{h}:{m:02d}:{s:05.2f}"

header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 2

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Main,Microsoft YaHei,20,&H00FFFFFF,&H00FFFFFF,&H80000000,&H80000000,0,0,0,0,100,100,0,0,3,1,0,7,18,18,14,1
Style: Legend,Microsoft YaHei,17,&H00FFFFFF,&H00FFFFFF,&H80000000,&H80000000,0,0,0,0,100,100,0,0,3,1,0,1,18,18,14,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
legend = (
    "绿球：沿 S 型路线移动的机械臂末端目标"
    if data["pattern"] == "serpentine"
    else "绿球：末端目标　旋转阶段保持机身位置，目标按世界坐标轨迹移动"
)
lines = [f"Dialogue: 0,0:00:00.00,{stamp(data['duration_s'])},Legend,,0,0,0,,{legend}"]
if data["pattern"] == "walk_run":
    lines = [f"Dialogue: 0,0:00:00.00,{stamp(data['duration_s'])},Legend,,0,0,0,,绿球：移动末端目标　速度为实际机身速度"]
elif data["pattern"] == "curves":
    lines = [f"Dialogue: 0,0:00:00.00,{stamp(data['duration_s'])},Legend,,0,0,0,,绿球：沿曲线移动的机械臂末端目标"]
trace = data["trace_env0"]
for index,row in enumerate(trace):
    if not row["active"]:
        continue
    start = max(0.,row["time_s"]-drop/50.)
    end = max(start+.01,(trace[index+1]["time_s"] if index+1<len(trace) else data["duration_s"])-drop/50.)
    vx,vy,_ = row["base_velocity_world"]
    roll,pitch,yaw = [math.degrees(x) for x in row["rpy_rad"]]
    text = (args.title+"\\N"+f"时间 {row['time_s']:4.1f} 秒   末端误差 {row['error_m']*100:4.1f} 厘米\\N"
            +f"机身高度 {row['base_world'][2]*100:4.1f} 厘米   速度 {math.hypot(vx,vy):.2f} 米/秒\\N"
            +f"俯仰 {pitch:+.1f}°   滚转 {roll:+.1f}°   偏航 {yaw:+.1f}°")
    lines.append(f"Dialogue: 0,{stamp(start)},{stamp(end)},Main,,0,0,0,,{text}")
subtitle = folder/"telemetry.ass"
subtitle.write_text(header+"\n".join(lines)+"\n",encoding="utf-8-sig")
output = folder/"annotated.mp4"
filters = f"trim=start_frame={drop},setpts=N/(50*TB),ass=telemetry.ass"
subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),"-y","-i",str(source),"-vf",filters,"-r","50",
                "-c:v","libx264","-crf","20","-pix_fmt","yuv420p","-movflags","+faststart",str(output)],
               cwd=folder,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
cap = cv2.VideoCapture(str(output))
assert abs(cap.get(cv2.CAP_PROP_FPS)-50.)<.01
for seconds in (2.,9.,17.,25.,34.):
    cap.set(cv2.CAP_PROP_POS_MSEC,seconds*1000)
    ok,frame = cap.read()
    assert ok and frame.mean()>5.
    cv2.imencode(".png",frame)[1].tofile(str(folder/f"annotated_preview_{int(seconds)}.png"))
cap.release()
print(json.dumps({"output":str(output),"frames":frames-drop,"fps":50,"telemetry_source":"actual first episode env0"}))
