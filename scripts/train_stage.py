"""按 scripts/stages.json 里的记录，逐阶段调用 run_adaptive.py 训练。

为什么要有这个入口：整条链有 10 次训练，每次的起点权重、seed、轮数、学习率、步态配方
都不一样，手抄命令行极易错一位数字；这里把参数固化成表，只用阶段名驱动。

    python scripts/train_stage.py --list                 # 看整条链
    python scripts/train_stage.py --stage stability_v8 --dry-run
    python scripts/train_stage.py --stage stability_v8 --python /path/to/isaaclab/python
    python scripts/train_stage.py --through user_mount_adapt_v8   # 从第一个缺的权重开始一路往下

真正的训练命令等价于：

    <IsaacLab>/env/Scripts/python.exe -u run_adaptive.py --mode train --profile adaptive \\
        --pattern combined --num_envs 512 --iterations 1000 --seed 381 \\
        --learning_rate 2.5e-05 --gait_recipe stability_support \\
        --checkpoint runs/stability_v1/model_final.pt --run_name stability_v8 --headless

注意：本脚本只按记录传参，不改 run_adaptive.py 的任何默认值。显卡显存 <8 GB 时把
--num_envs 调小（512 → 256）会改变训练动态，结果与仓库里的权重不可直接比较。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STAGES = json.loads((ROOT / "scripts" / "stages.json").read_text(encoding="utf-8"))
DEFAULTS = STAGES["defaults"]
TABLE = {stage["id"]: stage for stage in STAGES["stages"]}
ORDER = [stage["id"] for stage in STAGES["stages"]]
ENTRY = ROOT / "runs" / "spin_stage" / "model_final.pt"      # 发布链的起点权重


def ckpt_of(stage: dict) -> Path:
    parent = stage["parent"]
    if parent == "spin_stage":
        return ENTRY
    return ROOT / "runs" / parent / "model_final.pt"


def command_of(stage: dict, python: str) -> list[str]:
    return [python, "-u", "run_adaptive.py",
            "--mode", "train",
            "--profile", DEFAULTS["profile"],
            "--pattern", stage["pattern"],
            "--num_envs", str(DEFAULTS["num_envs"]),
            "--iterations", str(stage["iterations"]),
            "--seed", str(stage["seed"]),
            "--learning_rate", f"{stage['learning_rate']:g}",
            "--gait_recipe", stage["recipe"],
            "--duration", f"{DEFAULTS['duration']:g}",
            "--checkpoint", str(ckpt_of(stage).relative_to(ROOT)).replace("\\", "/"),
            "--run_name", stage["id"],
            "--headless"]


def print_table() -> None:
    print(f"{'stage':24s}{'parent':22s}{'pattern':12s}{'iters':>6s}{'seed':>6s}"
          f"{'lr':>10s}{'recipe':19s}{'mount':10s}{'min':>5s}")
    for name in ORDER:
        stage = TABLE[name]
        print(f"{name:24s}{stage['parent']:22s}{stage['pattern']:12s}{stage['iterations']:6d}"
              f"{stage['seed']:6d}{stage['learning_rate']:>10g}{stage['recipe']:19s}"
              f"{stage['mount']:10s}{stage['elapsed_min']:5d}")
    print(f"\n合计约 {sum(TABLE[n]['elapsed_min'] for n in ORDER)} 分钟（512 环境，RTX 5070 Laptop 8 GB 实测）")


def default_python() -> str:
    env = os.environ.get("ISAACLAB_PYTHON")
    if env:
        return env
    return sys.executable


def run_one(name: str, python: str, dry_run: bool) -> bool:
    stage = TABLE[name]
    checkpoint = ckpt_of(stage)
    if not checkpoint.exists():
        print(f"[skip] {name}: 缺少起点权重 {checkpoint.relative_to(ROOT)}")
        return False
    command = command_of(stage, python)
    print(f"\n=== {name}  ({stage['note']})")
    print(f"    安装座几何: {stage['mount']}；预计 {stage['elapsed_min']} 分钟")
    print("    " + " ".join(command))
    if dry_run:
        return True
    log = ROOT / "runs" / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as stream:
        code = subprocess.call(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    if code != 0:
        print(f"[fail] {name}: 退出码 {code}，见 {log.relative_to(ROOT)}")
        return False
    print(f"[ok] {name}: 权重 {Path('runs') / name / 'model_final.pt'}")
    return True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true", help="打印阶段表")
    parser.add_argument("--stage", choices=ORDER, help="只跑某一个阶段")
    parser.add_argument("--through", choices=ORDER, help="跑到该阶段为止（已存在的权重会跳过）")
    parser.add_argument("--python", default=default_python(), help="Isaac Lab 的 python 解释器")
    parser.add_argument("--dry-run", action="store_true", help="只打印命令不执行")
    parser.add_argument("--force", action="store_true", help="即使目标权重已存在也重跑")
    args = parser.parse_args(argv)

    if args.list or (not args.stage and not args.through):
        print_table()
        return 0

    targets = [args.stage] if args.stage else ORDER[:ORDER.index(args.through) + 1]
    for name in targets:
        if not args.force and (ROOT / "runs" / name / "model_final.pt").exists():
            print(f"[skip] {name}: runs/{name}/model_final.pt 已存在（--force 强制重跑）")
            continue
        if not run_one(name, args.python, args.dry_run):
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
