"""获取、整理、打补丁与校验 Go2 + Piper（RoboDuet 整机）资产。

本脚本只做文本/文件操作，不需要启动 Isaac Sim；把 URDF 转成 USD 的那一步在
convert_asset.py 里（那一步必须由 Isaac Lab 的 Python 执行）。

子命令
------
fetch    从上游仓库按固定 commit 下载 URDF 与网格，逐个校验 sha256
prepare  把网格路径改写成自包含路径，生成 assets/go2piper.urdf 与资产清单
mount    对转换出来的 go2piper.usda 施加安装座偏移补丁（original / lowered）
check    校验 URDF 与 USD 的关节顺序、刚体数量、安装座高度

典型流程（在仓库根目录）
------------------------
    python scripts/build_asset.py fetch
    python scripts/build_asset.py prepare
    <IsaacLab>/isaaclab.bat -p scripts/convert_asset.py --headless
    python scripts/build_asset.py mount --preset lowered
    python scripts/build_asset.py check

上游与许可证
------------
资产来自 Sivan666666/roboduet_go2piper（作者提供的 Go2 + Piper 整机 URDF）。
本仓库不转存网格文件，只在复现时按需下载；上游许可证见 references/。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

UPSTREAM_REPO = "Sivan666666/roboduet_go2piper"
UPSTREAM_COMMIT = "32f38984e878502c203260a9e441213c65dab794"
UPSTREAM_BASE = f"https://raw.githubusercontent.com/{UPSTREAM_REPO}/{UPSTREAM_COMMIT}/resources/robots/go2_piper/"

# 上游文件清单：大小与 sha256 均按该 commit 的原始字节记录（下载后逐个校验）
MANIFEST = {
    "go2/urdf/go2piper.urdf": (33807, "8281cee1fa0f77e1695207203b3850506c6b4d0eb83e5b956b721cd05ff2597f"),
    "go2/dae/base.dae": (10727399, "ac4d5de5870c3dca89c7bfa5f7670d9aeb336cabd5e862d4a0dfe8a1a17c5be4"),
    "go2/dae/calf.dae": (1125517, "0e15d069faa4984ec85890e2cb6119b757d75ff44480018ada3e13c3ebb8e537"),
    "go2/dae/calf_mirror.dae": (1125457, "69305f09d67ab5d27cf07ecdf91b7d17aa8faba6556b16757f31d9f57ea7beec"),
    "go2/dae/foot.dae": (506988, "1e621a5c8571eb486802a862f4f9fdd2f4feaecff0cb2ead85b5208cf41345a7"),
    "go2/dae/hip.dae": (4666421, "3e8c23a440cea642c4b276d42c0ed02968bc2718a86306e66c6a213fd2b235eb"),
    "go2/dae/thigh.dae": (3884136, "276a3b48122946f7c00cd20b186f96bad11fc027dc51c43a9ae7998b85840d69"),
    "go2/dae/thigh_mirror.dae": (3896575, "6b8bd43781322953a90c7f5258914a22b224dc95a33bb41b0431f83f4f3affaa"),
    "piper_description/meshes/base_link.STL": (607384, "fd4a3d6d0266205fe3f725f048645be66592c23564db22e08b0807d5569b2c64"),
    "piper_description/meshes/gripper_base.STL": (649484, "2895bfa6f1eef2d0ad2cb9dfdedd5f6c760ec07388bcde2b4d1e412e1189c9cb"),
    "piper_description/meshes/link1.STL": (447784, "9ee6e1d67c0241f42a0dbaf5c676d19230375ed078ee1943bb8c4c37ab303d14"),
    "piper_description/meshes/link2.STL": (3658384, "0d37675b699b643f2e15348de99aed535fac10dc5f09a79b8e4b0acfaf4e0378"),
    "piper_description/meshes/link3.STL": (1845784, "e8ae51f17c0c7f6d67b753922e2cc269f5e8372a65734f859231639a64fe8fad"),
    "piper_description/meshes/link4.STL": (877984, "efa33022976218f246f736a6133494b5d0e5486e8b50b52900c42f285ae85558"),
    "piper_description/meshes/link5.STL": (833684, "a1c69d1c7f6f8c60e2c82331c0f444725200ed4053d3e5ec23aa82cb5db5b29f"),
    "piper_description/meshes/link6.STL": (58884, "88374973bc3bd9e7389b5853cd6e555af9731c30b658b1ee6f85be34a126cb92"),
    "piper_description/meshes/link7.STL": (104384, "475a7d8db056524d7659fadf5a2c307d62791047b6214d733dfc7846ed1c613d"),
    "piper_description/meshes/link8.STL": (104384, "3f88d534a85e28a33ef7d2061adacbd1d25214aa9325f9e464aad0f9e4b74606"),
}

# 被驱动关节：12 条腿 + 6 个臂关节（两个夹爪平移关节参与物理但不参与动作）
LEG_JOINTS = ["FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
              "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
              "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
              "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"]
ARM_JOINTS = [f"piper_joint{i}" for i in range(1, 7)]
GRIPPER_JOINTS = ["piper_joint7", "piper_joint8"]
EXPECTED_JOINTS = LEG_JOINTS + ARM_JOINTS + GRIPPER_JOINTS   # 20，与 URDF 里的书写顺序一致
# 运行时关节顺序（= 策略 18 维动作向量的顺序）：不是 URDF 书写顺序，而是按"离根的可动关节层数"
# 分层后的顺序（Isaac Lab 加载 articulation 时就是这个顺序，已用真实环境核对：
#   env.scene["robot"].joint_names）。
ACTION_ORDER = ["FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint", "piper_joint1",
                "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint", "piper_joint2",
                "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
                "piper_joint3", "piper_joint4", "piper_joint5", "piper_joint6"]
EXPECTED_BODIES = ["base"] + [f"{leg}_{part}" for leg in ("FL", "FR", "RL", "RR")
                              for part in ("hip", "thigh", "calf")] + \
                  [f"piper_link{i}" for i in range(1, 9)]    # 21

# 安装座两种几何：机身局部坐标下的 z 高度（米）
#   original  上游 URDF 原样（piper_mount xyz -0.01 0 0.085 + piper_joint1 z 0.123）
#   lowered   把安装座整体下沉后的版本：可见/碰撞网格 0.0562，臂根锚点 0.1722
MOUNT_PRESETS = {
    "original": {"base_link": 0.085, "base_link_1": 0.085, "piper_link1": 0.208, "piper_joint1": 0.208},
    "lowered": {"base_link": 0.0562, "base_link_1": 0.0562, "piper_link1": 0.1722, "piper_joint1": 0.1722},
}
# 参考环境里训练时那份 USD 的 sha256。**不作为校验目标**：USD 的序列化结果每次都不同
#   （doc 字段里带转换时的临时目录名、Flattened_Prototype 编号顺序也变），而且训练用的那份
#   被 Isaac Lab 会话保存过，每个 link 上多写了 PhysxContactReportAPI / threshold=0
#   （运行时 activate_contact_sensors=True 会重新补上，行为一致）。
# 因此核对接验只认 mount 的 4 个数值与 check 的关节/刚体清单。
REFERENCE_USD_SHA = {
    "original": "24e583dc60fb6bf3be32f8dcbac02424c891a2e1e95c7b3c399bd1b953c5efcd",
    "lowered": "71a69636a7f5715ef94216f275def5bb270e182c2c29e7ccc47e43235d0c1bcc",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- fetch
def cmd_fetch(args) -> int:
    dest = Path(args.dest)
    total = sum(size for size, _ in MANIFEST.values())
    print(f"[fetch] {UPSTREAM_REPO}@{UPSTREAM_COMMIT[:8]} → {dest}（{len(MANIFEST)} 个文件，{total / 1e6:.1f} MB）")
    failed = []
    for rel, (size, expect) in MANIFEST.items():
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and sha256(target) == expect:
            print(f"  [skip] {rel}")
            continue
        url = UPSTREAM_BASE + rel
        data = None
        for attempt in range(1, args.retries + 1):
            try:
                with urllib.request.urlopen(url, timeout=180) as response:
                    data = response.read()
                break
            except Exception as error:                  # noqa: BLE001 - 网络错误原样报告
                print(f"  [retry {attempt}/{args.retries}] {rel}: {error}")
                time.sleep(2 * attempt)
        if data is None:
            failed.append(rel)
            continue
        got = hashlib.sha256(data).hexdigest()
        if got != expect or len(data) != size:
            print(f"  [fail] {rel}: 校验不符 size={len(data)}/{size} sha={got[:12]}/{expect[:12]}")
            failed.append(rel)
            continue
        target.write_bytes(data)
        print(f"  [ok]   {rel}  {size / 1e6:.1f} MB")
    if failed:
        print(f"[fetch] 失败 {len(failed)} 个文件；确认网络可访问 raw.githubusercontent.com 后重试")
        return 1
    print("[fetch] 完成")
    return 0


# ------------------------------------------------------------------------- prepare
def rewrite_meshes(urdf_text: str, root: Path) -> tuple[str, list[str]]:
    """把 URDF 里的 mesh filename 改成指向 root 的绝对路径（转换器直接按文件系统读取）。"""
    refs = sorted(set(re.findall(r'filename="([^"]+)"', urdf_text)))
    missing = []
    for ref in refs:
        if ref.startswith("package://piper_description/"):
            path = root / "piper_description" / ref.split("piper_description/", 1)[1]
        elif ref.startswith("../dae/"):
            path = root / "go2" / "dae" / Path(ref).name
        else:
            path = (root / "go2" / "urdf" / ref).resolve()
        if not path.exists():
            missing.append(str(path))
            continue
        urdf_text = urdf_text.replace(f'filename="{ref}"', f'filename="{path.as_posix()}"')
    return urdf_text, missing


def runtime_joint_order(text: str) -> list[str]:
    """按 Isaac Lab 加载 articulation 的顺序排出可动关节（分层：离根的可动关节层数 + 书写顺序）。"""
    root = ET.fromstring(text)
    joints = []
    for element in root:
        if element.tag.split("}")[-1] != "joint":
            continue
        joints.append((element.get("name"), element.get("type"),
                       element.find("parent").get("link"), element.find("child").get("link")))
    by_child = {joint[3]: joint for joint in joints}
    position = {joint[0]: index for index, joint in enumerate(joints)}
    movable_types = ("revolute", "continuous", "prismatic")

    def depth(joint) -> int:
        level, link, seen = 1, joint[2], set()
        while link in by_child and link not in seen:
            seen.add(link)
            parent = by_child[link]
            if parent[1] in movable_types:
                level += 1
            link = parent[2]
        return level

    movable = [joint for joint in joints if joint[1] in movable_types]
    return [joint[0] for joint in sorted(movable, key=lambda joint: (depth(joint), position[joint[0]]))]


def describe_urdf(text: str) -> dict:
    root = ET.fromstring(text)
    links, joints = [], []
    for element in root:
        tag = element.tag.split("}")[-1]
        if tag == "link":
            links.append(element.get("name"))
        elif tag == "joint":
            joints.append({"name": element.get("name"), "type": element.get("type"),
                           "parent": element.find("parent").get("link"),
                           "child": element.find("child").get("link")})
    movable = [j["name"] for j in joints if j["type"] in ("revolute", "continuous", "prismatic")]
    return {"n_links": len(links), "n_joints": len(joints), "movable_joint_order": movable,
            "links": links}


def cmd_prepare(args) -> int:
    src, out = Path(args.src), Path(args.out)
    source_urdf = src / "go2/urdf/go2piper.urdf"
    if not source_urdf.exists():
        print(f"[prepare] 缺少 {source_urdf}，先运行 fetch")
        return 1
    text = source_urdf.read_text(encoding="utf-8", errors="ignore")
    text, missing = rewrite_meshes(text, src.resolve())
    if missing:
        print("[prepare] 以下网格缺失，转换会失败：")
        for item in missing:
            print("   ", item)
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    info = describe_urdf(text)
    report = {"upstream": f"{UPSTREAM_REPO}@{UPSTREAM_COMMIT}", "source_urdf": str(source_urdf),
              "output_urdf": str(out), **info}
    (out.parent / "asset_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False),
                                                  encoding="utf-8")
    print(f"[prepare] {out}  links={info['n_links']} joints={info['n_joints']} "
          f"movable={len(info['movable_joint_order'])}")
    print("[prepare] 可动关节顺序:", ", ".join(info["movable_joint_order"]))
    return 0


# --------------------------------------------------------------------------- mount
TRANSLATE = re.compile(r"double3 xformOp:translate = \(([^)]*)\)")
LOCAL_POS = re.compile(r'def PhysicsRevoluteJoint "(?P<joint>[^"]+)".*?'
                       r"point3f physics:localPos0 = \(([^)]*)\)", re.S)


def set_prim_translate(text: str, kind: str, name: str, z: float) -> tuple[str, float]:
    head = re.search(rf'def {kind} "{re.escape(name)}"', text)
    if head is None:
        raise KeyError(f"USD 里找不到 prim {kind} \"{name}\"")
    match = TRANSLATE.search(text, head.end())
    if match is None or "def " in text[head.end():match.start()]:
        raise KeyError(f"prim {name} 没有紧跟的 xformOp:translate")
    values = [item.strip() for item in match.group(1).split(",")]
    old = float(values[2])
    values[2] = f"{z:g}"
    return text[:match.start(1)] + ", ".join(values) + text[match.end(1):], old


def set_joint_anchor(text: str, joint: str, z: float) -> tuple[str, float]:
    match = re.search(rf'def PhysicsRevoluteJoint "{joint}".*?'
                      rf"point3f physics:localPos0 = \(([^)]*)\)", text, re.S)
    if match is None:
        raise KeyError(f"USD 里找不到关节 {joint}")
    values = [item.strip() for item in match.group(1).split(",")]
    old = float(values[2])
    values[2] = f"{z:g}"
    return text[:match.start(1)] + ", ".join(values) + text[match.end(1):], old


def cmd_mount(args) -> int:
    usd = Path(args.usd)
    preset = MOUNT_PRESETS[args.preset]
    if not usd.exists():
        print(f"[mount] 找不到 {usd}；先运行 convert_asset.py")
        return 1
    def read_exact(path: Path) -> str:
        # newline="" 关掉换行翻译：USD 是 CRLF 写入的，翻译会改变字节数并与参考值不一致
        with path.open("r", encoding="utf-8", errors="replace", newline="") as stream:
            return stream.read()

    text = read_exact(usd)
    backup = usd.with_suffix(usd.suffix + ".orig")
    if not backup.exists():
        shutil.copy2(usd, backup)          # 只保留一份原始转换结果，重复运行不会叠加
        print(f"[mount] 备份原始转换结果 → {backup.name}")
    else:
        text = read_exact(backup)
        print(f"[mount] 以 {backup.name} 为基准重新打补丁（可重复运行）")
    changed = []
    for name in ("base_link", "base_link_1"):
        text, old = set_prim_translate(text, "Mesh", name, preset[name])
        changed.append((f"Mesh/{name}", old, preset[name]))
    text, old = set_prim_translate(text, "Xform", "piper_link1", preset["piper_link1"])
    changed.append(("Xform/piper_link1", old, preset["piper_link1"]))
    text, old = set_joint_anchor(text, "piper_joint1", preset["piper_joint1"])
    changed.append(("Physics/piper_joint1.localPos0", old, preset["piper_joint1"]))
    usd.write_text(text, encoding="utf-8", newline="")
    print(f"[mount] preset={args.preset}")
    for name, old, new in changed:
        print(f"   {name:34s} {old:.4f} → {new:.4f}  ({new - old:+.4f} m)")
    digest = sha256(usd)
    print(f"[mount] sha256 {digest}")
    print("[mount] 这个哈希是本次生成的指纹，不是固定校验值（USD 每次序列化都不同）")
    print("[mount] 核对几何请用：python scripts/build_asset.py check")
    return 0


# --------------------------------------------------------------------------- check
def usd_joint_order(text: str) -> list[str]:
    return re.findall(r'def Physics(?:\w+)Joint "([^"]+)"', text)


def usd_bodies(text: str) -> list[str]:
    names = []
    for kind, name in re.findall(r'def (Xform|Mesh) "([^"]+)"', text):
        if kind == "Xform" and name not in names and not name.endswith(("_1", "_2")):
            names.append(name)
    return names


def cmd_check(args) -> int:
    ok = True
    checked = 0
    urdf = Path(args.urdf)
    if urdf.exists():
        checked += 1
        urdf_text = urdf.read_text(encoding="utf-8", errors="ignore")
        info = describe_urdf(urdf_text)
        driven = [name for name in runtime_joint_order(urdf_text) if name not in GRIPPER_JOINTS]
        if driven == ACTION_ORDER:
            print(f"[check] URDF 可动关节 {len(info['movable_joint_order'])} 个，"
                  f"运行时动作顺序与参考一致（18 维）")
        else:
            ok = False
            print("[check] URDF 运行时动作顺序不符：")
            print("   实际:", driven)
            print("   期望:", ACTION_ORDER)
    else:
        print(f"[check] 跳过 URDF（找不到 {urdf}）")

    usd = Path(args.usd)
    if usd.exists():
        checked += 1
        with usd.open("r", encoding="utf-8", errors="replace", newline="") as stream:
            text = stream.read()
        joints = usd_joint_order(text)          # 文本顺序 ≠ 运行时顺序，这里只核对集合
        bodies = usd_bodies(text)
        missing = [name for name in EXPECTED_JOINTS if name not in joints]
        if missing:
            ok = False
            print(f"[check] USD 缺少关节：{missing}")
        else:
            print(f"[check] USD 关节 {len(EXPECTED_JOINTS)} 个齐全（文本顺序不参与动作顺序判定）")
        contact = text.count("PhysxContactReportAPI")
        if contact >= len(EXPECTED_BODIES):
            print(f"[check] 接触报告 API 已写入 {contact} 个 prim（≥ {len(EXPECTED_BODIES)} 个刚体）")
        else:
            ok = False
            print(f"[check] 接触报告 API 只有 {contact} 个 prim：接触传感器会只认 base，"
                  f"足端奖励失效（重跑 convert_asset.py）")
        absent = [name for name in EXPECTED_BODIES if name not in bodies]
        matched = len(EXPECTED_BODIES) - len(absent)
        print(f"[check] USD 刚体 {matched}/{len(EXPECTED_BODIES)} 个与参考清单一致"
              + (f"，缺少 {absent}" if absent else ""))
        if absent:
            ok = False
        for name in ("base_link", "base_link_1", "piper_link1"):
            match = re.search(rf'def (?:Xform|Mesh) "{name}"', text)
            if match is None:
                continue
            found = TRANSLATE.search(text, match.end())
            print(f"   安装座 {name:12s} z = {found.group(1).split(',')[-1].strip() if found else '?'}")
        joint = re.search(r'def PhysicsRevoluteJoint "piper_joint1".*?'
                          r"point3f physics:localPos0 = \(([^)]*)\)", text, re.S)
        if joint:
            print(f"   臂根锚点 piper_joint1 z = {joint.group(1).split(',')[-1].strip()}")
    else:
        print(f"[check] 跳过 USD（找不到 {usd}）")
    if checked == 0:
        print("[check] 没有可校验的文件：先跑 fetch / prepare / convert_asset.py")
        return 1
    print("[check] 通过" if ok else "[check] 未通过")
    return 0 if ok else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser("fetch", help="下载上游 URDF 与网格")
    fetch.add_argument("--dest", default="assets/upstream")
    fetch.add_argument("--retries", type=int, default=4, help="单个文件失败重试次数（GitHub raw 偶发超时）")
    fetch.set_defaults(func=cmd_fetch)

    prepare = sub.add_parser("prepare", help="改写网格路径，生成自包含 URDF")
    prepare.add_argument("--src", default="assets/upstream")
    prepare.add_argument("--out", default="assets/go2piper.urdf")
    prepare.set_defaults(func=cmd_prepare)

    mount = sub.add_parser("mount", help="给 USD 打安装座偏移补丁")
    mount.add_argument("--usd", default="isaaclab_extension/roboduet_go2piper/data/go2piper.usd/go2piper/go2piper.usda")
    mount.add_argument("--preset", choices=sorted(MOUNT_PRESETS), default="lowered")
    mount.set_defaults(func=cmd_mount)

    check = sub.add_parser("check", help="校验 URDF 与 USD 的关节/刚体清单")
    check.add_argument("--urdf", default="assets/go2piper.urdf")
    check.add_argument("--usd", default="isaaclab_extension/roboduet_go2piper/data/go2piper.usd/go2piper/go2piper.usda")
    check.set_defaults(func=cmd_check)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
