"""把 assets/go2piper.urdf 转成 Isaac Lab 使用的 USD，并补上接触传感器所需的接触报告 API。

    <IsaacLab>/env/Scripts/python.exe -u scripts/convert_asset.py --headless

转换参数固定为下面这组，和训练时使用的资产一致：
    fix_base=False            整机自由落体，不固定基座
    merge_fixed_joints=True   合并固定关节 → 单一 articulation（piper_mount 被并进 base）
    make_instanceable=False   instanceable prim 只读，接触传感器 API 写不进去
    run_asset_transformer=False  否则刚体被塞进 /Geometry/base/<link>/... 嵌套层级
    joint_drive target_type=none  驱动器增益由 ArticulationCfg 在运行时给，USD 里留空

转换完成后脚本会给每个刚体写上 PhysxContactReportAPI / physxContactReport:threshold=0
（与原训练资产一致）。这一步不能省：本资产的链接是嵌在 base 下面的，Isaac Lab 在 spawn 时
调用的 activate_contact_sensors 一碰到带 RigidBodyAPI 的 base 就停止下探，不会给 21 个
链节补 API —— 少了这一步，接触传感器只认得 base，足端接触/抬脚奖励全部失效。

输出目录是一个 USD "资产包"：
    <out>/go2piper.usd/go2piper/go2piper.usda    ← 真正被加载的主文件
    <out>/go2piper.usd/config.yaml               ← 本次转换参数（可追溯）
"""

from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--urdf", default="assets/go2piper.urdf", help="build_asset.py prepare 的产物")
parser.add_argument("--out", default="isaaclab_extension/roboduet_go2piper/data/go2piper.usd",
                    help="USD 资产包目录（robots.py 里的 data/go2piper.usd）")
parser.add_argument("--skip-contact", action="store_true", help="不补接触报告 API（仅用于排查）")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # noqa: E402  必须在 AppLauncher 之后导入


def bake_contact_report(usd_path: str, threshold: float = 0.0) -> int:
    """给所有刚体写 PhysxContactReportAPI，返回处理过的刚体数。"""
    from pxr import Sdf, Usd, UsdPhysics

    stage = Usd.Stage.Open(usd_path)
    count = 0
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        if "PhysxContactReportAPI" not in prim.GetAppliedSchemas():
            prim.AddAppliedSchema("PhysxContactReportAPI")
        attribute = prim.GetAttribute("physxContactReport:threshold")
        if not attribute.IsValid():
            attribute = prim.CreateAttribute("physxContactReport:threshold", Sdf.ValueTypeNames.Float)
        attribute.Set(threshold)
        count += 1
    stage.GetRootLayer().Save()
    return count


def main() -> None:
    urdf = os.path.abspath(args.urdf)
    if not os.path.exists(urdf):
        raise SystemExit(f"找不到 {urdf}，先运行 python scripts/build_asset.py prepare")
    cfg = UrdfConverterCfg(
        asset_path=urdf,
        usd_dir=os.path.abspath(args.out),
        fix_base=False,
        merge_fixed_joints=True,
        make_instanceable=False,
        run_asset_transformer=False,
        run_multi_physics_conversion=False,
        force_usd_conversion=True,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            target_type="none",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
        ),
    )
    converter = UrdfConverter(cfg)
    print("=" * 78)
    print("[convert] USD:", converter.usd_path)
    if args.skip_contact:
        print("[convert] 跳过接触报告 API（--skip-contact）")
    else:
        count = bake_contact_report(converter.usd_path)
        print(f"[convert] 已给 {count} 个刚体写入 PhysxContactReportAPI（应为 21）")
    print("[convert] 下一步：python scripts/build_asset.py mount --preset lowered")
    print("=" * 78)


if __name__ == "__main__":
    main()
    simulation_app.close()
