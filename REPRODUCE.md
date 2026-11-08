# 复现指南

**复现边界：** `runs/` 中保存的是历史实验配置记录，本机绝对路径已脱敏为相对路径或占位符，不能直接作为运行配置。部分历史阶段与当前脚本的奖励配置存在差异，请逐项比较，不能保证当前命令精确重现历史结果。`spin_stage` 的上游 `robust_stage` 权重未包含在本仓库。`requirements.txt` 是参考版本清单，并非一键安装文件；需先配置兼容的 Isaac Lab 和 Isaac Sim。

**简体中文** | [English](REPRODUCE.en.md)

尝试复现训练需要三块内容：机器人资产、Isaac Lab 扩展包、分阶段训练命令。本文记录相关命令、产物位置和校验值，历史配置差异见上方说明。仓库只提交代码、配置、权重与实验结果；USD 与网格由脚本按固定上游 commit 生成（约 37 MB 网格 + 约 83 MB USD），不入库。

已验证环境的其余固定版本见 `environment-freeze-reference.txt`；该快照排除了本地路径和 URL 依赖，只供核对，不保证可直接安装。

## 1. 环境

| 组件 | 本项目实测版本 | 检查命令 |
| --- | --- | --- |
| 操作系统 | Windows 11 | — |
| Isaac Lab | 3.0.0（`VERSION` 文件；`isaaclab.__version__` 打印 6.1.17 是扩展版本串） | `cat <IsaacLab>/VERSION` |
| Isaac Sim | 6.0.1 | `ls <IsaacLab>/env/Lib/site-packages \| grep isaacsim-6` |
| Python | 3.12.10 | `<IsaacLab>/env/Scripts/python.exe -V` |
| PyTorch | 2.11.0+cu128 | `<IsaacLab>/env/Scripts/python.exe -c "import torch;print(torch.__version__)"` |
| RSL-RL | 5.0.1 | `<IsaacLab>/env/Scripts/python.exe -m pip show rsl-rl-lib` |
| GPU | RTX 5070 Laptop 8 GB | 512 环境约 4.5 GB 显存；同卡只能跑一个实例 |

训练与回放的命令都用 Isaac Lab 自带的解释器直接执行（`<IsaacLab>/env/Scripts/python.exe -u run_adaptive.py ...`），和记录里 `train_*.ps1` 的启动方式一致。Linux 下换成对应的 `env/bin/python`。

## 2. 机器人资产

### 2.1 用的是哪份资产

Go2 + Piper 整机 URDF 取自 [Sivan666666/roboduet_go2piper](https://github.com/Sivan666666/roboduet_go2piper)，固定 commit `32f3898`，文件在 `resources/robots/go2_piper/`。

| 项 | 值 |
| --- | --- |
| 刚体 | 21（base + 4 hip + 4 thigh + 4 calf + piper_link1…8） |
| 关节 | 20（12 腿 + 6 臂转动 + 2 夹爪平移） |
| 被驱动 | 18（`DRIVEN_JOINT_NAMES = [".*_hip_joint", ".*_thigh_joint", ".*_calf_joint", "piper_joint[1-6]"]`） |
| 动作向量顺序 | 按"离根的可动关节层数"分层：FL_hip, FR_hip, RL_hip, RR_hip, piper_joint1, FL_thigh, FR_thigh, RL_thigh, RR_thigh, piper_joint2, FL_calf, FR_calf, RL_calf, RR_calf, piper_joint3…6（与 URDF 书写顺序不同，实测自 `env.scene["robot"].joint_names`） |
| 策略观测 | 92 维（含 2 维步态相位与目标速度/加速度项），见 `runs/*/effective_config.json` |

动作顺序既不是 URDF 的书写顺序，也不是 USD 文本里的关节顺序，而是 Isaac Lab 加载 articulation 时的分层顺序（上面这一串）。换任何一份资产都要先确认顺序一致，否则权重加载后动作全部错位；`scripts/build_asset.py check` 会推导出这个顺序并对比参考清单。

### 2.2 生成流程

```bash
python scripts/build_asset.py fetch      # 按 commit 下载 18 个文件，逐个校验 sha256
python scripts/build_asset.py prepare    # 把 package:// 与 ../dae/ 改写成绝对路径
<IsaacLab>/env/Scripts/python.exe -u scripts/convert_asset.py --headless   # URDF → USD（需要 Isaac Sim，等价于 isaaclab.bat -p）
python scripts/build_asset.py mount --preset lowered             # 安装座下沉补丁
python scripts/build_asset.py check                              # 校验关节顺序/刚体/安装座高度
```

产物：

```
assets/upstream/                                   # 上游 URDF + 7 个 dae + 10 个 STL（不入库）
assets/go2piper.urdf                               # 路径改写后的 URDF
assets/asset_report.json                           # link/joint 清单
isaaclab_extension/roboduet_go2piper/data/go2piper.usd/
    ├── config.yaml                                # 本次转换参数（可追溯）
    └── go2piper/go2piper.usda                     # 真正被加载的主文件
```

转换参数固定在 `scripts/convert_asset.py` 里，其中三项不能改：`merge_fixed_joints=True`（`piper_mount` 必须并进 base，否则整机不是单一 articulation）、`make_instanceable=False`（instanceable prim 只读，接触传感器 API 写不进去）、`run_asset_transformer=False`（否则刚体被塞进嵌套层级，`SceneEntityCfg("robot", body_names=".*_calf")` 匹配不到）。合并固定关节的副作用是足端刚体被并入小腿，所以所有足端判据都用 `.*_calf` 而不是 `*_foot`。

### 2.3 安装座两种几何

`mount` 子命令只改 4 个数值（机身局部坐标、单位 m）：

| prim | 含义 | `--preset original` | `--preset lowered` |
| --- | --- | --- | --- |
| `Mesh/base_link` | 安装座可见网格 | 0.085 | 0.0562 |
| `Mesh/base_link_1` | 安装座碰撞网格 | 0.085 | 0.0562 |
| `Xform/piper_link1` | 臂根刚体偏移 | 0.208 | 0.1722 |
| `Physics/piper_joint1` 的 `physics:localPos0` | 臂根关节锚点 | 0.208 | 0.1722 |

两种几何都要能生成：`original` 是上游 URDF 原样（0.085 + 0.123），链上除最后一个阶段外的训练都用它；`lowered` 是安装座下沉后贴合狗背的版本，`runs/user_mount_adapt_v8` 用它训练，顶部 6 段录像也都是在它上面回放的。

在 Isaac Sim 6.0.1 的转换器下重新生成，`check` 应打印：20 个关节齐全、18 维动作顺序与参考一致、21/21 个刚体、21 个 prim 带接触报告 API、以及所选预设的 4 个安装座数值。USD 文件本身没有可对标的固定哈希：每次转换的序列化结果都不同（文件头 `doc` 字段带转换时的临时目录名，`Flattened_Prototype` 编号顺序也会变）。训练时那份 `lowered` USD 的 sha256 是 `71a69636…`，仅作溯源记录。

接触报告 API 必须由转换脚本写进 USD，不能指望运行时补：本资产的 21 个链节是嵌在 `base` 下面的，Isaac Lab 在 spawn 时调用的 `activate_contact_sensors` 是"遇到带 `RigidBodyAPI` 的 prim 就不再下探"，于是只在 `base` 上打了 API。实测用未经处理的转换结果建环境时，接触传感器只认得 `base`（`sensor.body_names == ['base']`），所有足端接触与抬脚奖励失效；`scripts/convert_asset.py` 在转换后会给 21 个刚体全部写入 `PhysxContactReportAPI` 与 `physxContactReport:threshold = 0`，补完后 `body_names` 是完整的 21 个。`check` 会核对这个数字。

改资产几何后有两处不能忘：`isaaclab_extension/go2_piper/go2_piper_walk_env_cfg.py` 里的 `GO2_CALF_GROUND_Z` 是本资产实测的站立小腿刚体原点高度（0.2715 m），换资产或改安装座后要重新实测，否则相位抬脚高度项会系统性偏 2–3 cm；两个夹爪平移关节（`piper_joint7/8`）不参与动作，行走任务里保持全闭，只有拾取任务换成 `go2_piper_pick_env_cfg.py` 的半开配置。

## 3. 把扩展包放进 Isaac Lab

```bash
<IsaacLab>/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomanipulation/
    go2_piper/            ← 复制 isaaclab_extension/go2_piper
    roboduet_go2piper/    ← 复制 isaaclab_extension/roboduet_go2piper（含 data/go2piper.usd）
```

`robots.py` 用 `os.path.dirname(__file__)` 拼 USD 相对路径，所以整个包可以放到任意目录，但包内 `data/go2piper.usd` 不能挪走。`isaaclab_tasks` 会自动扫描并注册任务包，不需要改仓库里任何 `__init__.py`。

确认注册成功：

```bash
<IsaacLab>/env/Scripts/python.exe -c "import gymnasium as gym, isaaclab_tasks; print([k for k in gym.registry if 'Go2Piper' in k])"
# 应包含 Isaac-Go2Piper-Walk-v0 / -Play-v0
```

## 4. 权重谱系

`runs/` 下 11 个目录构成一条链，每个目录含 `model_final.pt`、`policy_jit.pt`、`invocation.json`（当次命令行）、`effective_config.json`（完整环境配置）、`export_validation.json`（导出校验）。

| 阶段 | 起点 | pattern | 轮数 | seed | lr | 配方 | 安装座 | 用时 | 产物 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `spin_stage` | 更早的协同策略（本仓库之前的任务版本） | combined | 600 | 91 | 8e-05 | baseline | original | 22 min | 发布链起点 |
| `natural_stage` | `spin_stage` | combined | 800 | 211 | 1e-04 | natural | original | 28 min | 分相位步态 |
| `natural_refine` | `natural_stage` | combined | 600 | 221 | 6e-05 | natural | original | 42 min | 步频前馈 + 落地冲击约束 |
| `natural_coordination` | `natural_refine` | combined | 400 | 241 | 4e-05 | natural | original | 15 min | 低学习率协调 |
| `natural_balance` | `natural_coordination` | combined | 800 | 261 | 6e-05 | natural | original | 33 min | 收紧低速跟踪 |
| `stability_v1` | `natural_balance` | combined | 800 | 311 | 4e-05 | stability | original | 29 min | 稳定性配方起点 |
| `stability_v8` | `stability_v1` | combined | 1000 | 381 | 2.5e-05 | stability_support | original | 39 min | **通用模型** |
| `s_curve_v1` | `stability_v8` | serpentine | 650 | 481 | 2.5e-05 | stability_support | original | 24 min | **S 型专项** |
| `walk_run_v1` | `s_curve_v1` | walk_run | 1000 | 581 | 5e-05 | natural | original | 37 min | 走跑第一级 |
| `walk_run_v3` | `walk_run_v1` | walk_run | 1000 | 601 | 4e-05 | natural | original | 36 min | **走跑专项** |
| `user_mount_adapt_v8` | `stability_v8` | combined | 600 | 691 | 2e-05 | stability_support | lowered | 21 min | **安装座适配** |

几点说明：

* `spin_stage` 是用重构前的任务版本训练的（当时的代码没有 `--gait_recipe` 与 `--pattern serpentine/walk_run`），无法用当前代码逐位重训，作为起点权重随仓库提供；它上面的两个更早阶段（协同、平滑旋转）同样是旧版本产物，不再展开。
* 加粗的 4 个是 README 里报告结果的模型；中间 7 个权重也都在仓库里，所以既可以整条链重跑，也可以只从某一级（例如 `stability_v1`）开始验证下游。
* 所有阶段都是 `--profile adaptive`、`--num_envs 512`、`--duration 38`（即每回合 38 s）。用时应按 512 环境、RTX 5070 Laptop 8 GB 实测的 2.0–2.8 s/迭代换算。

## 5. 分阶段训练命令

模板（10 个阶段只是参数不同）：

```bash
<IsaacLab>/env/Scripts/python.exe -u run_adaptive.py --mode train --profile adaptive \
    --pattern <combined|serpentine|walk_run> --num_envs 512 --iterations <N> \
    --seed <S> --learning_rate <LR> --gait_recipe <natural|stability|stability_support> \
    --duration 38 --checkpoint runs/<起点>/model_final.pt --run_name <阶段名> --headless
```

按表逐条执行，或直接用驱动脚本（读 `scripts/stages.json`，命令行与上表一一对应）：

```bash
python scripts/train_stage.py --list                                  # 打印阶段表
python scripts/train_stage.py --through stability_v8 --dry-run        # 只打印命令
python scripts/train_stage.py --through stability_v8 \
    --python <IsaacLab>/env/Scripts/python.exe                        # 真跑（已存在的权重会跳过）
```

三个分支的实际命令：

**主干：步态 → 稳定性 → 通用模型**

```bash
python run_adaptive.py --mode train --pattern combined --num_envs 512 --iterations 800 \
    --seed 211 --learning_rate 1e-4 --gait_recipe natural --duration 38 \
    --checkpoint runs/spin_stage/model_final.pt --run_name natural_stage --headless

python run_adaptive.py --mode train --pattern combined --num_envs 512 --iterations 600 \
    --seed 221 --learning_rate 6e-5 --gait_recipe natural --duration 38 \
    --checkpoint runs/natural_stage/model_final.pt --run_name natural_refine --headless

python run_adaptive.py --mode train --pattern combined --num_envs 512 --iterations 400 \
    --seed 241 --learning_rate 4e-5 --gait_recipe natural --duration 38 \
    --checkpoint runs/natural_refine/model_final.pt --run_name natural_coordination --headless

python run_adaptive.py --mode train --pattern combined --num_envs 512 --iterations 800 \
    --seed 261 --learning_rate 6e-5 --gait_recipe natural --duration 38 \
    --checkpoint runs/natural_coordination/model_final.pt --run_name natural_balance --headless

python run_adaptive.py --mode train --pattern combined --num_envs 512 --iterations 800 \
    --seed 311 --learning_rate 4e-5 --gait_recipe stability --duration 38 \
    --checkpoint runs/natural_balance/model_final.pt --run_name stability_v1 --headless

python run_adaptive.py --mode train --pattern combined --num_envs 512 --iterations 1000 \
    --seed 381 --learning_rate 2.5e-5 --gait_recipe stability_support --duration 38 \
    --checkpoint runs/stability_v1/model_final.pt --run_name stability_v8 --headless
```

**分支 A：S 型专项**（从通用模型继续，只走蛇形曲线）

```bash
python run_adaptive.py --mode train --pattern serpentine --num_envs 512 --iterations 650 \
    --seed 481 --learning_rate 2.5e-5 --gait_recipe stability_support --duration 38 \
    --checkpoint runs/stability_v8/model_final.pt --run_name s_curve_v1 --headless
```

**分支 B：走跑过渡**（接在 S 型专项之后，两段训练）

```bash
python run_adaptive.py --mode train --pattern walk_run --num_envs 512 --iterations 1000 \
    --seed 581 --learning_rate 5e-5 --gait_recipe natural --duration 38 \
    --checkpoint runs/s_curve_v1/model_final.pt --run_name walk_run_v1 --headless

python run_adaptive.py --mode train --pattern walk_run --num_envs 512 --iterations 1000 \
    --seed 601 --learning_rate 4e-5 --gait_recipe natural --duration 38 \
    --checkpoint runs/walk_run_v1/model_final.pt --run_name walk_run_v3 --headless
```

**分支 C：安装座适配**（必须先把资产打成 `--preset lowered`）

```bash
python scripts/build_asset.py mount --preset lowered
python run_adaptive.py --mode train --pattern combined --num_envs 512 --iterations 600 \
    --seed 691 --learning_rate 2e-5 --gait_recipe stability_support --duration 38 \
    --checkpoint runs/stability_v8/model_final.pt --run_name user_mount_adapt_v8 --headless
```

显存不够时把 `--num_envs` 降到 256 也能训，但批内步态统计量与 512 不同，等价于换了训练配方，结果不能与仓库权重直接对比；先按 512 跑。

## 6. 验证

回放（原速、带录像，seed 与发布视频一致）：

```bash
python run_adaptive.py --mode eval --pattern transition --num_envs 1 --seed 1419 \
    --checkpoint runs/user_mount_adapt_v8/model_final.pt --run_name replay_spin --video
python run_adaptive.py --mode eval --pattern serpentine --num_envs 1 --seed 1482 \
    --checkpoint runs/s_curve_v1/model_final.pt --run_name replay_s --video
python run_adaptive.py --mode eval --pattern walk_run --num_envs 1 --seed 1581 \
    --checkpoint runs/walk_run_v3/model_final.pt --run_name replay_run --video
python run_ground_pick.py --run-name replay_pick --seed 1161 --video
```

批量统计（128/256 独立场景）用 `--mode eval --num_envs 128` 加对应 pattern，输出写到 `results/<run_name>/evaluation.json`，阈值与判据在 `run_adaptive.py` 的评估段里；仓库已有的对照数据在 `results/`，例如：

| 对照项 | 仓库数据 | 达标判据 |
| --- | --- | --- |
| 通用模型原地旋转 | `results/stability_v8_fresh_spin/evaluation.json` | 128/128 达标，0 跌倒 |
| 通用模型行进切换 | `results/stability_v8_fresh_transition/evaluation.json` | 128/128 达标，0 跌倒 |
| S 型 | `results/s_curve_test/gait_quality.json` | 127/128 达标，偏航角加速度 RMS 1.65 rad/s² |
| 走跑 | `results/walk_run_v3_contact/walk_run_metrics.json` | 127/128 达标，速度 0.28 → 1.72 → 0.28 m/s |
| 拾取 | `results/ground_pick_stable_*/evaluation.json` | 3 个位置 48/48 成功 |

导出校验（确认 `policy_jit.pt` 与训练策略输出一致）：

```bash
python validate_export.py runs/stability_v8
```

用本流程重新生成的 `lowered` 资产（没有任何手工修改）回放 `runs/user_mount_adapt_v8`（`--pattern spin --seed 1409`），结果与 `results/github_user_mount_spin/evaluation.json` 逐项相同：`successful_episodes = 1`、`fall_count = 0`、`spin_rotation_mean_rad = 6.70748`、`mean_error_m = 0.0118312`，`reset_state_sha256` 里六项初始状态的哈希、以及四腿接触占空比也完全一致 —— 说明这套脚本生成资产的几何与接触设置与训练时等价。

重训得到的是同配方、同 seed 的另一份策略，不是逐位相同的文件：CUDA kernel 的非确定性使得同一 seed 在不同显卡/驱动上也会有差异。判断"复现成功"看上面这些统计量与录像，不看权重哈希。
