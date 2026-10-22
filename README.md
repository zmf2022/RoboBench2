<h1><picture><source media="(prefers-color-scheme: dark)" srcset="docs/images/ribble.gif"><img src="docs/images/ribble-dark.gif" alt="" height="42" align="absmiddle" /></picture> RoboLab</h1>

**RoboLab** 是基于 [NVIDIA Isaac Lab](https://github.com/isaac-sim/IsaacLab) 构建、面向机器人操作策略的任务式评估基准。它提供 100 多个具备自动成功检测功能的操作任务，采用服务端与客户端策略架构，支持多环境并行评估，旨在为通用机器人策略提供可复现、大规模的仿真基准评测。

<h4 align="center"><a href="assets/objects/README.md">物体</a> · <a href="assets/scenes/README.md">场景</a> · <a href="robolab/tasks/README.md">任务</a> · <a href="robolab/robots/README.md">机器人</a> · <a href="policies/README.md">策略客户端</a></h4>

<div align="center">
  <img src="docs/images/robolab.png" alt="RoboLab 概览" width="800"/>
</div>

<a id="upstream-news"></a>

## 上游动态

- **[2026/10/18]** 🔥 新增 [**Astranomy**](https://research.nvidia.com/labs/srl/projects/astranomy/) 策略代码，使用通用模型（如 GPT-6 Astra）直接作为机器人策略。
- **[2026/10/13]** 🔥 [**LADDER**](https://ladder.insait.ai/) 推出 RoboLab-Long，包含 44 个用于评估长时序操作的任务，是其覆盖三个仿真套件、共 145 个任务的基准的一部分。参见[任务库](docs/ecosystem.md#task-libraries)。
- **[2026/08/19]** 🤖 [**DROID+**](https://github.com/NVlabs/DroidPlus) 发布：一套简洁的 DROID 硬件方案。现有 RoboLab 策略配置也可用于 DROID+。
- **[2026/08/07]** 🔥 [**RoboVoLo**](https://github.com/NVlabs/RoboVoLo) 发布：来自 [VoLo](https://chicychen.github.io/VoLo/) 项目的任务库，为 RoboLab 扩展了长时序、强调推理能力的操作任务，可通过随附的 [VoLo 策略后端](policies/volo/README.md)进行评估。

完整更新（包括小幅改进和问题修复）请参见[更新日志](CHANGELOG.md)。

<a id="key-features"></a>

## 主要功能

- **RoboLab-120**：120 个基准[任务](robolab/tasks/README.md)，涵盖抓取放置、堆叠、重排、工具使用等。每个任务均包含语言指令，并通过可组合的条件谓词自动判断成功或失败。
- **支持自定义机器人**：任务不绑定特定本体，可使用任何兼容 Isaac Lab 的机器人。项目内置单臂 DROID、Franka、Kinova，以及两套双臂配置，参见[机器人文档](robolab/robots/README.md)。
- **丰富的资产库**：提供[物体](assets/objects/README.md)、[场景](assets/scenes/README.md)和精选[背景](assets/backgrounds/README.md)，用于创建新场景和任务。
- **AI 辅助工作流**：通过 [/robolab-scenegen](skills/robolab-scenegen/) 和 [/robolab-taskgen](skills/robolab-taskgen/) 两个 Claude Code 技能，根据自然语言生成场景和任务。
- **多环境并行评估**：支持并行运行多个回合、向量化条件判断，以及各环境独立终止。
- **结果面板**：内置独立的网页[面板](docs/dashboard.md)，用于浏览场景和任务、回放回合并比较实验。所有相机画面通过同一通道传输，事件时间轴可用于定位回放位置，并支持 `?t=` 时间定位链接。

基于 RoboLab 构建的相关项目，请参见[生态](docs/ecosystem.md)。

<a id="getting-started"></a>

## 快速开始

<a id="installation"></a>

### 安装

需要 [uv](https://docs.astral.sh/uv/getting-started/installation/)、`ffmpeg` 和 `git-lfs`。当前使用 Python 3.12、Isaac Sim 6.0.1 和 Isaac Lab 3.0 beta 2 patch 1。
无桌面的 Linux 系统还需安装图形运行库：`sudo apt install libegl1 libvulkan1`。

在仓库根目录执行：

```bash
git lfs pull
uv venv --python 3.12
source .venv/bin/activate
uv sync --extra isaac60 --extra test
```

验证安装：

```bash
python -m pytest tests/ -q         # 仿真器集成检查，需要 GPU
```

未激活环境时，在命令前加 `uv run --no-sync`。

<a id="astranomy"></a>

### Astranomy

提供 Direct-API（直接调用模型接口）和 Agentic（通过 Codex 持续对话控制机器人）两种方式。以下命令均在仓库根目录、激活 `.venv` 后执行，会产生模型调用费用。

**Direct-API**：配置支持 Chat Completions 的服务地址、密钥和模型名称，然后运行一个回合：

```bash
export OPENAI_API_KEY="你的密钥"
export OPENAI_BASE_URL="https://api.openai.com/v1"
export AI_NATIVE_MODEL="gpt-6-astra"

python policies/ai_native/direct_api/list_models.py
python policies/ai_native/direct_api/run.py \
  --headless --task BananaInBowlTask --num-envs 1 --num-runs 1

# GUI可视化
python policies/ai_native/direct_api/run.py \
  --visualizer kit --task RubiksCubeTask --num-envs 1 --num-runs 1
```

使用其他服务商时，替换 `OPENAI_BASE_URL` 和 `AI_NATIVE_MODEL`。动作类型、历史观测和推理参数见 [Direct-API 参数说明](policies/ai_native/direct_api/README.md)。

**Agentic**：默认 Codex 已安装并已登录，且当前账号可访问示例使用的 GPT-6.1 Sol。直接复用现有 Codex 配置和登录状态：

```bash
python policies/ai_native/agentic/run.py \
  --model gpt-6.1-sol-medium \
  --headless --embodiment droid --task RubiksCubeTask \
  --seed 0 --num-envs 1 --num-runs 1 \
  --video-mode sensor --renderer realtime \
  --output-folder-name gpt6-example --allow-model-requests
```

示例通过 `--model gpt-6.1-sol-medium` 选择 Sol 的中等思考强度；省略 `--model` 时，runner 默认使用 `gpt-6-astra-medium`，不会沿用 Codex 配置中的默认模型。现有 Codex 登录状态仍会复用。

每个并行进程使用独立的输出目录；无需重新登录或另设 Codex 登录目录。工具、历史记录及视频示范配置见 [Agentic 参数说明](policies/ai_native/agentic/README.md)。Orchestrated 策略尚未提供。

<a id="docker"></a>

### Docker

需要 Docker、NVIDIA Container Toolkit，以及 `nvcr.io/nvidia/isaac-lab:3.0.0-beta2` 镜像的访问权限。在仓库根目录构建并启动：

```bash
./docker/build_docker.sh my-tag
./docker/run_docker.sh my-tag
```

运行脚本会转发 X11 界面并挂载仓库和缓存。镜像内已包含项目代码、资产和 Python 依赖；构建前先执行 `git lfs pull`。如需推送镜像，设置 `ROBOLAB_REGISTRY` 后执行 `./docker/build_docker.sh my-tag --push`。无界面运行及镜像内容见 [Docker 参考](docker/README.md)。

<a id="version-compatibility"></a>

### 版本兼容

| Isaac Sim | Isaac Lab          | 当前状态                               |
| --------- | ------------------ | -------------------------------------- |
| 6.0.1     | 3.0 beta 2 patch 1 | 当前安装配置，使用 `--extra isaac60` |
| 6.2.0     | 3.0 正式版         | 尚未适配，不提供 `isaac62` 安装选项  |

6.2 的正式版移除了当前代码使用的 `sim.render` 配置，官方依赖采用 PyTorch 2.11／CUDA 13.0，不能只修改版本号升级。详见[接口审查和历史验证记录](docs/isaac_sim_6.md)。

<a id="run-without-a-policy"></a>

### 不加载策略运行

```bash
# 使用随机动作运行空回合（无界面模式）
python examples/run_empty.py --headless

# 使用图形界面运行（Isaac Sim 6 需要 --visualizer kit）
python examples/run_empty.py --visualizer kit

# 指定摩擦系数，并读回 PhysX 实际使用的值（参见 docs/physics.md）
python examples/run_empty.py --task BananaInBowlTask --headless --friction 0.5

# 回放录制的示范数据
python examples/run_recorded.py --headless

# 固定机械臂，交替打开和闭合夹爪，检查夹爪动作链路；
# 传感器及视口视频保存至 output/run_gripper_toggle/<task>/
python examples/run_gripper_toggle.py --task BananaInBowlTask --headless

# 使用脚本客户端驱动双 Franka 机器人
python policies/bimanual/run.py --task BimanualLiftToteTask --num-envs 2 --headless

# 使用脚本客户端驱动双臂 YAM 机器人（无需模型权重）
python policies/bimanual/run.py --robot yam --task YamPutEverythingInBoxTask --headless

# 使用 MolmoAct 2 驱动双臂 YAM（需先启动模型服务，参见 policies/molmoact2/README.md）
python policies/molmoact2/run.py --task YamPutEverythingInBoxTask --task-dirs bimanual --num-envs 4 --headless
```

> **回放说明**：`run_recorded.py` 会恢复录制时的初始状态，并以开环方式回放记录的动作，默认使用录制文件旁保存的环境配置（`env_cfg.json`）。不同 Isaac Sim / Isaac Lab 版本的接触动力学存在差异，回放结果可能随版本变化，参见[运行要求](#requirements)。要忠实复现轨迹，录制和回放均需使用单个环境。详情参见[录制回合回放](docs/replay.md)。

<a id="run-with-a-policy"></a>

### 加载策略运行

RoboLab 采用服务端与客户端架构：模型作为独立服务运行，RoboLab 通过推理客户端连接模型。快速测试可使用[基于 OpenPI 的 π0.5](policies/pi0_family/README.md)，并启动关节位置模型权重对应的服务（`scripts/serve_pi05.sh`）。OpenPI 的快捷参数 `--env DROID` 会提供增量动作，而此处的动作空间不接受这类动作。

```bash
cd robolab
uv run --no-sync python policies/pi0_family/run.py --policy pi05 --task BananaInBowlTask --num-envs 10
```

使用[结果面板](#dashboard)查看本地输出目录中的结果。

<a id="common-cli-options"></a>

### 常用命令行选项

```bash
# 无界面运行
python policies/pi0_family/run.py --policy pi05 --headless

# 指定任务（这两个任务适合做基本检查）
python policies/pi0_family/run.py --policy pi05 --task BananaInBowlTask RubiksCubeAndBananaTask

# 按标签选择任务
python policies/pi0_family/run.py --policy pi05 --tag semantics

# 每个任务并行运行 12 个回合
python policies/pi0_family/run.py --policy pi05 --headless --num-envs 12

# 指定摩擦配置（upstream | <mu> | realistic | table.json），配置会记录到运行目录
python policies/pi0_family/run.py --policy pi05 --headless --friction realistic

# 禁用子任务进度跟踪（默认启用；禁用后结果中不包含 score/reason）
python policies/pi0_family/run.py --policy pi05 --disable-subtask

# 恢复先前的运行，跳过已完成的回合
python policies/pi0_family/run.py --policy pi05 --output-folder-name my_previous_run
```

每个任务应使用独立进程。一次调用传入多个任务会预先实例化所有任务的环境，并导致崩溃。这是上游已知问题，参见[环境运行说明](docs/environment_run.md)。

<a id="check-a-run"></a>

### 检查运行结果

```bash
scripts/verify_patches.py output/<run>                # 根据录制数据逐项报告 PASS / FAIL / N/A
scripts/verify_patches.py --summary output/rc*        # 汇总多次运行；存在 FAIL 时退出码为 1
scripts/flag_regression.py --output-dir output        # 对照人工判定评估标记结果
```

详情参见[验证说明](docs/verified/verification.md)。

<a id="example-tasks"></a>

## 任务示例

全部 120 个任务请参见完整的[基准任务库](robolab/tasks/README.md)。

<div align="center">
  <img src="docs/images/Make_sure_all_the_white_mugs_are_upright_so_that_the_opening_is_facing_upwards_0_hstack_3X_fps24_width800.gif" alt="确保所有白色杯子都直立，杯口朝上" width="800"/>
  <br><em>“确保所有白色杯子都直立，杯口朝上。”</em>
</div>

<div align="center">
  <img src="docs/images/Put_all_plastic_bottles_away_in_the_bin_3_hstack_3X_fps24.gif" alt="将所有塑料瓶收进箱子" width="800"/>
  <br><em>“将所有塑料瓶收进箱子。”</em>
</div>

<div align="center">
  <img src="docs/images/Put_the_orange_measuring_cup_and_the_blue_measuring_cup_outside_of_the_plate_0_hstack_3X_fps24_width800.gif" alt="将橙色量杯和蓝色量杯放到盘子外面" width="800"/>
  <br><em>“将橙色量杯和蓝色量杯放到盘子外面。”</em>
</div>

<a id="dashboard"></a>

## 结果面板

项目内置独立的网页面板，用于浏览基准中的场景和任务，并分析实验结果。

```bash
uv run --no-sync robolab-dashboard --host 127.0.0.1
# 在浏览器中打开 http://localhost:8080
```

面板没有身份认证功能，在共享机器上应绑定到 `127.0.0.1`。

功能介绍、命令行参数及 API 接口请参见[面板文档](docs/dashboard.md)。

<a id="documentation"></a>

## 文档

完整文档入口为 [docs/README.md](docs/README.md)，内容包括：

- [环境安装](#installation)与 [Astranomy 配置](#astranomy)
- [物体](docs/objects.md)、[场景](docs/scene.md)、[任务](docs/task.md)：创建和管理资产及基准任务
- [机器人](docs/robots.md)、[相机](docs/camera.md)、[光照](docs/lighting.md)、[背景](docs/background.md)：配置仿真参数
- [环境注册](docs/environment_registration.md)：组合任务、机器人、观测和动作配置
- [推理客户端](policies/README.md)：支持的开源模型及客户端
- [录制回合回放](docs/replay.md)：忠实回放已记录的 HDF5 回合
- [分析与结果](docs/analysis.md)、[数据与输出](docs/data.md)：结果汇总、比较、审计及输出数据格式
- [结果面板](docs/dashboard.md)：交互式浏览基准、任务、场景及评估结果
- [子任务检查](docs/subtask.md)、[条件谓词](docs/task_conditionals.md)、[事件跟踪](docs/event_tracking.md)
- [生态](docs/ecosystem.md)：任务库及基于 RoboLab 构建的项目

<a id="requirements"></a>

## 运行要求

| 依赖      | 版本               |
| --------- | ------------------ |
| Isaac Sim | 6.0.1              |
| Isaac Lab | 3.0 beta 2 patch 1 |
| Python    | 3.12               |
| Linux     | Ubuntu 22.04+      |

> **仿真器版本说明**：不同版本使用不同的 PhysX 构建，抓取、物体静置等涉及接触的动力学行为可能存在差异。基准结果应尽量在相同环境版本下比较。RoboLab 可以转换不同版本录制数据中的四元数布局，但要保证数值层面的回放一致性，仍需使用原始仿真环境版本。

- **磁盘空间**：约 8 GB，其中资产约占 7 GB。
- **GPU**：需要 NVIDIA RTX GPU，建议显存至少 48 GB。参见 [Isaac Lab 硬件要求](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html#system-requirements)和各任务的 [`num_envs` 指南](docs/env_vram_size_guide.md)。
- **运行速度**：在 L40 上、推理耗时约 200 ms 时，单任务 4 环境运行约需 5–17 分钟；每种策略跑完整个基准约需 30 GPU 小时。

<a id="license"></a>

## 许可证

RoboLab 使用 [Apache License 2.0](./LICENSE) 发布，参见 [NOTICE](./NOTICE)。第三方依赖的许可证列于 [THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md)。

<a id="citation"></a>

## 引用

RoboLab 由 NVIDIA 西雅图机器人实验室开发。使用本基准时，请引用以下论文：

```bibtex
@inproceedings{yang2026robolab,
    author    = {Jenai Xuning Yang and Rishit Dagli and Alex Zook and Hugo Hadfield and Ankit Goyal and Stan Birchfield and Fabio Ramos and Jonathan Tremblay},
    title     = {{RoboLab: A High-Fidelity Simulation Benchmark for Analysis of Task Generalist Policies}},
    booktitle = {Proceedings of Robotics: Science and Systems},
    year      = {2026},
    address   = {Sydney, Australia},
    month     = {July},
    url       = {https://arxiv.org/abs/2604.09860}
}
```

报告结果时，请注明仓库提交版本及仿真器版本。

<a id="contributing"></a>

## 参与贡献

变更记录、验证要求及上游致谢请参见[贡献指南](CONTRIBUTING.md)。
