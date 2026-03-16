# 翼伞无人机规划与控制系统 (PNC)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

翼伞无人机路径规划与控制项目，包含 8-DOF 动力学建模、Kinodynamic RRT* 路径规划、轨迹平滑、ADRC 控制器、闭环仿真、Benchmark 测试框架、Web 可视化，以及基于 Transformer 的 **In-Context Dynamics Model** 学习模块。

> **维护者**: 张驰  
> **联系方式**: zhangchi9900@gmail.com  
> **实验室**: 南开大学智能预测自适应实验室

---

## 目录

- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [In-Context Dynamics Model](#in-context-dynamics-model)
- [Benchmark 测试框架](#benchmark-测试框架)
- [Web 可视化](#web-可视化)
- [命令行参数速查](#命令行参数速查)
- [模块详细说明](#模块详细说明)
- [代码调用示例](#代码调用示例)

---

## 项目结构

```
pnc/
├── cfg/                           # 配置文件
│   ├── config.yaml                # 翼伞动力学模型参数
│   └── map_config.yaml            # 地图与任务配置（支持场景随机化）
│
├── models/                        # 动力学模型
│   └── parafoil_model.py          # 8自由度翼伞模型（20维状态向量）
│
├── planning/                      # 规划模块
│   ├── map_manager.py             # 地图管理器（障碍物、约束、可达性分析）
│   ├── kinodynamic_rrt.py         # ★ Kinodynamic RRT*（Dubins曲线 + 滑翔比约束）
│   ├── trajectory_postprocess.py  # 轨迹后处理（平滑、螺旋消高、时间参数化）
│   └── trajectory.py              # 轨迹数据结构
│
├── control/                       # 控制模块
│   └── adrc_controller.py         # ADRC 航向/横向控制器
│
├── simulation/                    # 仿真模块
│   ├── closed_loop_sim.py         # ★ 闭环仿真（规划+控制+动力学）
│   └── open_loop_test.py          # 开环动力学测试
│
├── benchmark/                     # ★ Benchmark 测试框架
│   ├── runner.py                  # Benchmark 运行器（批量测试）
│   ├── metrics.py                 # 失败检测与质量指标计算
│   ├── outputs.py                 # 结果导出（metrics.json, case.json）
│   └── rng_manager.py             # 随机数管理（可复现性）
│
├── learning/                      # ★ In-Context Dynamics Learning
│   ├── configs/                   # 训练配置
│   │   ├── test.yaml              #   流水线验证（200条, d=64, 10 epochs）
│   │   ├── pilot.yaml             #   小规模实验（1000条, d=128, 50 epochs）
│   │   └── full.yaml              #   正式训练（50000条, d=256, 200 epochs）
│   ├── data_generation/           # 数据生成
│   │   ├── dataset_generator.py   #   轨迹生成器（HDF5 输出）
│   │   ├── domain_randomization.py#   域随机化（参数/风场/执行器/噪声）
│   │   └── control_policies.py    #   控制策略（随机/激励/闭环+噪声）
│   ├── dataset.py                 # PyTorch Dataset + 归一化 + 状态编码
│   ├── model.py                   # Transformer 架构 + MLP 基线
│   ├── trainer.py                 # 训练循环（课程学习/wandb/TensorBoard）
│   ├── train.py                   # 训练 CLI 入口
│   ├── evaluate.py                # 评测框架（IID/OOD/消融/图表）
│   ├── mpc_controller.py          # MPPI 闭环控制器
│   ├── visualize_dataset.py       # 数据集可视化
│   └── requirements.txt           # 依赖列表
│
└── visualization/                 # Web 可视化模块
    ├── server.py                  # Flask 服务器
    ├── templates/index.html       # 前端页面
    └── static/js/app.js           # Three.js 3D 渲染
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install numpy scipy matplotlib pyyaml flask tqdm
```

### 2. 运行单次闭环仿真

```bash
# 基本运行（自动规划 + 控制 + 可视化）
python simulation/closed_loop_sim.py

# 指定随机种子（可复现）
python simulation/closed_loop_sim.py --seed=42

# 导出仿真数据
python simulation/closed_loop_sim.py --output-dir=results/
```

### 3. 训练 In-Context Dynamics Model（详见 [专节](#in-context-dynamics-model)）

```bash
# 安装额外依赖
pip install -r learning/requirements.txt

# 快速验证流水线
python -m learning.data_generation.dataset_generator --n-trajs 200 --n-steps 500 --output learning/datasets/test.h5 --dataset-name test
python -m learning.train --config learning/configs/test.yaml

# 评测
python -m learning.evaluate --checkpoint learning/checkpoints/best.pt --generate-eval-data --K-values 5 10 20 30
```

### 4. 运行 Benchmark 批量测试

```bash
# 运行 10 个随机种子
python -m benchmark.runner --seeds 10

# 运行指定范围
python -m benchmark.runner --seeds 1-50

# 继续之前中断的测试
python -m benchmark.runner --seeds 100 --resume
```

### 5. 启动 Web 可视化

```bash
# 启动服务器
python visualization/server.py --port=8080

# 浏览器访问
# http://127.0.0.1:8080
```

---

## In-Context Dynamics Model

基于 Transformer 的翼伞动力学基础模型，核心思想：通过观察短历史序列（状态-动作对），让模型在测试时**自适应**未知的物理参数、风场等操作条件，无需重新训练。

### 核心思路

传统物理模型（`models/parafoil_model.py`）的参数在实际飞行中不精确（磨损、载荷变化、未知风场等）。本模块训练一个序列模型来**隐式辨识**这些未知因素：

```
输入:  最近 K 步 [(state_t, action_t), ..., (state_{t+K}, action_{t+K})]
输出:  未来 H 步的状态变化量 [Δstate_{t+K+1}, ..., Δstate_{t+K+H}]
```

训练时通过**域随机化**（Domain Randomization）生成大量不同条件下的轨迹，模型从中学会：给定一段历史，推断当前条件并做出准确预测。

### 完整流水线

```
Step 1: 数据生成 ──→ Step 2: 训练 ──→ Step 3: 评测 ──→ Step 4: MPC闭环
```

#### Step 1: 生成训练数据

```bash
# 生成 5000 条轨迹，每条 2000 步，保存为 HDF5
python -m learning.data_generation.dataset_generator \
    --n-trajs 5000 \
    --n-steps 2000 \
    --output learning/datasets/pilot.h5 \
    --dataset-name pilot
```

每条轨迹会随机化：
- **物理参数**：质量、翼面积、气动系数等（±10%~30%）
- **风场**：AR(1) 随机风，风速 0~5 m/s，随机方向和湍流
- **执行器**：一阶滞后 + 随机延迟
- **传感器噪声**：位置/速度/姿态加性高斯噪声
- **控制策略**：随机分段常值、频率扫描激励、带噪声的 ADRC 闭环

#### Step 2: 训练模型

```bash
# 使用配置文件训练（推荐）
python -m learning.train --config learning/configs/pilot.yaml --tensorboard

# TensorBoard 实时监控
tensorboard --logdir learning/checkpoints/runs/
```

三档配置：

| 配置 | 数据量 | 模型大小 | 训练轮次 | 用途 |
|------|--------|---------|---------|------|
| `test.yaml` | 200 条 | d=64, L=2 | 10 | 流水线验证 |
| `pilot.yaml` | 1000 条 | d=128, L=4 | 50 | 小规模实验 |
| `full.yaml` | 50000 条 | d=256, L=6 | 200 | 正式训练 |

训练自动保存到 `learning/checkpoints/`：`best.pt`（验证集最优）、`epoch_XXXX.pt`（周期保存）、`final.pt`。

#### Step 3: 评测

```bash
python -m learning.evaluate \
    --checkpoint learning/checkpoints/best.pt \
    --generate-eval-data \
    --K-values 5 10 20 30 50 \
    --max-trajs 50
```

自动生成三类评测数据并出图：

| 评测集 | 说明 | 目的 |
|--------|------|------|
| **IID** | 同分布：参数/风场范围与训练相同 | 验证基本拟合能力 |
| **OOD-Param** | 分布外参数：扰动范围扩大到 1.8 倍 | 验证对未知物理参数的适应 |
| **OOD-Wind** | 分布外风场：风速 5~10 m/s（训练仅 0~5） | 验证对未知环境的适应 |

输出保存到 `learning/eval_results/<timestamp>_<checkpoint>/`，包含：
- `run_info.json` — 模型配置与运行元信息
- `eval_metrics.json` — 量化指标（RMSE / MAE / 逐通道误差）
- `fig1_context_adaptation.png` — RMSE vs Context Length K（in-context 适应曲线）
- `fig2_horizon_error.png` — 误差随预测步长的增长
- `fig3_channel_heatmap.png` — 各状态通道 RMSE 热力图

#### Step 4: MPC 闭环验证

```bash
python -m learning.mpc_controller \
    --checkpoint learning/checkpoints/best.pt \
    --norm-stats learning/datasets/pilot_norm.npz \
    --n-trials 20
```

使用 MPPI（Model Predictive Path Integral）控制器，将学到的 Transformer 动力学模型作为内部预测器，在扰动 ODE 环境中闭环控制翼伞飞行。

### 模型架构

```
Input MLP ──→ Transformer Encoder (L layers) ──→ Cross-Attention ──→ Output Heads
  ↑                    ↑                              ↑                    ↓
(K, token_dim)   Learnable PosEmb                Query: horizon      (H, target_dim)
                                                  embeddings         状态变化量 Δstate
```

- **输入**：K 步历史 token（归一化状态 + 动作，sin/cos 角度编码），维度 = 22
- **输出**：H 步未来状态变化量（17 维 delta：角速率、速度、角速度变化）
- **训练目标**：多步 delta 预测，Huber loss + 时间衰减权重 + 课程学习

---

## Benchmark 测试框架

Benchmark 模块用于批量测试闭环系统性能，支持：

- **批量运行**：指定种子数量或范围，自动运行多次仿真
- **进度显示**：实时显示规划/仿真进度和成功率统计
- **结果导出**：每个种子生成 `metrics.json` 和 `case.json`
- **汇总报告**：生成 `summary.csv` 和终端统计报告
- **可复现性**：基于 `SeedSequence` 的随机数管理

### 使用方法

```bash
# 基本用法
python -m benchmark.runner --seeds 100

# 指定场景和配置
python -m benchmark.runner --seeds 50 --scene my_scene --map-config cfg/my_map.yaml

# 显示详细输出
python -m benchmark.runner --seeds 10 -v
```

### 输出结构

```
benchmark/outputs/
└── exp_20260206_143025/          # 实验目录（按时间戳命名）
    └── default/                   # 场景名称
        ├── summary.csv            # 汇总表（所有种子的指标）
        ├── seed_001/
        │   ├── metrics.json       # 关键指标（成功/失败、误差、耗时）
        │   └── case.json          # 完整数据（轨迹、控制量、配置）
        ├── seed_002/
        │   └── ...
        └── ...
```

### 评估指标

| 类别 | 指标 | 说明 |
|------|------|------|
| **成功判定** | `landing_error` | 落点误差 < 20m 为成功 |
| **质量指标** | `ADE` | 平均跟踪误差 (m) |
| | `FDE` | 终点偏差 (m) |
| | `control_effort` | 控制量积分 |
| **失败原因** | `hard_*` | 硬失败（禁飞区、姿态超限、数值爆炸） |
| | `soft_*` | 软失败（跟踪发散、长时间饱和、超时） |

### 进度条说明

运行时会显示实时进度：

```
Seed 042 [仿真 45.2%]: 100%|████████████| 50/50 [08:32<00:00] Plan:48✓/2✗ Ctrl:35✓/13✗
```

- `Seed 042 [仿真 45.2%]`：当前种子和阶段进度
- `50/50`：已完成/总数
- `Plan:48✓/2✗`：规划成功 48 次，失败 2 次
- `Ctrl:35✓/13✗`：控制成功 35 次，失败 13 次

---

## Web 可视化

基于 Three.js 的 3D 可视化界面，支持：

- 查看 RRT 树节点（按高度着色）
- 查看规划路径和实际轨迹
- 查看禁飞区和走廊
- 浏览 Benchmark 历史结果
- 鼠标交互（左键旋转、右键平移、滚轮缩放）

### 启动方法

```bash
python visualization/server.py --port=8080 --host=127.0.0.1
```

### 功能

1. **加载规划数据**：从 `visualization/data/` 加载 RRT 规划结果
2. **加载仿真数据**：从 `visualization/data/` 加载闭环仿真结果
3. **浏览 Benchmark**：从 `benchmark/outputs/` 浏览历史测试结果

---

## 命令行参数速查

### `python -m learning.data_generation.dataset_generator` - 数据生成

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--n-trajs` | `1000` | 轨迹数量 |
| `--n-steps` | `2000` | 每条轨迹仿真步数 |
| `--output` | `learning/datasets/data.h5` | 输出 HDF5 路径 |
| `--dataset-name` | `data` | 数据集名称（用于归一化文件命名） |
| `--n-workers` | `1` | 并行进程数（Windows 下强制为 1） |

### `python -m learning.train` - 模型训练

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--config` | *必选* | YAML 配置文件路径 |
| `--tensorboard` | `false` | 启用 TensorBoard 日志 |

### `python -m learning.evaluate` - 模型评测

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--checkpoint` | *必选* | 模型 checkpoint 路径 |
| `--eval-data-dir` | `learning/datasets` | 评测数据目录 |
| `--generate-eval-data` | `false` | 是否先生成评测数据 |
| `--K-values` | `5 10 20 30 50` | Context length 消融值 |
| `--max-trajs` | `50` | 每个 split 最多使用的轨迹数 |
| `--output-base-dir` | `learning/eval_results` | 输出根目录 |

### `python -m learning.mpc_controller` - MPC 闭环验证

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--checkpoint` | *必选* | 模型 checkpoint 路径 |
| `--norm-stats` | *必选* | 归一化统计文件路径 (`.npz`) |
| `--n-trials` | `20` | 闭环仿真试次数 |
| `--output-dir` | `learning/mpc_results` | 输出目录 |

### `python -m benchmark.runner` - Benchmark 运行器

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--seeds` | `10` | 种子数量或范围，如 `100`、`1-50`、`1,5,10` |
| `--scene` | `default` | 场景名称 |
| `--map-config` | `cfg/map_config.yaml` | 地图配置文件 |
| `--model-config` | `cfg/config.yaml` | 动力学模型配置 |
| `--output-dir` | `benchmark/outputs` | 输出目录 |
| `--resume` | - | 跳过已有结果，继续测试 |
| `-v, --verbose` | - | 显示详细输出 |

### `python simulation/closed_loop_sim.py` - 闭环仿真

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--map-config` | `cfg/map_config.yaml` | 地图配置文件 |
| `--model-config` | `cfg/config.yaml` | 动力学模型配置 |
| `--seed` | 随机 | 随机种子（用于复现） |
| `--max-time` | 自动 | 最大仿真时间 (s) |
| `--output-dir` | - | 数据导出目录 |

### `python visualization/server.py` - Web 可视化

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--port` | `8080` | 端口号 |
| `--host` | `127.0.0.1` | 主机地址 |

---

## 模块详细说明

### 1. 规划模块 (`planning/`)

#### `kinodynamic_rrt.py` - Kinodynamic RRT* 规划器

**特点：**
- 使用 **Dubins 曲线** 作为扩展原语，生成满足最小转弯半径约束的平滑路径
- 在规划阶段考虑 **滑翔比约束**，确保路径在翼伞物理能力范围内
- **引导采样**：50% 概率在起点-终点走廊内采样，提高规划效率
- 支持导出 JSON 数据用于 Web 可视化

#### `trajectory_postprocess.py` - 轨迹后处理器

**功能：**
- 路径平滑（三次样条插值）
- 螺旋消高注入（当高度盈余过多时）
- 时间参数化（基于参考速度）
- 滑翔比验证

#### `map_manager.py` - 地图管理器

**功能：**
- 加载和管理地图配置（边界、障碍物、走廊）
- 支持场景随机化（随机起点、终点、障碍物）
- 可达性分析（检查起点到终点是否可达）
- 碰撞检测

### 2. 控制模块 (`control/`)

#### `adrc_controller.py` - ADRC 控制器

自抗扰控制器，用于航向和横向跟踪。

**核心组件：**
| 组件 | 功能 |
|------|------|
| TD (跟踪微分器) | 平滑参考信号 |
| ESO (扩展状态观测器) | 估计扰动 |
| SEF (状态误差反馈) | 生成控制量 |

**输出：**
- `delta_left`：左操纵绳偏转 [0, 1]
- `delta_right`：右操纵绳偏转 [0, 1]

### 3. 学习模块 (`learning/`)

#### 数据生成 (`data_generation/`)

- `domain_randomization.py` — 物理参数扰动规格（`PARAM_PERTURBATION_SPEC`）、风场模型（`WindField`，支持恒定/AR(1)模式）、执行器滞后（`ActuatorModel`）、传感器噪声（`SensorNoiseModel`）
- `control_policies.py` — 随机分段常值、频率扫描激励信号（chirp/step/doublet）、带噪声 ADRC 闭环
- `dataset_generator.py` — 主生成器，集成以上模块，运行 ODE 仿真并输出 HDF5

#### 模型 (`model.py`)

- `InContextDynamicsTransformer` — Transformer Encoder + Cross-Attention + Per-Horizon Output Heads
- `MLPDynamicsModel` — 简单 MLP 基线

#### 训练 (`trainer.py`)

- `MultiStepLoss` — 多步预测损失（Huber + 时间衰减 + 通道权重）
- 课程学习（逐步增加预测步长 H）
- Warmup + CosineAnnealing 学习率调度
- 支持 wandb 和 TensorBoard 日志

#### 评测 (`evaluate.py`)

- 三类评测集生成（IID / OOD-Param / OOD-Wind）
- Context length 消融实验
- 自回归 rollout + 逐通道误差分析
- 自动生成 publication-quality 图表

#### MPC 控制 (`mpc_controller.py`)

- MPPI 采样优化控制器
- 支持 Learned Dynamics / ODE Dynamics 两种 rollout 后端
- 闭环仿真（控制器规划 + 扰动 ODE 环境执行）

### 4. 仿真模块 (`simulation/`)

#### `closed_loop_sim.py` - 闭环仿真器

整合 **规划 + 控制 + 动力学** 的完整仿真。

**流程：**
```
加载配置 → 规划路径 → 轨迹后处理 → 初始化状态
    ↓
循环: 控制器计算 → 设置操纵绳 → 动力学积分 → 状态更新 → 失败检测
    ↓
输出: 可视化 / 数据导出
```

### 5. Benchmark 模块 (`benchmark/`)

#### `runner.py` - Benchmark 运行器

批量运行仿真，收集统计数据。

#### `metrics.py` - 指标计算与失败检测

**硬失败 (Hard Fail)**：
- H1: 进入禁飞区 / 安全间距不足
- H2: 姿态超限（滚转 > 60°，俯仰 > 45°）
- H3: 数值爆炸

**软失败 (Soft Fail)**：
- S1: 跟踪发散（持续误差过大）
- S3: 长时间控制饱和
- S4: 超时

#### `rng_manager.py` - 随机数管理

基于 `numpy.random.SeedSequence` 的子 RNG 管理，确保：
- 相同种子产生相同结果
- 不同模块的随机数互不影响

---

## 代码调用示例

### 方式 1：使用闭环仿真器

```python
from simulation.closed_loop_sim import ClosedLoopSimulator

# 创建仿真器
sim = ClosedLoopSimulator(
    map_config_path="cfg/map_config.yaml",
    model_config_path="cfg/config.yaml",
    seed=42  # 可选，用于复现
)

# 规划
sim.plan(max_time=30.0)

# 初始化状态
sim.init_state()

# 运行仿真
sim.run(max_time=200.0)

# 可视化
sim.visualize()
```

### 方式 2：分模块调用

```python
from planning.map_manager import MapManager
from planning.kinodynamic_rrt import KinodynamicRRTStar
from planning.trajectory_postprocess import TrajectoryPostprocessor

# 加载地图
map_mgr = MapManager.from_yaml("cfg/map_config.yaml")

# 运行规划
planner = KinodynamicRRTStar(map_mgr)
path, info = planner.plan(max_time=30.0)

# 轨迹后处理
postprocessor = TrajectoryPostprocessor(map_mgr)
trajectory = postprocessor.process(path, reference_speed=9.0)
```

### 方式 3：运行 Benchmark

```python
from benchmark.runner import BenchmarkRunner, ExperimentConfig

config = ExperimentConfig(
    name="my_experiment",
    seeds=list(range(1, 101)),  # 1-100
    scene="default",
    map_config="cfg/map_config.yaml"
)

runner = BenchmarkRunner(config)
results = runner.run_all()
```

---

## 约束参数

| 参数 | 配置键 | 默认值 | 说明 |
|------|--------|--------|------|
| 最小转弯半径 | `min_turn_radius` | 50 m | 翼伞机动性限制 |
| 最低飞行高度 | `min_altitude` | 20 m | 安全高度 |
| 最大滑翔比 | `glide_ratio` | 6.48 | 最大水平距离/下降高度 |
| 最小滑翔比 | `min_glide_ratio` | 2.47 | 最小水平距离/下降高度 |
| 安全裕度 | `safety_margin` | 15 m | 障碍物距离 |
| 着陆半径 | `landing_radius` | 20 m | 成功判定阈值 |

---

## 依赖

**基础模块**（规划/控制/仿真/可视化）:

```bash
pip install numpy scipy matplotlib pyyaml flask tqdm
```

**学习模块**（额外依赖）:

```bash
pip install -r learning/requirements.txt
# 核心: torch>=2.0, h5py>=3.8, pyyaml, numpy, scipy, matplotlib
# 可选: wandb (实验跟踪)
```

---

## 许可证

本项目采用 [MIT 许可证](LICENSE)。
