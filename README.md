# 翼伞无人机规划与控制系统 (PNC)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

翼伞无人机路径规划与控制项目，包含动力学建模、Kinodynamic RRT* 路径规划、轨迹平滑、ADRC 控制器、闭环仿真、Benchmark 测试框架和 Web 可视化模块。

> **维护者**: 张驰  
> **联系方式**: zhangchi9900@gmail.com  
> **实验室**: 南开大学智能预测自适应实验室

---

## 目录

- [项目结构](#项目结构)
- [快速开始](#快速开始)
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
│   └── parafoil_model.py          # 8自由度翼伞模型
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
│   ├── rng_manager.py             # 随机数管理（可复现性）
│   └── outputs/                   # 测试结果输出目录
│       └── exp_YYYYMMDD_HHMMSS/   # 每次实验的结果
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

### 3. 运行 Benchmark 批量测试

```bash
# 运行 10 个随机种子
python -m benchmark.runner --seeds 10

# 运行指定范围
python -m benchmark.runner --seeds 1-50

# 继续之前中断的测试
python -m benchmark.runner --seeds 100 --resume
```

### 4. 启动 Web 可视化

```bash
# 启动服务器
python visualization/server.py --port=8080

# 浏览器访问
# http://127.0.0.1:8080
```

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

### 3. 仿真模块 (`simulation/`)

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

### 4. Benchmark 模块 (`benchmark/`)

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

```
numpy
scipy
matplotlib
pyyaml
flask
tqdm
```

安装全部依赖:
```bash
pip install numpy scipy matplotlib pyyaml flask tqdm
```

---

## 许可证

本项目采用 [MIT 许可证](LICENSE)。
