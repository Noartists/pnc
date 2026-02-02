# 翼伞无人机规划与控制系统 (PNC)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

翼伞无人机路径规划与控制项目，包含动力学建模、全局路径规划、轨迹平滑、ADRC控制器、闭环仿真和Web可视化模块。

> **维护者**: [张驰]  
> **联系方式**: [zhangchi9900@gmail.com]  
> **实验室**: [南开大学智能预测自适应实验室]

---

## 项目结构

```
pnc/
├── cfg/                           # 配置文件
│   ├── config.yaml                # 翼伞动力学模型参数
│   └── map_config.yaml            # 地图与任务配置
│
├── models/                        # 动力学模型
│   └── parafoil_model.py          # 8自由度翼伞模型
│
├── planning/                      # 规划模块
│   ├── __init__.py                # 模块导出
│   ├── map_manager.py             # 地图管理器 (障碍物、约束)
│   ├── global_planner.py          # RRT* 全局规划 (旧版)
│   ├── kinodynamic_rrt.py         # ★ Kinodynamic RRT* (新版，含Dubins曲线)
│   ├── path_smoother.py           # 路径平滑器 (样条/Dubins)
│   ├── trajectory_postprocess.py  # 轨迹后处理
│   └── trajectory.py              # 轨迹数据结构
│
├── control/                       # 控制模块
│   ├── __init__.py                # 模块导出
│   └── adrc_controller.py         # ADRC 航向/横向控制器
│
├── simulation/                    # 仿真模块
│   ├── __init__.py                # 模块导出
│   ├── closed_loop_sim.py         # 闭环仿真 (规划+控制+动力学)
│   └── open_loop_test.py          # 开环动力学测试
│
└── visualization/                 # ★ Web可视化模块
    ├── server.py                  # Flask服务器 (端口8080)
    ├── templates/
    │   └── index.html             # Web前端页面
    ├── static/js/
    │   └── app.js                 # Three.js 3D渲染
    └── data/                      # RRT数据文件 (自动生成)
        └── rrt_YYYYMMDD_HHMMSS_*.json
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install numpy scipy matplotlib pyyaml flask tqdm
```

### 2. 运行闭环仿真

```bash
# 使用新的Kinodynamic RRT*规划器（默认）
python simulation/closed_loop_sim.py

# 使用旧的RRT*规划器
python simulation/closed_loop_sim.py --planner=old

# 自定义参数
python simulation/closed_loop_sim.py --max-time=300 --control-dt=0.01
```

### 3. 单独运行规划器并查看可视化

```bash
# 步骤1: 运行规划器（自动导出JSON数据）
python planning/kinodynamic_rrt.py

# 步骤2: 启动Web可视化服务器
python visualization/server.py --port=8080

# 步骤3: 浏览器访问
# http://127.0.0.1:8080
```

---

## 命令行参数速查

### `simulation/closed_loop_sim.py` - 闭环仿真

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--map-config` | `cfg/map_config.yaml` | 地图配置文件 |
| `--model-config` | `cfg/config.yaml` | 动力学模型配置 |
| `--max-time` | 自动 | 最大仿真时间 (s) |
| `--control-dt` | `0.01` | 控制周期 (s) |
| `--dynamics-dt` | `0.002` | 动力学积分步长 (s) |
| `--planner` | `kinodynamic` | 规划器: `kinodynamic` 或 `old` |
| `--position-noise` | `10 -15 0` | 初始位置偏差 [dx, dy, dz] |
| `--heading-noise` | `0.1` | 初始航向偏差 (rad) |

### `visualization/server.py` - Web可视化服务器

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--port` | `8080` | 端口号 |
| `--host` | `127.0.0.1` | 主机地址 |

### `simulation/open_loop_test.py` - 开环测试

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--altitude` | `400` | 初始高度 (m) |
| `--left` | `0.0` | 左操纵绳偏转 [0,1] |
| `--right` | `0.0` | 右操纵绳偏转 [0,1] |

---

## 模块详细说明

### 1. 配置文件 (`cfg/`)

#### `config.yaml` - 翼伞动力学参数
| 参数类别 | 内容 |
|---------|------|
| 质量参数 | 伞体质量、负载质量 |
| 几何参数 | 翼展、面积、伞绳长度等 |
| 气动参数 | 升力、阻力、力矩系数 |

#### `map_config.yaml` - 地图与任务配置
| 参数类别 | 内容 |
|---------|------|
| 地图边界 | X/Y/Z 范围 |
| 起点/终点 | 位置、航向 |
| 禁飞区 | 圆柱体/多边形障碍物 |
| 约束参数 | 最小转弯半径、滑翔比范围、最低高度等 |

---

### 2. 规划模块 (`planning/`)

#### `kinodynamic_rrt.py` - ★ Kinodynamic RRT* 规划器 (推荐)

**特点：**
- 使用 **Dubins曲线** 作为扩展原语，生成平滑路径
- 在规划阶段考虑 **滑翔比约束**
- 支持导出JSON数据用于Web可视化

**主要方法：**
```python
from planning.kinodynamic_rrt import KinodynamicRRTStar

planner = KinodynamicRRTStar(map_manager)
path, info = planner.plan(max_time=30.0)

# 导出数据到visualization/data/目录
planner.export_to_json(path=path, path_length=info.get('path_length'))
```

#### `global_planner.py` - RRT* 规划器 (旧版)

基础的RRT*规划器，不含Dubins曲线。

#### `path_smoother.py` - 路径平滑器

将离散航点平滑为连续轨迹：
- 三次样条插值
- Dubins曲线连接
- 时间参数化

#### `trajectory.py` - 轨迹数据结构

```python
@dataclass
class TrajectoryPoint:
    t: float           # 时间戳 (s)
    position: [x,y,z]  # 位置 (m)
    velocity: [vx,vy,vz]  # 速度 (m/s)
    heading: float     # 航向角 (rad)
    curvature: float   # 曲率 (1/m)
```

---

### 3. 控制模块 (`control/`)

#### `adrc_controller.py` - ADRC 控制器

自抗扰控制器，用于航向和横向跟踪。

**核心组件：**
| 组件 | 功能 |
|------|------|
| TD (跟踪微分器) | 平滑参考信号 |
| ESO (扩展状态观测器) | 估计扰动 |
| SEF (状态误差反馈) | 生成控制量 |

**可调参数：**
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `heading_kp` | 0.8 | 航向比例增益 |
| `heading_kd` | 0.4 | 航向微分增益 |
| `lateral_kp` | 0.002 | 横向误差增益 |
| `lookahead_distance` | 100.0 | 前视距离 (m) |
| `max_deflection` | 1.0 | 最大操纵绳偏转 [0,1] |

---

### 4. 仿真模块 (`simulation/`)

#### `closed_loop_sim.py` - 闭环仿真

整合 **规划 + 控制 + 动力学** 的完整仿真。

**流程：**
```
加载配置 → 可达性检查 → 路径规划 → 轨迹平滑 → 初始化状态
    ↓
循环: 控制器计算 → 设置操纵绳 → 动力学积分 → 更新状态
    ↓
可视化: 3D轨迹、XY平面、跟踪误差、控制输入
```

---

### 5. 可视化模块 (`visualization/`)

#### Web可视化界面

基于 **Three.js** 的3D可视化，支持：
- 查看RRT树的所有节点（按高度着色）
- 查看最终路径
- 查看障碍物
- 鼠标交互（左键旋转、右键平移、滚轮缩放）
- 选择历史数据文件

**使用方法：**
```bash
# 1. 运行规划器生成数据
python planning/kinodynamic_rrt.py

# 2. 启动服务器
python visualization/server.py --port=8080

# 3. 浏览器访问 http://127.0.0.1:8080
```

**数据文件格式：**

每次运行规划器会生成唯一文件名：
```
visualization/data/rrt_20260203_143025_123456_ok.json   # 规划成功
visualization/data/rrt_20260203_143030_654321_fail.json # 规划失败
```

---

## 代码调用示例

```python
# 方式1: 使用闭环仿真器（推荐）
from simulation import ClosedLoopSimulator

sim = ClosedLoopSimulator(
    map_config_path="cfg/map_config.yaml",
    model_config_path="cfg/config.yaml"
)
sim.plan_kinodynamic()  # 使用新规划器
sim.init_state()
sim.run()
sim.visualize()

# 方式2: 分模块调用
from planning.map_manager import MapManager
from planning.kinodynamic_rrt import KinodynamicRRTStar

# 加载地图
map_mgr = MapManager.from_yaml("cfg/map_config.yaml")

# 运行规划
planner = KinodynamicRRTStar(map_mgr)
path, info = planner.plan(max_time=30.0)

# 导出可视化数据
planner.export_to_json(path=path, path_length=info.get('path_length'))
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
