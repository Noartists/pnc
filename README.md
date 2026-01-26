# 翼伞无人机规划与控制系统 (PNC)

翼伞无人机路径规划与控制项目，包含动力学建模、全局路径规划、轨迹平滑、ADRC控制器和闭环仿真模块。

---

## 项目结构

```
pnc/
├── cfg/                        # 配置文件
│   ├── config.yaml             # 翼伞动力学模型参数
│   └── map_config.yaml         # 地图与任务配置
├── models/                     # 动力学模型
│   └── parafoil_model.py       # 8自由度翼伞模型
├── planning/                   # 规划模块
│   ├── __init__.py             # 模块导出
│   ├── map_manager.py          # 地图管理器 (含可达性检查)
│   ├── global_planner.py       # RRT* 全局规划
│   ├── path_smoother.py        # 三次样条路径平滑
│   └── trajectory.py           # 轨迹数据结构
├── control/                    # 控制模块
│   ├── __init__.py             # 模块导出
│   └── adrc_controller.py      # ADRC 控制器
├── simulation/                 # 仿真模块
│   ├── __init__.py             # 模块导出
│   ├── closed_loop_sim.py      # 闭环仿真 (规划+控制+动力学)
│   └── open_loop_test.py       # 开环动力学测试
└── visualization/              # 可视化模块
    └── map_visualizer.py       # 地图与轨迹可视化
```

---

## 模块说明

### 1. 配置文件 (`cfg/`)

#### `config.yaml` - 翼伞动力学参数
| 参数类别 | 内容 |
|---------|------|
| 质量参数 | 伞体质量、负载质量 |
| 几何参数 | 翼展、面积、伞绳长度等 |
| 铰链参数 | 偏航/俯仰刚度和阻尼 |
| 气动参数 | 升力、阻力、力矩系数 |
| 环境参数 | 重力、空气密度 |

#### `map_config.yaml` - 地图与任务配置
| 参数类别 | 内容 |
|---------|------|
| 地图边界 | X/Y/Z 范围 |
| 起点/终点 | 位置、航向、随机化范围 |
| 禁飞区 | 圆柱体/多边形障碍物 |
| 空域走廊 | 允许飞行的多边形区域 |
| 约束参数 | 最小转弯半径、滑翔比、高度余量等 |
| 轨迹参数 | 参考速度 (12 m/s)、控制频率 (100 Hz) |
| 随机化配置 | 障碍物和起终点随机生成 |

---

### 2. 动力学模型 (`models/`)

#### `parafoil_model.py` - 8自由度翼伞模型

**状态向量 (20维):**
| 索引 | 变量 | 描述 |
|------|------|------|
| 0:3 | x, y, z | 惯性坐标系位置 |
| 3:6 | φ, θ, ψ | 伞体欧拉角 |
| 6:8 | θᵣ, ψᵣ | 伞体-负载相对角 |
| 8:11 | u, v, w | 伞体速度 |
| 11:14 | p, q, r | 伞体角速度 |
| 14:17 | - | 负载速度 |
| 17:20 | - | 负载角速度 |

**主要功能:**
- `ParafoilParams` 类：动力学参数管理
- 气动力计算（升力、阻力、侧力）
- 力矩计算（俯仰、滚转、偏航）
- 使用 `scipy.integrate.odeint` 进行数值积分

---

### 3. 规划模块 (`planning/`)

#### `map_manager.py` - 地图管理器

**主要类:**
- `MapManager`: 地图管理主类
- `Cylinder`, `Polygon`: 禁飞区几何体
- `Corridor`: 空域走廊
- `LandingTarget`: 着陆目标（含进场航向）
- `Constraints`: 飞行约束参数（含滑翔比）

**主要功能:**
- 从 YAML 加载地图配置
- 碰撞检测 (`is_collision`, `is_path_collision`)
- **可达性检查** (`check_reachability`, `print_reachability_report`)
- **智能随机化**：确保起终点满足滑翔比约束
- 距离查询

#### `global_planner.py` - RRT* 全局规划器

**算法:** RRT* (Rapidly-exploring Random Tree Star)

**特点:**
- 3D 空间规划
- 禁飞区避障
- 空域走廊约束
- 智能采样策略（目标偏置、路径偏置、进场点偏置）
- Rewire 优化

**输出:** 航点列表 `[[x, y, z], ...]`

#### `path_smoother.py` - 路径平滑器

**算法:** 三次样条全局平滑 (Cubic Spline)

**处理流程:**
1. 航点加密 (增加密度)
2. 弧长参数化
3. 三次样条插值 (X, Y, Z 分别)
4. 时间参数化 (100Hz 采样)
5. 计算速度、航向、曲率

**输出:** `Trajectory` 对象

#### `trajectory.py` - 轨迹数据结构

**`TrajectoryPoint` 数据类:**
```python
@dataclass
class TrajectoryPoint:
    t: float           # 时间戳 (s)
    position: [x,y,z]  # 位置 (m)
    velocity: [vx,vy,vz]  # 速度 (m/s)
    heading: float     # 航向角 (rad)
    curvature: float   # 曲率 (1/m)
```

**`Trajectory` 类功能:**
| 方法 | 描述 |
|------|------|
| `to_array()` | 转换为 [n, 9] 数组 |
| `to_legacy_array()` | 兼容旧格式 [n, 4] |
| `interpolate_at(t)` | 时间插值查询 |
| `resample(new_dt)` | 重采样 |
| `save(path)` | 保存 (.json/.csv/.npy) |
| `load(path)` | 加载文件 |
| `summary()` | 输出摘要信息 |

---

### 4. 控制模块 (`control/`)

#### `adrc_controller.py` - ADRC 控制器

**算法:** 自抗扰控制 (Active Disturbance Rejection Control)

**核心组件:**
| 组件 | 类名 | 功能 |
|------|------|------|
| 跟踪微分器 | `TD` | 平滑参考信号，提取微分 |
| 扩展状态观测器 | `ESO` / `LinearESO` | 估计系统状态和总扰动 |
| 非线性状态误差反馈 | `NLSEF` / `LinearSEF` | 根据误差生成控制量 |
| 组合控制器 | `ADRC` | 整合 TD + ESO + SEF |

**`ParafoilADRCController` 类:**

翼伞航向/横向跟踪控制器，基于 Pure Pursuit 风格的路径跟踪。

**可调超参数:**
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `heading_kp` | 1.0 | 航向比例增益，越大响应越快 |
| `heading_kd` | 0.3 | 航向微分增益，抑制震荡 |
| `heading_eso_omega` | 10.0 | ESO 带宽，越大扰动估计越快 |
| `heading_td_r` | 20.0 | TD 快速因子，越小参考越平滑 |
| `lateral_kp` | 0.003 | 横向误差→航向修正增益 |
| `lookahead_distance` | 100.0 | 前视距离 (m) |
| `max_deflection` | 0.6 | 最大操纵绳偏转量 [0,1] |

**控制输出:**
```python
@dataclass
class ControlOutput:
    delta_left: float      # 左操纵绳 [0,1]
    delta_right: float     # 右操纵绳 [0,1]
    heading_error: float   # 航向误差 (rad)
    cross_track_error: float  # 横向误差 (m)
    along_track_error: float  # 沿轨误差 (m)
    altitude_error: float  # 高度误差 (m)
    ref_heading: float     # 参考航向 (rad)
    ref_position: np.ndarray  # 参考位置
```

---

### 5. 仿真模块 (`simulation/`)

#### `closed_loop_sim.py` - 闭环仿真

整合 **规划 + 控制 + 动力学** 的完整仿真流程。

**`ClosedLoopSimulator` 类:**

| 方法 | 功能 |
|------|------|
| `__init__()` | 加载配置，初始化各模块 |
| `plan()` | 运行 RRT* 规划和路径平滑 |
| `init_state()` | 初始化翼伞状态 |
| `step()` | 单步仿真（控制+动力学积分） |
| `run()` | 运行完整仿真 |
| `visualize()` | 可视化结果 |

**仿真流程:**
```
加载配置 → 可达性检查 → RRT*规划 → 路径平滑 → 初始化状态
    ↓
循环: 控制器计算 → 设置操纵绳 → 动力学积分 → 更新状态
    ↓
可视化: 3D轨迹、XY平面、跟踪误差、控制输入、高度/速度曲线
```

#### `open_loop_test.py` - 开环测试

测试纯动力学模型响应（无控制），用于验证模型参数。

---

## 快速使用

### 1. 运行闭环仿真（推荐）

```bash
python simulation/closed_loop_sim.py
```

**可选参数:**
```bash
python simulation/closed_loop_sim.py \
    --map-config=cfg/map_config.yaml \
    --model-config=cfg/config.yaml \
    --max-time=200 \
    --control-dt=0.01 \
    --dynamics-dt=0.002
```

### 2. 运行开环测试

```bash
python simulation/open_loop_test.py
```

**可选参数:**
```bash
# 指定高度和控制输入
python simulation/open_loop_test.py --altitude=400 --left=0.1 --right=0.0
```

### 3. 单独运行规划

```bash
python planning/path_smoother.py --config=cfg/map_config.yaml
```

### 4. 代码调用示例

```python
from planning import MapManager, RRTStarPlanner, PathSmoother
from control import ParafoilADRCController
from simulation import ClosedLoopSimulator

# 方式1: 使用闭环仿真器 (推荐)
sim = ClosedLoopSimulator(
    map_config_path="cfg/map_config.yaml",
    model_config_path="cfg/config.yaml"
)
sim.plan()
sim.init_state()
sim.run()
sim.visualize()

# 方式2: 分模块调用
# 1. 加载地图并检查可达性
map_mgr = MapManager.from_yaml("cfg/map_config.yaml")
map_mgr.print_reachability_report()

# 2. 全局规划
planner = RRTStarPlanner(map_mgr)
path, info = planner.plan(max_time=30.0)

# 3. 路径平滑
smoother = PathSmoother(turn_radius=50.0, reference_speed=12.0)
trajectory = smoother.smooth(path)

# 4. 创建控制器
controller = ParafoilADRCController(dt=0.01)
controller.set_trajectory(trajectory)

# 5. 控制循环
ctrl = controller.update(current_pos, current_vel, current_heading)
print(f"左绳: {ctrl.delta_left}, 右绳: {ctrl.delta_right}")
```

---

## 可达性分析

系统会自动检查起点到终点的可达性（基于滑翔比约束）：

```
==================================================
  可达性分析报告
==================================================
  起点: (50, 80, 450)
  终点: (1500, 800, 0)
  水平距离: 1612m
  可用高度: 450m
  滑翔比: 6.5:1
--------------------------------------------------
  直线飞行需要高度: 248m
  考虑绕行需要高度: 373m
  高度盈余 (直线): 202m
  高度盈余 (含余量): 77m
--------------------------------------------------
  [✓] 可达
  [!] 建议消高: 77m (约 1.5 圈)
==================================================
```

**随机化时自动保证可达性：** 起点高度会根据终点位置和滑翔比自动调整。

---

## 约束参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| 最小转弯半径 | 50 m | 翼伞机动性限制 |
| 最低飞行高度 | 20 m | 安全高度 |
| 终端高度 | 20 m | 进场高度 |
| 安全裕度 | 15 m | 障碍物距离 |
| **滑翔比** | 6.5 | 水平距离/下降高度 |
| **高度余量** | 50 m | 用于路径绕行和消高 |

---

## 轨迹输出格式

### 完整格式 (9列)
| 列 | 变量 | 单位 |
|----|------|------|
| 0 | t | s |
| 1 | x | m |
| 2 | y | m |
| 3 | z | m |
| 4 | vx | m/s |
| 5 | vy | m/s |
| 6 | vz | m/s |
| 7 | heading | rad |
| 8 | curvature | 1/m |

### 时间参数
- 采样周期: **0.01s** (100Hz)
- 参考速度: **12 m/s**

---

## 依赖

```
numpy
scipy
matplotlib
pyyaml
```

安装依赖:
```bash
pip install numpy scipy matplotlib pyyaml
```
