# 翼伞无人机规划与控制系统 (PNC)

翼伞无人机路径规划与控制项目，包含动力学建模、全局路径规划、轨迹平滑和可视化模块。

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
│   ├── map_manager.py          # 地图管理器
│   ├── global_planner.py       # RRT* 全局规划
│   ├── path_smoother.py        # 三次样条路径平滑
│   └── trajectory.py           # 轨迹数据结构
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
| 约束参数 | 最小转弯半径、最低高度等 |
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
- `ParafoilModel` 类：完整动力学模型
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
- `Constraints`: 飞行约束参数

**主要功能:**
- 从 YAML 加载地图配置
- 碰撞检测 (`is_collision`, `is_path_collision`)
- 距离查询
- 支持随机生成障碍物和起终点

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

**可视化:** 6 图对比
- 3D 轨迹对比
- XY 平面轨迹
- 高度剖面
- 转弯半径分布
- 速度剖面
- 航向变化

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

### 4. 可视化模块 (`visualization/`)

#### `map_visualizer.py` - 地图可视化

**`MapVisualizer` 类功能:**
- `plot_2d()`: 2D 俯视图（禁飞区、走廊、轨迹）
- `plot_3d()`: 3D 视图
- `plot_combined()`: 组合视图

---

## 快速使用

### 运行路径规划和平滑

```bash
python planning/path_smoother.py --config=cfg/map_config.yaml
```

**可选参数:**
- `--save-trajectory=output.json`: 保存轨迹文件
- `--save-figure=output.png`: 保存可视化图像

### 代码调用示例

```python
from planning import MapManager, RRTStarPlanner, PathSmoother, Trajectory

# 1. 加载地图
map_mgr = MapManager.from_yaml("cfg/map_config.yaml")

# 2. 全局规划
planner = RRTStarPlanner(map_mgr)
path, info = planner.plan(max_time=30.0)

# 3. 路径平滑
smoother = PathSmoother(
    turn_radius=50.0,
    reference_speed=12.0,
    control_frequency=100
)
trajectory = smoother.smooth(path)

# 4. 使用轨迹
print(trajectory.summary())
trajectory.save("trajectory.json")

# 5. 控制器查询
point = trajectory.interpolate_at(t=5.0)
print(f"位置: {point.position}")
print(f"速度: {point.velocity}")
```

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

## 约束参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 最小转弯半径 | 50 m | 翼伞机动性限制 |
| 最低飞行高度 | 30 m | 安全高度 |
| 终端高度 | 100 m | 进场高度 |
| 安全裕度 | 20 m | 障碍物距离 |

---

## 依赖

```
numpy
scipy
matplotlib
pyyaml
```
