"""
地图管理模块

功能:
- 禁飞区管理 (圆形、多边形、圆柱体)
- 空域走廊管理
- 碰撞检测
- 距离查询
"""

import yaml
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from abc import ABC, abstractmethod


# ============================================================
#                     几何基元
# ============================================================

class Obstacle(ABC):
    """障碍物基类"""

    @abstractmethod
    def contains(self, point: np.ndarray) -> bool:
        """判断点是否在障碍物内"""
        pass

    @abstractmethod
    def distance(self, point: np.ndarray) -> float:
        """计算点到障碍物的距离 (负值表示在内部)"""
        pass

    @abstractmethod
    def get_boundary_2d(self, n_points: int = 50) -> np.ndarray:
        """获取2D边界点用于可视化"""
        pass


class Circle(Obstacle):
    """2D圆形障碍物"""

    def __init__(self, center: np.ndarray, radius: float, name: str = ""):
        self.center = np.array(center[:2])
        self.radius = radius
        self.name = name

    def contains(self, point: np.ndarray) -> bool:
        dist = np.linalg.norm(point[:2] - self.center)
        return dist <= self.radius

    def distance(self, point: np.ndarray) -> float:
        return np.linalg.norm(point[:2] - self.center) - self.radius

    def get_boundary_2d(self, n_points: int = 50) -> np.ndarray:
        theta = np.linspace(0, 2 * np.pi, n_points)
        x = self.center[0] + self.radius * np.cos(theta)
        y = self.center[1] + self.radius * np.sin(theta)
        return np.column_stack([x, y])


class Cylinder(Obstacle):
    """3D圆柱体障碍物"""

    def __init__(self, center: np.ndarray, radius: float,
                 z_min: float, z_max: float, name: str = ""):
        self.center = np.array(center[:2])
        self.radius = radius
        self.z_min = z_min
        self.z_max = z_max
        self.name = name

    def contains(self, point: np.ndarray) -> bool:
        xy_dist = np.linalg.norm(point[:2] - self.center)
        z = point[2] if len(point) > 2 else 0
        return xy_dist <= self.radius and self.z_min <= z <= self.z_max

    def distance(self, point: np.ndarray) -> float:
        xy_dist = np.linalg.norm(point[:2] - self.center) - self.radius
        z = point[2] if len(point) > 2 else 0

        if z < self.z_min:
            z_dist = self.z_min - z
        elif z > self.z_max:
            z_dist = z - self.z_max
        else:
            z_dist = 0

        if xy_dist < 0 and z_dist == 0:
            return xy_dist  # 在圆柱内部
        elif xy_dist >= 0 and z_dist == 0:
            return xy_dist  # 在高度范围内，返回xy距离
        elif xy_dist < 0 and z_dist > 0:
            return z_dist   # 在xy范围内但高度在外
        else:
            return np.sqrt(xy_dist**2 + z_dist**2)

    def get_boundary_2d(self, n_points: int = 50) -> np.ndarray:
        theta = np.linspace(0, 2 * np.pi, n_points)
        x = self.center[0] + self.radius * np.cos(theta)
        y = self.center[1] + self.radius * np.sin(theta)
        return np.column_stack([x, y])


class Polygon(Obstacle):
    """2D多边形障碍物"""

    def __init__(self, vertices: np.ndarray, name: str = ""):
        self.vertices = np.array(vertices)
        self.name = name
        self._precompute()

    def _precompute(self):
        """预计算用于快速检测"""
        self.n = len(self.vertices)
        self.edges = []
        for i in range(self.n):
            p1 = self.vertices[i]
            p2 = self.vertices[(i + 1) % self.n]
            self.edges.append((p1, p2))

    def contains(self, point: np.ndarray) -> bool:
        """射线法判断点是否在多边形内"""
        x, y = point[0], point[1]
        inside = False

        j = self.n - 1
        for i in range(self.n):
            xi, yi = self.vertices[i]
            xj, yj = self.vertices[j]

            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i

        return inside

    def distance(self, point: np.ndarray) -> float:
        """计算点到多边形的距离"""
        p = point[:2]

        # 如果在内部，返回负的最小边距离
        if self.contains(point):
            min_dist = float('inf')
            for p1, p2 in self.edges:
                dist = self._point_to_segment_distance(p, p1, p2)
                min_dist = min(min_dist, dist)
            return -min_dist

        # 在外部，返回到最近边的距离
        min_dist = float('inf')
        for p1, p2 in self.edges:
            dist = self._point_to_segment_distance(p, p1, p2)
            min_dist = min(min_dist, dist)
        return min_dist

    def _point_to_segment_distance(self, p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        """计算点到线段的距离"""
        ab = b - a
        ap = p - a
        t = np.clip(np.dot(ap, ab) / (np.dot(ab, ab) + 1e-10), 0, 1)
        closest = a + t * ab
        return np.linalg.norm(p - closest)

    def get_boundary_2d(self, n_points: int = 50) -> np.ndarray:
        # 闭合多边形
        return np.vstack([self.vertices, self.vertices[0]])


class Prism(Obstacle):
    """3D多边形棱柱障碍物"""

    def __init__(self, vertices: np.ndarray, z_min: float, z_max: float, name: str = ""):
        self.polygon = Polygon(vertices, name)
        self.z_min = z_min
        self.z_max = z_max
        self.name = name

    def contains(self, point: np.ndarray) -> bool:
        z = point[2] if len(point) > 2 else 0
        return self.polygon.contains(point) and self.z_min <= z <= self.z_max

    def distance(self, point: np.ndarray) -> float:
        xy_dist = self.polygon.distance(point)
        z = point[2] if len(point) > 2 else 0

        if z < self.z_min:
            z_dist = self.z_min - z
        elif z > self.z_max:
            z_dist = z - self.z_max
        else:
            z_dist = 0

        if xy_dist < 0 and z_dist == 0:
            return xy_dist
        elif xy_dist >= 0 and z_dist == 0:
            return xy_dist
        elif xy_dist < 0 and z_dist > 0:
            return z_dist
        else:
            return np.sqrt(xy_dist**2 + z_dist**2)

    def get_boundary_2d(self, n_points: int = 50) -> np.ndarray:
        return self.polygon.get_boundary_2d(n_points)


# ============================================================
#                     数据结构
# ============================================================

@dataclass
class Waypoint:
    """航点"""
    x: float
    y: float
    z: float
    heading: Optional[float] = None  # 航向角 (rad)

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    @classmethod
    def from_array(cls, arr: np.ndarray, heading: Optional[float] = None):
        return cls(arr[0], arr[1], arr[2], heading)


@dataclass
class LandingTarget:
    """着陆目标"""
    position: np.ndarray                # [x, y, z]
    radius: float                       # 着陆区半径
    desired_approach_heading: float     # 期望进场航向 (rad)
    max_approach_length: float          # 最大进场直线段长度 (m)
    min_approach_length: float = 50.0  # 最小进场直线段长度 (m)
    approach_heading_tolerance: float = np.pi / 2  # 航向调整容差 (rad)，默认±90°

    # 为了兼容性，保留旧的字段名作为属性
    @property
    def approach_heading(self) -> float:
        """兼容性：返回期望进场航向"""
        return self.desired_approach_heading

    @property
    def approach_length(self) -> float:
        """兼容性：返回最大进场长度"""
        return self.max_approach_length

    def get_approach_point(self, terminal_altitude: float = 80) -> np.ndarray:
        """获取进场点位置（基于期望参数，实际使用时应调用 MapManager.find_safe_approach_point）

        参数:
            terminal_altitude: 进场点高度 (m)，默认80m
        """
        dx = -self.max_approach_length * np.cos(self.desired_approach_heading)
        dy = -self.max_approach_length * np.sin(self.desired_approach_heading)
        return self.position + np.array([dx, dy, terminal_altitude])


@dataclass
class Constraints:
    """约束参数"""
    min_turn_radius: float = 50
    max_bank_angle: float = 30
    min_altitude: float = 30
    terminal_altitude: float = 100
    safety_margin: float = 20
    glide_ratio: float = 6.48          # 最大滑翔比 (不拉绳)
    min_glide_ratio: float = 2.47      # 最小滑翔比 (左右拉满)
    altitude_margin: float = 50        # 高度余量 (m)，用于路径绕行和消高


# ============================================================
#                     地图管理器
# ============================================================

class MapManager:
    """地图管理器"""

    def __init__(self):
        self.bounds = {
            'x_min': -np.inf, 'x_max': np.inf,
            'y_min': -np.inf, 'y_max': np.inf,
            'z_min': 0, 'z_max': np.inf
        }
        self.start: Optional[Waypoint] = None
        self.target: Optional[LandingTarget] = None
        self.obstacles: List[Obstacle] = []
        self.corridor: Optional[Polygon] = None
        self.corridor_z_range: Tuple[float, float] = (0, np.inf)
        self.constraints = Constraints()
        # 外部传入的 RNG（用于 benchmark 可复现性）
        self._rng = None
        # 静默模式
        self._quiet = False
    
    def _rand_uniform(self, low: float, high: float) -> float:
        """统一的随机数生成（支持外部 RNG）"""
        if self._rng is not None:
            return self._rng.uniform(low, high)
        return np.random.uniform(low, high)
    
    def _rand_int(self, low: int, high: int) -> int:
        """统一的随机整数生成（支持外部 RNG）"""
        if self._rng is not None:
            return self._rng.integers(low, high)
        return np.random.randint(low, high)
    
    def _rand_choice(self, a, size=None, replace=True, p=None):
        """统一的随机选择（支持外部 RNG）"""
        if self._rng is not None:
            return self._rng.choice(a, size=size, replace=replace, p=p)
        return np.random.choice(a, size=size, replace=replace, p=p)

    @classmethod
    def from_yaml(cls, yaml_path: str, rng=None, quiet: bool = False) -> 'MapManager':
        """
        从YAML文件加载地图
        
        参数:
            yaml_path: YAML配置文件路径
            rng: 可选的 numpy.random.Generator，用于 benchmark 的可复现随机化
                 如果提供了 rng，则忽略配置文件中的 seed
            quiet: 静默模式（不打印详细信息）
        """
        with open(yaml_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)

        manager = cls()
        manager._quiet = quiet

        # 边界
        if 'bounds' in cfg:
            manager.bounds.update(cfg['bounds'])

        # 起点
        if 'start' in cfg:
            s = cfg['start']
            heading = np.radians(s.get('heading', 0))
            manager.start = Waypoint(s['x'], s['y'], s['z'], heading)

        # 目标
        if 'target' in cfg:
            t = cfg['target']
            # 兼容旧配置：如果使用旧字段名，自动转换
            desired_heading = t.get('desired_approach_heading', t.get('approach_heading', 45))
            max_length = t.get('max_approach_length', t.get('approach_length', 150))
            min_length = t.get('min_approach_length', 50)
            heading_tolerance = t.get('approach_heading_tolerance', 90)

            manager.target = LandingTarget(
                position=np.array([t['x'], t['y'], t['z']]),
                radius=t['radius'],
                desired_approach_heading=np.radians(desired_heading),
                max_approach_length=max_length,
                min_approach_length=min_length,
                approach_heading_tolerance=np.radians(heading_tolerance)
            )

        # 禁飞区
        if 'no_fly_zones' in cfg:
            for zone in cfg['no_fly_zones']:
                obs = manager._create_obstacle(zone)
                if obs:
                    manager.obstacles.append(obs)

        # 空域走廊
        if 'corridor' in cfg and cfg['corridor'].get('enabled', False):
            c = cfg['corridor']
            manager.corridor = Polygon(np.array(c['boundary']), "corridor")
            manager.corridor_z_range = (c.get('z_min', 0), c.get('z_max', np.inf))

        # 约束参数
        if 'constraints' in cfg:
            c = cfg['constraints']
            manager.constraints = Constraints(
                min_turn_radius=c.get('min_turn_radius', 50),
                max_bank_angle=c.get('max_bank_angle', 30),
                min_altitude=c.get('min_altitude', 30),
                terminal_altitude=c.get('terminal_altitude', 100),
                safety_margin=c.get('safety_margin', 20),
                glide_ratio=c.get('glide_ratio', 6.48),
                min_glide_ratio=c.get('min_glide_ratio', 2.47),
                altitude_margin=c.get('altitude_margin', 50)
            )

        # 随机化
        if 'randomization' in cfg and cfg['randomization'].get('enabled', False):
            r = cfg['randomization']
            
            # 如果外部传入了 rng，则使用它；否则使用配置文件中的 seed
            if rng is not None:
                # 使用外部传入的 RNG（benchmark 模式）
                manager._rng = rng
            else:
                # 使用配置文件中的 seed
                seed = r.get('seed', None)
                if seed is not None:
                    np.random.seed(seed)
                manager._rng = None

            # 随机化起点终点
            if r.get('randomize_endpoints', False):
                manager._randomize_endpoints(cfg)

            # 随机化障碍物
            manager.randomize_obstacles(
                n_obstacles=r.get('n_obstacles', 15),
                radius_range=tuple(r.get('radius_range', [40, 100])),
                height_range=tuple(r.get('height_range', [200, 600])),
                clear_radius_start=r.get('clear_radius_start', 150),
                clear_radius_target=r.get('clear_radius_target', 200),
                flight_zone_ratio=r.get('flight_zone_ratio', 0.75),
                seed=None  # 已经设置过seed了
            )

        return manager

    def _randomize_endpoints(self, cfg: dict):
        """
        随机化起点和终点，确保可达性
        
        可达性条件 (基于滑翔比):
            起点高度 >= 水平距离 / 滑翔比 + 高度余量
        
        策略:
            1. 先随机终点
            2. 计算需要的最小起点高度
            3. 随机起点时确保高度足够
        """
        glide_ratio = self.constraints.glide_ratio
        altitude_margin = self.constraints.altitude_margin
        
        # 1. 先随机化终点
        target_x, target_y, target_z = 0, 0, 0
        if 'target' in cfg and 'random' in cfg['target']:
            t = cfg['target']
            r = t['random']
            target_x = self._rand_uniform(*r['x_range'])
            target_y = self._rand_uniform(*r['y_range'])
            target_z = r.get('z', 0)  # 地面高度固定
            desired_heading = self._rand_uniform(0, 2 * np.pi)

            # 兼容新旧配置格式
            max_length = t.get('max_approach_length', t.get('approach_length', 200))
            min_length = t.get('min_approach_length', 50)
            heading_tolerance = t.get('approach_heading_tolerance', 90)

            self.target = LandingTarget(
                position=np.array([target_x, target_y, target_z]),
                radius=t.get('radius', 20),
                desired_approach_heading=desired_heading,
                max_approach_length=max_length,
                min_approach_length=min_length,
                approach_heading_tolerance=np.radians(heading_tolerance)
            )
            if not self._quiet:
                print(f"随机终点: ({target_x:.0f}, {target_y:.0f}, {target_z:.0f})")
        elif self.target is not None:
            target_x, target_y, target_z = self.target.position
        
        # 2. 随机化起点，确保可达性
        if 'start' in cfg and 'random' in cfg['start']:
            s = cfg['start']
            r = s['random']
            
            max_attempts = 100
            for attempt in range(max_attempts):
                # 随机 XY
                start_x = self._rand_uniform(*r['x_range'])
                start_y = self._rand_uniform(*r['y_range'])
                
                # 计算水平距离
                horizontal_dist = np.sqrt((start_x - target_x)**2 + (start_y - target_y)**2)
                
                # 计算需要的最小高度 (考虑路径可能绕行，乘以 1.3 系数)
                path_factor = 1.3  # 路径绕行系数
                min_altitude_required = (horizontal_dist * path_factor) / glide_ratio + target_z + altitude_margin
                
                # 随机高度，确保不小于最小需要高度
                z_range = r['z_range']
                z_min = max(z_range[0], min_altitude_required)
                z_max = z_range[1]
                
                if z_min <= z_max:
                    # 可达，生成起点
                    start_z = self._rand_uniform(z_min, z_max)
                    heading = self._rand_uniform(0, 2 * np.pi)
                    self.start = Waypoint(start_x, start_y, start_z, heading)
                    
                    excess_altitude = start_z - (horizontal_dist / glide_ratio + target_z)
                    if not self._quiet:
                        print(f"随机起点: ({start_x:.0f}, {start_y:.0f}, {start_z:.0f})")
                        print(f"  水平距离: {horizontal_dist:.0f}m, 需要最小高度: {min_altitude_required:.0f}m")
                        print(f"  高度盈余: {excess_altitude:.0f}m (可用于路径绕行/消高)")
                    return
                
                # 当前 XY 无法满足可达性，重试
                if attempt == max_attempts - 1:
                    # 最后尝试：强制使用最大高度
                    start_z = z_range[1]
                    heading = self._rand_uniform(0, 2 * np.pi)
                    self.start = Waypoint(start_x, start_y, start_z, heading)
                    
                    if not self._quiet:
                        print(f"[警告] 可达性受限，使用最大高度")
                        print(f"随机起点: ({start_x:.0f}, {start_y:.0f}, {start_z:.0f})")
                        print(f"  水平距离: {horizontal_dist:.0f}m, 需要最小高度: {min_altitude_required:.0f}m")
                        
                        if start_z < min_altitude_required:
                            print(f"  [警告] 高度不足! 差 {min_altitude_required - start_z:.0f}m")
                    return

    def check_reachability(self, path_factor: float = 1.3) -> dict:
        """
        检查从起点到终点的可达性
        
        参数:
            path_factor: 路径绕行系数 (实际路径长度 / 直线距离)
        
        返回:
            dict: 可达性分析结果
        """
        if self.start is None or self.target is None:
            return {'reachable': False, 'reason': '起点或终点未设置'}
        
        start_pos = self.start.to_array()
        target_pos = self.target.position
        
        # 水平距离
        horizontal_dist = np.linalg.norm(start_pos[:2] - target_pos[:2])
        
        # 高度差
        altitude_available = start_pos[2] - target_pos[2]
        
        # 滑翔比约束
        glide_ratio = self.constraints.glide_ratio
        altitude_margin = self.constraints.altitude_margin
        
        # 直线飞行需要的高度
        altitude_needed_direct = horizontal_dist / glide_ratio
        
        # 考虑绕行的高度需求
        altitude_needed_with_path = (horizontal_dist * path_factor) / glide_ratio + altitude_margin
        
        # 高度盈余
        altitude_excess = altitude_available - altitude_needed_direct
        altitude_excess_with_margin = altitude_available - altitude_needed_with_path
        
        # 消高需求 (如果高度过剩)
        spiral_altitude = 0
        spiral_turns = 0
        if altitude_excess_with_margin > 0:
            # 高度盈余，可能需要消高
            # 每圈消耗高度 = 2π × 转弯半径 / 滑翔比
            turn_radius = self.constraints.min_turn_radius
            altitude_per_turn = 2 * np.pi * turn_radius / glide_ratio
            spiral_altitude = altitude_excess_with_margin
            spiral_turns = spiral_altitude / altitude_per_turn
        
        result = {
            'reachable': altitude_excess >= 0,
            'horizontal_distance': horizontal_dist,
            'altitude_available': altitude_available,
            'altitude_needed_direct': altitude_needed_direct,
            'altitude_needed_with_path': altitude_needed_with_path,
            'altitude_excess': altitude_excess,
            'altitude_excess_with_margin': altitude_excess_with_margin,
            'glide_ratio': glide_ratio,
            'spiral_altitude': spiral_altitude,
            'spiral_turns': spiral_turns,
        }
        
        return result
    
    def print_reachability_report(self):
        """打印可达性报告（静默模式下只返回结果不打印）"""
        r = self.check_reachability()
        
        if self._quiet:
            return r
        
        print("\n" + "=" * 50)
        print("  可达性分析报告")
        print("=" * 50)
        print(f"  起点: ({self.start.x:.0f}, {self.start.y:.0f}, {self.start.z:.0f})")
        print(f"  终点: ({self.target.position[0]:.0f}, {self.target.position[1]:.0f}, {self.target.position[2]:.0f})")
        print(f"  水平距离: {r['horizontal_distance']:.0f}m")
        print(f"  可用高度: {r['altitude_available']:.0f}m")
        print(f"  滑翔比: {r['glide_ratio']:.1f}:1")
        print("-" * 50)
        print(f"  直线飞行需要高度: {r['altitude_needed_direct']:.0f}m")
        print(f"  考虑绕行需要高度: {r['altitude_needed_with_path']:.0f}m")
        print(f"  高度盈余 (直线): {r['altitude_excess']:.0f}m")
        print(f"  高度盈余 (含余量): {r['altitude_excess_with_margin']:.0f}m")
        print("-" * 50)
        
        if r['reachable']:
            print("  [✓] 可达")
            if r['spiral_turns'] > 0.5:
                print(f"  [!] 建议消高: {r['spiral_altitude']:.0f}m (约 {r['spiral_turns']:.1f} 圈)")
        else:
            deficit = -r['altitude_excess']
            print(f"  [✗] 不可达! 高度不足 {deficit:.0f}m")
            print(f"      建议: 提高起点高度至 {self.start.z + deficit + 50:.0f}m 以上")
        
        print("=" * 50 + "\n")
        
        return r

    def _create_obstacle(self, zone: dict) -> Optional[Obstacle]:
        """根据配置创建障碍物"""
        zone_type = zone.get('type', 'circle')
        name = zone.get('name', '')

        if zone_type == 'circle':
            return Circle(zone['center'], zone['radius'], name)

        elif zone_type == 'cylinder':
            return Cylinder(
                zone['center'], zone['radius'],
                zone.get('z_min', 0), zone.get('z_max', np.inf),
                name
            )

        elif zone_type == 'polygon':
            vertices = np.array(zone['vertices'])
            z_min = zone.get('z_min', 0)
            z_max = zone.get('z_max', np.inf)
            if z_min == 0 and z_max == np.inf:
                return Polygon(vertices, name)
            else:
                return Prism(vertices, z_min, z_max, name)

        return None

    def randomize_obstacles(self, n_obstacles: int = 15,
                            radius_range: Tuple[float, float] = (40, 100),
                            height_range: Tuple[float, float] = (200, 600),
                            clear_radius_start: float = 150,
                            clear_radius_target: float = 200,
                            flight_zone_ratio: float = 0.75,
                            seed: Optional[int] = None):
        """
        随机生成障碍物 (圆柱体和多边形棱柱混合)

        参数:
            n_obstacles: 障碍物数量
            radius_range: 半径/尺寸范围 (min, max)
            height_range: 高度范围 (min, max)
            clear_radius_start: 起点周围清空半径
            clear_radius_target: 目标周围清空半径
            flight_zone_ratio: 飞行区域内障碍物比例
            seed: 随机种子
        """
        if seed is not None:
            np.random.seed(seed)

        # 清空现有障碍物
        self.obstacles = []

        # 获取生成区域
        x_min = self.bounds['x_min'] + radius_range[1]
        x_max = self.bounds['x_max'] - radius_range[1]
        y_min = self.bounds['y_min'] + radius_range[1]
        y_max = self.bounds['y_max'] - radius_range[1]

        # 起点和目标位置
        start_pos = self.start.to_array()[:2] if self.start else np.array([0, 0])
        target_pos = self.target.position[:2] if self.target else np.array([x_max, y_max])

        # 计算飞行区域 (起点到终点的椭圆区域)
        flight_center = (start_pos + target_pos) / 2
        flight_length = np.linalg.norm(target_pos - start_pos)
        flight_dir = (target_pos - start_pos) / (flight_length + 1e-6)
        flight_perp = np.array([-flight_dir[1], flight_dir[0]])

        generated = 0
        max_attempts = n_obstacles * 30

        for attempt in range(max_attempts):
            if generated >= n_obstacles:
                break

            # 决定是在飞行区域内还是区域外生成
            in_flight_zone = self._rand_uniform(0, 1) < flight_zone_ratio

            if in_flight_zone:
                # 在飞行区域内生成 (沿飞行路径的椭圆区域)
                t = self._rand_uniform(-0.6, 1.1)  # 沿飞行方向的位置
                s = self._rand_uniform(-0.4, 0.4)  # 垂直于飞行方向的偏移
                cx = flight_center[0] + t * flight_length * flight_dir[0] + s * flight_length * flight_perp[0]
                cy = flight_center[1] + t * flight_length * flight_dir[1] + s * flight_length * flight_perp[1]
            else:
                # 在整个区域内随机生成
                cx = self._rand_uniform(x_min, x_max)
                cy = self._rand_uniform(y_min, y_max)

            # 确保在边界内
            cx = np.clip(cx, x_min, x_max)
            cy = np.clip(cy, y_min, y_max)
            center = np.array([cx, cy])

            # 随机尺寸和高度
            size = self._rand_uniform(*radius_range)
            height = self._rand_uniform(*height_range)

            # 检查是否与起点/目标太近
            if start_pos is not None:
                if np.linalg.norm(center - start_pos) < clear_radius_start + size:
                    continue

            if target_pos is not None:
                if np.linalg.norm(center - target_pos) < clear_radius_target + size:
                    continue

            # 检查是否与已有障碍物重叠
            overlap = False
            for obs in self.obstacles:
                obs_center = obs.center if hasattr(obs, 'center') else np.mean(obs.polygon.vertices, axis=0)
                obs_size = obs.radius if hasattr(obs, 'radius') else np.max(np.linalg.norm(obs.polygon.vertices - obs_center, axis=1))
                dist = np.linalg.norm(center - obs_center)
                if dist < size + obs_size + 30:  # 30m间隙
                    overlap = True
                    break

            if overlap:
                continue

            # 随机选择障碍物类型 (40% 圆柱, 60% 多边形)
            name = f"建筑{generated + 1}"
            obs_type = self._rand_choice(['cylinder', 'polygon'], p=[0.4, 0.6])

            if obs_type == 'cylinder':
                obs = Cylinder(center, size, 0, height, name)
            else:
                # 生成随机多边形
                vertices = self._generate_random_polygon(center, size)
                obs = Prism(vertices, 0, height, name)

            self.obstacles.append(obs)
            generated += 1

        if not self._quiet:
            print(f"随机生成了 {generated} 个障碍物")

    def _generate_random_polygon(self, center: np.ndarray, size: float) -> np.ndarray:
        """生成随机多边形顶点"""
        # 随机选择形状类型
        shape_type = self._rand_choice(['rectangle', 'l_shape', 'random'])

        if shape_type == 'rectangle':
            # 随机矩形
            w = size * self._rand_uniform(0.6, 1.4)
            h = size * self._rand_uniform(0.6, 1.4)
            angle = self._rand_uniform(0, np.pi / 2)
            corners = np.array([[-w/2, -h/2], [w/2, -h/2], [w/2, h/2], [-w/2, h/2]])
            # 旋转
            rot = np.array([[np.cos(angle), -np.sin(angle)],
                           [np.sin(angle), np.cos(angle)]])
            vertices = (rot @ corners.T).T + center

        elif shape_type == 'l_shape':
            # L形建筑
            w = size * self._rand_uniform(0.8, 1.2)
            h = size * self._rand_uniform(0.8, 1.2)
            cut = self._rand_uniform(0.3, 0.5)
            vertices = np.array([
                [0, 0], [w, 0], [w, h * cut],
                [w * cut, h * cut], [w * cut, h], [0, h]
            ])
            vertices = vertices - np.array([w/2, h/2]) + center
            # 随机旋转
            angle = self._rand_choice([0, np.pi/2, np.pi, 3*np.pi/2])
            rot = np.array([[np.cos(angle), -np.sin(angle)],
                           [np.sin(angle), np.cos(angle)]])
            vertices = (rot @ (vertices - center).T).T + center

        else:
            # 随机凸多边形
            n_vertices = self._rand_int(4, 7)
            angles = np.sort(np.array([self._rand_uniform(0, 2 * np.pi) for _ in range(n_vertices)]))
            radii = np.array([size * self._rand_uniform(0.5, 1.0) for _ in range(n_vertices)])
            vertices = np.column_stack([
                center[0] + radii * np.cos(angles),
                center[1] + radii * np.sin(angles)
            ])

        return vertices

    # -------- 碰撞检测 --------

    def is_collision(self, point: np.ndarray, margin: Optional[float] = None) -> bool:
        """
        检查点是否发生碰撞

        参数:
            point: [x, y, z] 位置
            margin: 安全裕度，None则使用默认值

        返回:
            True 如果碰撞
        """
        if margin is None:
            margin = self.constraints.safety_margin

        # 检查边界
        if not self._in_bounds(point):
            return True

        # 检查走廊
        if not self._in_corridor(point):
            return True

        # 检查禁飞区
        for obs in self.obstacles:
            if obs.distance(point) < margin:
                return True

        return False

    def is_path_collision(self, p1: np.ndarray, p2: np.ndarray,
                          n_samples: int = 20, margin: Optional[float] = None) -> bool:
        """
        检查路径段是否发生碰撞

        参数:
            p1, p2: 路径起点和终点
            n_samples: 采样点数
            margin: 安全裕度
        """
        for t in np.linspace(0, 1, n_samples):
            point = p1 + t * (p2 - p1)
            if self.is_collision(point, margin):
                return True
        return False

    def _in_bounds(self, point: np.ndarray) -> bool:
        """检查点是否在边界内"""
        x, y = point[0], point[1]
        z = point[2] if len(point) > 2 else 0

        return (self.bounds['x_min'] <= x <= self.bounds['x_max'] and
                self.bounds['y_min'] <= y <= self.bounds['y_max'] and
                self.bounds['z_min'] <= z <= self.bounds['z_max'])

    def _in_corridor(self, point: np.ndarray) -> bool:
        """检查点是否在走廊内"""
        if self.corridor is None:
            return True

        z = point[2] if len(point) > 2 else 0
        z_min, z_max = self.corridor_z_range

        return self.corridor.contains(point) and z_min <= z <= z_max

    # -------- 距离查询 --------

    def min_obstacle_distance(self, point: np.ndarray) -> Tuple[float, Optional[str]]:
        """
        计算点到最近障碍物的距离

        返回:
            (距离, 障碍物名称)
        """
        min_dist = np.inf
        closest_name = None

        for obs in self.obstacles:
            dist = obs.distance(point)
            if dist < min_dist:
                min_dist = dist
                closest_name = obs.name

        return min_dist, closest_name

    def corridor_distance(self, point: np.ndarray) -> float:
        """计算点到走廊边界的距离 (负值表示在外部)"""
        if self.corridor is None:
            return np.inf
        return -self.corridor.distance(point)  # 取反，在内部为正

    # -------- 进场点搜索 --------

    def is_in_bounds(self, point: np.ndarray) -> bool:
        """检查点是否在地图边界内（已有_in_bounds，此为公开接口）"""
        return self._in_bounds(point)

    def is_safe_approach(self, approach_point: np.ndarray, target_pos: np.ndarray) -> bool:
        """
        检查进场点及进场路径是否安全

        参数:
            approach_point: 进场点 [x, y, z]
            target_pos: 目标点 [x, y, z]

        返回:
            True 如果安全，False 否则
        """
        # 1. 进场点在地图边界内
        if not self._in_bounds(approach_point):
            return False

        # 2. 进场点在空域走廊内
        if not self._in_corridor(approach_point):
            return False

        # 3. 进场点无碰撞
        if self.is_collision(approach_point):
            return False

        # 4. 进场直线段无碰撞
        if self.is_path_collision(approach_point, target_pos):
            return False

        return True

    def find_safe_approach_point(self,
                                 target_pos: np.ndarray,
                                 altitude: float,
                                 desired_heading: Optional[float] = None,
                                 max_length: Optional[float] = None,
                                 min_length: Optional[float] = None,
                                 heading_tolerance: Optional[float] = None) -> Tuple[Optional[np.ndarray], float, float]:
        """
        动态查找安全的进场点

        策略：
        1. 沿期望航向从短到长采样不同长度的进场段
        2. 如果都不安全，尝试调整航向（±30°、±60°等）
        3. 返回找到的最长的安全进场段

        参数:
            target_pos: 目标位置 [x, y, z]
            altitude: 进场点高度
            desired_heading: 期望进场航向 (rad)，None则使用target配置
            max_length: 最大进场长度，None则使用target配置
            min_length: 最小进场长度，None则使用target配置
            heading_tolerance: 航向调整容差 (rad)，None则使用target配置

        返回:
            (approach_point, actual_heading, actual_length) 或 (None, 0, 0)
        """
        if self.target is None:
            if not self._quiet:
                print("    [警告] 未定义目标，无法查找进场点")
            return None, 0.0, 0.0

        # 使用target配置作为默认值
        if desired_heading is None:
            desired_heading = self.target.desired_approach_heading
        if max_length is None:
            max_length = self.target.max_approach_length
        if min_length is None:
            min_length = self.target.min_approach_length
        if heading_tolerance is None:
            heading_tolerance = self.target.approach_heading_tolerance

        if not self._quiet:
            print(f"    [进场点搜索] 期望航向={np.degrees(desired_heading):.1f}°, "
                  f"长度范围=[{min_length:.0f}, {max_length:.0f}]m, "
                  f"容差=±{np.degrees(heading_tolerance):.1f}°")

        # 定义采样长度序列（从长到短，优先选择长的）
        length_samples = []
        step = 25  # 每25m采样一个长度
        current_length = max_length
        while current_length >= min_length:
            length_samples.append(current_length)
            current_length -= step
        # 确保包含最小长度
        if min_length not in length_samples:
            length_samples.append(min_length)

        # 定义航向偏移序列（从0开始，逐渐增大偏移）
        heading_offsets = [0]  # 首先尝试期望航向
        for offset_deg in [15, 30, 45, 60, 75, 90]:
            offset_rad = np.radians(offset_deg)
            if offset_rad <= heading_tolerance:
                heading_offsets.extend([offset_rad, -offset_rad])

        # 搜索安全的进场点
        best_approach = None
        best_heading = 0.0
        best_length = 0.0

        for heading_offset in heading_offsets:
            test_heading = desired_heading + heading_offset

            for length in length_samples:
                # 计算候选进场点
                dx = -length * np.cos(test_heading)
                dy = -length * np.sin(test_heading)
                candidate = target_pos.copy()
                candidate[0] += dx
                candidate[1] += dy
                candidate[2] = altitude

                # 检查是否安全
                if self.is_safe_approach(candidate, target_pos):
                    # 找到安全点，如果比当前最佳更长，则更新
                    if length > best_length:
                        best_approach = candidate
                        best_heading = test_heading
                        best_length = length
                        if not self._quiet:
                            print(f"    [进场点搜索] 找到安全进场点: "
                                  f"航向={np.degrees(test_heading):.1f}° (偏移{np.degrees(heading_offset):.1f}°), "
                                  f"长度={length:.0f}m")
                    # 由于从长到短搜索，找到就可以跳过后续更短的
                    break

            # 如果在当前航向找到了最大长度，直接返回（最优解）
            if best_length == max_length:
                break

        if best_approach is not None:
            if not self._quiet:
                print(f"    [进场点搜索] ✓ 最终选择: "
                      f"航向={np.degrees(best_heading):.1f}°, 长度={best_length:.0f}m")
            return best_approach, best_heading, best_length
        else:
            if not self._quiet:
                print(f"    [进场点搜索] ✗ 未找到安全进场点！")
            return None, 0.0, 0.0

    # -------- 可视化数据 --------

    def get_visualization_data(self) -> dict:
        """获取可视化所需的数据"""
        data = {
            'bounds': self.bounds,
            'obstacles': [],
            'corridor': None,
            'start': None,
            'target': None
        }

        # 障碍物边界
        for obs in self.obstacles:
            data['obstacles'].append({
                'name': obs.name,
                'boundary': obs.get_boundary_2d(),
                'z_range': (obs.z_min, obs.z_max) if hasattr(obs, 'z_min') else (0, np.inf)
            })

        # 走廊边界
        if self.corridor:
            data['corridor'] = {
                'boundary': self.corridor.get_boundary_2d(),
                'z_range': self.corridor_z_range
            }

        # 起点终点
        if self.start:
            data['start'] = self.start.to_array()
        if self.target:
            data['target'] = {
                'position': self.target.position,
                'radius': self.target.radius,
                'approach_heading': self.target.approach_heading,
                'approach_point': self.target.get_approach_point()
            }

        return data


# ============================================================
#                     测试
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="地图管理测试")
    parser.add_argument("--config", type=str, required=True, help="地图配置文件路径")
    args = parser.parse_args()

    # 加载地图
    map_mgr = MapManager.from_yaml(args.config)

    print("=" * 50)
    print("地图加载成功")
    print("=" * 50)

    print(f"\n边界: x=[{map_mgr.bounds['x_min']}, {map_mgr.bounds['x_max']}], "
          f"y=[{map_mgr.bounds['y_min']}, {map_mgr.bounds['y_max']}], "
          f"z=[{map_mgr.bounds['z_min']}, {map_mgr.bounds['z_max']}]")

    if map_mgr.start:
        print(f"\n起点: ({map_mgr.start.x}, {map_mgr.start.y}, {map_mgr.start.z})")

    if map_mgr.target:
        print(f"目标: ({map_mgr.target.position[0]}, {map_mgr.target.position[1]}, "
              f"{map_mgr.target.position[2]}), 半径={map_mgr.target.radius}m")
        print(f"进场航向: {np.degrees(map_mgr.target.approach_heading):.1f}°")

    print(f"\n禁飞区数量: {len(map_mgr.obstacles)}")
    for obs in map_mgr.obstacles:
        print(f"  - {obs.name}")

    print(f"\n走廊启用: {map_mgr.corridor is not None}")

    print(f"\n约束参数:")
    print(f"  最小转弯半径: {map_mgr.constraints.min_turn_radius}m")
    print(f"  安全裕度: {map_mgr.constraints.safety_margin}m")

    # 碰撞检测测试
    print("\n" + "=" * 50)
    print("碰撞检测测试")
    print("=" * 50)

    test_points = [
        np.array([0, 0, 1000]),      # 起点
        np.array([600, 400, 100]),   # 建筑物A内
        np.array([600, 400, 200]),   # 建筑物A上方
        np.array([1500, 800, 50]),   # 目标附近
        np.array([1000, 1000, 500]), # 走廊内
    ]

    for p in test_points:
        collision = map_mgr.is_collision(p)
        dist, name = map_mgr.min_obstacle_distance(p)
        print(f"点 ({p[0]}, {p[1]}, {p[2]}): "
              f"碰撞={'是' if collision else '否'}, "
              f"最近障碍物距离={dist:.1f}m ({name})")
