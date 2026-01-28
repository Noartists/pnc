"""
全局路径规划模块

基于 RRT* 算法，考虑:
- 禁飞区避障
- 空域走廊约束
- 最小转弯半径约束
- 3D空间规划
"""

import os
import sys
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import time

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# 添加项目根目录到路径
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from planning.map_manager import MapManager, Waypoint


@dataclass
class RRTNode:
    """RRT树节点"""
    position: np.ndarray      # [x, y, z]
    parent: Optional[int]     # 父节点索引
    cost: float = 0.0         # 从起点到该节点的代价

    def __hash__(self):
        return id(self)


class RRTStarPlanner:
    """RRT* 全局路径规划器"""

    def __init__(self, map_manager: MapManager):
        self.map = map_manager
        self.nodes: List[RRTNode] = []
        self.goal_reached = False

        # 规划参数
        self.step_size = 15.0           # 扩展步长 (m)，小步长获得更多航点
        self.goal_sample_rate = 0.12    # 目标采样概率
        self.approach_sample_rate = 0.18  # 进场点采样概率
        self.path_bias_rate = 0.30      # 路径偏置采样概率
        self.search_radius = 60.0       # 近邻搜索半径
        self.max_iterations = 20000     # 最大迭代次数
        self.goal_threshold = 25.0      # 到达目标阈值

        # 约束参数
        self.min_turn_radius = map_manager.constraints.min_turn_radius
        self.min_altitude = map_manager.constraints.min_altitude
        self.vertical_weight = 1.0      # 垂直方向权重

        # 进场点 (考虑入场方向)
        self.approach_point = None
        if map_manager.target:
            terminal_alt = map_manager.constraints.terminal_altitude
            self.approach_point = map_manager.target.get_approach_point(terminal_alt)

    def plan(self, start: np.ndarray = None, goal: np.ndarray = None,
             max_time: float = 30.0, add_final_target: bool = True) -> Tuple[Optional[List[np.ndarray]], dict]:
        """
        规划从起点到目标的路径 (考虑入场方向)

        规划策略：起点 -> 进场点 -> 目标
        进场点确保从正确方向接近着陆区

        参数:
            start: 起点 [x, y, z]，默认使用地图起点
            goal: 目标 [x, y, z]，默认使用地图目标
            max_time: 最大规划时间 (秒)
            add_final_target: 是否在路径末尾添加最终目标点（默认True）

        返回:
            path: 路径点列表，失败返回 None
            info: 规划信息字典
        """
        self._add_final_target = add_final_target  # 保存参数供_extract_path使用
        # 初始化
        if start is None:
            start = self.map.start.to_array()
        if goal is None:
            # 先规划到进场点，再到目标
            goal = self.approach_point.copy() if self.approach_point is not None else self.map.target.position.copy()

        self.start = start
        self.goal = goal
        self.final_target = self.map.target.position.copy() if self.map.target else goal
        self.nodes = [RRTNode(start, None, 0.0)]
        self.goal_reached = False
        self.goal_node_idx = None

        start_time = time.time()
        iterations = 0

        # 创建进度条
        pbar = None
        if HAS_TQDM:
            from tqdm import tqdm
            pbar = tqdm(
                desc="    RRT* 规划中",
                unit=" iter",
                dynamic_ncols=True,
                leave=True
            )

        # RRT* 主循环
        for i in range(self.max_iterations):
            iterations = i + 1

            # 超时检查
            if time.time() - start_time > max_time:
                if pbar is not None:
                    pbar.set_postfix_str(f"节点:{len(self.nodes)} | 超时!")
                    pbar.close()
                break

            # 随机采样
            if np.random.random() < self.goal_sample_rate:
                sample = goal.copy()
            else:
                sample = self._random_sample()

            # 找最近节点
            nearest_idx = self._nearest_node(sample)
            nearest = self.nodes[nearest_idx]

            # 向采样点扩展
            new_pos = self._steer(nearest.position, sample)

            # 碰撞检测
            if self.map.is_collision(new_pos):
                if pbar is not None:
                    pbar.update(1)
                continue
            if self.map.is_path_collision(nearest.position, new_pos):
                if pbar is not None:
                    pbar.update(1)
                continue

            # 在搜索半径内找近邻节点
            near_indices = self._near_nodes(new_pos)

            # 选择最优父节点
            min_cost = nearest.cost + self._cost(nearest.position, new_pos)
            best_parent = nearest_idx

            for idx in near_indices:
                node = self.nodes[idx]
                cost = node.cost + self._cost(node.position, new_pos)
                if cost < min_cost and not self.map.is_path_collision(node.position, new_pos):
                    min_cost = cost
                    best_parent = idx

            # 添加新节点
            new_node = RRTNode(new_pos, best_parent, min_cost)
            new_idx = len(self.nodes)
            self.nodes.append(new_node)

            # 重新布线 (Rewire)
            self._rewire(new_idx, near_indices)

            # 更新进度条
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix_str(f"节点:{len(self.nodes)}")

            # 检查是否到达目标
            if not self.goal_reached:  # 只在第一次找到时处理
                if np.linalg.norm(new_pos - goal) < self.goal_threshold:
                    if not self.map.is_path_collision(new_pos, goal):
                        self.goal_reached = True
                        self.goal_node_idx = new_idx  # 记录连接到目标的节点
                        # 添加目标节点
                        goal_cost = new_node.cost + self._cost(new_pos, goal)
                        goal_node = RRTNode(goal, new_idx, goal_cost)
                        self.nodes.append(goal_node)

                        # 更新进度条显示成功
                        if pbar is not None:
                            pbar.set_postfix_str(f"节点:{len(self.nodes)} | ✓ 找到路径!")
                            pbar.close()
                        break

        # 确保进度条关闭
        if pbar is not None:
            try:
                pbar.close()
            except:
                pass

        # 提取路径
        elapsed = time.time() - start_time
        info = {
            'iterations': iterations,
            'nodes': len(self.nodes),
            'time': elapsed,
            'success': self.goal_reached
        }

        if self.goal_reached:
            # 使用plan调用时传入的参数
            path = self._extract_path(add_final_target=self._add_final_target)
            info['path_length'] = self._path_length(path)
            return path, info
        else:
            return None, info

    def _random_sample(self) -> np.ndarray:
        """
        智能随机采样策略：
        - path_bias_rate: 沿起点到目标连线附近采样
        - approach_sample_rate: 在进场点附近采样
        - goal_sample_rate: 直接采样目标
        - 其余: 全局随机采样
        """
        rand = np.random.random()

        if rand < self.goal_sample_rate:
            # 直接采样目标
            return self.goal.copy()

        elif rand < self.goal_sample_rate + self.approach_sample_rate:
            # 进场点附近采样
            if self.approach_point is not None:
                offset = np.random.randn(3) * np.array([100, 100, 30])
                sample = self.approach_point + offset
                sample[2] = max(sample[2], self.min_altitude)
                return sample

        elif rand < self.goal_sample_rate + self.approach_sample_rate + self.path_bias_rate:
            # 沿起点到目标连线附近采样 (椭圆高斯分布)
            t = np.random.uniform(0, 1)
            base = self.start + t * (self.goal - self.start)

            # 添加垂直于路径的随机偏移
            direction = self.goal - self.start
            dist = np.linalg.norm(direction[:2])
            if dist > 1e-6:
                perp = np.array([-direction[1], direction[0], 0]) / dist
                lateral_offset = np.random.randn() * dist * 0.3  # 横向偏移
                vertical_offset = np.random.randn() * 50  # 垂直偏移
                sample = base + perp * lateral_offset
                sample[2] += vertical_offset
            else:
                sample = base

            sample[2] = np.clip(sample[2], self.min_altitude, self.start[2])
            return sample

        # 全局随机采样
        x = np.random.uniform(self.map.bounds['x_min'], self.map.bounds['x_max'])
        y = np.random.uniform(self.map.bounds['y_min'], self.map.bounds['y_max'])

        # z 方向：偏向从起点高度下降到目标高度
        z_min = max(self.min_altitude, self.goal[2] - 50)
        z_max = self.start[2] + 100
        z = np.random.uniform(z_min, z_max)

        return np.array([x, y, z])

    def _nearest_node(self, point: np.ndarray) -> int:
        """找最近的节点"""
        min_dist = np.inf
        nearest_idx = 0

        for i, node in enumerate(self.nodes):
            dist = self._distance(node.position, point)
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i

        return nearest_idx

    def _near_nodes(self, point: np.ndarray) -> List[int]:
        """找搜索半径内的所有节点"""
        indices = []
        for i, node in enumerate(self.nodes):
            if self._distance(node.position, point) < self.search_radius:
                indices.append(i)
        return indices

    def _steer(self, from_pos: np.ndarray, to_pos: np.ndarray) -> np.ndarray:
        """从 from_pos 向 to_pos 方向扩展一步"""
        direction = to_pos - from_pos
        dist = np.linalg.norm(direction)

        if dist < self.step_size:
            return to_pos.copy()

        direction = direction / dist * self.step_size
        new_pos = from_pos + direction

        # 确保高度不低于最小值
        new_pos[2] = max(new_pos[2], self.min_altitude)

        return new_pos

    def _distance(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """计算两点距离 (考虑垂直方向权重)"""
        d_xy = np.linalg.norm(p1[:2] - p2[:2])
        d_z = abs(p1[2] - p2[2]) * self.vertical_weight
        return np.sqrt(d_xy**2 + d_z**2)

    def _cost(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """计算两点之间的代价"""
        return self._distance(p1, p2)

    def _rewire(self, new_idx: int, near_indices: List[int]):
        """重新布线：检查是否可以通过新节点降低近邻节点的代价"""
        new_node = self.nodes[new_idx]

        for idx in near_indices:
            node = self.nodes[idx]
            new_cost = new_node.cost + self._cost(new_node.position, node.position)

            if new_cost < node.cost:
                if not self.map.is_path_collision(new_node.position, node.position):
                    node.parent = new_idx
                    node.cost = new_cost

    def _extract_path(self, add_final_target: bool = True) -> List[np.ndarray]:
        """从目标回溯提取路径
        
        参数:
            add_final_target: 是否添加最终着陆点（默认True，plan_with_loiter时设为False）
        """
        path = []
        idx = len(self.nodes) - 1  # 目标节点

        while idx is not None:
            path.append(self.nodes[idx].position.copy())
            idx = self.nodes[idx].parent

        path.reverse()

        # 添加最终着陆点（如果需要且进场点和目标不同）
        if add_final_target and self.map.target is not None:
            final = self.map.target.position.copy()
            if np.linalg.norm(path[-1] - final) > 1.0:
                path.append(final)

        return path
    
    def plan_with_loiter(self, start: np.ndarray = None, 
                         max_time: float = 30.0) -> Tuple[Optional[List[np.ndarray]], dict]:
        """
        规划包含画圆消高的完整路径
        
        规划流程：
        1. 计算画圆参数（位置、半径、圈数）
        2. RRT*规划：起点 → 画圆入口点
        3. 生成画圆航点（检查碰撞）
        4. 添加进场航点（检查碰撞）
        5. 返回完整路径
        
        返回:
            path: 完整路径点列表（包含画圆），失败返回 None
            info: 规划信息字典
        """
        if start is None:
            start = self.map.start.to_array()
        
        target = self.map.target
        constraints = self.map.constraints
        
        if target is None:
            return None, {'success': False, 'error': 'No target defined'}
        
        target_pos = target.position.copy()
        approach_heading = target.approach_heading
        approach_length = target.approach_length
        terminal_altitude = constraints.terminal_altitude
        glide_ratio = constraints.glide_ratio
        altitude_margin = constraints.altitude_margin
        min_turn_radius = constraints.min_turn_radius
        
        # ======== Step 1: 计算画圆参数 ========
        # 计算需要的额外水平距离
        start_z = start[2]
        target_z = target_pos[2]
        
        # 基于滑翔比计算需要的最小水平距离
        min_horizontal = (start_z - target_z - altitude_margin) * glide_ratio
        
        # 估算直线距离（起点到目标）
        direct_distance = np.linalg.norm(start[:2] - target_pos[:2])
        
        # 需要的额外距离（用于画圆消高）
        extra_distance = max(0.0, min_horizontal - direct_distance)
        
        # 计算画圆参数
        loiter_radius = 1.5 * min_turn_radius  # 使用1.5倍最小转弯半径
        loiter_loops = 0
        if extra_distance > 1e-3:
            loiter_loops = int(np.ceil(extra_distance / (2 * np.pi * loiter_radius)))
        
        # ======== Step 2: 计算画圆位置和入口点 ========
        # 画圆位置在进场点附近，高度为进场高度
        loiter_altitude = terminal_altitude
        
        # 进场点（画圆出口 = 进场点）
        approach_point = target_pos.copy()
        approach_point[0] -= approach_length * np.cos(approach_heading)
        approach_point[1] -= approach_length * np.sin(approach_heading)
        approach_point[2] = loiter_altitude
        
        # 画圆中心（在进场点的侧面）
        # CCW方向：中心在进场点的左侧
        theta_center = approach_heading - np.pi / 2.0
        loiter_center = approach_point.copy()
        loiter_center[:2] -= loiter_radius * np.array([np.cos(theta_center), np.sin(theta_center)])
        
        # 检查画圆区域是否有碰撞
        loiter_direction = "ccw"
        if not self._circle_ok(loiter_center, loiter_radius, loiter_altitude):
            # 尝试CW方向
            theta_center = approach_heading + np.pi / 2.0
            loiter_center = approach_point.copy()
            loiter_center[:2] -= loiter_radius * np.array([np.cos(theta_center), np.sin(theta_center)])
            loiter_direction = "cw"
            
            if not self._circle_ok(loiter_center, loiter_radius, loiter_altitude):
                # 两个方向都不行，尝试更大的半径
                loiter_radius = 2.0 * min_turn_radius
                loiter_loops = int(np.ceil(extra_distance / (2 * np.pi * loiter_radius))) if extra_distance > 1e-3 else 0
                
                theta_center = approach_heading - np.pi / 2.0
                loiter_center = approach_point.copy()
                loiter_center[:2] -= loiter_radius * np.array([np.cos(theta_center), np.sin(theta_center)])
                loiter_direction = "ccw"
                
                if not self._circle_ok(loiter_center, loiter_radius, loiter_altitude):
                    # 还是不行，跳过画圆
                    loiter_loops = 0
                    print(f"    [警告] 无法找到安全的画圆区域，跳过画圆消高")
        
        # 计算画圆入口点
        if loiter_loops > 0:
            total_angle = 2 * np.pi * loiter_loops
            if loiter_direction == "ccw":
                theta_end = approach_heading - np.pi / 2.0
                theta_start = theta_end - total_angle
                tangent_sign = 1.0
            else:
                theta_end = approach_heading + np.pi / 2.0
                theta_start = theta_end + total_angle
                tangent_sign = -1.0
            
            # 画圆入口点（loiter_center已经在正确高度，z方向偏移为0）
            loiter_entry = loiter_center + np.array([
                loiter_radius * np.cos(theta_start),
                loiter_radius * np.sin(theta_start),
                0.0  # z方向偏移为0
            ])
            
            # RRT*的目标是画圆入口点
            rrt_goal = loiter_entry.copy()
        else:
            # 不需要画圆，RRT*的目标是进场点
            rrt_goal = approach_point.copy()
            theta_start = 0
            theta_end = 0
            tangent_sign = 1.0
        
        # ======== Step 3: RRT*规划到画圆入口 ========
        print(f"    [规划] RRT*目标: {'画圆入口点' if loiter_loops > 0 else '进场点'} (高度{rrt_goal[2]:.1f}m)")

        # 不添加最终目标，因为后面会手动添加画圆和进场航点
        path_to_entry, info = self.plan(start=start, goal=rrt_goal, max_time=max_time, add_final_target=False)

        if path_to_entry is None:
            return None, info

        # ======== Step 3.5: 检查并修复RRT*路径末端到画圆高度的平滑过渡 ========
        # 问题：RRT*可能在距离目标较远处连接，导致多个点低于画圆高度，造成突然跳升
        # 解决：从后往前找到所有低于画圆高度的点，从第一个高点开始平滑下降
        if loiter_loops > 0 and len(path_to_entry) >= 2:
            print(f"    [调试] 画圆高度={loiter_altitude:.1f}m")
            print(f"    [调试] RRT*路径最后5个点的高度: ", end="")
            for i in range(max(0, len(path_to_entry)-5), len(path_to_entry)):
                print(f"[{i}]={path_to_entry[i][2]:.1f}m ", end="")
            print()

            # 从后往前扫描，找到第一个高度>=画圆高度的点
            split_idx = -1
            for i in range(len(path_to_entry) - 1, -1, -1):
                if path_to_entry[i][2] >= loiter_altitude:
                    split_idx = i
                    break

            if split_idx == -1:
                # 所有点都低于画圆高度，这不应该发生（起点应该更高）
                print(f"    [警告] RRT*路径所有点都低于画圆高度，强制最后一个点为画圆高度")
                path_to_entry[-1] = path_to_entry[-1].copy()
                path_to_entry[-1][2] = loiter_altitude
            elif split_idx == len(path_to_entry) - 1:
                # 最后一个点已经>=画圆高度，无需调整
                print(f"    [调试] 最后一个点高度已>=画圆高度，无需调整")
            else:
                # 有点低于画圆高度，需要调整
                n_low_points = len(path_to_entry) - 1 - split_idx
                start_point = path_to_entry[split_idx].copy()
                end_point = path_to_entry[-1].copy()  # 画圆入口点

                print(f"    [调试] 发现{n_low_points}个点低于画圆高度")
                print(f"    [调试] 从点[{split_idx}](高度{start_point[2]:.1f}m)到点[{len(path_to_entry)-1}](高度{end_point[2]:.1f}m)")

                # 移除split_idx之后的所有点（包括画圆入口点）
                path_to_entry = path_to_entry[:split_idx+1]

                # 插入平滑过渡点：从start_point平滑下降到loiter_altitude
                horizontal_dist = np.linalg.norm(end_point[:2] - start_point[:2])
                n_transition = max(int(horizontal_dist / 10), n_low_points + 2)  # 至少和原来一样多，每10m至少1个点

                for i in range(1, n_transition + 1):
                    alpha = i / n_transition
                    # 水平位置线性插值
                    new_pt = start_point + alpha * (end_point - start_point)
                    # 高度从start_point[2]线性下降到loiter_altitude
                    new_pt[2] = start_point[2] + alpha * (loiter_altitude - start_point[2])
                    path_to_entry.append(new_pt)

                print(f"    [规划] 插入{n_transition}个过渡点，从{start_point[2]:.1f}m平滑下降到{loiter_altitude:.1f}m")
        
        # ======== Step 4: 生成画圆航点 ========
        loiter_waypoints = []
        if loiter_loops > 0:
            n_points_per_loop = 36  # 每圈36个点（每10度一个点）
            n_total_points = loiter_loops * n_points_per_loop
            thetas = np.linspace(theta_start, theta_end, n_total_points + 1)
            
            for theta in thetas:
                wp = loiter_center + np.array([
                    loiter_radius * np.cos(theta),
                    loiter_radius * np.sin(theta),
                    0.0  # z方向偏移为0，loiter_center已经在正确高度
                ])
                loiter_waypoints.append(wp)
            
            print(f"    [规划] 画圆消高: {loiter_loops}圈, 半径={loiter_radius:.1f}m, {len(loiter_waypoints)}个航点")
        
        # ======== Step 5: 生成进场航点 ========
        approach_waypoints = []
        # 从画圆出口（或进场点）到目标的直线
        if loiter_loops > 0:
            start_approach = loiter_waypoints[-1]  # 画圆出口
        else:
            start_approach = path_to_entry[-1] if path_to_entry else rrt_goal
        
        # 检查进场路径是否有碰撞
        if not self.map.is_path_collision(start_approach, target_pos):
            # 生成进场航点（从进场点到目标）
            n_approach_points = max(int(approach_length / 10), 2)  # 每10m一个点
            for i in range(1, n_approach_points + 1):
                alpha = i / n_approach_points
                wp = start_approach + alpha * (target_pos - start_approach)
                approach_waypoints.append(wp)
            print(f"    [规划] 进场直线: {len(approach_waypoints)}个航点")
        else:
            print(f"    [警告] 进场路径有碰撞，需要额外规划")
            # TODO: 可以在这里添加额外的RRT*规划
            approach_waypoints = [target_pos.copy()]
        
        # ======== Step 6: 合并所有航点 ========
        full_path = path_to_entry.copy()
        full_path.extend(loiter_waypoints)
        full_path.extend(approach_waypoints)
        
        # 更新info
        info['loiter_loops'] = loiter_loops
        info['loiter_radius'] = loiter_radius if loiter_loops > 0 else 0
        info['loiter_direction'] = loiter_direction if loiter_loops > 0 else None
        info['total_waypoints'] = len(full_path)
        info['path_length'] = self._path_length(full_path)
        
        return full_path, info
    
    def _circle_ok(self, center: np.ndarray, radius: float, z: float) -> bool:
        """检查画圆区域是否有碰撞"""
        for theta in np.linspace(0, 2 * np.pi, 36, endpoint=False):
            pt = center + np.array([radius * np.cos(theta), radius * np.sin(theta), 0.0])
            pt[2] = z
            if self.map.is_collision(pt):
                return False
        return True

    def _path_length(self, path: List[np.ndarray]) -> float:
        """计算路径长度"""
        length = 0
        for i in range(len(path) - 1):
            length += np.linalg.norm(path[i + 1] - path[i])
        return length

    def get_tree_edges(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """获取RRT树的所有边 (用于可视化)"""
        edges = []
        for node in self.nodes:
            if node.parent is not None:
                parent = self.nodes[node.parent]
                edges.append((parent.position, node.position))
        return edges


# ============================================================
#                     测试
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="全局路径规划测试")
    parser.add_argument("--config", type=str, required=True, help="地图配置文件路径")
    parser.add_argument("--max-time", type=float, default=30.0, help="最大规划时间")
    args = parser.parse_args()

    # 加载地图
    map_mgr = MapManager.from_yaml(args.config)

    # 创建规划器
    planner = RRTStarPlanner(map_mgr)

    # 规划路径
    path, info = planner.plan(max_time=args.max_time)

    print("\n" + "=" * 50)
    print("规划结果:")
    print("=" * 50)
    print(f"成功: {info['success']}")
    print(f"迭代次数: {info['iterations']}")
    print(f"节点数: {info['nodes']}")
    print(f"耗时: {info['time']:.2f}s")
    if path:
        print(f"路径长度: {info['path_length']:.1f}m")
        print(f"航点数: {len(path)}")

    # 可视化
    if path:
        import matplotlib.pyplot as plt
        from visualization.map_visualizer import MapVisualizer

        path_array = np.array(path)
        visualizer = MapVisualizer(map_mgr)
        visualizer.plot_combined(trajectory=path_array)
        plt.show()
