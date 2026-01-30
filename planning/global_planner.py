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
        glide_ratio = constraints.glide_ratio  # 最大滑翔比
        min_glide_ratio = constraints.min_glide_ratio  # 最小滑翔比
        altitude_margin = constraints.altitude_margin
        min_turn_radius = constraints.min_turn_radius

        # ======== Step 0: 快速滑翔比判断 ========
        horizontal_dist = np.linalg.norm(start[:2] - target_pos[:2])
        altitude_diff = start[2] - target_pos[2]

        if altitude_diff <= 0:
            print(f"    [规划] 起点高度({start[2]:.1f}m) <= 终点高度({target_pos[2]:.1f}m)，无法到达！")
            return None, {'success': False, 'error': 'Start altitude too low'}

        direct_glide_ratio = horizontal_dist / altitude_diff

        print(f"\n    [快速判断] 起点到终点:")
        print(f"      水平距离: {horizontal_dist:.1f}m, 高度差: {altitude_diff:.1f}m")
        print(f"      直线滑翔比: {direct_glide_ratio:.2f}, 范围: [{min_glide_ratio:.2f}, {glide_ratio:.2f}]")

        if direct_glide_ratio > glide_ratio:
            print(f"    [规划] ✗ 终点不可达！滑翔比{direct_glide_ratio:.2f} > 最大{glide_ratio:.2f}")
            return None, {'success': False, 'error': 'Target unreachable'}
        elif direct_glide_ratio >= min_glide_ratio:
            print(f"    [规划] ✓ 可以直飞，不需要画圆消高\n")
            force_no_loiter = True
        else:
            print(f"    [规划] ⚠ 需要画圆消高（滑翔比{direct_glide_ratio:.2f} < 最小{min_glide_ratio:.2f})\n")
            force_no_loiter = False

        # ======== Step 1: 基于最大滑翔比约束计算画圆高度范围 ========
        if force_no_loiter:
            # 跳过画圆计算
            loiter_loops = 0
            loiter_radius = min_turn_radius * 1.5
            loiter_direction = None
            loiter_altitude = terminal_altitude
        else:
            # 需要画圆，执行完整计算
            start_z = start[2]
            target_z = target_pos[2]

            # 估算直线距离（起点到目标）
            direct_distance = np.linalg.norm(start[:2] - target_pos[:2])

            # 关键约束：所有路径段的滑翔比都不能超过翼伞的最大滑翔比
            # 约束1：进场段滑翔比 <= glide_ratio
            # approach_length / (loiter_exit - target_z) <= glide_ratio
            # => loiter_exit >= target_z + approach_length / glide_ratio
            min_loiter_exit = target_z + approach_length / glide_ratio

            # 约束2：起点到画圆入口段的滑翔比 <= glide_ratio
            # 注意：RRT*会绕行避障，实际路径长度会比直线距离长
            # 保守估计：实际路径 = 直线距离 × 1.5（50%绕行，更保守）
            rrt_detour_factor = 1.5  # RRT*绕行系数（保守估计）
            estimated_rrt_distance = (direct_distance - approach_length) * rrt_detour_factor

            # estimated_rrt_distance / (start_z - loiter_entry) <= glide_ratio
            # => loiter_entry <= start_z - estimated_rrt_distance / glide_ratio
            max_loiter_entry = start_z - estimated_rrt_distance / glide_ratio

            print(f"    [规划] 估算RRT*距离={estimated_rrt_distance:.0f}m (直线{direct_distance-approach_length:.0f}m × 绕行系数{rrt_detour_factor})")
            print(f"    [规划] 满足滑翔比{glide_ratio:.1f}约束: 画圆入口高度<={max_loiter_entry:.1f}m")

            # 画圆要消耗高度，所以 loiter_entry > loiter_exit
            # 同时确保至少消耗一定的高度
            min_altitude_loss = 20.0  # 画圆至少消耗20m高度

            # 理论最优：在满足约束的前提下，尽量低（省距离）
            optimal_loiter_exit = max(min_loiter_exit, self.min_altitude)
            optimal_loiter_entry = min(max_loiter_entry, start_z - 10)

            # 检查是否可行
            if optimal_loiter_entry <= optimal_loiter_exit + min_altitude_loss:
                # 不可行：即使在理论最优情况下，画圆也消耗不了足够的高度
                # 说明直线距离太短，需要通过画圆补充距离
                print(f"    [规划] 直线距离不足，需要画圆补充距离")
                # 强制设定：画圆出口尽量低，入口尽量高
                optimal_loiter_exit = max(min_loiter_exit, self.min_altitude)
                optimal_loiter_entry = start_z - 10
            else:
                print(f"    [规划] 计算画圆高度范围: 入口<={optimal_loiter_entry:.1f}m, 出口>={optimal_loiter_exit:.1f}m")

            # 使用入口高度作为搜索的目标（后面会调整出口高度）
            optimal_loiter_altitude = optimal_loiter_entry
            print(f"    [规划] 目标画圆入口高度={optimal_loiter_altitude:.1f}m (满足最大滑翔比{glide_ratio:.1f}约束)")

            # ======== Step 2: 搜索可行的画圆高度（避障） ========
            # 画圆中心的XY位置（固定，基于进场方向）
            loiter_center_xy = target_pos[:2].copy()
            loiter_center_xy[0] -= approach_length * np.cos(approach_heading)
            loiter_center_xy[1] -= approach_length * np.sin(approach_heading)

            # 画圆半径
            loiter_radius = 1.5 * min_turn_radius

            # 搜索可行的画圆高度：在 [terminal_altitude, start_z-10] 范围内
            # 优先级：最接近 optimal_loiter_altitude 的高度
            search_altitudes = []

            # 以 optimal_loiter_altitude 为中心，向上下搜索
            step = 10.0  # 每10m采样一个高度
            for offset in np.arange(0, start_z - terminal_altitude, step):
                # 向上搜索
                if optimal_loiter_altitude + offset <= start_z - 10:
                    search_altitudes.append(optimal_loiter_altitude + offset)
                # 向下搜索
                if optimal_loiter_altitude - offset >= terminal_altitude:
                    search_altitudes.append(optimal_loiter_altitude - offset)

            # 确保包含边界值
            if terminal_altitude not in search_altitudes:
                search_altitudes.append(terminal_altitude)

            # 尝试找到可行的画圆高度和方向
            loiter_altitude = None
            loiter_direction = None
            theta_center = None

            for candidate_altitude in search_altitudes:
                # 尝试CCW方向（逆时针）
                theta_ccw = approach_heading - np.pi / 2.0
                center_ccw = np.array([
                    loiter_center_xy[0] - loiter_radius * np.cos(theta_ccw),
                    loiter_center_xy[1] - loiter_radius * np.sin(theta_ccw),
                    candidate_altitude
                ])

                if self._circle_ok(center_ccw, loiter_radius, candidate_altitude):
                    loiter_altitude = candidate_altitude
                    loiter_direction = "ccw"
                    theta_center = theta_ccw
                    loiter_center = center_ccw
                    print(f"    [规划] 找到可行画圆高度={loiter_altitude:.1f}m (CCW方向)")
                    break

                # 尝试CW方向（顺时针）
                theta_cw = approach_heading + np.pi / 2.0
                center_cw = np.array([
                    loiter_center_xy[0] - loiter_radius * np.cos(theta_cw),
                    loiter_center_xy[1] - loiter_radius * np.sin(theta_cw),
                    candidate_altitude
                ])

                if self._circle_ok(center_cw, loiter_radius, candidate_altitude):
                    loiter_altitude = candidate_altitude
                    loiter_direction = "cw"
                    theta_center = theta_cw
                    loiter_center = center_cw
                    print(f"    [规划] 找到可行画圆高度={loiter_altitude:.1f}m (CW方向)")
                    break

            # 如果找不到可行高度，尝试更大的半径
            if loiter_altitude is None:
                print(f"    [警告] 标准半径({loiter_radius:.1f}m)无可行高度，尝试更大半径")
                loiter_radius = 2.0 * min_turn_radius

                for candidate_altitude in search_altitudes:
                    theta_ccw = approach_heading - np.pi / 2.0
                    center_ccw = np.array([
                        loiter_center_xy[0] - loiter_radius * np.cos(theta_ccw),
                        loiter_center_xy[1] - loiter_radius * np.sin(theta_ccw),
                        candidate_altitude
                    ])

                    if self._circle_ok(center_ccw, loiter_radius, candidate_altitude):
                        loiter_altitude = candidate_altitude
                        loiter_direction = "ccw"
                        theta_center = theta_ccw
                        loiter_center = center_ccw
                        print(f"    [规划] 找到可行画圆高度={loiter_altitude:.1f}m (大半径{loiter_radius:.1f}m, CCW)")
                        break

                    theta_cw = approach_heading + np.pi / 2.0
                    center_cw = np.array([
                        loiter_center_xy[0] - loiter_radius * np.cos(theta_cw),
                        loiter_center_xy[1] - loiter_radius * np.sin(theta_cw),
                        candidate_altitude
                    ])

                    if self._circle_ok(center_cw, loiter_radius, candidate_altitude):
                        loiter_altitude = candidate_altitude
                        loiter_direction = "cw"
                        theta_center = theta_cw
                        loiter_center = center_cw
                        print(f"    [规划] 找到可行画圆高度={loiter_altitude:.1f}m (大半径{loiter_radius:.1f}m, CW)")
                        break

            # 如果还是找不到，跳过画圆
            if loiter_altitude is None:
                print(f"    [警告] 无法找到安全的画圆高度，跳过画圆消高")
                loiter_altitude = terminal_altitude
                loiter_loops = 0
            else:
                # ======== Step 3: 基于确定的画圆高度计算画圆圈数 ========
                # 计算总共需要的直线距离（起点→画圆入口 + 画圆出口→目标）
                altitude_loss_to_loiter = start_z - loiter_altitude
                altitude_loss_from_loiter = loiter_altitude - target_z - altitude_margin

                # 总共需要的直线距离
                total_required_horizontal = (altitude_loss_to_loiter + altitude_loss_from_loiter) * glide_ratio

                # 需要通过画圆补充的距离
                extra_distance = max(0.0, total_required_horizontal - direct_distance)

                # 计算画圆圈数
                loiter_loops = 0
                if extra_distance > 1e-3:
                    loiter_loops = int(np.ceil(extra_distance / (2 * np.pi * loiter_radius)))
                    print(f"    [规划] 需要画圆消高: 总需距离{total_required_horizontal:.0f}m > 直线距离{direct_distance:.0f}m")
                    print(f"    [规划] 通过画圆补充{extra_distance:.0f}m，圈数={loiter_loops}，半径={loiter_radius:.1f}m")
                else:
                    print(f"    [规划] 无需画圆消高: 直线距离{direct_distance:.0f}m >= 总需距离{total_required_horizontal:.0f}m")

        # Step 1-3 结束

        # ======== Step 4: 计算画圆入口点和RRT*目标 ========
        if loiter_loops > 0:
            # 需要画圆：RRT*目标是画圆入口点
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
            # 不需要画圆：RRT*目标是进场点
            # 进场起点高度由滑翔比决定
            required_approach_altitude = target_z + approach_length / glide_ratio if glide_ratio > 0 else terminal_altitude
            approach_altitude = np.clip(required_approach_altitude,
                                       self.min_altitude,  # 最低飞行高度
                                       loiter_altitude)  # 不超过当前画圆高度

            # 动态查找安全的进场点
            safe_approach_point, actual_heading, actual_length = self.map.find_safe_approach_point(
                target_pos=target_pos,
                altitude=approach_altitude,
                desired_heading=approach_heading,
                max_length=approach_length,
                min_length=target.min_approach_length,
                heading_tolerance=target.approach_heading_tolerance
            )

            if safe_approach_point is None:
                print(f"    [警告] 未找到安全进场点，使用期望参数")
                approach_point = target_pos.copy()
                approach_point[0] -= approach_length * np.cos(approach_heading)
                approach_point[1] -= approach_length * np.sin(approach_heading)
                approach_point[2] = approach_altitude
            else:
                approach_point = safe_approach_point
                approach_heading = actual_heading
                approach_length = actual_length

            rrt_goal = approach_point.copy()
            theta_start = 0
            theta_end = 0
            tangent_sign = 1.0

            actual_glide_ratio = approach_length / (approach_altitude - target_z) if approach_altitude > target_z else 0
            print(f"    [规划] 无需画圆，进场起点高度={approach_altitude:.1f}m (滑翔比{actual_glide_ratio:.2f})")

        # ======== Step 5: RRT*规划到画圆入口 ========
        print(f"    [规划] RRT*目标: {'画圆入口点' if loiter_loops > 0 else '进场点'} (高度{rrt_goal[2]:.1f}m)")

        # 不添加最终目标，因为后面会手动添加画圆和进场航点
        path_to_entry, info = self.plan(start=start, goal=rrt_goal, max_time=max_time, add_final_target=False)

        if path_to_entry is None:
            return None, info

        # ======== Step 5.5: 验证RRT*路径的滑翔比约束 ========
        # 检查起点到画圆入口的整体滑翔比是否满足约束
        path_horizontal_dist = 0.0
        for i in range(len(path_to_entry) - 1):
            path_horizontal_dist += np.linalg.norm(path_to_entry[i+1][:2] - path_to_entry[i][:2])

        path_altitude_loss = start[2] - path_to_entry[-1][2]
        path_glide_ratio = path_horizontal_dist / path_altitude_loss if path_altitude_loss > 1.0 else 0.0

        print(f"    [验证] 起点→画圆入口: 水平{path_horizontal_dist:.0f}m, 下降{path_altitude_loss:.0f}m, 滑翔比{path_glide_ratio:.2f}")

        if path_glide_ratio > glide_ratio * 1.1:
            print(f"    [警告] RRT*路径滑翔比{path_glide_ratio:.2f}超过最大值{glide_ratio:.2f}！")
            print(f"    [警告] 翼伞可能无法跟踪此轨迹，建议：")
            print(f"    [警告]   1. 降低画圆入口高度")
            print(f"    [警告]   2. 或增加路径绕行距离")

        # ======== Step 6: 检查并修复RRT*路径末端到画圆高度的平滑过渡 ========
        # 问题：RRT*的goal_threshold=25m，可能在距离目标较远处连接，导致多个点低于画圆高度
        # 解决：扫描整个路径，找到所有低于画圆高度的点，统一处理
        if loiter_loops > 0 and len(path_to_entry) >= 2:
            # 打印调试信息
            print(f"    [调试] 画圆入口目标高度={loiter_altitude:.1f}m")
            n_show = min(8, len(path_to_entry))
            print(f"    [调试] RRT*路径最后{n_show}个点的高度: ", end="")
            for i in range(len(path_to_entry) - n_show, len(path_to_entry)):
                print(f"[{i}]={path_to_entry[i][2]:.1f}m ", end="")
            print()

            # 扫描整个路径：从后往前找，找到最后一个高度>=画圆入口高度的"安全点"
            # 注意：最后一个点是RRT*直接添加的目标点（高度=loiter_altitude），不参与扫描
            split_idx = None
            for i in range(len(path_to_entry) - 2, -1, -1):
                if path_to_entry[i][2] >= loiter_altitude:
                    split_idx = i
                    break

            if split_idx is None:
                # 除了最后一个点，所有点都低于画圆高度 - 异常情况
                print(f"    [警告] RRT*路径所有点都低于画圆高度！从起点强制平滑上升")
                split_idx = 0

            # 计算需要调整的点数（split_idx+1 到 倒数第二个点）
            n_low_points = len(path_to_entry) - 2 - split_idx

            if n_low_points > 0:
                # 有点低于画圆高度，需要调整
                start_point = path_to_entry[split_idx].copy()
                end_point = path_to_entry[-1].copy()  # 画圆入口点

                print(f"    [规划] 检测到路径末端{n_low_points}个点低于画圆高度，需要平滑过渡")
                print(f"    [规划] 从点[{split_idx}](高度{start_point[2]:.1f}m) → 画圆入口(高度{loiter_altitude:.1f}m)")

                # 删除所有低点（split_idx+1 到 倒数第二个点）
                path_to_entry = path_to_entry[:split_idx+1]

                # 插入平滑过渡点：从start_point到画圆入口
                horizontal_dist = np.linalg.norm(end_point[:2] - start_point[:2])
                # 过渡点数量：至少替换原来的低点数量，每10m水平距离至少1个点
                n_transition = max(int(horizontal_dist / 10), n_low_points, 3)

                for i in range(1, n_transition + 1):
                    alpha = i / n_transition
                    # 水平位置线性插值
                    new_pt = start_point + alpha * (end_point - start_point)
                    # 高度线性过渡
                    new_pt[2] = start_point[2] + alpha * (loiter_altitude - start_point[2])
                    path_to_entry.append(new_pt)

                print(f"    [规划] 已插入{n_transition}个过渡点，平滑过渡到画圆入口")
            else:
                # 路径末端高度已经合适
                print(f"    [调试] 路径末端所有点高度合适，无需调整")

        # ======== Step 7: 生成画圆消高航点（螺旋下降）========
        loiter_waypoints = []
        if loiter_loops > 0:
            # 计算画圆出口高度：必须满足进场段滑翔比约束
            # 进场段滑翔比 = approach_length / (loiter_exit - target_z) <= glide_ratio
            # => loiter_exit >= target_z + approach_length / glide_ratio
            required_approach_altitude = target_z + approach_length / glide_ratio

            # 画圆出口高度：满足滑翔比约束，且至少消耗20m高度
            loiter_exit_altitude = max(
                required_approach_altitude,  # 满足进场段滑翔比<=glide_ratio
                self.min_altitude  # 不低于最低飞行高度
            )

            # 确保画圆能消耗足够的高度
            if loiter_altitude - loiter_exit_altitude < 20:
                # 画圆消耗高度不足，调整画圆入口高度
                print(f"    [警告] 画圆入口高度{loiter_altitude:.1f}m过低，调整出口高度")
                loiter_exit_altitude = min(loiter_exit_altitude, loiter_altitude - 20)

            # 计算实际的进场段滑翔比
            actual_approach_glide = approach_length / (loiter_exit_altitude - target_z) if loiter_exit_altitude > target_z else 0

            # 检查并报告
            if actual_approach_glide > glide_ratio * 1.1:
                print(f"    [警告] 进场段滑翔比{actual_approach_glide:.2f}超过最大值{glide_ratio:.2f}！")
            else:
                print(f"    [规划] 画圆出口高度={loiter_exit_altitude:.1f}m (进场段滑翔比{actual_approach_glide:.2f}，最大{glide_ratio:.2f})")

            n_points_per_loop = 36  # 每圈36个点（每10度一个点）
            n_total_points = loiter_loops * n_points_per_loop
            thetas = np.linspace(theta_start, theta_end, n_total_points + 1)

            for i, theta in enumerate(thetas):
                # 计算当前点的高度（螺旋下降）
                alpha = i / max(len(thetas) - 1, 1)
                current_altitude = loiter_altitude + alpha * (loiter_exit_altitude - loiter_altitude)

                # 计算当前点的位置
                wp = np.array([
                    loiter_center[0] + loiter_radius * np.cos(theta),
                    loiter_center[1] + loiter_radius * np.sin(theta),
                    current_altitude
                ])
                loiter_waypoints.append(wp)

            altitude_loss = loiter_altitude - loiter_exit_altitude
            print(f"    [规划] 画圆消高: {loiter_loops}圈, 半径={loiter_radius:.1f}m, 从{loiter_altitude:.1f}m螺旋降至{loiter_exit_altitude:.1f}m(消高{altitude_loss:.1f}m), {len(loiter_waypoints)}个航点")

        # ======== Step 8: 生成进场航点 ========
        approach_waypoints = []
        # 从画圆出口（或进场点）到目标的直线
        if loiter_loops > 0:
            start_approach = loiter_waypoints[-1]  # 画圆出口
        else:
            if path_to_entry and len(path_to_entry) > 0:
                last_rrt_point = path_to_entry[-1].copy()
                dist_to_goal = np.linalg.norm(last_rrt_point[:2] - rrt_goal[:2])
                if dist_to_goal > 5.0:
                    n_transition = max(int(dist_to_goal / 10), 2)
                    for i in range(1, n_transition + 1):
                        alpha = i / n_transition
                        transition_pt = last_rrt_point + alpha * (rrt_goal - last_rrt_point)
                        path_to_entry.append(transition_pt)
                    print(f"    [规划] 插入{n_transition}个过渡点到进场点(距离{dist_to_goal:.1f}m)")
                else:
                    path_to_entry.append(rrt_goal.copy())
            start_approach = rrt_goal        
        # 检查进场路径是否有碰撞
        if not self.map.is_path_collision(start_approach, target_pos):
            # 生成进场航点（从进场点到目标）
            n_approach_points = max(int(approach_length / 10), 2)  # 每10m一个点
            for i in range(1, n_approach_points + 1):
                alpha = i / n_approach_points
                wp = start_approach + alpha * (target_pos - start_approach)
                approach_waypoints.append(wp)
            # 计算进场段的滑翔比
            approach_altitude_loss = start_approach[2] - target_pos[2]
            approach_horizontal = np.linalg.norm(start_approach[:2] - target_pos[:2])
            approach_glide_ratio = approach_horizontal / approach_altitude_loss if approach_altitude_loss > 1e-3 else 0
            print(f"    [规划] 进场直线: {len(approach_waypoints)}个航点, 从{start_approach[2]:.1f}m到{target_pos[2]:.1f}m")
            print(f"    [规划] 进场滑翔比: {approach_glide_ratio:.2f} (水平{approach_horizontal:.0f}m / 下降{approach_altitude_loss:.0f}m)")
        else:
            print(f"    [警告] 进场路径有碰撞，需要额外规划")
            # TODO: 可以在这里添加额外的RRT*规划
            approach_waypoints = [target_pos.copy()]

        # ======== Step 9: 合并所有航点 ========
        full_path = path_to_entry.copy()
        full_path.extend(loiter_waypoints)
        full_path.extend(approach_waypoints)
        
        # 更新info
        info['loiter_loops'] = loiter_loops
        info['loiter_radius'] = loiter_radius if loiter_loops > 0 else 0
        info['loiter_direction'] = loiter_direction if loiter_loops > 0 else None
        info['total_waypoints'] = len(full_path)
        info['path_length'] = self._path_length(full_path)

        # ======== 最终验证：检查所有路径段的滑翔比 ========
        print(f"\n    [总结] 路径滑翔比验证 (最大允许{glide_ratio:.1f}):")
        max_segment_glide = 0.0
        if 'path_glide_ratio' in locals():
            print(f"      起点→画圆入口: {path_glide_ratio:.2f} {'✓' if path_glide_ratio <= glide_ratio * 1.05 else '✗ 超限!'}")
            max_segment_glide = max(max_segment_glide, path_glide_ratio)
        if loiter_loops > 0 and 'approach_glide_ratio' in locals():
            print(f"      画圆出口→目标: {approach_glide_ratio:.2f} {'✓' if approach_glide_ratio <= glide_ratio * 1.05 else '✗ 超限!'}")
            max_segment_glide = max(max_segment_glide, approach_glide_ratio)

        if max_segment_glide > glide_ratio * 1.1:
            print(f"    [警告] 最大段滑翔比{max_segment_glide:.2f}超过翼伞能力{glide_ratio:.1f}，控制器可能无法跟踪！")
        else:
            print(f"    [✓] 所有路径段滑翔比均在翼伞能力范围内")

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
