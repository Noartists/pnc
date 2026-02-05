"""
Kinodynamic RRT* 规划器 (改进版)

专为翼伞设计，考虑运动学约束。
使用Dubins曲线作为扩展原语，保证转弯平滑。
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass, field
import time
from tqdm import tqdm


@dataclass
class KinoState:
    """状态: 位置 + 航向"""
    x: float
    y: float
    z: float
    heading: float
    
    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])
    
    @property
    def pos_2d(self) -> np.ndarray:
        return np.array([self.x, self.y])


class DubinsPath:
    """
    Dubins曲线计算器
    
    连接两个2D位姿 (x, y, heading)，满足最小转弯半径约束。
    """
    
    def __init__(self, turn_radius: float):
        self.turn_radius = turn_radius
    
    def compute(self, start: Tuple[float, float, float], 
                end: Tuple[float, float, float]) -> Optional[dict]:
        """
        计算Dubins曲线
        
        参数:
            start: (x, y, heading) 起点
            end: (x, y, heading) 终点
            
        返回:
            最优Dubins路径参数，或None（不可达）
        """
        x0, y0, h0 = start
        x1, y1, h1 = end
        r = self.turn_radius
        
        # 转换到局部坐标系
        dx = x1 - x0
        dy = y1 - y0
        D = np.sqrt(dx**2 + dy**2)
        d = D / r  # 归一化距离
        
        if d < 1e-6:
            return None
        
        theta = np.arctan2(dy, dx)
        alpha = self._normalize_angle(h0 - theta)
        beta = self._normalize_angle(h1 - theta)
        
        # 尝试所有6种Dubins路径类型
        paths = [
            self._LSL(alpha, beta, d),
            self._RSR(alpha, beta, d),
            self._LSR(alpha, beta, d),
            self._RSL(alpha, beta, d),
            self._LRL(alpha, beta, d),
            self._RLR(alpha, beta, d),
        ]
        
        # 选择最短的有效路径
        best = None
        best_len = float('inf')
        
        for path in paths:
            if path is not None and path['length'] < best_len:
                best = path
                best_len = path['length']
        
        if best is None:
            return None
        
        # 转换回世界坐标
        best['start'] = start
        best['end'] = end
        best['turn_radius'] = r
        best['length'] *= r  # 实际长度
        
        return best
    
    def sample(self, path: dict, num_points: int = 20) -> List[np.ndarray]:
        """
        沿Dubins曲线采样点
        
        返回:
            2D点列表 [(x, y), ...]
        """
        if path is None:
            return []
        
        x0, y0, h0 = path['start']
        r = path['turn_radius']
        segments = path['segments']  # [(length, direction), ...]
        
        points = []
        x, y, h = x0, y0, h0
        
        total_len = sum(s[0] for s in segments)
        step = total_len / (num_points - 1) if num_points > 1 else total_len
        
        accumulated = 0.0
        seg_idx = 0
        seg_progress = 0.0
        
        for i in range(num_points):
            target_dist = i * step
            
            # 前进到目标距离
            while accumulated + (segments[seg_idx][0] * r - seg_progress) < target_dist and seg_idx < len(segments) - 1:
                # 完成当前段
                seg_len, direction = segments[seg_idx]
                remaining = seg_len * r - seg_progress
                
                if direction == 'S':  # 直线
                    x += remaining * np.cos(h)
                    y += remaining * np.sin(h)
                elif direction == 'L':  # 左转
                    dtheta = remaining / r
                    cx = x - r * np.sin(h)
                    cy = y + r * np.cos(h)
                    h += dtheta
                    x = cx + r * np.sin(h)
                    y = cy - r * np.cos(h)
                elif direction == 'R':  # 右转
                    dtheta = remaining / r
                    cx = x + r * np.sin(h)
                    cy = y - r * np.cos(h)
                    h -= dtheta
                    x = cx - r * np.sin(h)
                    y = cy + r * np.cos(h)
                
                accumulated += remaining
                seg_progress = 0.0
                seg_idx += 1
            
            # 在当前段内前进
            if seg_idx < len(segments):
                seg_len, direction = segments[seg_idx]
                advance = target_dist - accumulated
                
                if direction == 'S':
                    px = x + advance * np.cos(h)
                    py = y + advance * np.sin(h)
                elif direction == 'L':
                    dtheta = advance / r
                    cx = x - r * np.sin(h)
                    cy = y + r * np.cos(h)
                    px = cx + r * np.sin(h + dtheta)
                    py = cy - r * np.cos(h + dtheta)
                elif direction == 'R':
                    dtheta = advance / r
                    cx = x + r * np.sin(h)
                    cy = y - r * np.cos(h)
                    px = cx - r * np.sin(h - dtheta)
                    py = cy + r * np.cos(h - dtheta)
                
                points.append(np.array([px, py]))
            else:
                points.append(np.array([x, y]))
        
        return points
    
    def _normalize_angle(self, angle: float) -> float:
        """归一化角度到 [-pi, pi]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
    
    def _LSL(self, alpha: float, beta: float, d: float) -> Optional[dict]:
        """Left-Straight-Left"""
        ca, sa = np.cos(alpha), np.sin(alpha)
        cb, sb = np.cos(beta), np.sin(beta)
        
        tmp = 2 + d**2 - 2*(ca*cb + sa*sb - d*(sa - sb))
        if tmp < 0:
            return None
        
        p = np.sqrt(tmp)
        theta = np.arctan2(cb - ca, d + sa - sb)
        t = self._normalize_angle(-alpha + theta)
        q = self._normalize_angle(beta - theta)
        
        if t < 0 or q < 0:
            return None
        
        return {
            'type': 'LSL',
            'segments': [(t, 'L'), (p, 'S'), (q, 'L')],
            'length': t + p + q
        }
    
    def _RSR(self, alpha: float, beta: float, d: float) -> Optional[dict]:
        """Right-Straight-Right"""
        ca, sa = np.cos(alpha), np.sin(alpha)
        cb, sb = np.cos(beta), np.sin(beta)
        
        tmp = 2 + d**2 - 2*(ca*cb + sa*sb - d*(sb - sa))
        if tmp < 0:
            return None
        
        p = np.sqrt(tmp)
        theta = np.arctan2(ca - cb, d - sa + sb)
        t = self._normalize_angle(alpha - theta)
        q = self._normalize_angle(-beta + theta)
        
        if t < 0 or q < 0:
            return None
        
        return {
            'type': 'RSR',
            'segments': [(t, 'R'), (p, 'S'), (q, 'R')],
            'length': t + p + q
        }
    
    def _LSR(self, alpha: float, beta: float, d: float) -> Optional[dict]:
        """Left-Straight-Right"""
        ca, sa = np.cos(alpha), np.sin(alpha)
        cb, sb = np.cos(beta), np.sin(beta)
        
        tmp = -2 + d**2 + 2*(ca*cb + sa*sb + d*(sa + sb))
        if tmp < 0:
            return None
        
        p = np.sqrt(tmp)
        theta = np.arctan2(-ca - cb, d + sa + sb) - np.arctan2(-2, p)
        t = self._normalize_angle(-alpha + theta)
        q = self._normalize_angle(-beta + theta)
        
        if t < 0 or q < 0:
            return None
        
        return {
            'type': 'LSR',
            'segments': [(t, 'L'), (p, 'S'), (q, 'R')],
            'length': t + p + q
        }
    
    def _RSL(self, alpha: float, beta: float, d: float) -> Optional[dict]:
        """Right-Straight-Left"""
        ca, sa = np.cos(alpha), np.sin(alpha)
        cb, sb = np.cos(beta), np.sin(beta)
        
        tmp = -2 + d**2 + 2*(ca*cb + sa*sb - d*(sa + sb))
        if tmp < 0:
            return None
        
        p = np.sqrt(tmp)
        theta = np.arctan2(ca + cb, d - sa - sb) - np.arctan2(2, p)
        t = self._normalize_angle(alpha - theta)
        q = self._normalize_angle(beta - theta)
        
        if t < 0 or q < 0:
            return None
        
        return {
            'type': 'RSL',
            'segments': [(t, 'R'), (p, 'S'), (q, 'L')],
            'length': t + p + q
        }
    
    def _LRL(self, alpha: float, beta: float, d: float) -> Optional[dict]:
        """Left-Right-Left"""
        ca, sa = np.cos(alpha), np.sin(alpha)
        cb, sb = np.cos(beta), np.sin(beta)
        
        tmp = (6 - d**2 + 2*(ca*cb + sa*sb + d*(sa - sb))) / 8
        if abs(tmp) > 1:
            return None
        
        p = np.arccos(tmp)
        theta = np.arctan2(ca - cb, d + sa - sb)
        t = self._normalize_angle(-alpha + theta + p/2)
        q = self._normalize_angle(beta - theta + p/2)
        
        if t < 0 or q < 0:
            return None
        
        return {
            'type': 'LRL',
            'segments': [(t, 'L'), (p, 'R'), (q, 'L')],
            'length': t + p + q
        }
    
    def _RLR(self, alpha: float, beta: float, d: float) -> Optional[dict]:
        """Right-Left-Right"""
        ca, sa = np.cos(alpha), np.sin(alpha)
        cb, sb = np.cos(beta), np.sin(beta)
        
        tmp = (6 - d**2 + 2*(ca*cb + sa*sb - d*(sa - sb))) / 8
        if abs(tmp) > 1:
            return None
        
        p = np.arccos(tmp)
        theta = np.arctan2(ca - cb, d - sa + sb)
        t = self._normalize_angle(alpha - theta + p/2)
        q = self._normalize_angle(-beta + theta + p/2)
        
        if t < 0 or q < 0:
            return None
        
        return {
            'type': 'RLR',
            'segments': [(t, 'R'), (p, 'L'), (q, 'R')],
            'length': t + p + q
        }


@dataclass
class KinoNode:
    """RRT节点"""
    state: KinoState
    parent: Optional[int] = None
    cost: float = 0.0


class KinodynamicRRTStar:
    """Kinodynamic RRT* 规划器（使用Dubins曲线扩展）"""
    
    def __init__(self, map_manager,
                 step_size: float = 100.0,
                 goal_sample_rate: float = 0.3,  # 增加目标采样率
                 max_iterations: int = 5000,
                 quiet: bool = False):
        
        self.map = map_manager
        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate
        self.max_iterations = max_iterations
        self.quiet = quiet
        
        # 约束
        self.min_turn_radius = map_manager.constraints.min_turn_radius
        self.max_glide = map_manager.constraints.glide_ratio
        self.min_glide = map_manager.constraints.min_glide_ratio
        self.min_altitude = map_manager.constraints.min_altitude
        
        # Dubins曲线计算器
        self.dubins = DubinsPath(self.min_turn_radius)
        
        self.nodes: List[KinoNode] = []
        
        # 存储每条边的Dubins路径点（用于最终输出平滑路径）
        self.edge_paths: dict = {}  # {(parent_idx, child_idx): [points]}
    
    def plan(self, max_time: float = 30.0, progress_callback: callable = None):
        """
        规划路径
        
        参数:
            max_time: 最大规划时间
            progress_callback: 进度回调 callback(iteration, max_iterations)
        """
        # 起点终点
        start = np.array([self.map.start.x, self.map.start.y, self.map.start.z])
        goal = self.map.target.position.copy()
        
        # 计算航向
        dx, dy = goal[0] - start[0], goal[1] - start[1]
        start_heading = np.arctan2(dy, dx)
        goal_heading = self.map.target.approach_heading
        
        if not self.quiet:
            print(f"\n{'='*60}")
            print(f"  Kinodynamic RRT* 规划")
            print(f"{'='*60}")
            print(f"  起点: ({start[0]:.0f}, {start[1]:.0f}, {start[2]:.0f})")
            print(f"  终点: ({goal[0]:.0f}, {goal[1]:.0f}, {goal[2]:.0f})")
        
        # 可达性检查
        h_dist = np.linalg.norm(goal[:2] - start[:2])
        v_dist = start[2] - goal[2]
        if v_dist <= 0:
            if not self.quiet:
                print("  [错误] 高度不足")
            return None, {'success': False}
        
        glide = h_dist / v_dist
        if not self.quiet:
            print(f"  直线滑翔比: {glide:.2f} (范围: {self.min_glide:.2f}~{self.max_glide:.2f})")
        
        if glide > self.max_glide:
            if not self.quiet:
                print("  [错误] 不可达")
            return None, {'success': False}
        
        # 存储目标信息供 _steer 使用
        self._goal_xy = goal[:2].copy()
        self._goal_z = goal[2]

        # 初始化
        start_state = KinoState(start[0], start[1], start[2], start_heading)
        goal_state = KinoState(goal[0], goal[1], goal[2], goal_heading)
        self.nodes = [KinoNode(start_state, None, 0.0)]
        self.edge_paths = {}  # 清空边路径缓存
        
        # 统计
        stats = {'steer': 0, 'feasible': 0, 'collision': 0, 'added': 0}
        
        start_time = time.time()
        goal_reached = False
        best_idx = None
        best_cost = float('inf')
        
        pbar = tqdm(total=self.max_iterations, desc="RRT*", disable=self.quiet)
        last_callback_time = 0
        
        for i in range(self.max_iterations):
            if time.time() - start_time > max_time:
                break
            
            # 进度回调（每0.5秒更新一次）
            current_time = time.time()
            if progress_callback and current_time - last_callback_time >= 0.5:
                progress_callback(i, self.max_iterations)
                last_callback_time = current_time

            # 采样
            if np.random.random() < self.goal_sample_rate:
                sample = goal_state
            else:
                sample = self._random_sample(start[2], goal[2], goal)

            # 尝试最多3个最近节点（避免卡在低高度前沿节点上）
            tried = set()
            extended = False
            nearest_idx = None
            nearest = None
            new_state = None
            path_3d = None

            for _attempt in range(3):
                nearest_idx = self._nearest(sample, exclude=tried)
                if nearest_idx is None:
                    break
                tried.add(nearest_idx)
                nearest = self.nodes[nearest_idx]

                steer_result = self._steer(nearest.state, sample)
                if steer_result is None:
                    stats['steer'] += 1
                    continue

                new_state, path_3d = steer_result

                if not self._is_feasible(nearest.state, new_state, goal_z=goal[2]):
                    stats['feasible'] += 1
                    continue

                if self._has_collision_path(path_3d):
                    stats['collision'] += 1
                    continue

                extended = True
                break

            if not extended:
                pbar.update(1)
                continue

            # 计算路径长度作为代价
            path_len = sum(np.linalg.norm(path_3d[j+1] - path_3d[j])
                          for j in range(len(path_3d)-1))
            cost = nearest.cost + path_len

            best_parent_idx = nearest_idx
            best_parent_cost = cost
            best_parent_path = path_3d

            # --- RRT* best parent selection (仅在前期且节点不多时) ---
            elapsed = time.time() - start_time
            n_nodes = len(self.nodes)
            if elapsed < 15.0 and n_nodes < 800:
                rewire_radius = min(self.step_size * 2, 300.0)
                neighbor_idxs = self._near(new_state, rewire_radius)
                for n_idx in neighbor_idxs:
                    if n_idx == nearest_idx:
                        continue
                    n_node = self.nodes[n_idx]
                    alt_steer = self._steer(n_node.state, new_state)
                    if alt_steer is None:
                        continue
                    alt_state, alt_path = alt_steer
                    if np.linalg.norm(alt_state.position - new_state.position) > 30.0:
                        continue
                    if not self._is_feasible(n_node.state, alt_state, goal_z=goal[2]):
                        continue
                    if self._has_collision_path(alt_path):
                        continue
                    alt_len = sum(np.linalg.norm(alt_path[j+1] - alt_path[j])
                                  for j in range(len(alt_path)-1))
                    alt_cost = n_node.cost + alt_len
                    if alt_cost < best_parent_cost:
                        best_parent_idx = n_idx
                        best_parent_cost = alt_cost
                        best_parent_path = alt_path
                        new_state = alt_state

            new_node = KinoNode(new_state, best_parent_idx, best_parent_cost)
            new_idx = len(self.nodes)
            self.nodes.append(new_node)

            # 存储路径点
            self.edge_paths[(best_parent_idx, new_idx)] = best_parent_path

            stats['added'] += 1

            # --- RRT* rewire (仅在找到路径后且时间充裕时) ---
            if goal_reached and elapsed < 25.0 and n_nodes < 1500:
                rewire_radius = min(self.step_size * 2, 300.0)
                neighbor_idxs = self._near(new_state, rewire_radius)
                for n_idx in neighbor_idxs:
                    if n_idx == best_parent_idx or n_idx == 0:
                        continue
                    n_node = self.nodes[n_idx]
                    rw_steer = self._steer(new_state, n_node.state)
                    if rw_steer is None:
                        continue
                    rw_state, rw_path = rw_steer
                    if np.linalg.norm(rw_state.position - n_node.state.position) > 30.0:
                        continue
                    if not self._is_feasible(new_state, rw_state, goal_z=goal[2]):
                        continue
                    if self._has_collision_path(rw_path):
                        continue
                    rw_len = sum(np.linalg.norm(rw_path[j+1] - rw_path[j])
                                 for j in range(len(rw_path)-1))
                    rw_cost = best_parent_cost + rw_len
                    if rw_cost < n_node.cost:
                        old_parent = n_node.parent
                        n_node.parent = new_idx
                        n_node.cost = rw_cost
                        if old_parent is not None:
                            self.edge_paths.pop((old_parent, n_idx), None)
                        self.edge_paths[(new_idx, n_idx)] = rw_path

            # 检查目标 — 只检查2D距离和高度，不要求航向
            dist_to_goal = np.linalg.norm(new_state.position[:2] - goal_state.position[:2])
            dz_to_goal = abs(new_state.z - goal_state.z)

            if dist_to_goal < 200 and dz_to_goal < 80:
                # 尝试用Dubins/直线连接到精确目标点
                connected = False
                goal_steer = self._steer(new_state, goal_state)
                if goal_steer is not None:
                    g_state, g_path = goal_steer
                    g_dist = np.linalg.norm(g_state.position[:2] - goal_state.position[:2])
                    if g_dist < 80 and not self._has_collision_path(g_path):
                        g_len = sum(np.linalg.norm(g_path[j+1] - g_path[j])
                                    for j in range(len(g_path)-1))
                        g_cost = best_parent_cost + g_len
                        g_node = KinoNode(g_state, new_idx, g_cost)
                        g_idx = len(self.nodes)
                        self.nodes.append(g_node)
                        self.edge_paths[(new_idx, g_idx)] = g_path
                        if g_cost < best_cost:
                            best_cost = g_cost
                            best_idx = g_idx
                            goal_reached = True
                            connected = True

                # 即使直连没成功，只要足够近也算到达
                if not connected and best_parent_cost < best_cost:
                    best_cost = best_parent_cost
                    best_idx = new_idx
                    goal_reached = True

            pbar.update(1)
            pbar.set_postfix_str(f"节点:{len(self.nodes)}, 距目标:{dist_to_goal:.0f}m")
        
        pbar.close()
        
        # 打印统计
        if not self.quiet:
            print(f"\n  === 统计 ===")
            print(f"  扩展失败: {stats['steer']}")
            print(f"  可行性失败: {stats['feasible']}")
            print(f"  碰撞失败: {stats['collision']}")
            print(f"  成功添加: {stats['added']}")

            # 高度分布诊断
            if self.nodes:
                altitudes = [n.state.z for n in self.nodes]
                print(f"  节点高度: min={min(altitudes):.1f}, max={max(altitudes):.1f}, "
                      f"mean={np.mean(altitudes):.1f}")
                low_count = sum(1 for z in altitudes if z < 30)
                print(f"  低高度节点 (<30m): {low_count}/{len(self.nodes)}")
                # 最近目标距离
                min_dist = min(np.linalg.norm(n.state.position[:2] - goal[:2])
                              for n in self.nodes)
                print(f"  最近目标2D距离: {min_dist:.0f}m")
        
        if not goal_reached:
            if not self.quiet:
                print(f"  [失败] 未到达目标")
            return None, {'success': False, 'stats': stats}

        # 提取路径
        path = self._extract_path(best_idx)
        path.append(goal)  # 添加终点

        # --- 后处理：如果终点高度过高，注入螺旋下降段 ---
        path = self._inject_spiral_descent(path, goal, goal_state.heading)

        if not self.quiet:
            print(f"  [成功] 路径点数: {len(path)}")

        return path, {'success': True, 'stats': stats, 'path_length': best_cost}
    
    def _random_sample(self, z_max: float, z_min: float, goal: np.ndarray) -> KinoState:
        """
        混合采样策略（在非目标采样的70%中）:
        - ~14% 盘旋采样（目标附近圆轨道）
        - ~43% 引导采样（起点到目标走廊，z与进度关联）
        - ~43% 完全随机
        """
        bounds = self.map.bounds
        r = np.random.random()

        if r < 1.0 / 7.0:
            # 盘旋采样: 在目标附近的圆轨道上采样
            loiter_radius = self.min_turn_radius * np.random.uniform(1.0, 2.5)
            angle = np.random.uniform(-np.pi, np.pi)
            x = goal[0] + loiter_radius * np.cos(angle)
            y = goal[1] + loiter_radius * np.sin(angle)
            # z 在目标高度上方，留有消高空间
            z = goal[2] + np.random.uniform(20, max(z_max - goal[2], 50))
            z = np.clip(z, max(z_min, 0.0), z_max)
            # 航向设为轨道切线方向（顺时针或逆时针）
            tangent_dir = 1.0 if np.random.random() < 0.5 else -1.0
            heading = angle + tangent_dir * np.pi / 2
            x = np.clip(x, bounds['x_min'], bounds['x_max'])
            y = np.clip(y, bounds['y_min'], bounds['y_max'])
            return KinoState(x, y, z, heading)

        elif r < 4.0 / 7.0:
            # 引导式采样：z坐标与alpha（到目标进度）关联
            alpha = np.random.uniform(0.2, 1.0)
            start = np.array([self.map.start.x, self.map.start.y, self.map.start.z])

            # 基础位置（z与进度线性关联）
            base_x = start[0] + alpha * (goal[0] - start[0])
            base_y = start[1] + alpha * (goal[1] - start[1])
            base_z = start[2] + alpha * (goal[2] - start[2])

            # 添加随机偏移（z偏移较小，保持与进度关联）
            x = base_x + np.random.uniform(-200, 200)
            y = base_y + np.random.uniform(-200, 200)
            z = base_z + np.random.uniform(-20, 20)

            x = np.clip(x, bounds['x_min'], bounds['x_max'])
            y = np.clip(y, bounds['y_min'], bounds['y_max'])
            z = np.clip(z, max(z_min, 0.0), z_max)
        else:
            # 完全随机采样
            x = np.random.uniform(bounds['x_min'], bounds['x_max'])
            y = np.random.uniform(bounds['y_min'], bounds['y_max'])
            z = np.random.uniform(max(z_min, 0.0), z_max)

        heading = np.random.uniform(-np.pi, np.pi)
        return KinoState(x, y, z, heading)
    
    def _normalize_angle(self, angle: float) -> float:
        """归一化角度到 [-pi, pi]"""
        angle = angle % (2 * np.pi)
        if angle > np.pi:
            angle -= 2 * np.pi
        return angle

    def _nearest(self, state: KinoState, exclude: set = None) -> Optional[int]:
        """找最近节点（含航向权重），可排除已尝试过的节点"""
        heading_weight = self.min_turn_radius  # 约50m/rad，将航向差转换为等效弧长
        min_dist = float('inf')
        idx = None
        for i, node in enumerate(self.nodes):
            if exclude and i in exclude:
                continue
            dp = np.linalg.norm(state.position - node.state.position)
            dh = abs(self._normalize_angle(state.heading - node.state.heading))
            d = dp + heading_weight * dh
            if d < min_dist:
                min_dist = d
                idx = i
        return idx

    def _near(self, state: KinoState, radius: float) -> List[int]:
        """返回半径内所有节点索引，按代价排序，最多返回15个"""
        heading_weight = self.min_turn_radius
        neighbors = []
        for i, node in enumerate(self.nodes):
            dp = np.linalg.norm(state.position - node.state.position)
            dh = abs(self._normalize_angle(state.heading - node.state.heading))
            d = dp + heading_weight * dh
            if d < radius:
                neighbors.append((i, node.cost))
        # 按代价排序，限制数量
        neighbors.sort(key=lambda x: x[1])
        return [idx for idx, _ in neighbors[:15]]

    def _steer(self, from_s: KinoState, to_s: KinoState) -> Optional[Tuple[KinoState, List[np.ndarray]]]:
        """
        使用Dubins曲线（或直线）扩展。

        下降量由"到目标的剩余高度预算"决定，而非采样点的z。
        小航向偏差时用直线代替Dubins，减少弧长浪费。

        返回:
            (新状态, 路径点列表) 或 None
        """
        dx = to_s.x - from_s.x
        dy = to_s.y - from_s.y
        dist_2d = np.sqrt(dx**2 + dy**2)

        if dist_2d < 10.0:
            return None

        # 截断到 step_size
        if dist_2d > self.step_size:
            ratio = self.step_size / dist_2d
            target_x = from_s.x + dx * ratio
            target_y = from_s.y + dy * ratio
            target_heading = np.arctan2(dy, dx)
        else:
            target_x = to_s.x
            target_y = to_s.y
            target_heading = to_s.heading

        # 判断航向偏差，小偏差用直线（更省高度）
        heading_to_target = np.arctan2(target_y - from_s.y, target_x - from_s.x)
        heading_diff = abs(self._normalize_angle(from_s.heading - heading_to_target))

        if heading_diff < 0.25:  # ~14度，几乎同向，直线更高效
            new_heading = heading_to_target
            path_2d = [np.array([from_s.x, from_s.y]), np.array([target_x, target_y])]
            path_len = np.sqrt((target_x - from_s.x)**2 + (target_y - from_s.y)**2)
        else:
            # 计算Dubins曲线
            dubins_path = self.dubins.compute(
                (from_s.x, from_s.y, from_s.heading),
                (target_x, target_y, target_heading)
            )

            if dubins_path is None:
                new_heading = heading_to_target
                path_2d = [np.array([from_s.x, from_s.y]), np.array([target_x, target_y])]
                path_len = np.sqrt((target_x - from_s.x)**2 + (target_y - from_s.y)**2)
            else:
                path_2d = self.dubins.sample(dubins_path, num_points=15)
                path_len = dubins_path['length']

                if len(path_2d) < 2:
                    return None

                last_dx = path_2d[-1][0] - path_2d[-2][0]
                last_dy = path_2d[-1][1] - path_2d[-2][1]
                if abs(last_dx) > 1e-6 or abs(last_dy) > 1e-6:
                    new_heading = np.arctan2(last_dy, last_dx)
                else:
                    new_heading = target_heading

        # --- 目标感知的下降量计算 ---
        # 根据"从当前节点到目标还需要什么滑翔比"来决定下降率
        remaining_h = np.linalg.norm(
            np.array([from_s.x, from_s.y]) - self._goal_xy)
        remaining_v = from_s.z - self._goal_z

        if remaining_v > 1.0:
            budget_glide = remaining_h / remaining_v
        else:
            budget_glide = self.max_glide

        # 目标滑翔比 = 预算滑翔比 clamped 到物理范围
        target_glide = np.clip(budget_glide, self.min_glide, self.max_glide)

        # 下降量 = 弧长 / 目标滑翔比，再 clamp 到物理极限
        descent = path_len / target_glide
        # 物理极限：不能比 max_glide 下降更少，不能比 min_glide 下降更多
        min_descent = path_len / self.max_glide
        max_descent = path_len / self.min_glide
        descent = np.clip(descent, min_descent, max_descent)

        new_z = from_s.z - descent
        new_x = path_2d[-1][0]
        new_y = path_2d[-1][1]

        # 构建3D路径点
        path_3d = []
        for idx, pt2d in enumerate(path_2d):
            alpha = idx / (len(path_2d) - 1) if len(path_2d) > 1 else 1.0
            z = from_s.z - alpha * descent
            path_3d.append(np.array([pt2d[0], pt2d[1], z]))

        new_state = KinoState(new_x, new_y, new_z, new_heading)
        return new_state, path_3d
    
    def _is_feasible(self, from_s: KinoState, to_s: KinoState, goal_z: float = 0.0) -> bool:
        """
        可行性检查
        
        注意：滑翔比已在_steer中保证，这里只检查高度限制
        """
        # 不能爬升（翼伞只能下降）
        if to_s.z > from_s.z + 1.0:
            return False
        
        # 不能太低（但目标附近允许）
        # 如果目标高度低于min_altitude，则允许在目标高度+一定余量
        effective_min = min(self.min_altitude, goal_z + 5.0)
        if to_s.z < effective_min:
            return False
        
        return True
    
    def _has_collision_path(self, path_3d: List[np.ndarray]) -> bool:
        """检测路径上的碰撞"""
        for pt in path_3d:
            if self.map.is_collision(pt):
                return True
        return False
    
    def _distance(self, s1: KinoState, s2: KinoState) -> float:
        """计算距离"""
        return np.linalg.norm(s2.position - s1.position)
    
    def _extract_path(self, goal_idx: int) -> List[np.ndarray]:
        """
        提取平滑路径
        
        利用存储的Dubins路径点，输出平滑的路径
        """
        # 先提取节点索引链
        indices = []
        idx = goal_idx
        while idx is not None:
            indices.append(idx)
            idx = self.nodes[idx].parent
        indices.reverse()
        
        # 拼接路径点
        path = []
        for i in range(len(indices) - 1):
            parent_idx = indices[i]
            child_idx = indices[i + 1]
            
            edge_key = (parent_idx, child_idx)
            if edge_key in self.edge_paths:
                edge_path = self.edge_paths[edge_key]
                # 除了最后一条边，不添加最后一个点（避免重复）
                if i < len(indices) - 2:
                    path.extend(edge_path[:-1])
                else:
                    path.extend(edge_path)
            else:
                # 没有存储的路径，直接用节点位置
                path.append(self.nodes[parent_idx].state.position.copy())
        
        # 确保最后一个节点在路径中
        if len(path) == 0 or not np.allclose(path[-1], self.nodes[goal_idx].state.position):
            path.append(self.nodes[goal_idx].state.position.copy())
        
        return path

    def export_to_json(self, output_dir: str = None, path: List[np.ndarray] = None, 
                       path_length: float = None) -> str:
        """
        导出RRT数据为JSON文件（用于Web可视化）
        
        参数:
            output_dir: 输出目录（默认: visualization/data/）
            path: 最终路径（可选）
            path_length: 路径长度（可选）
            
        返回:
            生成的文件路径
        """
        import json
        import os
        from datetime import datetime
        
        # 默认输出目录
        if output_dir is None:
            output_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'visualization', 'data'
            )
        
        # 创建目录（如果不存在）
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成唯一文件名：rrt_YYYYMMDD_HHMMSS_微秒.json
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        success_tag = 'ok' if path else 'fail'
        filename = f'rrt_{timestamp}_{success_tag}.json'
        filepath = os.path.join(output_dir, filename)
        
        # 节点数据
        nodes_data = []
        for i, node in enumerate(self.nodes):
            nodes_data.append({
                'x': float(node.state.x),
                'y': float(node.state.y),
                'z': float(node.state.z),
                'heading': float(node.state.heading),
                'parent_idx': node.parent,
                'cost': float(node.cost)
            })
        
        # 路径数据
        path_data = []
        if path:
            for pt in path:
                path_data.append([float(pt[0]), float(pt[1]), float(pt[2])])
        
        # 障碍物数据
        obstacles_data = []
        for obs in self.map.obstacles:
            if hasattr(obs, 'radius'):  # Cylinder
                obstacles_data.append({
                    'type': 'cylinder',
                    'center': [float(obs.center[0]), float(obs.center[1])],
                    'radius': float(obs.radius),
                    'z_min': float(obs.z_min),
                    'z_max': float(obs.z_max)
                })
            elif hasattr(obs, 'polygon'):  # Prism
                obstacles_data.append({
                    'type': 'prism',
                    'vertices': obs.polygon.vertices.tolist(),
                    'z_min': float(obs.z_min),
                    'z_max': float(obs.z_max)
                })
        
        # 构建完整数据
        data = {
            'nodes': nodes_data,
            'path': path_data,
            'path_length': path_length,
            'start': {
                'x': float(self.map.start.x),
                'y': float(self.map.start.y),
                'z': float(self.map.start.z)
            },
            'goal': {
                'x': float(self.map.target.position[0]),
                'y': float(self.map.target.position[1]),
                'z': float(self.map.target.position[2])
            },
            'obstacles': obstacles_data,
            'config': {
                'min_glide': float(self.min_glide),
                'max_glide': float(self.max_glide),
                'min_turn_radius': float(self.min_turn_radius),
                'step_size': float(self.step_size)
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        if not self.quiet:
            print(f"  数据已导出: {filepath}")
        return filepath

    def _inject_spiral_descent(self, path: List[np.ndarray], goal: np.ndarray,
                                goal_heading: float) -> List[np.ndarray]:
        """
        后处理：当路径末段的滑翔比低于 min_glide（下降过陡，高度远多于水平距离
        能消耗的量）时，在目标附近注入螺旋下降段消掉多余高度。

        只消掉"多余"的高度，即正常以 min_glide 滑过去消不掉的那部分。
        """
        if len(path) < 2:
            return path

        # 取路径倒数第二个点（螺旋插入点）到目标的关系
        # path[-1] 是 goal 本身，取 path[-2] 作为"到达目标前的最后一个路径点"
        entry_pt = path[-2] if len(path) >= 2 else path[-1]

        h_dist = np.linalg.norm(entry_pt[:2] - goal[:2])
        v_dist = entry_pt[2] - goal[2]

        if v_dist <= 0:
            return path  # 不需要下降

        needed_glide = h_dist / v_dist if v_dist > 1e-6 else float('inf')

        # 如果滑翔比 >= min_glide，正常滑翔就能到达，不需要盘旋
        if needed_glide >= self.min_glide:
            return path

        # 多余高度 = 总高度差 - 以 min_glide 飞完水平距离能消耗的高度
        consumable_alt = h_dist / self.min_glide
        excess_alt = v_dist - consumable_alt

        if excess_alt < 10.0:
            return path  # 多余量太小，忽略

        if not self.quiet:
            print(f"  [后处理] 滑翔比 {needed_glide:.2f} < min_glide {self.min_glide:.2f}，"
                  f"需螺旋消高 {excess_alt:.1f}m")

        # 螺旋参数
        loiter_radius = self.min_turn_radius * 1.5
        glide_for_spiral = self.max_glide  # 螺旋时用最大滑翔比（最缓下降率）

        # 螺旋消掉 excess_alt 所需的水平飞行距离
        h_needed = excess_alt * glide_for_spiral
        circumference = 2 * np.pi * loiter_radius
        num_loops = h_needed / circumference

        # 螺旋中心在目标附近
        cx, cy = goal[0], goal[1]

        # 生成螺旋点
        points_per_loop = 20
        total_points = max(int(num_loops * points_per_loop), 10)
        total_angle = num_loops * 2 * np.pi

        spiral_path = []
        start_z = entry_pt[2]
        start_angle = np.arctan2(entry_pt[1] - cy, entry_pt[0] - cx)

        for k in range(total_points + 1):
            frac = k / total_points
            angle = start_angle - frac * total_angle  # 顺时针
            x = cx + loiter_radius * np.cos(angle)
            y = cy + loiter_radius * np.sin(angle)
            z = start_z - frac * excess_alt  # 只消掉多余高度
            spiral_path.append(np.array([x, y, z]))

        # 检查螺旋段碰撞
        if self._has_collision_path(spiral_path):
            if not self.quiet:
                print(f"  [后处理] 螺旋段有碰撞，跳过注入")
            return path

        # 拼接：path[:-1]（去掉goal） + 螺旋段 + goal
        # 螺旋结束后高度 = entry_pt.z - excess_alt = goal.z + consumable_alt
        # 此时从螺旋终点到goal的滑翔比 ≈ min_glide，可以正常滑到
        result = path[:-1] + spiral_path + [goal]
        return result


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from planning.map_manager import MapManager, Cylinder, Prism
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle as MplCircle, Polygon as MplPolygon
    
    map_mgr = MapManager.from_yaml("cfg/map_config.yaml")
    planner = KinodynamicRRTStar(map_mgr)
    
    path, info = planner.plan(max_time=30.0)
    
    # 导出JSON（用于Web可视化）
    planner.export_to_json(path=path, path_length=info.get('path_length'))
    
    # 可视化
    fig = plt.figure(figsize=(18, 6))
    
    # 3D图
    ax0 = fig.add_subplot(131, projection='3d')
    
    # 绘制3D障碍物（圆柱体）
    for obs in map_mgr.obstacles:
        if isinstance(obs, Cylinder):
            # 绘制圆柱体
            theta = np.linspace(0, 2*np.pi, 30)
            z_cyl = np.linspace(obs.z_min, obs.z_max, 10)
            theta, z_cyl = np.meshgrid(theta, z_cyl)
            x_cyl = obs.center[0] + obs.radius * np.cos(theta)
            y_cyl = obs.center[1] + obs.radius * np.sin(theta)
            ax0.plot_surface(x_cyl, y_cyl, z_cyl, alpha=0.3, color='gray')
        elif isinstance(obs, Prism):
            # 绘制棱柱的侧面
            verts = obs.polygon.vertices
            n = len(verts)
            for i in range(n):
                j = (i + 1) % n
                xs = [verts[i, 0], verts[j, 0], verts[j, 0], verts[i, 0]]
                ys = [verts[i, 1], verts[j, 1], verts[j, 1], verts[i, 1]]
                zs = [obs.z_min, obs.z_min, obs.z_max, obs.z_max]
                ax0.plot_surface(np.array([[xs[0], xs[1]], [xs[3], xs[2]]]),
                                np.array([[ys[0], ys[1]], [ys[3], ys[2]]]),
                                np.array([[zs[0], zs[1]], [zs[3], zs[2]]]),
                                alpha=0.3, color='orange')
    
    # 绘制3D路径
    if path:
        path_arr = np.array(path)
        ax0.plot(path_arr[:, 0], path_arr[:, 1], path_arr[:, 2], 
                'b-', linewidth=2, label='Path')
        ax0.scatter(path_arr[:, 0], path_arr[:, 1], path_arr[:, 2], 
                   c='blue', s=20)
    
    # 起点终点
    start_pos = np.array([map_mgr.start.x, map_mgr.start.y, map_mgr.start.z])
    goal_pos = map_mgr.target.position
    ax0.scatter(*start_pos, c='green', s=100, marker='o', label='Start')
    ax0.scatter(*goal_pos, c='red', s=150, marker='*', label='Goal')
    
    ax0.set_xlabel('X (m)')
    ax0.set_ylabel('Y (m)')
    ax0.set_zlabel('Z (m)')
    ax0.legend()
    ax0.set_title('3D View')
    
    # XY平面图
    ax1 = fig.add_subplot(132)
    
    # 绘制障碍物
    for obs in map_mgr.obstacles:
        if isinstance(obs, Cylinder):
            circle = MplCircle((obs.center[0], obs.center[1]), obs.radius, 
                              fill=True, facecolor='gray', edgecolor='black', 
                              alpha=0.5, linewidth=1)
            ax1.add_patch(circle)
            # 标注高度
            ax1.text(obs.center[0], obs.center[1], f'{obs.z_max:.0f}m', 
                    ha='center', va='center', fontsize=7, color='white')
        elif isinstance(obs, Prism):
            poly = MplPolygon(obs.polygon.vertices, fill=True, facecolor='orange', 
                             edgecolor='black', alpha=0.5, linewidth=1)
            ax1.add_patch(poly)
            # 标注高度
            cx = np.mean(obs.polygon.vertices[:, 0])
            cy = np.mean(obs.polygon.vertices[:, 1])
            ax1.text(cx, cy, f'{obs.z_max:.0f}m', 
                    ha='center', va='center', fontsize=7, color='white')
    
    # 绘制路径
    if path:
        path_arr = np.array(path)
        ax1.plot(path_arr[:, 0], path_arr[:, 1], 'b-', linewidth=2, label='Path')
        ax1.plot(path_arr[:, 0], path_arr[:, 1], 'bo', markersize=4)
    
    # 起点终点
    start = np.array([map_mgr.start.x, map_mgr.start.y])
    goal = map_mgr.target.position[:2]
    ax1.plot(start[0], start[1], 'go', markersize=12, label='Start', zorder=10)
    ax1.plot(goal[0], goal[1], 'r*', markersize=18, label='Goal', zorder=10)
    
    # 设置范围
    bounds = map_mgr.bounds
    ax1.set_xlim(bounds['x_min'] - 50, bounds['x_max'] + 50)
    ax1.set_ylim(bounds['y_min'] - 50, bounds['y_max'] + 50)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.legend(loc='upper left')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('XY Plane with Obstacles')
    
    # 高度剖面
    ax2 = fig.add_subplot(133)
    if path:
        # 计算累积距离
        distances = [0]
        for i in range(1, len(path_arr)):
            d = np.linalg.norm(path_arr[i, :2] - path_arr[i-1, :2])
            distances.append(distances[-1] + d)
        
        ax2.plot(distances, path_arr[:, 2], 'b-o', markersize=4, linewidth=2)
        ax2.fill_between(distances, 0, path_arr[:, 2], alpha=0.3)
        
        # 标注滑翔比
        for i in range(1, len(path_arr)):
            dz = path_arr[i-1, 2] - path_arr[i, 2]
            dxy = distances[i] - distances[i-1]
            if dz > 1:
                glide = dxy / dz
                mid_x = (distances[i] + distances[i-1]) / 2
                mid_y = (path_arr[i, 2] + path_arr[i-1, 2]) / 2
                if i % 3 == 0:  # 每隔几个点标注一次
                    ax2.text(mid_x, mid_y + 20, f'{glide:.1f}', fontsize=8, 
                            ha='center', color='red')
    
    ax2.set_xlabel('Distance (m)')
    ax2.set_ylabel('Altitude (m)')
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Altitude Profile (red numbers = glide ratio)')
    
    plt.tight_layout()
    plt.show()
