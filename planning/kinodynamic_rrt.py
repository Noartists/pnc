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

from planning.dubins import DubinsPath


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
        # 有效最小滑翔比（考虑航向/下降共享控制预算）
        # 翼伞弯道时 delta_a 占用大量行程 → delta_s 受限 → 下降能力降低
        # 整条路径的实际平均滑翔比远高于理论 min_glide
        # 经验值约 5.0（基于 ~67% 饱和率下的加权平均），
        # 对比理论中点 4.48 更保守，确保路径长度充足
        self.effective_min_glide = 5.0
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

            if dist_to_goal < 300 and dz_to_goal < 200:
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

        # --- 最终尝试：如果主循环未到达目标，从最近节点尝试连接 ---
        if not goal_reached:
            candidates = []
            for ni, node in enumerate(self.nodes):
                d2d = np.linalg.norm(
                    node.state.position[:2] - goal_state.position[:2])
                candidates.append((d2d, ni))
            candidates.sort()

            for d2d, ni in candidates[:50]:
                if d2d > 500:
                    break
                n_node = self.nodes[ni]
                gs = self._steer(n_node.state, goal_state)
                if gs is None:
                    continue
                gs_state, gs_path = gs
                gs_dist = np.linalg.norm(
                    gs_state.position[:2] - goal_state.position[:2])
                if gs_dist < 100 and not self._has_collision_path(gs_path):
                    gs_len = sum(
                        np.linalg.norm(gs_path[j+1] - gs_path[j])
                        for j in range(len(gs_path)-1))
                    gs_cost = n_node.cost + gs_len
                    gs_node = KinoNode(gs_state, ni, gs_cost)
                    gs_idx = len(self.nodes)
                    self.nodes.append(gs_node)
                    self.edge_paths[(ni, gs_idx)] = gs_path
                    if gs_cost < best_cost:
                        best_cost = gs_cost
                        best_idx = gs_idx
                        goal_reached = True
                        if not self.quiet:
                            print(f"  [最终连接] 从节点{ni}成功, "
                                  f"2D距离={d2d:.0f}m")
                        break

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

        # 提取路径（此时高度是 _steer 中用名义滑翔比粗算的，需要重分配）
        path = self._extract_path(best_idx)
        path.append(goal)  # 添加终点

        # 提取节点链（供后处理器使用）
        node_chain = self._extract_node_chain(best_idx)

        # --- 后处理：基于实际路径长度重新分配高度 ---
        z_start = start[2]
        z_goal = goal[2]
        path, actual_glide = self._reassign_altitude(
            path, z_start, z_goal, goal_state.heading
        )

        if not self.quiet:
            print(f"  [成功] 路径点数: {len(path)}, 实际滑翔比: {actual_glide:.2f}")

        return path, {
            'success': True,
            'stats': stats,
            'path_length': best_cost,
            'edge_paths': self.edge_paths,
            'node_chain': node_chain,
        }
    
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
            # 引导式采样：z坐标基于滑翔比可行锥
            alpha = np.random.uniform(0.2, 1.0)
            start = np.array([self.map.start.x, self.map.start.y, self.map.start.z])

            base_x = start[0] + alpha * (goal[0] - start[0])
            base_y = start[1] + alpha * (goal[1] - start[1])
            x = base_x + np.random.uniform(-200, 200)
            y = base_y + np.random.uniform(-200, 200)
            x = np.clip(x, bounds['x_min'], bounds['x_max'])
            y = np.clip(y, bounds['y_min'], bounds['y_max'])

            # z 基于到目标的剩余2D距离和滑翔比可行锥
            # 使用 effective_min_glide（考虑控制预算共享），
            # 避免采样到物理上不可达的高度范围
            remaining = np.linalg.norm(np.array([x, y]) - goal[:2])
            z_low = goal[2] + remaining / self.max_glide
            z_high = goal[2] + remaining / self.effective_min_glide
            z_high = min(z_high, z_max)
            z_low = max(z_low, max(z_min, 0.0))
            if z_low <= z_high:
                z = np.random.uniform(z_low, z_high)
            else:
                z = z_low
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

        # --- 自适应下降率：根据当前节点到目标的剩余距离动态调整 ---
        end_pt = path_2d[-1]
        remaining_2d = np.linalg.norm(
            np.array([end_pt[0], end_pt[1]]) - self._goal_xy)
        altitude_above_goal = from_s.z - self._goal_z

        if altitude_above_goal > 1.0 and remaining_2d > 1.0:
            # 当前位置飞到目标所需的滑翔比，留 10% 余量
            needed_glide = remaining_2d / altitude_above_goal
            # 使用 effective_min_glide 作为下限，确保分配到的下降率
            # 在控制预算共享约束下物理可达
            adaptive_glide = np.clip(needed_glide * 1.1,
                                     self.effective_min_glide, self.max_glide)
        else:
            # 已在目标正上方或高度极低，用有效最小滑翔比下降
            adaptive_glide = self.effective_min_glide

        descent = path_len / adaptive_glide
        min_descent = path_len / self.max_glide
        max_descent = path_len / self.effective_min_glide
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

    def _extract_node_chain(self, goal_idx: int) -> List[Tuple[int, int]]:
        """
        提取有序的边键列表 [(parent_idx, child_idx), ...]

        与 _extract_path 逻辑类似，但返回节点索引对而非路径点。
        """
        indices = []
        idx = goal_idx
        while idx is not None:
            indices.append(idx)
            idx = self.nodes[idx].parent
        indices.reverse()

        chain = []
        for i in range(len(indices) - 1):
            chain.append((indices[i], indices[i + 1]))
        return chain

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

    # ================================================================
    #  后处理：高度重分配（核心改进）
    # ================================================================

    def _reassign_altitude(self, path: List[np.ndarray],
                           z_start: float, z_goal: float,
                           goal_heading: float) -> Tuple[List[np.ndarray], float]:
        """
        基于实际 2D 路径长度，重新分配高度剖面（线性分配）。

        螺旋消高已移至后处理器 (_inject_spiral_dubins)，
        此处仅做粗略的线性 z 分配供规划器输出。

        返回:
            (path, actual_glide_ratio)
        """
        if len(path) < 2:
            return path, 0.0

        delta_z = z_start - z_goal
        if delta_z <= 0:
            return path, float('inf')

        # 计算 2D 路径总长
        total_len_2d = self._path_length_2d(path)

        if total_len_2d < 1.0:
            return path, 0.0

        required_glide = total_len_2d / delta_z

        if not self.quiet:
            print(f"  [高度重分配] 2D路径长={total_len_2d:.0f}m, "
                  f"高度差={delta_z:.0f}m, 所需滑翔比={required_glide:.2f}, "
                  f"有效最小滑翔比={self.effective_min_glide:.2f}")

        # 线性分配高度（按 2D 距离比例）
        cum_dist = 0.0
        path[0][2] = z_start
        for i in range(1, len(path)):
            seg_len = np.linalg.norm(path[i][:2] - path[i - 1][:2])
            cum_dist += seg_len
            frac = cum_dist / total_len_2d if total_len_2d > 0 else 1.0
            frac = min(frac, 1.0)
            path[i][2] = z_start - frac * delta_z

        # 确保终点精确
        path[-1][2] = z_goal

        return path, required_glide

    @staticmethod
    def _path_length_2d(path: List[np.ndarray]) -> float:
        """计算路径的 2D 总长度"""
        total = 0.0
        for i in range(len(path) - 1):
            total += np.linalg.norm(path[i + 1][:2] - path[i][:2])
        return total


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
