"""
轨迹后处理模块

功能：
1. 接收 Kinodynamic RRT* 返回的路径点列表
2. Dubins 感知流水线（保留 Dubins 曲线的动力学结构）
3. 三次样条平滑（legacy 回退）
4. 按控制频率重采样
5. 时间参数化
6. 生成 Trajectory 对象

用法：
    from planning.trajectory_postprocess import TrajectoryPostprocessor

    postprocessor = TrajectoryPostprocessor(
        reference_speed=12.0,
        control_frequency=100.0,
        min_turn_radius=50.0
    )

    trajectory = postprocessor.process(path, smooth=True)
"""

import numpy as np
from typing import List, Optional, Tuple, Dict
from scipy.interpolate import CubicSpline
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from planning.trajectory import Trajectory, TrajectoryPoint
from planning.dubins import DubinsPath


class TrajectoryPostprocessor:
    """
    轨迹后处理器

    将 Kinodynamic RRT* 生成的路径点转换为可用于控制的平滑轨迹。
    支持两条流水线：
    - Dubins 感知流水线（当 edge_paths/node_chain 可用时）
    - Legacy 三次样条流水线（向后兼容）
    """

    def __init__(self,
                 reference_speed: float = 12.0,
                 control_frequency: float = 100.0,
                 min_turn_radius: float = 50.0,
                 max_glide_ratio: float = 6.48,
                 min_glide_ratio: float = 2.47,
                 map_manager=None,
                 quiet: bool = False):
        """
        参数:
            reference_speed: 参考飞行速度 (m/s)
            control_frequency: 控制频率 (Hz)
            min_turn_radius: 最小转弯半径 (m)，用于约束检查
            max_glide_ratio: 最大滑翔比（直线段）
            min_glide_ratio: 最小滑翔比（最大偏转转弯）
            map_manager: 地图管理器（用于螺旋碰撞检查）
            quiet: 是否静默模式（不打印信息）
        """
        self.reference_speed = reference_speed
        self.control_frequency = control_frequency
        self.dt = 1.0 / control_frequency
        self.min_turn_radius = min_turn_radius
        self.max_glide = max_glide_ratio
        self.min_glide = min_glide_ratio
        self.quiet = quiet
        self.dubins = DubinsPath(self.min_turn_radius)
        self.map_manager = map_manager

    def process(self,
                path: List[np.ndarray],
                smooth: bool = True,
                end_heading: Optional[float] = None,
                edge_paths: Optional[Dict] = None,
                node_chain: Optional[List[Tuple[int, int]]] = None) -> Trajectory:
        """
        处理路径点，生成平滑轨迹

        参数:
            path: Kinodynamic RRT* 返回的路径点列表 [np.array([x,y,z]), ...]
            smooth: 是否进行三次样条平滑（默认True）
            end_heading: 终点期望航向（可选，用于进场对齐）
            edge_paths: RRT* 的边路径字典 {(parent_idx, child_idx): [3D points]}
            node_chain: RRT* 的节点链 [(parent_idx, child_idx), ...]

        返回:
            Trajectory 对象
        """
        if not path or len(path) < 2:
            if not self.quiet:
                print("    [后处理] 路径点太少，返回空轨迹")
            return Trajectory(dt=self.dt)

        # 分发逻辑
        if edge_paths is not None and node_chain is not None:
            if not self.quiet:
                print(f"\n    [后处理] 使用 Dubins 感知流水线")
            return self._process_dubins_aware(path, edge_paths, node_chain, end_heading)
        else:
            if not self.quiet:
                print(f"\n    [后处理] 使用 Legacy 流水线")
            return self._process_legacy(path, smooth, end_heading)

    # ==================================================================
    #  新流水线: Dubins 感知
    # ==================================================================

    def _process_dubins_aware(self,
                              path: List[np.ndarray],
                              edge_paths: Dict,
                              node_chain: List[Tuple[int, int]],
                              end_heading: Optional[float] = None) -> Trajectory:
        """
        Dubins 感知流水线:
          1. _stitch_edge_paths     — 拼接 Dubins 边路径 (2D)
          2. _fix_junctions         — 修复接缝航向不连续
          3. _inject_spiral_dubins  — 螺旋消高 + Dubins 出口弧
          4. _resample_arc_length   — 均匀弧长重采样
          5. _redistribute_altitude_constrained — 约束内高度分配
          6. _create_trajectory     — 生成 Trajectory 对象
          7. _validate_trajectory   — 验证约束
        """
        waypoints = [np.asarray(wp, dtype=np.float64) for wp in path]
        z_start = waypoints[0][2]
        z_goal = waypoints[-1][2]

        if not self.quiet:
            print(f"    [后处理] 输入: {len(path)} 个路径点, z: {z_start:.0f} -> {z_goal:.0f}")

        # Step 1: 拼接 Dubins 边路径 (只取 XY)
        points_2d, edge_lengths = self._stitch_edge_paths(edge_paths, node_chain, waypoints)
        if not self.quiet:
            print(f"    [Step1] 拼接后: {len(points_2d)} 个 2D 点")

        # Step 2: 修复接缝
        points_2d = self._fix_junctions(points_2d, edge_lengths)
        if not self.quiet:
            print(f"    [Step2] 修复接缝后: {len(points_2d)} 个 2D 点")

        # Step 3: 螺旋消高（如果需要）
        points_2d = self._inject_spiral_dubins(points_2d, z_start, z_goal, end_heading)
        if not self.quiet:
            print(f"    [Step3] 螺旋消高后: {len(points_2d)} 个 2D 点")

        # Step 4: 均匀弧长重采样
        points_2d = self._resample_arc_length(points_2d)
        if not self.quiet:
            print(f"    [Step4] 重采样后: {len(points_2d)} 个 2D 点")

        # Step 5: 约束内高度分配 → 生成 3D 点
        points_3d = self._redistribute_altitude_constrained(points_2d, z_start, z_goal)
        if not self.quiet:
            print(f"    [Step5] 高度分配后: {len(points_3d)} 个 3D 点")

        # Step 6: 生成轨迹
        trajectory = self._create_trajectory(points_3d, end_heading)
        if not self.quiet:
            print(f"    [后处理] 生成轨迹: {len(trajectory)} 点, 时长 {trajectory.duration:.1f}s")

        # Step 7: 验证约束
        self._validate_trajectory(trajectory)

        return trajectory

    def _stitch_edge_paths(self,
                           edge_paths: Dict,
                           node_chain: List[Tuple[int, int]],
                           waypoints: List[np.ndarray]
                           ) -> Tuple[List[np.ndarray], List[int]]:
        """
        按 node_chain 顺序拼接 Dubins 边路径（只取 XY）。

        返回:
            (points_2d, edge_lengths)
            edge_lengths[i] = 第 i 条边在 points_2d 中的点数
        """
        all_points = []
        edge_lengths = []

        for idx, (parent_idx, child_idx) in enumerate(node_chain):
            edge_key = (parent_idx, child_idx)
            if edge_key in edge_paths:
                edge_pts = edge_paths[edge_key]
                # 取 XY
                pts_2d = [np.array([pt[0], pt[1]]) for pt in edge_pts]
            else:
                # 回退: 起终点直线
                # 从 waypoints 中尝试找合理的起终点
                if idx < len(waypoints) - 1:
                    p_start = waypoints[idx][:2]
                    p_end = waypoints[idx + 1][:2]
                else:
                    p_start = waypoints[-2][:2]
                    p_end = waypoints[-1][:2]
                pts_2d = [p_start.copy(), p_end.copy()]

            if idx == 0:
                all_points.extend(pts_2d)
                edge_lengths.append(len(pts_2d))
            else:
                # 跳过第一个点（与前一条边末点重复）
                if len(pts_2d) > 1:
                    all_points.extend(pts_2d[1:])
                    edge_lengths.append(len(pts_2d) - 1)
                else:
                    edge_lengths.append(0)

        # 如果路径末尾有终点不在 edge_paths 中覆盖的范围
        # （plan() 中 append(goal)），确保终点在列表里
        if len(waypoints) > 0:
            goal_xy = waypoints[-1][:2]
            if len(all_points) > 0:
                dist_to_goal = np.linalg.norm(all_points[-1] - goal_xy)
                if dist_to_goal > 1.0:
                    all_points.append(goal_xy.copy())

        return all_points, edge_lengths

    def _fix_junctions(self,
                       points: List[np.ndarray],
                       edge_lengths: List[int]) -> List[np.ndarray]:
        """
        检测接缝处航向不连续，插入 Dubins 过渡弧。
        """
        if len(points) < 4 or len(edge_lengths) < 2:
            return points

        # 计算接缝索引（每条边结束位置）
        junction_indices = []
        cum = 0
        for i, el in enumerate(edge_lengths):
            cum += el
            if i < len(edge_lengths) - 1:
                # 接缝在 cum-1（当前边末点）和 cum（下一条边首点）之间
                # 但由于我们跳过了重复点，实际接缝就在 cum-1 处
                junction_indices.append(min(cum - 1, len(points) - 2))

        heading_threshold = 0.1  # rad (~5.7°)
        result = list(points)
        offset = 0  # 插入点导致的索引偏移

        for j_idx in junction_indices:
            j = j_idx + offset
            if j < 1 or j >= len(result) - 1:
                continue

            # 出口航向（从 j-1 到 j）
            dx_out = result[j][0] - result[j - 1][0]
            dy_out = result[j][1] - result[j - 1][1]
            if abs(dx_out) < 1e-6 and abs(dy_out) < 1e-6:
                continue
            heading_out = np.arctan2(dy_out, dx_out)

            # 入口航向（从 j 到 j+1）
            dx_in = result[j + 1][0] - result[j][0]
            dy_in = result[j + 1][1] - result[j][1]
            if abs(dx_in) < 1e-6 and abs(dy_in) < 1e-6:
                continue
            heading_in = np.arctan2(dy_in, dx_in)

            # 航向差
            dh = heading_in - heading_out
            while dh > np.pi:
                dh -= 2 * np.pi
            while dh < -np.pi:
                dh += 2 * np.pi

            if abs(dh) < heading_threshold:
                continue

            # 用 Dubins 曲线过渡
            start_pose = (result[j][0], result[j][1], heading_out)
            end_pose = (result[j + 1][0], result[j + 1][1], heading_in)
            dubins_path = self.dubins.compute(start_pose, end_pose)
            if dubins_path is not None:
                transition_pts = self.dubins.sample(dubins_path, num_points=10)
                if len(transition_pts) > 2:
                    # 替换接缝点: 移除 j 和 j+1，插入过渡弧
                    new_pts = [np.array([pt[0], pt[1]]) for pt in transition_pts]
                    result = result[:j] + new_pts + result[j + 2:]
                    offset += len(new_pts) - 2

        return result

    def _inject_spiral_dubins(self,
                              points: List[np.ndarray],
                              z_start: float, z_goal: float,
                              end_heading: Optional[float]) -> List[np.ndarray]:
        """
        若所需滑翔比 < 有效最小滑翔比，生成螺旋消高 + Dubins 出口弧。
        """
        if len(points) < 2:
            return points

        delta_z = z_start - z_goal
        if delta_z <= 0:
            return points

        # 计算 2D 路径总长
        total_len_2d = 0.0
        for i in range(len(points) - 1):
            total_len_2d += np.linalg.norm(points[i + 1] - points[i])

        if total_len_2d < 1.0:
            return points

        # 路径加权有效最小滑翔比（动态计算）
        n = len(points)
        G_max = self.max_glide
        G_min = self.min_glide
        R_min = self.min_turn_radius

        weighted_g_eff_min = 0.0
        total_weight = 0.0
        for i in range(n - 1):
            ds = np.linalg.norm(points[i + 1] - points[i])
            if ds < 1e-6:
                continue
            # 计算局部曲率
            if 0 < i < n - 1:
                kappa = self._compute_curvature_2d(
                    points[max(i - 1, 0)], points[i], points[min(i + 1, n - 1)])
            else:
                kappa = 0.0
            f_sym = max(0.0, 1.0 - kappa * R_min)
            g_eff_min_i = G_max - f_sym * (G_max - G_min)
            weighted_g_eff_min += g_eff_min_i * ds
            total_weight += ds

        if total_weight > 1e-6:
            effective_min_glide = weighted_g_eff_min / total_weight
        else:
            effective_min_glide = 5.0

        # 留一些余量
        effective_min_glide = max(effective_min_glide, 4.5)

        required_glide = total_len_2d / delta_z

        if not self.quiet:
            print(f"    [螺旋检查] 所需滑翔比={required_glide:.2f}, "
                  f"有效最小滑翔比={effective_min_glide:.2f}")

        if required_glide >= effective_min_glide:
            return points  # 不需要螺旋

        # 需要螺旋消高
        needed_len = delta_z * effective_min_glide
        shortfall = needed_len - total_len_2d

        if not self.quiet:
            print(f"    [螺旋消高] 需额外路径 {shortfall:.0f}m")

        loiter_radius = self.min_turn_radius * 2.0
        circumference = 2 * np.pi * loiter_radius
        num_loops = max(shortfall / circumference, 0.5)

        goal_xy = points[-1]
        goal_x, goal_y = goal_xy[0], goal_xy[1]

        # 进场航向
        if len(points) >= 3:
            approach_heading = np.arctan2(
                points[-1][1] - points[-3][1],
                points[-1][0] - points[-3][0])
        elif len(points) >= 2:
            approach_heading = np.arctan2(
                points[-1][1] - points[-2][1],
                points[-1][0] - points[-2][0])
        elif end_heading is not None:
            approach_heading = end_heading
        else:
            approach_heading = 0.0

        # 螺旋中心
        cx = goal_x - loiter_radius * np.cos(approach_heading)
        cy = goal_y - loiter_radius * np.sin(approach_heading)
        exit_angle = np.arctan2(goal_y - cy, goal_x - cx)

        # 找插入位置
        insert_idx = len(points) - 2
        min_dist = float('inf')
        for i in range(max(1, len(points) // 2), len(points) - 1):
            d = np.sqrt((points[i][0] - cx) ** 2 + (points[i][1] - cy) ** 2)
            if d < min_dist:
                min_dist = d
                insert_idx = i

        # 螺旋入口方向
        if insert_idx > 0:
            entry_pt = points[insert_idx - 1]
            start_angle = np.arctan2(entry_pt[1] - cy, entry_pt[0] - cx)
        else:
            start_angle = approach_heading + np.pi

        # 调整总角度使螺旋出口对齐
        total_angle = num_loops * 2 * np.pi
        desired_end = exit_angle
        actual_end = start_angle - total_angle
        angle_correction = (desired_end - actual_end) % (2 * np.pi)
        total_angle += angle_correction

        # 生成螺旋 2D 点
        points_per_loop = 36
        total_spiral_pts = max(int((total_angle / (2 * np.pi)) * points_per_loop), 18)

        spiral_path = []
        for k in range(total_spiral_pts + 1):
            frac = k / total_spiral_pts
            angle = start_angle - frac * total_angle
            x = cx + loiter_radius * np.cos(angle)
            y = cy + loiter_radius * np.sin(angle)
            spiral_path.append(np.array([x, y]))

        # 螺旋出口切线航向
        if len(spiral_path) >= 2:
            exit_dx = spiral_path[-1][0] - spiral_path[-2][0]
            exit_dy = spiral_path[-1][1] - spiral_path[-2][1]
            spiral_exit_heading = np.arctan2(exit_dy, exit_dx)
        else:
            spiral_exit_heading = exit_angle - np.pi / 2  # 顺时针切线

        # 进场段起始航向
        approach_start_heading = approach_heading

        # 用 Dubins 曲线连接螺旋出口到进场段起始
        spiral_exit_pt = spiral_path[-1] if spiral_path else np.array([
            cx + loiter_radius * np.cos(exit_angle),
            cy + loiter_radius * np.sin(exit_angle)])

        dubins_transition = self.dubins.compute(
            (spiral_exit_pt[0], spiral_exit_pt[1], spiral_exit_heading),
            (goal_x, goal_y, approach_start_heading))

        transition_pts = []
        if dubins_transition is not None:
            sampled = self.dubins.sample(dubins_transition, num_points=15)
            transition_pts = [np.array([pt[0], pt[1]]) for pt in sampled]
        else:
            # 回退: 直线连接
            dist_to_goal = np.linalg.norm(spiral_exit_pt - goal_xy)
            n_approach = max(3, int(dist_to_goal / 10.0))
            for k in range(1, n_approach + 1):
                frac = k / n_approach
                x = spiral_exit_pt[0] + frac * (goal_x - spiral_exit_pt[0])
                y = spiral_exit_pt[1] + frac * (goal_y - spiral_exit_pt[1])
                transition_pts.append(np.array([x, y]))

        # 碰撞检查螺旋段
        if self.map_manager is not None:
            approx_z = max(50.0, delta_z * 0.3)
            for pt in spiral_path:
                if self.map_manager.is_collision(np.array([pt[0], pt[1], approx_z])):
                    if not self.quiet:
                        print(f"    [螺旋消高] 螺旋段有碰撞，跳过")
                    return points

        if not self.quiet:
            print(f"    [螺旋消高] 中心=({cx:.0f},{cy:.0f}), "
                  f"半径={loiter_radius:.0f}m, "
                  f"圈数={total_angle / (2 * np.pi):.1f}")

        # 拼接: path[:insert] + spiral + dubins_transition + [goal]
        result = points[:insert_idx] + spiral_path
        if transition_pts:
            # 跳过第一个点（与螺旋末点重复）
            result.extend(transition_pts[1:] if len(transition_pts) > 1 else transition_pts)
        result.append(points[-1].copy())
        return result

    def _resample_arc_length(self,
                             points: List[np.ndarray],
                             spacing: float = None) -> List[np.ndarray]:
        """
        按均匀弧长间隔重采样 2D 点（不改变形状）。
        """
        if len(points) < 2:
            return points

        if spacing is None:
            spacing = self.reference_speed * self.dt  # ~0.08m for 8m/s @ 100Hz

        # 计算累积弧长
        arc_lengths = [0.0]
        for i in range(1, len(points)):
            d = np.linalg.norm(points[i] - points[i - 1])
            arc_lengths.append(arc_lengths[-1] + d)

        total_length = arc_lengths[-1]
        if total_length < 1e-6:
            return points

        n_samples = max(int(total_length / spacing), 2)

        resampled = []
        for i in range(n_samples + 1):
            target_s = (i / n_samples) * total_length
            pt = self._interpolate_at_arc_length_2d(points, arc_lengths, target_s)
            resampled.append(pt)

        return resampled

    def _interpolate_at_arc_length_2d(self,
                                       points: List[np.ndarray],
                                       arc_lengths: List[float],
                                       s: float) -> np.ndarray:
        """在指定弧长处对 2D 点列表线性插值"""
        for i in range(1, len(arc_lengths)):
            if arc_lengths[i] >= s:
                s0, s1 = arc_lengths[i - 1], arc_lengths[i]
                if s1 - s0 < 1e-6:
                    alpha = 0.0
                else:
                    alpha = (s - s0) / (s1 - s0)
                return points[i - 1] + alpha * (points[i] - points[i - 1])
        return points[-1].copy()

    def _redistribute_altitude_constrained(self,
                                           points_2d: List[np.ndarray],
                                           z_start: float,
                                           z_goal: float) -> List[np.ndarray]:
        """
        约束内高度分配。

        在现有曲率感知算法基础上增加下降率平滑约束：
        相邻段下降率变化不超过 max_descent_rate_change。

        输入 2D 点列表，输出 3D 点列表。
        """
        n = len(points_2d)
        if n < 2:
            return [np.array([points_2d[0][0], points_2d[0][1], z_start])]

        delta_z = z_start - z_goal
        if abs(delta_z) < 1e-6:
            # 无高度差，全部设为 z_start
            return [np.array([pt[0], pt[1], z_start]) for pt in points_2d]

        R_min = self.min_turn_radius
        G_max = self.max_glide
        G_min = self.min_glide

        # 计算每段 2D 弧长
        ds_2d = np.zeros(n - 1)
        for i in range(n - 1):
            ds_2d[i] = np.linalg.norm(points_2d[i + 1] - points_2d[i])

        total_2d = np.sum(ds_2d)
        if total_2d < 1e-6:
            return [np.array([pt[0], pt[1], z_start]) for pt in points_2d]

        # 计算曲率（Dubins 路径的曲率现在是准确的）
        curvatures = np.zeros(n)
        for i in range(1, n - 1):
            curvatures[i] = self._compute_curvature_2d(
                points_2d[i - 1], points_2d[i], points_2d[i + 1])

        # 曲率感知下降分配 + FAF 加权
        curv_dz = np.zeros(n - 1)
        dz_floor = np.zeros(n - 1)
        dz_ceil = np.zeros(n - 1)

        cum_dist = np.zeros(n - 1)
        cum_sum = 0.0
        for i in range(n - 1):
            cum_sum += ds_2d[i]
            cum_dist[i] = cum_sum

        final_approach_threshold = 0.85
        final_approach_glide = 3.0

        for i in range(n - 1):
            kappa = 0.5 * (curvatures[i] + curvatures[min(i + 1, n - 1)])
            progress = cum_dist[i] / total_2d if total_2d > 1e-6 else 0.0

            f_sym = max(0.0, 1.0 - kappa * R_min)
            g_eff_min = G_max - f_sym * (G_max - G_min)

            if progress > final_approach_threshold:
                g_target = max(final_approach_glide, g_eff_min)
            else:
                g_target = 0.5 * (G_max + g_eff_min)

            if ds_2d[i] > 1e-6:
                curv_dz[i] = ds_2d[i] / g_target
                dz_floor[i] = ds_2d[i] / G_max
                dz_ceil[i] = ds_2d[i] / g_eff_min
            else:
                curv_dz[i] = 0.0
                dz_floor[i] = 0.0
                dz_ceil[i] = delta_z

        # 归一化
        total_curv = np.sum(curv_dz)
        if total_curv > 1e-9:
            curv_dz *= delta_z / total_curv

        # 线性下降量
        linear_dz = np.zeros(n - 1)
        for i in range(n - 1):
            linear_dz[i] = delta_z * (ds_2d[i] / total_2d)

        # 下降率平滑约束
        max_descent_rate_change = 0.05

        # 二分搜索最大 alpha
        alpha_lo, alpha_hi = 0.0, 1.0
        for _ in range(30):
            alpha = (alpha_lo + alpha_hi) * 0.5
            blended = alpha * curv_dz + (1.0 - alpha) * linear_dz
            valid = True

            for i in range(n - 1):
                if blended[i] < dz_floor[i] - 1e-6 or blended[i] > dz_ceil[i] + 1e-6:
                    valid = False
                    break

            # 下降率平滑检查
            if valid:
                for i in range(1, n - 1):
                    if ds_2d[i] > 1e-6 and ds_2d[i - 1] > 1e-6:
                        rate_i = blended[i] / ds_2d[i]
                        rate_prev = blended[i - 1] / ds_2d[i - 1]
                        if abs(rate_i - rate_prev) > max_descent_rate_change:
                            valid = False
                            break

            if valid:
                alpha_lo = alpha
            else:
                alpha_hi = alpha

        final_dz = alpha_lo * curv_dz + (1.0 - alpha_lo) * linear_dz
        total_final = np.sum(final_dz)
        if total_final > 1e-9:
            final_dz *= delta_z / total_final

        # 生成 3D 点
        result = []
        cum = 0.0
        for i in range(n):
            z = z_start - cum
            result.append(np.array([points_2d[i][0], points_2d[i][1], z]))
            if i < n - 1:
                cum += final_dz[i]
        result[-1][2] = z_goal

        if not self.quiet and alpha_lo < 0.99:
            print(f"    [高度重分配] 曲率混合系数 α={alpha_lo:.2f}"
                  f"（1.0=纯曲率, 受滑翔比/平滑约束回退）")

        return result

    def _compute_curvature_2d(self, p0: np.ndarray, p1: np.ndarray,
                              p2: np.ndarray) -> float:
        """计算三个 2D 点的 Menger 曲率"""
        v1 = p1 - p0
        v2 = p2 - p1

        d01 = np.linalg.norm(v1)
        d12 = np.linalg.norm(v2)
        d02 = np.linalg.norm(p2 - p0)

        if d01 < 1e-6 or d12 < 1e-6 or d02 < 1e-6:
            return 0.0

        cross = abs(v1[0] * v2[1] - v1[1] * v2[0])
        area = cross / 2
        return 4 * area / (d01 * d12 * d02)

    # ==================================================================
    #  Legacy 流水线 (原始 process() 主体)
    # ==================================================================

    def _process_legacy(self,
                        path: List[np.ndarray],
                        smooth: bool = True,
                        end_heading: Optional[float] = None) -> Trajectory:
        """原始三次样条平滑流水线（向后兼容）"""
        # 转换为numpy数组
        waypoints = [np.asarray(wp, dtype=np.float64) for wp in path]

        # Step 1: 去除重复点
        waypoints = self._remove_duplicates(waypoints)
        if len(waypoints) < 2:
            if not self.quiet:
                print("    [后处理] 去重后路径点太少")
            return Trajectory(dt=self.dt)

        if not self.quiet:
            print(f"    [后处理] 输入: {len(path)} 个路径点")

        # Step 2: 平滑处理
        if smooth:
            if not self.quiet:
                print(f"    [后处理] 执行三次样条平滑...")
            smooth_points = self._smooth_spline(waypoints)
            if not self.quiet:
                print(f"    [后处理] 平滑后: {len(smooth_points)} 个点")
        else:
            if not self.quiet:
                print(f"    [后处理] 跳过平滑，直接重采样...")
            smooth_points = self._resample_linear(waypoints)

        # Step 2.5: 曲率感知高度重分配
        smooth_points = self._redistribute_altitude(smooth_points)

        # Step 3: 时间参数化，生成轨迹
        trajectory = self._create_trajectory(smooth_points, end_heading)

        if not self.quiet:
            print(f"    [后处理] 生成轨迹: {len(trajectory)} 点, 时长 {trajectory.duration:.1f}s")

        # Step 4: 验证轨迹约束
        self._validate_trajectory(trajectory)

        return trajectory

    # ==================================================================
    #  共用方法
    # ==================================================================

    def _remove_duplicates(self, waypoints: List[np.ndarray],
                           threshold: float = 0.5) -> List[np.ndarray]:
        """去除距离过近的重复点"""
        if len(waypoints) <= 1:
            return waypoints

        filtered = [waypoints[0]]
        for wp in waypoints[1:]:
            dist = np.linalg.norm(wp - filtered[-1])
            if dist > threshold:
                filtered.append(wp)

        # 确保终点保留
        if len(waypoints) > 1:
            final_dist = np.linalg.norm(waypoints[-1] - filtered[-1])
            if final_dist > threshold * 0.1:
                filtered.append(waypoints[-1])

        return filtered

    def _smooth_spline(self, waypoints: List[np.ndarray]) -> List[np.ndarray]:
        """
        三次样条平滑

        使用弧长参数化的三次样条插值
        """
        # 计算累积弧长
        arc_lengths = [0.0]
        for i in range(1, len(waypoints)):
            dist = np.linalg.norm(waypoints[i] - waypoints[i-1])
            arc_lengths.append(arc_lengths[-1] + dist)

        total_length = arc_lengths[-1]
        if total_length < 1e-6:
            return waypoints

        # 归一化弧长参数
        s = np.array(arc_lengths) / total_length

        # 提取坐标
        xs = np.array([wp[0] for wp in waypoints])
        ys = np.array([wp[1] for wp in waypoints])
        zs = np.array([wp[2] for wp in waypoints])

        # 创建三次样条（自然边界条件）
        try:
            cs_x = CubicSpline(s, xs, bc_type='natural')
            cs_y = CubicSpline(s, ys, bc_type='natural')
            cs_z = CubicSpline(s, zs, bc_type='natural')
        except Exception as e:
            if not self.quiet:
                print(f"    [后处理] 样条插值失败: {e}，使用线性插值")
            return self._resample_linear(waypoints)

        # 计算采样点数（基于参考速度和控制频率）
        spacing = self.reference_speed * self.dt
        n_samples = max(int(total_length / spacing), len(waypoints))

        # 均匀采样
        smooth_points = []
        for i in range(n_samples + 1):
            t = min(i / n_samples, 1.0)
            x = float(cs_x(t))
            y = float(cs_y(t))
            z = float(cs_z(t))
            smooth_points.append(np.array([x, y, z]))

        return smooth_points

    def _resample_linear(self, waypoints: List[np.ndarray]) -> List[np.ndarray]:
        """线性重采样（不平滑）"""
        arc_lengths = [0.0]
        for i in range(1, len(waypoints)):
            dist = np.linalg.norm(waypoints[i] - waypoints[i-1])
            arc_lengths.append(arc_lengths[-1] + dist)

        total_length = arc_lengths[-1]
        if total_length < 1e-6:
            return waypoints

        spacing = self.reference_speed * self.dt
        n_samples = max(int(total_length / spacing), len(waypoints))

        resampled = []
        for i in range(n_samples + 1):
            target_s = (i / n_samples) * total_length
            pt = self._interpolate_at_arc_length(waypoints, arc_lengths, target_s)
            resampled.append(pt)

        return resampled

    def _interpolate_at_arc_length(self, waypoints: List[np.ndarray],
                                    arc_lengths: List[float],
                                    s: float) -> np.ndarray:
        """在指定弧长处线性插值"""
        for i in range(1, len(arc_lengths)):
            if arc_lengths[i] >= s:
                s0, s1 = arc_lengths[i-1], arc_lengths[i]
                if s1 - s0 < 1e-6:
                    alpha = 0.0
                else:
                    alpha = (s - s0) / (s1 - s0)

                p0, p1 = waypoints[i-1], waypoints[i]
                return p0 + alpha * (p1 - p0)

        return waypoints[-1].copy()

    def _create_trajectory(self,
                           points: List[np.ndarray],
                           end_heading: Optional[float] = None) -> Trajectory:
        """
        从平滑后的点创建 Trajectory 对象

        包含：时间戳、位置、速度、航向、曲率
        """
        trajectory_points = []
        t = 0.0
        n = len(points)

        for i in range(n):
            position = points[i].copy()

            # 计算航向角
            if i < n - 1:
                dx = points[i+1][0] - points[i][0]
                dy = points[i+1][1] - points[i][1]
                heading = np.arctan2(dy, dx)
            elif end_heading is not None:
                heading = end_heading
            else:
                if i > 0:
                    dx = points[i][0] - points[i-1][0]
                    dy = points[i][1] - points[i-1][1]
                    heading = np.arctan2(dy, dx)
                else:
                    heading = 0.0

            # 计算速度向量
            if i < n - 1:
                direction = points[i+1] - points[i]
                dist = np.linalg.norm(direction)
                if dist > 1e-6:
                    velocity = direction / dist * self.reference_speed
                else:
                    velocity = self.reference_speed * np.array([
                        np.cos(heading), np.sin(heading), 0.0
                    ])
            else:
                velocity = self.reference_speed * np.array([
                    np.cos(heading), np.sin(heading), 0.0
                ])

            # 计算曲率（三点法）
            curvature = 0.0
            if 0 < i < n - 1:
                curvature = self._compute_curvature(points[i-1], points[i], points[i+1])

            trajectory_points.append(TrajectoryPoint(
                t=t,
                position=position,
                velocity=velocity,
                heading=heading,
                curvature=curvature
            ))

            if i < n - 1:
                dist = np.linalg.norm(points[i+1] - points[i])
                t += dist / self.reference_speed

        return Trajectory(points=trajectory_points, dt=self.dt)

    def _compute_curvature(self, p0: np.ndarray, p1: np.ndarray,
                           p2: np.ndarray) -> float:
        """
        计算三点确定的曲率（Menger曲率公式）

        κ = 4 * Area / (|P0P1| * |P1P2| * |P0P2|)
        """
        v1 = p1[:2] - p0[:2]
        v2 = p2[:2] - p1[:2]

        d01 = np.linalg.norm(v1)
        d12 = np.linalg.norm(v2)
        d02 = np.linalg.norm(p2[:2] - p0[:2])

        if d01 < 1e-6 or d12 < 1e-6 or d02 < 1e-6:
            return 0.0

        cross = abs(v1[0] * v2[1] - v1[1] * v2[0])
        area = cross / 2

        curvature = 4 * area / (d01 * d12 * d02)
        return curvature

    def _redistribute_altitude(self, points: List[np.ndarray]) -> List[np.ndarray]:
        """
        曲率感知高度重分配（基于执行器预算耦合模型）— Legacy 版本
        """
        if len(points) < 3:
            return points

        z_start = points[0][2]
        z_goal = points[-1][2]
        delta_z = z_start - z_goal

        if abs(delta_z) < 1e-6:
            return points

        n = len(points)
        R_min = self.min_turn_radius
        G_max = self.max_glide
        G_min = self.min_glide

        ds_2d = np.zeros(n - 1)
        for i in range(n - 1):
            ds_2d[i] = np.linalg.norm(points[i + 1][:2] - points[i][:2])

        total_2d = np.sum(ds_2d)
        if total_2d < 1e-6:
            return points

        curvatures = np.zeros(n)
        for i in range(1, n - 1):
            curvatures[i] = self._compute_curvature(points[i - 1], points[i], points[i + 1])

        curv_dz = np.zeros(n - 1)
        dz_floor = np.zeros(n - 1)
        dz_ceil = np.zeros(n - 1)

        cum_dist = np.zeros(n - 1)
        cum_sum = 0.0
        for i in range(n - 1):
            cum_sum += ds_2d[i]
            cum_dist[i] = cum_sum

        final_approach_threshold = 0.85
        final_approach_glide = 3.0

        for i in range(n - 1):
            kappa = 0.5 * (curvatures[i] + curvatures[min(i + 1, n - 1)])
            progress = cum_dist[i] / total_2d if total_2d > 1e-6 else 0.0

            f_sym = max(0.0, 1.0 - kappa * R_min)
            g_eff_min = G_max - f_sym * (G_max - G_min)

            if progress > final_approach_threshold:
                g_target = max(final_approach_glide, g_eff_min)
            else:
                g_target = 0.5 * (G_max + g_eff_min)

            if ds_2d[i] > 1e-6:
                curv_dz[i] = ds_2d[i] / g_target
                dz_floor[i] = ds_2d[i] / G_max
                dz_ceil[i] = ds_2d[i] / g_eff_min
            else:
                curv_dz[i] = 0.0
                dz_floor[i] = 0.0
                dz_ceil[i] = delta_z

        total_curv = np.sum(curv_dz)
        if total_curv > 1e-9:
            curv_dz *= delta_z / total_curv

        linear_dz = np.zeros(n - 1)
        for i in range(n - 1):
            linear_dz[i] = delta_z * (ds_2d[i] / total_2d)

        alpha_lo, alpha_hi = 0.0, 1.0
        for _ in range(30):
            alpha = (alpha_lo + alpha_hi) * 0.5
            blended = alpha * curv_dz + (1.0 - alpha) * linear_dz
            valid = True
            for i in range(n - 1):
                if blended[i] < dz_floor[i] - 1e-6 or blended[i] > dz_ceil[i] + 1e-6:
                    valid = False
                    break
            if valid:
                alpha_lo = alpha
            else:
                alpha_hi = alpha

        final_dz = alpha_lo * curv_dz + (1.0 - alpha_lo) * linear_dz
        total_final = np.sum(final_dz)
        if total_final > 1e-9:
            final_dz *= delta_z / total_final

        result = []
        cum = 0.0
        for i in range(n):
            pt = points[i].copy()
            pt[2] = z_start - cum
            result.append(pt)
            if i < n - 1:
                cum += final_dz[i]
        result[-1][2] = z_goal

        if not self.quiet and alpha_lo < 0.99:
            print(f"    [高度重分配] 曲率混合系数 α={alpha_lo:.2f}"
                  f"（1.0=纯曲率, 受滑翔比约束回退）")

        return result

    def _validate_trajectory(self, trajectory: Trajectory):
        """验证轨迹是否满足约束"""
        if len(trajectory) < 2:
            return

        positions = trajectory.to_position_array()
        n = len(positions)

        glide_ratios = []
        turn_radii = []
        violations = 0

        window = 10
        for i in range(0, n - window, window):
            p1 = positions[i]
            p2 = positions[i + window]

            dz = p1[2] - p2[2]
            dxy = np.linalg.norm(p2[:2] - p1[:2])

            if dz > 1.0:
                glide = dxy / dz
                glide_ratios.append(glide)

        curvatures = trajectory.get_curvatures()
        for c in curvatures:
            if c > 1e-6:
                r = 1.0 / c
                turn_radii.append(r)
                if r < self.min_turn_radius:
                    violations += 1

        if not self.quiet:
            if glide_ratios:
                print(f"    [验证] 滑翔比: [{min(glide_ratios):.2f}, {max(glide_ratios):.2f}], 均值={np.mean(glide_ratios):.2f}")

            if turn_radii:
                min_r = min(turn_radii)
                print(f"    [验证] 最小转弯半径: {min_r:.1f}m (约束: {self.min_turn_radius:.1f}m)")
                if violations > 0:
                    print(f"    [警告] 有 {violations} 处转弯半径违反约束")


def validate_trajectory(trajectory: Trajectory,
                       min_glide_ratio: float = 2.47,
                       max_glide_ratio: float = 6.48,
                       min_turn_radius: float = 50.0) -> dict:
    """
    验证轨迹是否满足运动学约束

    参数:
        trajectory: 轨迹对象
        min_glide_ratio: 最小滑翔比
        max_glide_ratio: 最大滑翔比
        min_turn_radius: 最小转弯半径

    返回:
        验证结果字典
    """
    result = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'statistics': {}
    }

    if len(trajectory) < 2:
        result['warnings'].append("轨迹点数太少")
        return result

    positions = trajectory.to_position_array()
    n_points = len(positions)

    glide_ratios = []
    turn_radii = []

    window_size = 10

    for i in range(0, n_points - window_size, window_size):
        p1 = positions[i]
        p2 = positions[i + window_size]

        dz = p1[2] - p2[2]
        dxy = np.linalg.norm(p2[:2] - p1[:2])

        if dz > 1.0:
            glide = dxy / dz
            glide_ratios.append(glide)

            if glide > max_glide_ratio:
                result['errors'].append(
                    f"段{i//window_size}: 滑翔比{glide:.2f} > 最大{max_glide_ratio:.2f}")
                result['valid'] = False
            elif glide < min_glide_ratio:
                result['warnings'].append(
                    f"段{i//window_size}: 滑翔比{glide:.2f} < 最小{min_glide_ratio:.2f}")

    # 曲率检查
    for i in range(len(trajectory)):
        pt = trajectory[i]
        if pt.curvature > 1e-6:
            radius = 1.0 / pt.curvature
            turn_radii.append(radius)

            if radius < min_turn_radius:
                result['warnings'].append(
                    f"点{i}: 转弯半径{radius:.1f}m < 最小{min_turn_radius:.1f}m")

    # 统计信息
    if glide_ratios:
        result['statistics']['glide_ratio'] = {
            'min': min(glide_ratios),
            'max': max(glide_ratios),
            'mean': np.mean(glide_ratios)
        }

    if turn_radii:
        result['statistics']['turn_radius'] = {
            'min': min(turn_radii),
            'max': max(turn_radii),
            'mean': np.mean(turn_radii)
        }

    return result


# ============================================================
#                     测试
# ============================================================

if __name__ == "__main__":
    from planning.kinodynamic_rrt import KinodynamicRRTStar
    from planning.map_manager import MapManager
    import matplotlib.pyplot as plt

    print("=" * 60)
    print("  轨迹后处理模块测试")
    print("=" * 60)

    # 加载地图
    print("\n[1/3] 加载地图...")
    map_mgr = MapManager.from_yaml("cfg/map_config.yaml")

    # 规划
    print("\n[2/3] 运行 Kinodynamic RRT*...")
    planner = KinodynamicRRTStar(map_mgr)
    path, info = planner.plan(max_time=30.0)

    if path is None:
        print("规划失败!")
        exit(1)

    print(f"    规划成功: {len(path)} 个路径点")

    # 后处理
    print("\n[3/3] 轨迹后处理...")
    postprocessor = TrajectoryPostprocessor(
        reference_speed=12.0,
        control_frequency=100.0,
        min_turn_radius=map_mgr.constraints.min_turn_radius,
        max_glide_ratio=map_mgr.constraints.glide_ratio,
        min_glide_ratio=map_mgr.constraints.min_glide_ratio,
        map_manager=map_mgr
    )

    end_heading = map_mgr.target.approach_heading if map_mgr.target else None
    edge_paths = info.get('edge_paths', None)
    node_chain = info.get('node_chain', None)
    trajectory = postprocessor.process(
        path, smooth=True, end_heading=end_heading,
        edge_paths=edge_paths, node_chain=node_chain
    )

    # 验证
    print("\n[验证结果]")
    result = validate_trajectory(
        trajectory,
        min_glide_ratio=map_mgr.constraints.min_glide_ratio,
        max_glide_ratio=map_mgr.constraints.glide_ratio,
        min_turn_radius=map_mgr.constraints.min_turn_radius
    )
    print(f"  有效: {result['valid']}")
    if result['errors']:
        print(f"  错误: {result['errors'][:3]}...")
    if result['warnings']:
        print(f"  警告数: {len(result['warnings'])}")

    # 可视化
    print("\n[可视化]")
    path_arr = np.array(path)
    traj_arr = trajectory.to_position_array()

    fig = plt.figure(figsize=(15, 5))

    # 3D对比
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot(path_arr[:, 0], path_arr[:, 1], path_arr[:, 2],
             'r.--', markersize=4, linewidth=1, alpha=0.7, label='RRT* 原始')
    ax1.plot(traj_arr[:, 0], traj_arr[:, 1], traj_arr[:, 2],
             'b-', linewidth=1.5, alpha=0.9, label='平滑后')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.legend()
    ax1.set_title('3D 轨迹对比')

    # XY平面
    ax2 = fig.add_subplot(132)
    ax2.plot(path_arr[:, 0], path_arr[:, 1], 'r.--',
             markersize=6, linewidth=1, alpha=0.7, label='RRT* 原始')
    ax2.plot(traj_arr[:, 0], traj_arr[:, 1], 'b-',
             linewidth=2, alpha=0.9, label='平滑后')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.legend()
    ax2.axis('equal')
    ax2.grid(True, alpha=0.3)
    ax2.set_title('XY 平面')

    # 高度剖面
    ax3 = fig.add_subplot(133)
    timestamps = trajectory.get_timestamps()
    ax3.plot(timestamps, traj_arr[:, 2], 'b-', linewidth=2)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Altitude (m)')
    ax3.grid(True, alpha=0.3)
    ax3.set_title('高度剖面')

    plt.tight_layout()
    plt.show()

    print("\n测试完成!")
