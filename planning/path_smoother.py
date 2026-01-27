"""
路径平滑模块

将 RRT* 生成的离散路径点平滑为连续轨迹:
- 三次样条全局平滑 (XY, Z)
- 时间参数化 (100Hz 采样)
- 输出包含时间、位置、速度、曲率的完整轨迹
"""

import os
import sys
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from scipy.interpolate import CubicSpline
import math

# 添加项目根目录到路径
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from planning.trajectory import Trajectory, TrajectoryPoint


class PathSmoother:
    """路径平滑器 - 三次样条全局平滑"""

    def __init__(self, turn_radius: float = 50.0,
                 reference_speed: float = 12.0,
                 control_frequency: float = 100.0):
        """
        参数:
            turn_radius: 最小转弯半径 (m)
            reference_speed: 参考飞行速度 (m/s)
            control_frequency: 控制频率 (Hz)
        """
        self.turn_radius = turn_radius
        self.reference_speed = reference_speed
        self.control_frequency = control_frequency
        self.dt = 1.0 / control_frequency  # 采样周期 (s)

    def smooth(self, waypoints: List[np.ndarray],
               start_heading: Optional[float] = None,
               end_heading: Optional[float] = None,
               waypoint_density: int = 3,
               map_manager=None) -> Trajectory:
        """
        平滑路径并生成带时间信息的轨迹

        参数:
            waypoints: 路径点列表 [[x, y, z], ...]
            start_heading: 起点航向 (rad)，None 则自动计算
            end_heading: 终点航向 (rad)，None 则自动计算
            waypoint_density: 航点加密倍数 (在原始航点间插入的点数+1)
            map_manager: 地图管理器（可选），如果提供则添加进场点、盘旋消高和进场直线段

        返回:
            Trajectory 对象，包含完整的时间、位置、速度、曲率信息
        """
        if len(waypoints) < 2:
            # 单点或空，返回空轨迹
            return Trajectory(dt=self.dt)

        waypoints = [np.array(wp) for wp in waypoints]
        
        # 如果提供了map_manager，移除最后的目标点（如果高度低于进场点），只保留到进场点
        if map_manager is not None and map_manager.target is not None:
            approach_point = map_manager.target.get_approach_point(
                map_manager.constraints.terminal_altitude
            )
            target_pos = map_manager.target.position
            approach_altitude = approach_point[2]
            
            # 如果最后一点是目标点（高度0m），且高度低于进场点，移除它
            if len(waypoints) > 0:
                last_point = waypoints[-1]
                # 检查最后一点是否是目标点（高度接近0m，且位置接近目标）
                if (abs(last_point[2] - target_pos[2]) < 5.0 and 
                    np.linalg.norm(last_point[:2] - target_pos[:2]) < 50.0 and
                    last_point[2] < approach_altitude - 5.0):
                    # 移除最后的目标点
                    waypoints = waypoints[:-1]
                    print(f"    [路径优化] 移除路径最后的目标点（高度{last_point[2]:.1f}m），保留到进场点（高度{approach_altitude:.1f}m）")
            
            # 调整最后几个路径点的高度，平滑过渡到进场点高度
            # 最后30%的路径点逐渐过渡到进场点高度
            n_points = len(waypoints)
            if n_points > 1:
                transition_start_idx = max(1, int(n_points * 0.7))  # 最后30%的点
                
                for i in range(transition_start_idx, n_points):
                    alpha = (i - transition_start_idx) / max(1, n_points - transition_start_idx - 1)
                    original_z = waypoints[i][2]
                    target_z = approach_altitude
                    # 如果原始高度高于目标高度，平滑下降到目标高度
                    # 如果原始高度低于目标高度，保持（翼伞不能抬升）
                    if original_z > target_z:
                        waypoints[i][2] = original_z - alpha * (original_z - target_z)
                    # 如果原始高度低于目标高度但接近，可以稍微调整（但不能超过物理限制）
                    elif original_z < target_z - 5.0:  # 如果高度差大于5m，说明有问题
                        # 保持原始高度，但记录警告
                        pass
                
                # 确保最后一点的高度就是进场点高度（如果可能）
                if waypoints[-1][2] >= approach_altitude - 5.0:  # 如果高度差小于5m
                    waypoints[-1][2] = approach_altitude
                
                # 调试：打印调整后的航点高度范围
                if len(waypoints) > 0:
                    wp_z_values = [wp[2] for wp in waypoints]
                    print(f"    [路径优化] 航点高度调整后: 最小={min(wp_z_values):.1f}m, 最大={max(wp_z_values):.1f}m, 最后={waypoints[-1][2]:.1f}m")

        # Step 0: 去除重复或极近的航点（避免弧长参数化出现重复值）
        waypoints = self._remove_duplicate_waypoints(waypoints, min_dist=1e-6)
        if len(waypoints) < 2:
            return Trajectory(dt=self.dt)

        # Step 1: 增加航点密度
        dense_waypoints = self._densify_waypoints(waypoints, waypoint_density)

        # Step 2: 计算累积弧长参数
        arc_lengths = self._compute_arc_lengths(dense_waypoints)
        total_length = arc_lengths[-1]

        if total_length < 1e-6:
            return Trajectory(dt=self.dt)

        # Step 3: 三次样条插值 (参数化为弧长)
        s_normalized = arc_lengths / total_length  # 归一化到 [0, 1]

        # 提取坐标
        xs = np.array([wp[0] for wp in dense_waypoints])
        ys = np.array([wp[1] for wp in dense_waypoints])
        zs = np.array([wp[2] for wp in dense_waypoints])

        # 创建三次样条 (自然边界条件)
        cs_x = CubicSpline(s_normalized, xs, bc_type='natural')
        cs_y = CubicSpline(s_normalized, ys, bc_type='natural')
        cs_z = CubicSpline(s_normalized, zs, bc_type='natural')

        # Step 4: 时间参数化
        # 根据参考速度计算总时间
        total_time = total_length / self.reference_speed
        n_samples = max(int(total_time / self.dt), 2)

        # Step 5: 生成轨迹点
        trajectory_points = []
        
        # 如果提供了map_manager，确保基础轨迹最后高度不低于进场点高度
        min_end_altitude = None
        if map_manager is not None and map_manager.target is not None:
            min_end_altitude = map_manager.target.get_approach_point(
                map_manager.constraints.terminal_altitude
            )[2]

        for i in range(n_samples + 1):
            t = i * self.dt
            # 弧长参数 (归一化)
            s = min(t / total_time, 1.0)

            # 位置
            x = float(cs_x(s))
            y = float(cs_y(s))
            z = float(cs_z(s))
            
            # 如果提供了最小高度约束，确保高度不低于最小高度
            if min_end_altitude is not None:
                # 在整个轨迹中，如果高度低于进场高度，强制提升到进场高度（防止样条插值产生低高度点）
                if z < min_end_altitude - 0.1:  # 如果高度低于进场高度超过0.1m
                    # 找到最后一个高度>=进场高度的点，从那里开始截断
                    # 但这里我们直接限制高度，不允许低于进场高度
                    z = min_end_altitude
                
                # 如果接近轨迹末端，平滑过渡到进场高度
                if s > 0.9:  # 最后10%的轨迹
                    transition_alpha = (s - 0.9) / 0.1  # 0到1的过渡
                    # 确保高度不低于最小高度，平滑过渡（更严格：只允许0.5m的过渡）
                    z = max(z, min_end_altitude - (1 - transition_alpha) * 0.5)  # 只允许最后0.5m的过渡
            
            position = np.array([x, y, z])

            # 一阶导数 (用于计算速度和航向)
            dx_ds = float(cs_x(s, 1))
            dy_ds = float(cs_y(s, 1))
            dz_ds = float(cs_z(s, 1))

            # 速度 (沿轨迹方向，大小为参考速度)
            ds_dt = 1.0 / total_time  # ds/dt
            tangent = np.array([dx_ds, dy_ds, dz_ds])
            tangent_norm = np.linalg.norm(tangent)
            if tangent_norm > 1e-6:
                velocity = tangent / tangent_norm * self.reference_speed
            else:
                velocity = np.array([self.reference_speed, 0.0, 0.0])

            # 航向角 (XY平面)
            heading = np.arctan2(dy_ds, dx_ds)

            # 二阶导数 (用于计算曲率)
            d2x_ds2 = float(cs_x(s, 2))
            d2y_ds2 = float(cs_y(s, 2))

            # 曲率计算 (XY平面)
            # kappa = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
            cross = abs(dx_ds * d2y_ds2 - dy_ds * d2x_ds2)
            denom = (dx_ds**2 + dy_ds**2) ** 1.5
            if denom > 1e-10:
                curvature = cross / denom / total_length  # 考虑弧长归一化
            else:
                curvature = 0.0

            trajectory_points.append(TrajectoryPoint(
                t=t,
                position=position,
                velocity=velocity,
                heading=heading,
                curvature=curvature
            ))

        # Step 6: 应用起点和终点航向约束（如果指定）
        if start_heading is not None and len(trajectory_points) > 0:
            trajectory_points[0].heading = start_heading
            # 调整起始速度方向
            speed = np.linalg.norm(trajectory_points[0].velocity)
            trajectory_points[0].velocity = speed * np.array([
                np.cos(start_heading), np.sin(start_heading),
                trajectory_points[0].velocity[2] / speed if speed > 1e-6 else 0
            ])

        if end_heading is not None and len(trajectory_points) > 0:
            trajectory_points[-1].heading = end_heading
            # 调整终点速度方向
            speed = np.linalg.norm(trajectory_points[-1].velocity)
            trajectory_points[-1].velocity = speed * np.array([
                np.cos(end_heading), np.sin(end_heading),
                trajectory_points[-1].velocity[2] / speed if speed > 1e-6 else 0
            ])
        
        base_trajectory = Trajectory(points=trajectory_points, dt=self.dt)
        
        # 如果提供了map_manager，先截断低于进场高度的点，并平滑航向角
        if map_manager is not None and map_manager.target is not None:
            approach_altitude = map_manager.target.get_approach_point(
                map_manager.constraints.terminal_altitude
            )[2]
            
            # 1. 截断低于进场高度的点（在样条插值后）
            points = list(base_trajectory.points)
            
            # 调试：打印样条插值后的高度范围
            if len(points) > 0:
                z_values = [p.position[2] for p in points]
                min_z = min(z_values)
                max_z = max(z_values)
                last_z = points[-1].position[2]
                print(f"    [路径优化] 样条插值后高度范围: {min_z:.1f}m ~ {max_z:.1f}m, 最后高度: {last_z:.1f}m")
            
            transition_start_idx = len(points) - 1
            
            # 从后往前找，找到最后一个高度>=进场高度的点（更严格的容差）
            for i in range(len(points) - 1, -1, -1):
                if points[i].position[2] >= approach_altitude - 0.1:  # 只允许0.1m的容差
                    transition_start_idx = i
                    break
            
            # 如果找到了合适的过渡起点，截断基础轨迹
            if transition_start_idx < len(points) - 1:
                print(f"    [路径优化] 样条插值后找到过渡起点: 第{transition_start_idx+1}个点 (高度{points[transition_start_idx].position[2]:.1f}m)")
                print(f"    [路径优化] 移除{len(points) - transition_start_idx - 1}个低于进场高度的样条点")
                points = points[:transition_start_idx + 1]
                base_trajectory = Trajectory(points=points, dt=self.dt)
            else:
                # 检查最后一点的高度
                if len(points) > 0 and points[-1].position[2] < approach_altitude - 0.1:
                    print(f"    [警告] 样条插值后最后高度({points[-1].position[2]:.1f}m)低于进场高度({approach_altitude:.1f}m)，强制截断")
                    # 强制找到最后一个高度>=进场高度的点
                    for i in range(len(points) - 1, -1, -1):
                        if points[i].position[2] >= approach_altitude - 0.1:
                            transition_start_idx = i
                            points = points[:transition_start_idx + 1]
                            base_trajectory = Trajectory(points=points, dt=self.dt)
                            print(f"    [路径优化] 强制截断到第{transition_start_idx+1}个点 (高度{points[-1].position[2]:.1f}m)")
                            break
                else:
                    print(f"    [路径优化] 样条插值后所有点高度都>=进场高度，无需截断")
            
            # 2. 平滑样条插值后的航向角（在添加盘旋前）
            smoothed_points = self._smooth_heading_changes(
                list(base_trajectory.points), 
                max_heading_rate=np.radians(20)  # 更严格的限制：20度/秒
            )
            base_trajectory = Trajectory(points=smoothed_points, dt=self.dt)
        
        # 如果提供了map_manager，添加进场点、盘旋消高和进场直线段
        if map_manager is not None:
            return self.add_approach_and_loiter(base_trajectory, map_manager)
        
        return base_trajectory

    def add_approach_and_loiter(self, 
                                 trajectory: Trajectory,
                                 map_manager) -> Trajectory:
        """
        添加进场点、盘旋消高和进场直线段到轨迹
        
        关键改进：
        1. 盘旋高度从基础轨迹最后高度平滑过渡到进场点高度（避免突然抬升）
        2. 添加平滑的航向角过渡（避免欧拉角突变）
        3. 添加从基础轨迹到盘旋的过渡圆弧（避免尖点）
        4. 盘旋应该在基础轨迹结束前开始，高度平滑下降
        
        参数:
            trajectory: 基础轨迹
            map_manager: 地图管理器，包含目标和约束信息
        
        返回:
            优化后的轨迹
        """
        if map_manager is None or map_manager.target is None:
            return trajectory
        
        target = map_manager.target
        constraints = map_manager.constraints
        approach_heading = target.approach_heading if target else None
        approach_point = target.get_approach_point(constraints.terminal_altitude) if target else None
        target_pos = target.position.copy() if target else None
        
        if approach_point is None or approach_heading is None or len(trajectory.points) == 0:
            return trajectory
        
        # 复制轨迹点
        points = list(trajectory.points)
        dt = self.dt
        speed = self.reference_speed
        
        # 注意：基础轨迹已经在smooth方法中截断了低于进场高度的点
        # 这里直接使用基础轨迹的最后状态
        approach_altitude = approach_point[2]

        # Avoid climb before loiter; lower approach altitude if needed.
        approach_point = approach_point.copy()
        
        # 获取基础轨迹的最后状态
        last_point = points[-1]
        last_pos = last_point.position
        last_heading = last_point.heading
        t = last_point.t if points else 0.0

        # 关键：盘旋高度不能高于基础轨迹的最后高度（翼伞不能抬升）
        loiter_altitude = min(approach_altitude, last_pos[2])
        if loiter_altitude < approach_altitude - 1e-3:
            approach_point[2] = loiter_altitude
            print(f"    [路径优化] 调整盘旋高度: {loiter_altitude:.1f}m (基础轨迹最后高度，低于进场高度{approach_altitude:.1f}m)")

        
        # 关键检查：确保基础轨迹最后高度不低于进场高度（允许0.1m容差）
        if last_pos[2] < approach_altitude - 0.1:
            print(f"    [警告] 基础轨迹最后高度({last_pos[2]:.1f}m)低于进场高度({approach_altitude:.1f}m)，这是不应该发生的！")
            # 强制调整最后一点的高度到进场高度（但这是最后手段，不应该发生）
            # 实际上应该在前面的截断逻辑中处理
        
        print(f"    [路径优化] 基础轨迹最后高度: {last_pos[2]:.1f}m, 进场点高度: {approach_altitude:.1f}m")
        
        # 1. 计算是否需要盘旋消高
        extra_distance = 0.0
        if target_pos is not None and map_manager.start is not None:
            start_z = map_manager.start.z
            target_z = target_pos[2]
            required_horizontal = max(
                0.0,
                (start_z - target_z - constraints.altitude_margin) * constraints.glide_ratio
            )
            path_length_xy = 0.0
            for i in range(len(points) - 1):
                p1, p2 = points[i].position, points[i + 1].position
                path_length_xy += np.linalg.norm(p2[:2] - p1[:2])
            extra_distance = max(0.0, required_horizontal - path_length_xy)
        
        # 2. 计算盘旋参数（如果需要）
        loiter_loops = 0
        loiter_radius = None
        loiter_direction = None  # "ccw" or "cw"
        loiter_center = None
        
        if extra_distance > 1e-3:
            base_radius = constraints.min_turn_radius
            radius_candidates = [1.5 * base_radius, 2.0 * base_radius, 2.5 * base_radius]
            
            def circle_ok(center: np.ndarray, radius: float, z: float) -> bool:
                for theta in np.linspace(0, 2 * np.pi, 36, endpoint=False):
                    pt = center + np.array([radius * np.cos(theta), radius * np.sin(theta), 0.0])
                    pt[2] = z
                    if map_manager.is_collision(pt):
                        return False
                return True
            
            for radius in radius_candidates:
                # CCW: tangent aligns with approach_heading at approach_point
                theta_ccw = approach_heading - np.pi / 2.0
                center_ccw = approach_point.copy()
                center_ccw[:2] -= radius * np.array([np.cos(theta_ccw), np.sin(theta_ccw)])
                if circle_ok(center_ccw, radius, approach_point[2]):
                    loiter_radius = radius
                    loiter_center = center_ccw
                    loiter_direction = "ccw"
                    break
                
                # CW: tangent aligns with approach_heading at approach_point
                theta_cw = approach_heading + np.pi / 2.0
                center_cw = approach_point.copy()
                center_cw[:2] -= radius * np.array([np.cos(theta_cw), np.sin(theta_cw)])
                if circle_ok(center_cw, radius, approach_point[2]):
                    loiter_radius = radius
                    loiter_center = center_cw
                    loiter_direction = "cw"
                    break
            
            if loiter_radius is not None:
                loiter_loops = int(np.ceil(extra_distance / (2 * np.pi * loiter_radius)))
        
        # 3. 添加盘旋消高（如果需要）
        if extra_distance > 1e-3:
            print(f"    [路径优化] 需要额外距离: {extra_distance:.1f}m")
        
        if loiter_loops > 0 and loiter_center is not None:
            print(f"    [路径优化] 添加盘旋消高: {loiter_loops}圈, 半径={loiter_radius:.1f}m, 方向={loiter_direction}")
            print(f"    [路径优化] 基础轨迹最后高度: {last_pos[2]:.1f}m, 进场点高度: {approach_point[2]:.1f}m")
            print(f"    [路径优化] 基础轨迹最后航向: {np.degrees(last_heading):.1f}°, 进场航向: {np.degrees(approach_heading):.1f}°")
            total_angle = 2 * np.pi * loiter_loops
            arc_len = loiter_radius * total_angle
            n_samples = max(int(arc_len / (speed * dt)), 1)
            
            # 计算盘旋的起始和结束角度
            if loiter_direction == "ccw":
                theta_end = approach_heading - np.pi / 2.0
                theta_start = theta_end - total_angle
                tangent_sign = 1.0
            else:
                theta_end = approach_heading + np.pi / 2.0
                theta_start = theta_end + total_angle
                tangent_sign = -1.0
            
            # 计算盘旋起始点的位置和航向
            loiter_start_pos = loiter_center + np.array([
                loiter_radius * np.cos(theta_start),
                loiter_radius * np.sin(theta_start),
                loiter_altitude
            ])
            loiter_start_heading = theta_start + tangent_sign * np.pi / 2.0
            loiter_start_heading = self._wrap_angle(loiter_start_heading)
            
            # 3.1 添加从基础轨迹到盘旋起始点的平滑过渡
            # 关键：限制航向角变化率，避免欧拉角突变
            dist_to_loiter_start = np.linalg.norm(last_pos[:2] - loiter_start_pos[:2])
            
            # 计算航向角变化
            heading_to_loiter = np.arctan2(
                loiter_start_pos[1] - last_pos[1],
                loiter_start_pos[0] - last_pos[0]
            )
            heading_change = self._wrap_angle(heading_to_loiter - last_heading)
            
            # 限制航向角变化率：最大角速度（rad/s）- 更严格的限制
            max_heading_rate = np.radians(15)  # 最大15度/秒（更保守）
            min_transition_time = abs(heading_change) / max_heading_rate if abs(heading_change) > 1e-6 else 0.0
            min_transition_distance = speed * min_transition_time
            
            # 如果距离不够，延长过渡距离
            transition_distance = max(dist_to_loiter_start, min_transition_distance)
            transition_time = transition_distance / speed
            n_transition = max(int(transition_time / dt), 1)
            
            print(f"    [路径优化] 添加平滑过渡: {n_transition}个点, 航向变化={np.degrees(heading_change):.1f}°, 距离={dist_to_loiter_start:.1f}m, 最大角速度={np.degrees(max_heading_rate):.1f}°/s")
            
            # 添加平滑过渡段
            for i in range(1, n_transition + 1):
                alpha = i / n_transition
                
                # 水平位置平滑过渡到盘旋起始点
                pos_xy = last_pos[:2] + alpha * (loiter_start_pos[:2] - last_pos[:2])
                
                # 高度：关键修复 - 翼伞不能抬升，只能下降或保持
                # 如果当前高度高于进场点，平滑下降到进场点高度
                # 如果当前高度低于或等于进场点，保持当前高度（不能抬升）
                if last_pos[2] > loiter_altitude + 0.1:  # 如果明显高于进场高度
                    z = last_pos[2] + alpha * (loiter_altitude - last_pos[2])  # 平滑下降
                else:
                    # 不能抬升！保持当前高度（或稍微下降，但不能低于最低安全高度）
                    z = last_pos[2]  # 保持基础轨迹的最后高度，不抬升
                    # 如果基础轨迹最后高度太低，说明有问题，但也不能抬升
                    if z < loiter_altitude - 10.0:  # 如果高度差太大，说明基础轨迹有问题
                        print(f"    [警告] 基础轨迹最后高度({z:.1f}m)远低于进场高度({approach_altitude:.1f}m)，但翼伞不能抬升！")
                
                pos = np.array([pos_xy[0], pos_xy[1], z])
                
                # 航向角平滑过渡（限制变化率）
                heading = last_heading + alpha * heading_change
                heading = self._wrap_angle(heading)
                
                vel = speed * np.array([np.cos(heading), np.sin(heading), 0.0])
                
                # 曲率：过渡段逐渐增加曲率
                curvature = alpha / max(loiter_radius, constraints.min_turn_radius)
                
                t += dt
                points.append(TrajectoryPoint(
                    t=t,
                    position=pos,
                    velocity=vel,
                    heading=heading,
                    curvature=curvature
                ))
            
            # 3.2 添加盘旋轨迹点（高度保持在进场点高度）
            print(f"    [路径优化] 添加盘旋轨迹: {n_samples}个点, 总角度={np.degrees(total_angle):.1f}°")
            thetas = np.linspace(theta_start, theta_end, n_samples + 1)
            
            # 获取过渡后的最后航向（用于平滑连接）
            last_heading_before_loiter = points[-1].heading if points else last_heading
            
            # 计算第一个盘旋点的期望航向
            first_loiter_theta = thetas[1] if len(thetas) > 1 else theta_start
            first_loiter_heading = first_loiter_theta + tangent_sign * np.pi / 2.0
            first_loiter_heading = self._wrap_angle(first_loiter_heading)
            
            # 计算航向角变化
            heading_change_to_loiter = self._wrap_angle(first_loiter_heading - last_heading_before_loiter)
            
            # 限制航向角变化率：每步最多变化的角度
            max_heading_change_per_step = np.radians(1.5)  # 每步最多1.5度（非常保守）
            
            # 如果航向角变化太大，添加额外的平滑过渡点
            if abs(heading_change_to_loiter) > max_heading_change_per_step:
                n_smooth_steps = int(abs(heading_change_to_loiter) / max_heading_change_per_step) + 1
                print(f"    [路径优化] 航向角变化较大({np.degrees(heading_change_to_loiter):.1f}°)，添加{n_smooth_steps}个平滑过渡点")
                
                for j in range(1, n_smooth_steps + 1):
                    alpha_smooth = j / n_smooth_steps
                    # 航向角平滑过渡
                    smooth_heading = last_heading_before_loiter + alpha_smooth * heading_change_to_loiter
                    smooth_heading = self._wrap_angle(smooth_heading)
                    
                    # 位置：从过渡终点平滑过渡到第一个盘旋点
                    # 关键修复：使用loiter_altitude而不是approach_altitude，避免抬升
                    smooth_theta = theta_start + alpha_smooth * (first_loiter_theta - theta_start)
                    smooth_pos = loiter_center + np.array([
                        loiter_radius * np.cos(smooth_theta),
                        loiter_radius * np.sin(smooth_theta),
                        loiter_altitude  # 使用loiter_altitude，不能抬升
                    ])
                    
                    vel = speed * np.array([np.cos(smooth_heading), np.sin(smooth_heading), 0.0])
                    curvature = 1.0 / loiter_radius if loiter_radius > 1e-6 else 0.0
                    t += dt
                    points.append(TrajectoryPoint(
                        t=t,
                        position=smooth_pos,
                        velocity=vel,
                        heading=smooth_heading,
                        curvature=curvature
                    ))
            
            # 添加剩余的盘旋轨迹点（高度恒定在loiter_altitude，不能抬升）
            start_idx = 1 if abs(heading_change_to_loiter) <= max_heading_change_per_step else 2
            for theta in thetas[start_idx:]:
                pos = loiter_center + np.array([
                    loiter_radius * np.cos(theta),
                    loiter_radius * np.sin(theta),
                    loiter_altitude  # 关键：使用loiter_altitude，不能抬升
                ])
                
                heading = theta + tangent_sign * np.pi / 2.0
                heading = self._wrap_angle(heading)
                vel = speed * np.array([np.cos(heading), np.sin(heading), 0.0])
                curvature = 1.0 / loiter_radius if loiter_radius > 1e-6 else 0.0
                t += dt
                points.append(TrajectoryPoint(
                    t=t,
                    position=pos,
                    velocity=vel,
                    heading=heading,
                    curvature=curvature
                ))
        
        # 4. 添加从最后位置到进场点的过渡（如果没有盘旋）
        if loiter_loops == 0:
            last_pos = points[-1].position
            if np.linalg.norm(last_pos[:2] - approach_point[:2]) > 1e-3:
                dist = np.linalg.norm(approach_point - last_pos)
                n_transition = max(int(dist / (speed * dt)), 1)
                for i in range(1, n_transition + 1):
                    alpha = i / n_transition
                    pos = last_pos + alpha * (approach_point - last_pos)
                    # 航向角平滑过渡
                    heading_to_approach = np.arctan2(
                        approach_point[1] - last_pos[1], 
                        approach_point[0] - last_pos[0]
                    )
                    heading = last_point.heading + alpha * self._wrap_angle(heading_to_approach - last_point.heading)
                    heading = self._wrap_angle(heading)
                    vel = speed * np.array([np.cos(heading), np.sin(heading), 0.0])
                    t += dt
                    points.append(TrajectoryPoint(
                        t=t,
                        position=pos,
                        velocity=vel,
                        heading=heading,
                        curvature=0.0
                    ))
        
        # 5. 添加进场直线段（从进场点开始）
        if target.approach_length > 1e-3:
            print(f"    [路径优化] 添加进场直线段: 长度={target.approach_length:.1f}m")
            approach_length = target.approach_length
            n_line = max(int(approach_length / (speed * dt)), 1)
            dir_xy = np.array([np.cos(approach_heading), np.sin(approach_heading)])
            z0 = loiter_altitude  # 从进场点高度开始
            z1 = target_pos[2]
            
            # 获取当前最后一点的状态
            last_point_before_approach = points[-1]
            last_pos_before_approach = last_point_before_approach.position
            last_heading_before_approach = last_point_before_approach.heading
            
            # 确保最后一点在进场点（如果没有盘旋，可能已经到达）
            if np.linalg.norm(last_pos_before_approach[:2] - approach_point[:2]) > 1e-3:
                # 添加平滑过渡到进场点
                dist_to_approach = np.linalg.norm(last_pos_before_approach[:2] - approach_point[:2])
                heading_to_approach = np.arctan2(
                    approach_point[1] - last_pos_before_approach[1],
                    approach_point[0] - last_pos_before_approach[0]
                )
                heading_change_to_approach = self._wrap_angle(heading_to_approach - last_heading_before_approach)
                
                # 限制航向角变化率（更严格）
                max_heading_rate = np.radians(15)  # 最大15度/秒
                min_transition_time = abs(heading_change_to_approach) / max_heading_rate if abs(heading_change_to_approach) > 1e-6 else 0.0
                transition_time = max(dist_to_approach / speed, min_transition_time)
                n_transition_to_approach = max(int(transition_time / dt), 1)
                
                for i in range(1, n_transition_to_approach + 1):
                    alpha = i / n_transition_to_approach
                    pos_xy = last_pos_before_approach[:2] + alpha * (approach_point[:2] - last_pos_before_approach[:2])
                    # 关键修复：使用loiter_altitude而不是approach_altitude，避免抬升
                    z = last_pos_before_approach[2] + alpha * (loiter_altitude - last_pos_before_approach[2])
                    pos = np.array([pos_xy[0], pos_xy[1], z])
                    
                    # 航向角平滑过渡
                    heading = last_heading_before_approach + alpha * heading_change_to_approach
                    heading = self._wrap_angle(heading)
                    vel = speed * np.array([np.cos(heading), np.sin(heading), 0.0])
                    curvature = 0.0
                    t += dt
                    points.append(TrajectoryPoint(
                        t=t,
                        position=pos,
                        velocity=vel,
                        heading=heading,
                        curvature=curvature
                    ))
            
            # 添加进场直线段（航向角平滑过渡到进场航向）
            last_heading_before_line = points[-1].heading
            heading_change_to_line = self._wrap_angle(approach_heading - last_heading_before_line)
            
            for i in range(1, n_line + 1):
                s = approach_length * i / n_line
                pos = approach_point.copy()
                pos[:2] += dir_xy * s
                if approach_length > 1e-6:
                    pos[2] = z0 + (z1 - z0) * (s / approach_length)
                
                # 航向角平滑过渡到进场航向（前10%的线段）
                if i <= max(1, int(n_line * 0.1)):
                    alpha_heading = i / max(1, int(n_line * 0.1))
                    heading = last_heading_before_line + alpha_heading * heading_change_to_line
                    heading = self._wrap_angle(heading)
                else:
                    heading = approach_heading
                
                vel = speed * np.array([np.cos(heading), np.sin(heading), 0.0])
                curvature = 0.0
                t += dt
                points.append(TrajectoryPoint(
                    t=t,
                    position=pos,
                    velocity=vel,
                    heading=heading,
                    curvature=curvature
                ))
        
        # 6. 后处理：检查并修复航向角突变（更严格的限制）
        points = self._smooth_heading_changes(points, max_heading_rate=np.radians(15))
        
        return Trajectory(points=points, dt=self.dt)
    
    def _smooth_heading_changes(self, points: List, max_heading_rate: float = np.radians(30)) -> List:
        """
        平滑航向角变化，避免欧拉角突变
        
        参数:
            points: 轨迹点列表
            max_heading_rate: 最大航向角变化率 (rad/s)
        
        返回:
            平滑后的轨迹点列表
        """
        if len(points) < 2:
            return points
        
        smoothed_points = [points[0]]  # 保留第一个点
        
        # 统计信息
        max_heading_change = 0.0
        n_violations = 0
        
        for i in range(1, len(points)):
            prev_point = smoothed_points[-1]
            curr_point = points[i]
            
            prev_heading = prev_point.heading
            curr_heading = curr_point.heading
            
            # 计算航向角变化
            heading_change = self._wrap_angle(curr_heading - prev_heading)
            
            # 计算时间差
            dt_actual = curr_point.t - prev_point.t
            if dt_actual < 1e-6:
                dt_actual = self.dt
            
            # 计算允许的最大航向角变化
            max_allowed_change = max_heading_rate * dt_actual
            
            # 如果变化太大，添加中间点平滑过渡
            if abs(heading_change) > max_allowed_change:
                n_violations += 1
                max_heading_change = max(max_heading_change, abs(heading_change))
                
                # 计算需要插入的中间点数量
                n_smooth = max(int(abs(heading_change) / max_allowed_change) + 1, 2)  # 至少2个点
                
                for j in range(1, n_smooth + 1):
                    alpha = j / n_smooth
                    # 位置插值
                    pos = prev_point.position + alpha * (curr_point.position - prev_point.position)
                    # 航向角平滑过渡（关键：确保每步变化不超过限制）
                    smooth_heading = prev_heading + alpha * heading_change
                    smooth_heading = self._wrap_angle(smooth_heading)
                    # 速度方向跟随航向
                    speed = np.linalg.norm(curr_point.velocity) if np.linalg.norm(curr_point.velocity) > 1e-6 else self.reference_speed
                    vel = speed * np.array([np.cos(smooth_heading), np.sin(smooth_heading), 0.0])
                    # 曲率插值
                    curvature = prev_point.curvature + alpha * (curr_point.curvature - prev_point.curvature)
                    # 时间（均匀分布）
                    t = prev_point.t + alpha * dt_actual
                    
                    smoothed_points.append(TrajectoryPoint(
                        t=t,
                        position=pos,
                        velocity=vel,
                        heading=smooth_heading,
                        curvature=curvature
                    ))
            else:
                # 变化在允许范围内，直接添加
                smoothed_points.append(curr_point)
        
        if n_violations > 0:
            print(f"    [航向平滑] 修复了{n_violations}处航向角突变，最大变化={np.degrees(max_heading_change):.1f}°")
        
        return smoothed_points
    
    @staticmethod
    def _wrap_angle(angle: float) -> float:
        """将角度归一化到 [-pi, pi]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

    def _densify_waypoints(self, waypoints: List[np.ndarray],
                           density: int) -> List[np.ndarray]:
        """
        增加航点密度

        参数:
            waypoints: 原始航点列表
            density: 每两个原始航点之间插入的点数 + 1

        返回:
            加密后的航点列表
        """
        if density <= 1 or len(waypoints) < 2:
            return waypoints

        dense = [waypoints[0]]

        for i in range(len(waypoints) - 1):
            p1, p2 = waypoints[i], waypoints[i + 1]

            # 在两点之间插入 density-1 个点
            for j in range(1, density):
                t = j / density
                new_point = p1 + t * (p2 - p1)
                dense.append(new_point)

            dense.append(p2)

        return dense

    def _compute_arc_lengths(self, waypoints: List[np.ndarray]) -> np.ndarray:
        """计算累积弧长"""
        arc_lengths = [0.0]
        for i in range(len(waypoints) - 1):
            dist = np.linalg.norm(waypoints[i + 1] - waypoints[i])
            arc_lengths.append(arc_lengths[-1] + dist)
        return np.array(arc_lengths)

    def _remove_duplicate_waypoints(self, waypoints: List[np.ndarray],
                                     min_dist: float = 1e-6) -> List[np.ndarray]:
        """
        移除距离过近的相邻航点，确保弧长严格递增
        
        参数:
            waypoints: 航点列表
            min_dist: 最小距离阈值，小于此值的相邻点会被合并
            
        返回:
            去重后的航点列表
        """
        if len(waypoints) <= 1:
            return waypoints
        
        filtered = [waypoints[0]]
        for i in range(1, len(waypoints)):
            dist = np.linalg.norm(waypoints[i] - filtered[-1])
            if dist > min_dist:
                filtered.append(waypoints[i])
        
        # 确保终点被保留（如果原始终点和过滤后最后一点不同）
        if len(waypoints) > 1:
            final_dist = np.linalg.norm(waypoints[-1] - filtered[-1])
            if final_dist > min_dist and not np.allclose(waypoints[-1], filtered[-1]):
                filtered.append(waypoints[-1])
        
        return filtered

    def smooth_legacy(self, waypoints: List[np.ndarray],
                      sample_step: float = 5.0,
                      start_heading: Optional[float] = None,
                      end_heading: Optional[float] = None) -> np.ndarray:
        """
        平滑路径 (兼容旧接口)

        参数:
            waypoints: 路径点列表 [[x, y, z], ...]
            sample_step: 采样步长 (m)，用于计算大致点数
            start_heading: 起点航向 (rad)
            end_heading: 终点航向 (rad)

        返回:
            平滑后的轨迹 [n, 4] (x, y, z, heading)
        """
        trajectory = self.smooth(waypoints, start_heading, end_heading)
        return trajectory.to_legacy_array()

    def compute_curvature(self, trajectory: np.ndarray) -> np.ndarray:
        """
        计算轨迹曲率 (兼容旧接口)

        参数:
            trajectory: [n, 4] 数组 (x, y, z, heading)

        返回:
            曲率数组 [n]
        """
        if len(trajectory) < 3:
            return np.zeros(len(trajectory))

        curvatures = [0]
        for i in range(1, len(trajectory) - 1):
            p1 = trajectory[i - 1, :2]
            p2 = trajectory[i, :2]
            p3 = trajectory[i + 1, :2]

            # 三点计算曲率
            v1 = p2 - p1
            v2 = p3 - p2
            cross = v1[0] * v2[1] - v1[1] * v2[0]
            d1 = np.linalg.norm(v1)
            d2 = np.linalg.norm(v2)
            d3 = np.linalg.norm(p3 - p1)

            if d1 * d2 * d3 > 1e-6:
                curvature = 2 * abs(cross) / (d1 * d2 * d3)
            else:
                curvature = 0

            curvatures.append(curvature)

        curvatures.append(0)
        return np.array(curvatures)


def visualize_trajectory(trajectory: Trajectory,
                        raw_waypoints: List[np.ndarray],
                        smoother: PathSmoother,
                        save_path: Optional[str] = None):
    """
    可视化轨迹

    参数:
        trajectory: 平滑后的轨迹
        raw_waypoints: 原始 RRT* 航点
        smoother: PathSmoother 对象（用于获取约束参数）
        save_path: 保存路径（可选）
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Set font
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12

    fig = plt.figure(figsize=(20, 10))

    # 转换数据
    traj_arr = trajectory.to_position_array()
    wp_arr = np.array(raw_waypoints)
    distances = np.cumsum([0] + [
        np.linalg.norm(traj_arr[i + 1] - traj_arr[i])
        for i in range(len(traj_arr) - 1)
    ])

    # ===== 1. 3D 轨迹对比 =====
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    # 原始航点：红色方块+虚线
    ax1.plot(wp_arr[:, 0], wp_arr[:, 1], wp_arr[:, 2],
             'rs--', markersize=5, linewidth=1, alpha=0.7, label='RRT* Waypoints')
    # 平滑轨迹：蓝色实线
    ax1.plot(traj_arr[:, 0], traj_arr[:, 1], traj_arr[:, 2],
             'b-', linewidth=1.5, alpha=0.9, label='Smoothed Trajectory')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Trajectory Comparison')
    ax1.legend()

    # ===== 2. XY 平面对比 =====
    ax2 = fig.add_subplot(2, 3, 2)
    # 原始航点：红色方块+虚线（有拐点）
    ax2.plot(wp_arr[:, 0], wp_arr[:, 1], 'rs--', markersize=6,
             linewidth=1.5, alpha=0.7, label='RRT* Waypoints (with corners)')
    # 平滑轨迹：蓝色实线（无拐点）
    ax2.plot(traj_arr[:, 0], traj_arr[:, 1], 'b-',
             linewidth=2, alpha=0.9, label='Smoothed Trajectory (smooth)')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('XY Plane: Smoothing Effect')
    ax2.legend()
    ax2.axis('equal')
    ax2.grid(True, alpha=0.3)

    # ===== 3. 高度剖面 =====
    ax3 = fig.add_subplot(2, 3, 3)
    wp_dist = np.cumsum([0] + [np.linalg.norm(wp_arr[i + 1] - wp_arr[i])
                                for i in range(len(wp_arr) - 1)])
    ax3.plot(wp_dist, wp_arr[:, 2], 'rs--', markersize=5,
             linewidth=1, alpha=0.7, label='RRT* Waypoints')
    ax3.plot(distances, traj_arr[:, 2], 'b-',
             linewidth=2, alpha=0.9, label='Smoothed Trajectory')
    ax3.set_xlabel('Distance (m)')
    ax3.set_ylabel('Altitude (m)')
    ax3.set_title('Altitude Profile')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # ===== 4. 转弯半径分布 =====
    ax4 = fig.add_subplot(2, 3, 4)
    turning_radii = trajectory.get_turning_radii()
    # 限制显示范围，避免无穷大
    turning_radii_clipped = np.clip(turning_radii, 0, 500)
    ax4.plot(distances, turning_radii_clipped, 'g-', linewidth=2)
    ax4.axhline(y=smoother.turn_radius, color='r', linestyle='--', linewidth=2,
                label=f'Min Turn Radius = {smoother.turn_radius}m')
    ax4.fill_between(distances, 0, smoother.turn_radius, alpha=0.2, color='red',
                     label='Constraint Violation Zone')
    ax4.set_xlabel('Distance (m)')
    ax4.set_ylabel('Turning Radius (m)')
    ax4.set_title('Turning Radius Distribution')
    ax4.set_ylim([0, min(500, turning_radii_clipped.max() * 1.1)])
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # ===== 5. 速度剖面 =====
    ax5 = fig.add_subplot(2, 3, 5)
    velocities = trajectory.get_velocities()
    speeds = np.linalg.norm(velocities, axis=1)
    timestamps = trajectory.get_timestamps()
    ax5.plot(timestamps, speeds, 'b-', linewidth=2, label='Speed')
    ax5.axhline(y=smoother.reference_speed, color='g', linestyle='--',
                linewidth=2, label=f'Reference Speed = {smoother.reference_speed} m/s')
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Speed (m/s)')
    ax5.set_title('Speed Profile')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # ===== 6. 航向角变化 =====
    ax6 = fig.add_subplot(2, 3, 6)
    headings = np.degrees(trajectory.get_headings())
    ax6.plot(timestamps, headings, 'm-', linewidth=2)
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Heading (deg)')
    ax6.set_title('Heading Profile')
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图像已保存到: {save_path}")

    plt.show()


# ============================================================
#                     测试
# ============================================================

def print_header(title: str, width: int = 70):
    """打印格式化标题"""
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)

def print_section(title: str, width: int = 70):
    """打印子标题"""
    print(f"\n{'─' * width}")
    print(f"  {title}")
    print(f"{'─' * width}")

def print_kv(key: str, value, indent: int = 4):
    """打印键值对"""
    print(f"{' ' * indent}{key:<20} : {value}")

def print_status(msg: str, status: str = "OK"):
    """打印状态信息"""
    symbols = {"OK": "✓", "WARN": "⚠", "ERROR": "✗", "INFO": "●"}
    symbol = symbols.get(status, "●")
    print(f"    [{symbol}] {msg}")


if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt
    from planning.map_manager import MapManager
    from planning.global_planner import RRTStarPlanner

    parser = argparse.ArgumentParser(description="路径平滑测试")
    parser.add_argument("--config", type=str, required=True, help="地图配置文件路径")
    parser.add_argument("--save-trajectory", type=str, default=None,
                        help="保存轨迹文件路径 (.json/.csv/.npy)")
    parser.add_argument("--save-figure", type=str, default=None,
                        help="保存可视化图像路径")
    args = parser.parse_args()

    print_header("翼伞无人机路径规划系统 (PNC)")

    # ========== 1. 加载配置 ==========
    print_section("Step 1/3: 加载地图配置")
    map_mgr = MapManager.from_yaml(args.config)

    import yaml
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    traj_config = config.get('trajectory', {})
    reference_speed = traj_config.get('reference_speed', 12.0)
    control_frequency = traj_config.get('control_frequency', 100)

    print_kv("起点", f"({map_mgr.start.x:.0f}, {map_mgr.start.y:.0f}, {map_mgr.start.z:.0f})")
    print_kv("目标", f"({map_mgr.target.position[0]:.0f}, {map_mgr.target.position[1]:.0f}, {map_mgr.target.position[2]:.0f})")
    print_kv("障碍物数量", len(map_mgr.obstacles))
    print_kv("参考速度", f"{reference_speed} m/s")
    print_kv("控制频率", f"{control_frequency} Hz")
    print_status("配置加载完成", "OK")

    # ========== 2. RRT* 全局规划（包含画圆消高） ==========
    print_section("Step 2/3: RRT* 全局路径规划")
    planner = RRTStarPlanner(map_mgr)
    
    print_kv("步长", f"{planner.step_size} m")
    print_kv("最大迭代", f"{planner.max_iterations}")
    print()
    
    # 使用plan_with_loiter，前端就包含画圆消高
    path, info = planner.plan_with_loiter(max_time=30.0)

    if path is None:
        print_status("规划失败!", "ERROR")
        exit(1)

    print()
    print_kv("规划耗时", f"{info['time']:.2f} s")
    print_kv("迭代次数", info['iterations'])
    print_kv("树节点数", info['nodes'])
    print_kv("路径长度", f"{info['path_length']:.1f} m")
    print_kv("原始航点数", len(path))
    if info.get('loiter_loops', 0) > 0:
        print_kv("画圆消高", f"{info['loiter_loops']}圈, 半径={info['loiter_radius']:.1f}m")
    print_status("路径规划成功", "OK")

    # ========== 3. 路径平滑 ==========
    print_section("Step 3/3: 三次样条路径平滑")
    smoother = PathSmoother(
        turn_radius=map_mgr.constraints.min_turn_radius,
        reference_speed=reference_speed,
        control_frequency=control_frequency
    )

    end_heading = map_mgr.target.approach_heading if map_mgr.target else None
    # 注意：不传递 map_manager，因为画圆已经在前端处理了
    # 后端只负责平滑，不再添加额外的画圆
    trajectory = smoother.smooth(path, end_heading=end_heading, waypoint_density=15)

    print_kv("轨迹点数", len(trajectory.points))
    print_kv("轨迹时长", f"{trajectory.duration:.2f} s")
    print_kv("轨迹长度", f"{trajectory.total_length:.1f} m")
    print_kv("采样周期", f"{smoother.dt*1000:.1f} ms ({control_frequency} Hz)")

    # 约束检查
    curvatures = trajectory.get_curvatures()
    max_curvature = np.max(curvatures)
    min_radius = 1.0 / max_curvature if max_curvature > 1e-6 else np.inf

    print()
    print("    约束检查:")
    print_kv("最小转弯半径约束", f"{smoother.turn_radius} m", indent=6)
    print_kv("实际最小转弯半径", f"{min_radius:.1f} m", indent=6)
    
    if min_radius >= smoother.turn_radius:
        print_status("满足转弯半径约束", "OK")
    else:
        print_status("违反转弯半径约束!", "WARN")

    # ========== 输出汇总 ==========
    print_header("规划完成")
    print(f"""
    ┌─────────────────────────────────────────────────────────┐
    │  规划结果汇总                                           │
    ├─────────────────────────────────────────────────────────┤
    │  路径长度         :  {info['path_length']:>10.1f} m                   │
    │  原始航点         :  {len(path):>10} 个                   │
    │  轨迹点数         :  {len(trajectory.points):>10} 个                   │
    │  飞行时间         :  {trajectory.duration:>10.1f} s                    │
    │  规划耗时         :  {info['time']:>10.2f} s                    │
    │  最小转弯半径     :  {min_radius:>10.1f} m                   │
    └─────────────────────────────────────────────────────────┘
    """)

    # 保存轨迹
    if args.save_trajectory:
        trajectory.save(args.save_trajectory)
        print_status(f"轨迹已保存到: {args.save_trajectory}", "OK")

    # 可视化
    print_status("正在生成可视化图表...", "INFO")
    visualize_trajectory(trajectory, path, smoother, save_path=args.save_figure)
