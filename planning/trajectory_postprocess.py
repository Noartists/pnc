"""
轨迹后处理模块

功能：
1. 接收 Kinodynamic RRT* 返回的路径点列表
2. 三次样条平滑（可选）
3. 按控制频率重采样
4. 时间参数化
5. 生成 Trajectory 对象

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
from typing import List, Optional, Tuple
from scipy.interpolate import CubicSpline
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from planning.trajectory import Trajectory, TrajectoryPoint


class TrajectoryPostprocessor:
    """
    轨迹后处理器

    将 Kinodynamic RRT* 生成的路径点转换为可用于控制的平滑轨迹
    """

    def __init__(self,
                 reference_speed: float = 12.0,
                 control_frequency: float = 100.0,
                 min_turn_radius: float = 50.0,
                 quiet: bool = False):
        """
        参数:
            reference_speed: 参考飞行速度 (m/s)
            control_frequency: 控制频率 (Hz)
            min_turn_radius: 最小转弯半径 (m)，用于约束检查
            quiet: 是否静默模式（不打印信息）
        """
        self.reference_speed = reference_speed
        self.control_frequency = control_frequency
        self.dt = 1.0 / control_frequency
        self.min_turn_radius = min_turn_radius
        self.quiet = quiet

    def process(self,
                path: List[np.ndarray],
                smooth: bool = True,
                end_heading: Optional[float] = None) -> Trajectory:
        """
        处理路径点，生成平滑轨迹

        参数:
            path: Kinodynamic RRT* 返回的路径点列表 [np.array([x,y,z]), ...]
            smooth: 是否进行三次样条平滑（默认True）
            end_heading: 终点期望航向（可选，用于进场对齐）

        返回:
            Trajectory 对象
        """
        if not path or len(path) < 2:
            if not self.quiet:
                print("    [后处理] 路径点太少，返回空轨迹")
            return Trajectory(dt=self.dt)

        # 转换为numpy数组
        waypoints = [np.asarray(wp, dtype=np.float64) for wp in path]

        # Step 1: 去除重复点
        waypoints = self._remove_duplicates(waypoints)
        if len(waypoints) < 2:
            if not self.quiet:
                print("    [后处理] 去重后路径点太少")
            return Trajectory(dt=self.dt)

        if not self.quiet:
            print(f"\n    [后处理] 输入: {len(path)} 个路径点")

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

        # Step 3: 时间参数化，生成轨迹
        trajectory = self._create_trajectory(smooth_points, end_heading)

        if not self.quiet:
            print(f"    [后处理] 生成轨迹: {len(trajectory)} 点, 时长 {trajectory.duration:.1f}s")

        # Step 4: 验证轨迹约束
        self._validate_trajectory(trajectory)

        return trajectory

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
        # 采样间距 = reference_speed * dt
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
        # 计算总弧长
        arc_lengths = [0.0]
        for i in range(1, len(waypoints)):
            dist = np.linalg.norm(waypoints[i] - waypoints[i-1])
            arc_lengths.append(arc_lengths[-1] + dist)

        total_length = arc_lengths[-1]
        if total_length < 1e-6:
            return waypoints

        # 采样间距
        spacing = self.reference_speed * self.dt
        n_samples = max(int(total_length / spacing), len(waypoints))

        # 按弧长均匀采样
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
                # 最后一点使用前一点的航向
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

            # 计算到下一点的时间
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

        # 三角形面积（叉积 / 2）
        cross = abs(v1[0] * v2[1] - v1[1] * v2[0])
        area = cross / 2

        curvature = 4 * area / (d01 * d12 * d02)
        return curvature

    def _validate_trajectory(self, trajectory: Trajectory):
        """验证轨迹是否满足约束"""
        if len(trajectory) < 2:
            return

        positions = trajectory.to_position_array()
        n = len(positions)

        # 统计
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

        # 曲率检查
        curvatures = trajectory.get_curvatures()
        for c in curvatures:
            if c > 1e-6:
                r = 1.0 / c
                turn_radii.append(r)
                if r < self.min_turn_radius:
                    violations += 1

        # 打印统计
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
        min_turn_radius=map_mgr.constraints.min_turn_radius
    )

    end_heading = map_mgr.target.approach_heading if map_mgr.target else None
    trajectory = postprocessor.process(path, smooth=True, end_heading=end_heading)

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
