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
               waypoint_density: int = 3) -> Trajectory:
        """
        平滑路径并生成带时间信息的轨迹

        参数:
            waypoints: 路径点列表 [[x, y, z], ...]
            start_heading: 起点航向 (rad)，None 则自动计算
            end_heading: 终点航向 (rad)，None 则自动计算
            waypoint_density: 航点加密倍数 (在原始航点间插入的点数+1)

        返回:
            Trajectory 对象，包含完整的时间、位置、速度、曲率信息
        """
        if len(waypoints) < 2:
            # 单点或空，返回空轨迹
            return Trajectory(dt=self.dt)

        waypoints = [np.array(wp) for wp in waypoints]

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

        for i in range(n_samples + 1):
            t = i * self.dt
            # 弧长参数 (归一化)
            s = min(t / total_time, 1.0)

            # 位置
            x = float(cs_x(s))
            y = float(cs_y(s))
            z = float(cs_z(s))
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

        return Trajectory(points=trajectory_points, dt=self.dt)

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

    # ========== 2. RRT* 全局规划 ==========
    print_section("Step 2/3: RRT* 全局路径规划")
    planner = RRTStarPlanner(map_mgr)
    
    print_kv("步长", f"{planner.step_size} m")
    print_kv("最大迭代", f"{planner.max_iterations}")
    print()
    
    path, info = planner.plan(max_time=30.0)

    if path is None:
        print_status("规划失败!", "ERROR")
        exit(1)

    print()
    print_kv("规划耗时", f"{info['time']:.2f} s")
    print_kv("迭代次数", info['iterations'])
    print_kv("树节点数", info['nodes'])
    print_kv("路径长度", f"{info['path_length']:.1f} m")
    print_kv("原始航点数", len(path))
    print_status("路径规划成功", "OK")

    # ========== 3. 路径平滑 ==========
    print_section("Step 3/3: 三次样条路径平滑")
    smoother = PathSmoother(
        turn_radius=map_mgr.constraints.min_turn_radius,
        reference_speed=reference_speed,
        control_frequency=control_frequency
    )

    end_heading = map_mgr.target.approach_heading if map_mgr.target else None
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
