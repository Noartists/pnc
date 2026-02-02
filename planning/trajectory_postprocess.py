"""
轨迹后处理模块

功能：
1. 拼接Kinodynamic RRT*返回的路径段
2. 按控制频率重采样
3. 时间参数化
4. 生成Trajectory对象

注意：不做样条平滑！路径段本身已经满足运动学约束。
"""

import numpy as np
from typing import List, Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from planning.trajectory import Trajectory, TrajectoryPoint
from planning.kinodynamic_rrt import PathSegment, DubinsSegment, LoiterSegment


class TrajectoryPostprocessor:
    """轨迹后处理器"""
    
    def __init__(self, 
                 reference_speed: float = 9.0,
                 control_frequency: float = 100.0):
        """
        参数:
            reference_speed: 参考飞行速度 (m/s)
            control_frequency: 控制频率 (Hz)
        """
        self.reference_speed = reference_speed
        self.control_frequency = control_frequency
        self.dt = 1.0 / control_frequency
    
    def process(self, segments: List[PathSegment]) -> Trajectory:
        """
        处理路径段列表，生成轨迹
        
        参数:
            segments: Kinodynamic RRT*返回的路径段列表
        
        返回:
            Trajectory对象
        """
        if not segments:
            return Trajectory()
        
        # Step 1: 拼接所有段的点
        all_points = []
        for seg in segments:
            # 每个点是 [x, y, z, heading]
            all_points.extend(seg.points)
        
        if not all_points:
            return Trajectory()
        
        # Step 2: 去除重复点
        all_points = self._remove_duplicates(all_points)
        
        # Step 3: 按控制频率重采样
        resampled_points = self._resample(all_points)
        
        # Step 4: 生成Trajectory对象
        trajectory = self._create_trajectory(resampled_points)
        
        return trajectory
    
    def _remove_duplicates(self, points: List[np.ndarray], 
                           threshold: float = 0.1) -> List[np.ndarray]:
        """去除重复点"""
        if not points:
            return []
        
        result = [points[0]]
        for pt in points[1:]:
            dist = np.linalg.norm(pt[:3] - result[-1][:3])
            if dist > threshold:
                result.append(pt)
        
        return result
    
    def _resample(self, points: List[np.ndarray]) -> List[np.ndarray]:
        """
        按控制频率重采样
        
        每隔 reference_speed * dt 米采样一个点
        """
        if len(points) < 2:
            return points
        
        # 计算总弧长
        arc_lengths = [0.0]
        for i in range(1, len(points)):
            dist = np.linalg.norm(points[i][:3] - points[i-1][:3])
            arc_lengths.append(arc_lengths[-1] + dist)
        
        total_length = arc_lengths[-1]
        
        # 计算采样间距
        spacing = self.reference_speed * self.dt
        
        # 采样
        resampled = []
        current_s = 0.0
        
        while current_s <= total_length:
            pt = self._interpolate_at_arc_length(points, arc_lengths, current_s)
            resampled.append(pt)
            current_s += spacing
        
        # 确保包含终点
        if len(resampled) > 0:
            last_dist = np.linalg.norm(resampled[-1][:3] - points[-1][:3])
            if last_dist > spacing * 0.5:
                resampled.append(points[-1].copy())
        
        return resampled
    
    def _interpolate_at_arc_length(self, points: List[np.ndarray],
                                    arc_lengths: List[float],
                                    s: float) -> np.ndarray:
        """在弧长s处插值"""
        # 找到s所在的段
        for i in range(1, len(arc_lengths)):
            if arc_lengths[i] >= s:
                # 在段 [i-1, i] 中
                s0, s1 = arc_lengths[i-1], arc_lengths[i]
                if s1 - s0 < 1e-6:
                    alpha = 0.0
                else:
                    alpha = (s - s0) / (s1 - s0)
                
                p0, p1 = points[i-1], points[i]
                
                # 线性插值位置和高度
                x = p0[0] + alpha * (p1[0] - p0[0])
                y = p0[1] + alpha * (p1[1] - p0[1])
                z = p0[2] + alpha * (p1[2] - p0[2])
                
                # 航向插值（处理环绕）
                h0, h1 = p0[3], p1[3]
                dh = h1 - h0
                while dh > np.pi:
                    dh -= 2 * np.pi
                while dh < -np.pi:
                    dh += 2 * np.pi
                heading = h0 + alpha * dh
                
                return np.array([x, y, z, heading])
        
        # s超出范围，返回终点
        return points[-1].copy()
    
    def _create_trajectory(self, points: List[np.ndarray]) -> Trajectory:
        """从采样点创建Trajectory对象"""
        trajectory = Trajectory()
        
        t = 0.0
        for i, pt in enumerate(points):
            position = np.array([pt[0], pt[1], pt[2]])
            heading = pt[3]
            
            # 计算速度向量
            if i < len(points) - 1:
                next_pt = points[i + 1]
                direction = np.array([next_pt[0] - pt[0], 
                                      next_pt[1] - pt[1], 
                                      next_pt[2] - pt[2]])
                dist = np.linalg.norm(direction)
                if dist > 1e-6:
                    velocity = direction / dist * self.reference_speed
                else:
                    velocity = np.array([self.reference_speed * np.cos(heading),
                                         self.reference_speed * np.sin(heading),
                                         0.0])
            else:
                # 最后一个点，用航向计算速度方向
                velocity = np.array([self.reference_speed * np.cos(heading),
                                     self.reference_speed * np.sin(heading),
                                     0.0])
            
            # 计算曲率（简化：用相邻点估算）
            if 0 < i < len(points) - 1:
                prev_pt = points[i - 1]
                next_pt = points[i + 1]
                curvature = self._compute_curvature(prev_pt, pt, next_pt)
            else:
                curvature = 0.0
            
            # 添加轨迹点
            traj_pt = TrajectoryPoint(
                position=position,
                velocity=velocity,
                heading=heading,
                curvature=curvature,
                timestamp=t
            )
            trajectory.add_point(traj_pt)
            
            t += self.dt
        
        return trajectory
    
    def _compute_curvature(self, p0: np.ndarray, p1: np.ndarray, 
                           p2: np.ndarray) -> float:
        """计算三点确定的曲率"""
        # 使用Menger曲率公式
        # κ = 4 * Area / (|P0P1| * |P1P2| * |P0P2|)
        
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


def validate_trajectory(trajectory: Trajectory,
                       min_glide_ratio: float = 2.47,
                       max_glide_ratio: float = 6.48,
                       max_glide_change: float = 0.5,
                       min_turn_radius: float = 50.0) -> dict:
    """
    验证轨迹是否满足运动学约束
    
    参数:
        trajectory: 轨迹对象
        min_glide_ratio: 最小滑翔比
        max_glide_ratio: 最大滑翔比
        max_glide_change: 最大滑翔比变化（每秒）
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
    
    # 统计
    glide_ratios = []
    turn_radii = []
    
    window_size = 10  # 用10个点计算局部滑翔比和转弯半径
    
    for i in range(0, n_points - window_size, window_size):
        p1 = positions[i]
        p2 = positions[i + window_size]
        
        # 滑翔比
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
                    f"段{i//window_size}: 滑翔比{glide:.2f} < 最小{min_glide_ratio:.2f} (需要画圈)")
    
    # 滑翔比连续性
    for i in range(1, len(glide_ratios)):
        change = abs(glide_ratios[i] - glide_ratios[i-1])
        if change > max_glide_change * window_size * 0.1:  # 考虑时间间隔
            result['warnings'].append(
                f"段{i}: 滑翔比变化{change:.2f}过大")
    
    # 曲率 → 转弯半径
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
    from planning.kinodynamic_rrt import KinodynamicRRTStar, KinoState
    from planning.map_manager import MapManager
    import matplotlib.pyplot as plt
    
    # 加载地图
    map_mgr = MapManager.from_yaml("cfg/map_config.yaml")
    
    # 规划
    planner = KinodynamicRRTStar(map_mgr)
    segments, info = planner.plan(max_time=30.0)
    
    if segments:
        # 后处理
        postprocessor = TrajectoryPostprocessor(
            reference_speed=9.0,
            control_frequency=100.0
        )
        trajectory = postprocessor.process(segments)
        
        print(f"\n轨迹生成完成:")
        print(f"  点数: {len(trajectory)}")
        print(f"  时长: {trajectory.duration:.1f}s")
        
        # 验证
        result = validate_trajectory(trajectory)
        print(f"\n轨迹验证:")
        print(f"  有效: {result['valid']}")
        if result['errors']:
            print(f"  错误: {result['errors']}")
        if result['warnings']:
            print(f"  警告: {result['warnings'][:5]}...")  # 只显示前5个
        if 'glide_ratio' in result['statistics']:
            stats = result['statistics']['glide_ratio']
            print(f"  滑翔比: [{stats['min']:.2f}, {stats['max']:.2f}], 均值={stats['mean']:.2f}")
        
        # 可视化
        positions = trajectory.to_position_array()
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        axes[0].plot(positions[:, 0], positions[:, 1], 'b-', linewidth=1)
        axes[0].set_xlabel('X (m)')
        axes[0].set_ylabel('Y (m)')
        axes[0].set_title('XY Plane')
        axes[0].axis('equal')
        axes[0].grid(True)
        
        timestamps = trajectory.get_timestamps()
        axes[1].plot(timestamps, positions[:, 2], 'b-', linewidth=1)
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Altitude (m)')
        axes[1].set_title('Altitude Profile')
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.show()
