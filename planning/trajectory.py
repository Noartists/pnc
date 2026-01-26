"""
轨迹数据结构模块

定义统一的轨迹类，包含控制器需要的所有信息:
- 时间戳
- 位置 (x, y, z)
- 速度 (vx, vy, vz)
- 航向角
- 曲率
"""

import os
import sys
import numpy as np
from typing import List, Optional
from dataclasses import dataclass, field
import json

# 添加项目根目录到路径
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


@dataclass
class TrajectoryPoint:
    """轨迹点数据结构"""
    t: float                           # 时间戳 (s)
    position: np.ndarray               # 位置 [x, y, z] (m)
    velocity: np.ndarray               # 速度 [vx, vy, vz] (m/s)
    heading: float                     # 航向角 (rad)
    curvature: float                   # 曲率 (1/m)

    def __post_init__(self):
        """确保数组类型正确"""
        self.position = np.asarray(self.position, dtype=np.float64)
        self.velocity = np.asarray(self.velocity, dtype=np.float64)

    def to_dict(self) -> dict:
        """转换为字典（用于序列化）"""
        return {
            't': self.t,
            'position': self.position.tolist(),
            'velocity': self.velocity.tolist(),
            'heading': self.heading,
            'curvature': self.curvature
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'TrajectoryPoint':
        """从字典创建（用于反序列化）"""
        return cls(
            t=data['t'],
            position=np.array(data['position']),
            velocity=np.array(data['velocity']),
            heading=data['heading'],
            curvature=data['curvature']
        )

    def to_array(self) -> np.ndarray:
        """转换为数组 [t, x, y, z, vx, vy, vz, heading, curvature]"""
        return np.array([
            self.t,
            self.position[0], self.position[1], self.position[2],
            self.velocity[0], self.velocity[1], self.velocity[2],
            self.heading,
            self.curvature
        ])


class Trajectory:
    """轨迹类"""

    def __init__(self, points: List[TrajectoryPoint] = None, dt: float = 0.01):
        """
        参数:
            points: 轨迹点列表
            dt: 采样周期 (s)，默认0.01s (100Hz)
        """
        self.points: List[TrajectoryPoint] = points if points is not None else []
        self.dt = dt

    def __len__(self) -> int:
        return len(self.points)

    def __getitem__(self, idx) -> TrajectoryPoint:
        return self.points[idx]

    def __iter__(self):
        return iter(self.points)

    def append(self, point: TrajectoryPoint):
        """添加轨迹点"""
        self.points.append(point)

    @property
    def duration(self) -> float:
        """轨迹总时长 (s)"""
        if len(self.points) == 0:
            return 0.0
        return self.points[-1].t - self.points[0].t

    @property
    def total_length(self) -> float:
        """轨迹总长度 (m)"""
        if len(self.points) < 2:
            return 0.0
        length = 0.0
        for i in range(len(self.points) - 1):
            length += np.linalg.norm(
                self.points[i + 1].position - self.points[i].position
            )
        return length

    def to_array(self) -> np.ndarray:
        """
        转换为数组格式

        返回:
            [n, 9] 数组: [t, x, y, z, vx, vy, vz, heading, curvature]
        """
        if len(self.points) == 0:
            return np.array([]).reshape(0, 9)
        return np.array([pt.to_array() for pt in self.points])

    def to_position_array(self) -> np.ndarray:
        """
        仅提取位置数组

        返回:
            [n, 3] 数组: [x, y, z]
        """
        if len(self.points) == 0:
            return np.array([]).reshape(0, 3)
        return np.array([pt.position for pt in self.points])

    def to_legacy_array(self) -> np.ndarray:
        """
        转换为旧版格式（兼容现有代码）

        返回:
            [n, 4] 数组: [x, y, z, heading]
        """
        if len(self.points) == 0:
            return np.array([]).reshape(0, 4)
        return np.array([
            [pt.position[0], pt.position[1], pt.position[2], pt.heading]
            for pt in self.points
        ])

    def get_timestamps(self) -> np.ndarray:
        """获取所有时间戳"""
        return np.array([pt.t for pt in self.points])

    def get_positions(self) -> np.ndarray:
        """获取所有位置 [n, 3]"""
        return self.to_position_array()

    def get_velocities(self) -> np.ndarray:
        """获取所有速度 [n, 3]"""
        if len(self.points) == 0:
            return np.array([]).reshape(0, 3)
        return np.array([pt.velocity for pt in self.points])

    def get_headings(self) -> np.ndarray:
        """获取所有航向角"""
        return np.array([pt.heading for pt in self.points])

    def get_curvatures(self) -> np.ndarray:
        """获取所有曲率"""
        return np.array([pt.curvature for pt in self.points])

    def get_turning_radii(self) -> np.ndarray:
        """获取所有转弯半径 (曲率的倒数，限制最大值)"""
        curvatures = self.get_curvatures()
        # 避免除零，曲率为0时返回无穷大（直线）
        with np.errstate(divide='ignore'):
            radii = np.where(curvatures > 1e-6, 1.0 / curvatures, np.inf)
        return radii

    def interpolate_at(self, t: float) -> Optional[TrajectoryPoint]:
        """
        在指定时间插值获取轨迹点

        参数:
            t: 时间戳 (s)

        返回:
            插值后的轨迹点，超出范围返回 None
        """
        if len(self.points) < 2:
            return None

        if t <= self.points[0].t:
            return self.points[0]
        if t >= self.points[-1].t:
            return self.points[-1]

        # 二分查找
        left, right = 0, len(self.points) - 1
        while left < right - 1:
            mid = (left + right) // 2
            if self.points[mid].t <= t:
                left = mid
            else:
                right = mid

        # 线性插值
        p1, p2 = self.points[left], self.points[right]
        alpha = (t - p1.t) / (p2.t - p1.t + 1e-10)

        return TrajectoryPoint(
            t=t,
            position=p1.position + alpha * (p2.position - p1.position),
            velocity=p1.velocity + alpha * (p2.velocity - p1.velocity),
            heading=p1.heading + alpha * self._angle_diff(p2.heading, p1.heading),
            curvature=p1.curvature + alpha * (p2.curvature - p1.curvature)
        )

    def _angle_diff(self, a: float, b: float) -> float:
        """计算角度差（处理2π跳变）"""
        diff = a - b
        while diff > np.pi:
            diff -= 2 * np.pi
        while diff < -np.pi:
            diff += 2 * np.pi
        return diff

    def save(self, path: str):
        """
        保存轨迹到文件

        支持格式:
            - .json: JSON格式（完整信息）
            - .csv: CSV格式（数组格式）
            - .npy: NumPy二进制格式

        参数:
            path: 文件路径
        """
        ext = os.path.splitext(path)[1].lower()

        if ext == '.json':
            data = {
                'dt': self.dt,
                'n_points': len(self.points),
                'duration': self.duration,
                'total_length': self.total_length,
                'points': [pt.to_dict() for pt in self.points]
            }
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)

        elif ext == '.csv':
            arr = self.to_array()
            header = 't,x,y,z,vx,vy,vz,heading,curvature'
            np.savetxt(path, arr, delimiter=',', header=header, comments='')

        elif ext == '.npy':
            arr = self.to_array()
            np.save(path, arr)

        else:
            raise ValueError(f"不支持的文件格式: {ext}")

        print(f"轨迹已保存到: {path}")

    @classmethod
    def load(cls, path: str) -> 'Trajectory':
        """
        从文件加载轨迹

        参数:
            path: 文件路径

        返回:
            Trajectory 对象
        """
        ext = os.path.splitext(path)[1].lower()

        if ext == '.json':
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            points = [TrajectoryPoint.from_dict(pt) for pt in data['points']]
            return cls(points=points, dt=data.get('dt', 0.01))

        elif ext == '.csv':
            arr = np.loadtxt(path, delimiter=',', skiprows=1)
            return cls.from_array(arr)

        elif ext == '.npy':
            arr = np.load(path)
            return cls.from_array(arr)

        else:
            raise ValueError(f"不支持的文件格式: {ext}")

    @classmethod
    def from_array(cls, arr: np.ndarray, dt: float = 0.01) -> 'Trajectory':
        """
        从数组创建轨迹

        参数:
            arr: [n, 9] 数组 [t, x, y, z, vx, vy, vz, heading, curvature]
            dt: 采样周期

        返回:
            Trajectory 对象
        """
        points = []
        for row in arr:
            points.append(TrajectoryPoint(
                t=row[0],
                position=row[1:4],
                velocity=row[4:7],
                heading=row[7],
                curvature=row[8]
            ))
        return cls(points=points, dt=dt)

    def resample(self, new_dt: float) -> 'Trajectory':
        """
        重新采样轨迹

        参数:
            new_dt: 新的采样周期 (s)

        返回:
            重新采样后的轨迹
        """
        if len(self.points) < 2:
            return Trajectory(points=list(self.points), dt=new_dt)

        t_start = self.points[0].t
        t_end = self.points[-1].t
        n_points = int((t_end - t_start) / new_dt) + 1

        new_points = []
        for i in range(n_points):
            t = t_start + i * new_dt
            pt = self.interpolate_at(t)
            if pt is not None:
                new_points.append(pt)

        return Trajectory(points=new_points, dt=new_dt)

    def summary(self) -> str:
        """返回轨迹摘要信息"""
        if len(self.points) == 0:
            return "空轨迹"

        curvatures = self.get_curvatures()
        max_curv = np.max(curvatures)
        min_radius = 1.0 / max_curv if max_curv > 1e-6 else np.inf

        speeds = np.linalg.norm(self.get_velocities(), axis=1)

        return (
            f"轨迹摘要:\n"
            f"  点数: {len(self.points)}\n"
            f"  采样周期: {self.dt:.4f}s ({1/self.dt:.0f}Hz)\n"
            f"  时长: {self.duration:.2f}s\n"
            f"  总长度: {self.total_length:.1f}m\n"
            f"  速度范围: [{speeds.min():.1f}, {speeds.max():.1f}] m/s\n"
            f"  最大曲率: {max_curv:.6f} (1/m)\n"
            f"  最小转弯半径: {min_radius:.1f}m"
        )


# ============================================================
#                     测试
# ============================================================

if __name__ == "__main__":
    # 创建测试轨迹
    print("创建测试轨迹...")
    points = []
    dt = 0.01
    v_ref = 12.0

    for i in range(100):
        t = i * dt
        theta = t * 0.5  # 圆弧运动
        r = 100  # 半径

        points.append(TrajectoryPoint(
            t=t,
            position=np.array([r * np.cos(theta), r * np.sin(theta), 500 - t * 10]),
            velocity=np.array([-r * 0.5 * np.sin(theta), r * 0.5 * np.cos(theta), -10]),
            heading=theta + np.pi / 2,
            curvature=1.0 / r
        ))

    traj = Trajectory(points=points, dt=dt)

    # 显示摘要
    print(traj.summary())

    # 测试保存和加载
    print("\n测试保存/加载...")
    traj.save("test_trajectory.json")
    traj.save("test_trajectory.csv")
    traj.save("test_trajectory.npy")

    traj_loaded = Trajectory.load("test_trajectory.json")
    print(f"JSON加载: {len(traj_loaded)} 点")

    traj_loaded = Trajectory.load("test_trajectory.csv")
    print(f"CSV加载: {len(traj_loaded)} 点")

    traj_loaded = Trajectory.load("test_trajectory.npy")
    print(f"NPY加载: {len(traj_loaded)} 点")

    # 清理测试文件
    import os
    for ext in ['.json', '.csv', '.npy']:
        os.remove(f"test_trajectory{ext}")

    print("\n测试完成!")
