"""
输出数据结构定义

定义 metrics.json 和 case.json 的数据结构
"""

import os
import json
import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class ReproducibilityInfo:
    """复现信息"""
    seed: int
    config_hash: str
    git_commit: Optional[str]


@dataclass 
class HardFailResults:
    """硬失败检测结果"""
    H1_nfz_violation: bool = False
    H1_clearance_violation: bool = False
    H2_attitude_violation: bool = False
    H3_numerical_explosion: bool = False


@dataclass
class SoftFailResults:
    """软失败检测结果"""
    S1_tracking_divergence: bool = False
    S3_saturation_divergence: bool = False
    S4_timeout: bool = False


@dataclass
class QualityMetrics:
    """质量指标"""
    ADE: float = 0.0                    # 平均位移误差
    RMSE: float = 0.0                   # 均方根误差
    FDE: float = 0.0                    # 终点位移误差
    FDE_horizontal: float = 0.0         # 终点水平误差
    FDE_vertical: float = 0.0           # 终点垂直误差
    mean_cross_track_error: float = 0.0 # 平均横向误差
    max_cross_track_error: float = 0.0  # 最大横向误差
    delta_u_sum: float = 0.0            # 控制变化总和
    mean_control_effort: float = 0.0    # 平均控制量
    max_roll: float = 0.0               # 最大滚转角
    max_pitch: float = 0.0              # 最大俯仰角
    max_yaw_rate: float = 0.0           # 最大偏航角速率
    saturation_ratio: float = 0.0       # 饱和占比


@dataclass
class TimingInfo:
    """时间信息"""
    planning_time: float = 0.0          # 规划耗时 (s)
    flight_time: float = 0.0            # 飞行时长 (s)
    wall_time: float = 0.0              # 计算耗时 (s)


class MetricsOutput:
    """
    metrics.json 输出类
    
    包含复现信息、成功判定、失败检测结果、质量指标、时间信息
    """
    
    def __init__(self):
        # 复现信息
        self.seed: int = 0
        self.config_hash: str = ""
        self.git_commit: Optional[str] = None
        
        # 场景信息
        self.scene: str = "default"
        self.wind_speed: float = 0.0
        self.controller: str = "adrc"
        
        # 成功判定
        self.success: bool = False
        self.termination_reason: str = ""
        self.termination_time: Optional[float] = None
        
        # 失败检测结果
        self.hard_fail = HardFailResults()
        self.soft_fail = SoftFailResults()
        
        # 质量指标
        self.quality = QualityMetrics()
        
        # 时间信息
        self.timing = TimingInfo()
        
        # 时间戳
        self.timestamp: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            # 复现信息
            'seed': self.seed,
            'config_hash': self.config_hash,
            'git_commit': self.git_commit,
            
            # 场景信息
            'scene': self.scene,
            'wind_speed': self.wind_speed,
            'controller': self.controller,
            
            # 成功判定
            'success': self.success,
            'termination_reason': self.termination_reason,
            'termination_time': self.termination_time,
            
            # 失败检测结果
            'hard_fail': asdict(self.hard_fail),
            'soft_fail': asdict(self.soft_fail),
            
            # 质量指标
            'quality': asdict(self.quality),
            
            # 时间信息
            'timing': asdict(self.timing),
            
            # 时间戳
            'timestamp': self.timestamp
        }
    
    def save(self, filepath: str):
        """保存到 JSON 文件"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, cls=NumpyEncoder, indent=2, ensure_ascii=False)


class NumpyEncoder(json.JSONEncoder):
    """处理 numpy 类型的 JSON 编码器"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


class CaseOutput:
    """
    case.json 输出类
    
    包含详细的轨迹数据、控制数据、事件日志等
    """
    
    def __init__(self):
        # 配置快照
        self.config: Dict = {}
        
        # 障碍物/禁飞区
        self.obstacles: List[Dict] = []
        self.no_fly_zones: List[Dict] = []
        
        # 起终点
        self.start_position: List[float] = [0, 0, 0]
        self.target_position: List[float] = [0, 0, 0]
        
        # 参考轨迹 (规划输出)
        self.reference_trajectory: List[Dict] = []
        
        # 实际轨迹
        self.actual_trajectory: List[Dict] = []
        
        # 控制历史
        self.control_history: List[Dict] = []
        
        # 事件日志
        self.events: List[Dict] = []
    
    def add_trajectory_point(self, t: float, position: np.ndarray, 
                            velocity: np.ndarray, euler: np.ndarray,
                            ref_position: np.ndarray, ref_heading: float):
        """添加轨迹点"""
        self.actual_trajectory.append({
            't': float(t),
            'x': float(position[0]),
            'y': float(position[1]),
            'z': float(position[2]),
            'vx': float(velocity[0]),
            'vy': float(velocity[1]),
            'vz': float(velocity[2]),
            'roll': float(euler[0]),
            'pitch': float(euler[1]),
            'yaw': float(euler[2]),
            'ref_x': float(ref_position[0]),
            'ref_y': float(ref_position[1]),
            'ref_z': float(ref_position[2]),
            'ref_heading': float(ref_heading)
        })
    
    def add_control_point(self, t: float, delta_left: float, delta_right: float,
                         cross_track_error: float, heading_error: float):
        """添加控制点"""
        self.control_history.append({
            't': float(t),
            'delta_left': float(delta_left),
            'delta_right': float(delta_right),
            'cross_track_error': float(cross_track_error),
            'heading_error': float(heading_error)
        })
    
    def add_event(self, t: float, event_type: str, details: Dict = None):
        """添加事件"""
        event = {
            't': float(t),
            'type': event_type
        }
        if details:
            event.update(details)
        self.events.append(event)
    
    def set_reference_trajectory(self, trajectory):
        """设置参考轨迹（从 Trajectory 对象）"""
        self.reference_trajectory = []
        for point in trajectory:
            self.reference_trajectory.append({
                't': float(point.t),
                'x': float(point.position[0]),
                'y': float(point.position[1]),
                'z': float(point.position[2]),
                'heading': float(point.heading)
            })
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'config': self.config,
            'obstacles': self.obstacles,
            'no_fly_zones': self.no_fly_zones,
            'start_position': self.start_position,
            'target_position': self.target_position,
            'reference_trajectory': self.reference_trajectory,
            'actual_trajectory': self.actual_trajectory,
            'control_history': self.control_history,
            'events': self.events
        }
    
    def save(self, filepath: str):
        """保存到 JSON 文件"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, cls=NumpyEncoder, ensure_ascii=False)


def create_output_dir(base_dir: str, scene: str, seed: int) -> str:
    """
    创建输出目录
    
    参数:
        base_dir: 基础目录 (例如 benchmark/outputs/exp_20260127)
        scene: 场景名称
        seed: 种子
    
    返回:
        输出目录路径
    """
    output_dir = os.path.join(base_dir, scene, f'seed_{seed:03d}')
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def create_experiment_dir(base_dir: str = 'benchmark/outputs') -> str:
    """
    创建实验目录（按时间戳命名）
    
    参数:
        base_dir: 基础目录
    
    返回:
        实验目录路径
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = os.path.join(base_dir, f'exp_{timestamp}')
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir
