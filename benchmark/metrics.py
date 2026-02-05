"""
指标计算与失败判定模块

包含:
- FailureDetector: 硬失败(Hard Fail)和软发散(Soft Fail)检测
- MetricsCalculator: Quality指标计算（ADE, RMSE, FDE等）
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum


class TerminationReason(Enum):
    """终止原因枚举"""
    SUCCESS = "success"                      # 成功着陆
    
    # Hard Failures
    H1_NFZ_VIOLATION = "hard_nfz_violation"           # 进入禁飞区
    H1_CLEARANCE_VIOLATION = "hard_clearance_violation"  # 安全间距不足
    H2_ATTITUDE_VIOLATION = "hard_attitude_violation"    # 姿态超限
    H3_NUMERICAL_EXPLOSION = "hard_numerical_explosion"  # 数值爆炸
    
    # Soft Failures
    S1_TRACKING_DIVERGENCE = "soft_tracking_divergence"  # 跟踪发散
    S3_SATURATION_DIVERGENCE = "soft_saturation_divergence"  # 长时间饱和
    S4_TIMEOUT = "soft_timeout"                          # 超时
    
    # 其他
    GROUND_CONTACT = "ground_contact"        # 落地（未到达目标）


@dataclass
class FailureThresholds:
    """失败判定阈值配置"""
    # 成功判定
    landing_radius: float = 20.0             # 落点误差阈值 (m)
    
    # H1: 约束违规
    safety_margin: float = 15.0              # 安全间距阈值 (m)
    
    # H2: 姿态超限（持续时间防抖）
    roll_max_hard: float = np.radians(60)    # 最大滚转角 (rad)
    pitch_max_hard: float = np.radians(45)   # 最大俯仰角 (rad)
    yaw_rate_max_hard: float = 2.0           # 最大偏航角速率 (rad/s)
    attitude_debounce_time: float = 0.2      # 姿态超限防抖时间 (s)
    
    # H3: 数值爆炸
    velocity_sanity_bound: float = 80.0      # 速度绝对值上限 (m/s)
    position_sanity_bound: float = 10000.0   # 位置绝对值上限 (m)
    
    # S1: 跟踪发散（滑动窗口）
    tracking_window: float = 10.0            # 窗口时长 (s) - 放宽
    tracking_error_threshold: float = 50.0   # 误差阈值 (m) - 进一步放宽
    tracking_violation_ratio: float = 0.9    # 超限占比阈值 - 放宽
    
    # S3: 长时间饱和
    saturation_window: float = 15.0          # 窗口时长 (s) - 放宽
    saturation_ratio_threshold: float = 0.98 # 饱和占比阈值 - 几乎全程饱和才失败
    
    # 软失败检测的 warmup 时间（在此期间不检测软失败）
    soft_fail_warmup: float = 30.0           # 预热时长 (s) - 给控制器更多收敛时间
    
    # S4: 超时
    timeout_margin: float = 30.0             # 超时余量 (在轨迹时长基础上加)


@dataclass
class FailureState:
    """失败检测状态（用于在线检测）"""
    # H2 防抖计数
    roll_violation_duration: float = 0.0
    pitch_violation_duration: float = 0.0
    yaw_rate_violation_duration: float = 0.0
    
    # S1 滑动窗口
    tracking_errors: List[float] = field(default_factory=list)
    tracking_times: List[float] = field(default_factory=list)
    
    # S3 滑动窗口
    saturation_flags: List[bool] = field(default_factory=list)
    saturation_times: List[float] = field(default_factory=list)


class FailureDetector:
    """
    失败检测器
    
    支持在线检测（每步调用）和离线检测（批量分析）
    """
    
    def __init__(self, 
                 thresholds: FailureThresholds = None,
                 no_fly_zones: List[Dict] = None,
                 target_position: np.ndarray = None):
        """
        参数:
            thresholds: 失败判定阈值
            no_fly_zones: 禁飞区列表
            target_position: 目标位置 [x, y, z]
        """
        self.thresholds = thresholds or FailureThresholds()
        self.no_fly_zones = no_fly_zones or []
        self.target_position = target_position
        
        # 在线检测状态
        self.state = FailureState()
        self.dt = 0.01  # 默认控制周期
        
        # 检测结果
        self.hard_fail_detected = False
        self.soft_fail_detected = False
        self.termination_reason: Optional[TerminationReason] = None
        self.termination_time: Optional[float] = None
        
        # 详细检测结果
        self.detection_results = {
            'H1_nfz_violation': False,
            'H1_clearance_violation': False,
            'H2_attitude_violation': False,
            'H3_numerical_explosion': False,
            'S1_tracking_divergence': False,
            'S3_saturation_divergence': False,
            'S4_timeout': False,
        }
    
    def set_dt(self, dt: float):
        """设置控制周期"""
        self.dt = dt
    
    def reset(self):
        """重置检测状态"""
        self.state = FailureState()
        self.hard_fail_detected = False
        self.soft_fail_detected = False
        self.termination_reason = None
        self.termination_time = None
        for key in self.detection_results:
            self.detection_results[key] = False
    
    def check_step(self,
                   t: float,
                   position: np.ndarray,
                   velocity: np.ndarray,
                   euler: np.ndarray,
                   euler_rate: np.ndarray,
                   control: Tuple[float, float],
                   cross_track_error: float,
                   max_time: float) -> Optional[TerminationReason]:
        """
        单步检测（在线调用）
        
        参数:
            t: 当前时间
            position: 位置 [x, y, z]
            velocity: 速度 [vx, vy, vz]
            euler: 姿态角 [roll, pitch, yaw]
            euler_rate: 姿态角速率 [roll_rate, pitch_rate, yaw_rate]
            control: 控制量 (delta_left, delta_right)
            cross_track_error: 横向跟踪误差
            max_time: 最大仿真时间
        
        返回:
            如果检测到失败，返回 TerminationReason；否则返回 None
        """
        if self.hard_fail_detected or self.soft_fail_detected:
            return self.termination_reason
        
        # === H1: 约束违规 ===
        # 禁飞区检测
        if self._check_nfz_violation(position):
            self._trigger_failure(TerminationReason.H1_NFZ_VIOLATION, t)
            self.detection_results['H1_nfz_violation'] = True
            return self.termination_reason
        
        # === H2: 姿态超限 ===
        if self._check_attitude_violation(euler, euler_rate):
            self._trigger_failure(TerminationReason.H2_ATTITUDE_VIOLATION, t)
            self.detection_results['H2_attitude_violation'] = True
            return self.termination_reason
        
        # === H3: 数值爆炸 ===
        if self._check_numerical_explosion(position, velocity):
            self._trigger_failure(TerminationReason.H3_NUMERICAL_EXPLOSION, t)
            self.detection_results['H3_numerical_explosion'] = True
            return self.termination_reason
        
        # === S1: 跟踪发散 ===
        if self._check_tracking_divergence(t, cross_track_error):
            self._trigger_failure(TerminationReason.S1_TRACKING_DIVERGENCE, t)
            self.detection_results['S1_tracking_divergence'] = True
            return self.termination_reason
        
        # === S3: 长时间饱和 ===
        if self._check_saturation_divergence(t, control):
            self._trigger_failure(TerminationReason.S3_SATURATION_DIVERGENCE, t)
            self.detection_results['S3_saturation_divergence'] = True
            return self.termination_reason
        
        # === S4: 超时 ===
        if t >= max_time:
            self._trigger_failure(TerminationReason.S4_TIMEOUT, t)
            self.detection_results['S4_timeout'] = True
            return self.termination_reason
        
        return None
    
    def check_success(self, final_position: np.ndarray) -> bool:
        """
        检查是否成功
        
        参数:
            final_position: 最终位置 [x, y, z]
        
        返回:
            是否成功
        """
        if self.target_position is None:
            return False
        
        # 计算落点误差（水平距离）
        error = np.linalg.norm(final_position[:2] - self.target_position[:2])
        
        # 检查是否在目标区域内且无失败
        return (error <= self.thresholds.landing_radius and 
                not self.hard_fail_detected and 
                not self.soft_fail_detected)
    
    def _trigger_failure(self, reason: TerminationReason, t: float):
        """触发失败"""
        if reason.value.startswith('hard'):
            self.hard_fail_detected = True
        else:
            self.soft_fail_detected = True
        self.termination_reason = reason
        self.termination_time = t
    
    def _check_nfz_violation(self, position: np.ndarray) -> bool:
        """检查禁飞区违规"""
        x, y, z = position
        
        for nfz in self.no_fly_zones:
            nfz_type = nfz.get('type', 'cylinder')
            z_min = nfz.get('z_min', 0)
            z_max = nfz.get('z_max', float('inf'))
            
            # 高度检查
            if z < z_min or z > z_max:
                continue
            
            if nfz_type == 'cylinder':
                center = nfz['center']
                radius = nfz['radius']
                dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
                if dist < radius:
                    return True
            
            elif nfz_type == 'polygon':
                vertices = np.array(nfz['vertices'])
                if self._point_in_polygon(x, y, vertices):
                    return True
        
        return False
    
    def _check_attitude_violation(self, euler: np.ndarray, euler_rate: np.ndarray) -> bool:
        """检查姿态超限（带防抖）"""
        roll, pitch, _ = euler
        _, _, yaw_rate = euler_rate
        
        th = self.thresholds
        
        # 滚转角检测
        if abs(roll) > th.roll_max_hard:
            self.state.roll_violation_duration += self.dt
        else:
            self.state.roll_violation_duration = 0.0
        
        # 俯仰角检测
        if abs(pitch) > th.pitch_max_hard:
            self.state.pitch_violation_duration += self.dt
        else:
            self.state.pitch_violation_duration = 0.0
        
        # 偏航角速率检测
        if abs(yaw_rate) > th.yaw_rate_max_hard:
            self.state.yaw_rate_violation_duration += self.dt
        else:
            self.state.yaw_rate_violation_duration = 0.0
        
        # 防抖判定
        debounce = th.attitude_debounce_time
        return (self.state.roll_violation_duration > debounce or
                self.state.pitch_violation_duration > debounce or
                self.state.yaw_rate_violation_duration > debounce)
    
    def _check_numerical_explosion(self, position: np.ndarray, velocity: np.ndarray) -> bool:
        """检查数值爆炸"""
        th = self.thresholds
        
        # NaN/Inf 检查
        if np.any(np.isnan(position)) or np.any(np.isinf(position)):
            return True
        if np.any(np.isnan(velocity)) or np.any(np.isinf(velocity)):
            return True
        
        # Sanity bound 检查
        if np.any(np.abs(position) > th.position_sanity_bound):
            return True
        if np.linalg.norm(velocity) > th.velocity_sanity_bound:
            return True
        
        return False
    
    def _check_tracking_divergence(self, t: float, cross_track_error: float) -> bool:
        """检查跟踪发散（滑动窗口）"""
        th = self.thresholds
        
        # 添加到窗口
        self.state.tracking_errors.append(abs(cross_track_error))
        self.state.tracking_times.append(t)
        
        # 移除过期数据
        window_start = t - th.tracking_window
        while (self.state.tracking_times and 
               self.state.tracking_times[0] < window_start):
            self.state.tracking_times.pop(0)
            self.state.tracking_errors.pop(0)
        
        # Warmup 期间不检测
        if t < th.soft_fail_warmup:
            return False
        
        # 需要足够数据（至少填满一半窗口）
        min_samples = int(th.tracking_window / self.dt * 0.5)
        if len(self.state.tracking_errors) < max(50, min_samples):
            return False
        
        violations = sum(1 for e in self.state.tracking_errors 
                        if e > th.tracking_error_threshold)
        ratio = violations / len(self.state.tracking_errors)
        
        return ratio > th.tracking_violation_ratio
    
    def _check_saturation_divergence(self, t: float, control: Tuple[float, float]) -> bool:
        """检查长时间饱和"""
        th = self.thresholds
        
        # 判断是否饱和（控制量接近边界）
        delta_left, delta_right = control
        is_saturated = (abs(delta_left - 1.0) < 0.01 or abs(delta_left) < 0.01 or
                       abs(delta_right - 1.0) < 0.01 or abs(delta_right) < 0.01)
        
        # 添加到窗口
        self.state.saturation_flags.append(is_saturated)
        self.state.saturation_times.append(t)
        
        # 移除过期数据
        window_start = t - th.saturation_window
        while (self.state.saturation_times and 
               self.state.saturation_times[0] < window_start):
            self.state.saturation_times.pop(0)
            self.state.saturation_flags.pop(0)
        
        # Warmup 期间不检测
        if t < th.soft_fail_warmup:
            return False
        
        # 需要足够数据（至少填满一半窗口）
        min_samples = int(th.saturation_window / self.dt * 0.5)
        if len(self.state.saturation_flags) < max(50, min_samples):
            return False
        
        ratio = sum(self.state.saturation_flags) / len(self.state.saturation_flags)
        
        return ratio > th.saturation_ratio_threshold
    
    @staticmethod
    def _point_in_polygon(x: float, y: float, vertices: np.ndarray) -> bool:
        """射线法判断点是否在多边形内"""
        n = len(vertices)
        inside = False
        
        j = n - 1
        for i in range(n):
            xi, yi = vertices[i]
            xj, yj = vertices[j]
            
            if ((yi > y) != (yj > y) and
                x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        
        return inside
    
    def get_results(self) -> Dict[str, Any]:
        """获取检测结果"""
        return {
            'hard_fail': self.hard_fail_detected,
            'soft_fail': self.soft_fail_detected,
            'termination_reason': self.termination_reason.value if self.termination_reason else None,
            'termination_time': self.termination_time,
            'detection_results': self.detection_results.copy()
        }


class MetricsCalculator:
    """
    Quality 指标计算器
    """
    
    def __init__(self, target_position: np.ndarray = None):
        """
        参数:
            target_position: 目标位置 [x, y, z]
        """
        self.target_position = target_position
    
    def compute_all(self,
                    positions: np.ndarray,
                    ref_positions: np.ndarray,
                    controls: np.ndarray,
                    euler: np.ndarray,
                    times: np.ndarray,
                    cross_track_errors: np.ndarray) -> Dict[str, float]:
        """
        计算所有 Quality 指标
        
        参数:
            positions: 实际位置序列 (N, 3)
            ref_positions: 参考位置序列 (N, 3)
            controls: 控制量序列 (N, 2) [delta_left, delta_right]
            euler: 姿态角序列 (N, 3) [roll, pitch, yaw]
            times: 时间序列 (N,)
            cross_track_errors: 横向误差序列 (N,)
        
        返回:
            指标字典
        """
        metrics = {}
        
        # === 跟踪误差指标 ===
        # ADE (Average Displacement Error)
        displacements = np.linalg.norm(positions - ref_positions, axis=1)
        metrics['ADE'] = float(np.mean(displacements))
        
        # RMSE
        metrics['RMSE'] = float(np.sqrt(np.mean(displacements**2)))
        
        # Cross-track 专用
        metrics['mean_cross_track_error'] = float(np.mean(np.abs(cross_track_errors)))
        metrics['max_cross_track_error'] = float(np.max(np.abs(cross_track_errors)))
        
        # === 终点误差指标 ===
        if self.target_position is not None:
            final_pos = positions[-1]
            # FDE (Final Displacement Error) - 3D
            metrics['FDE'] = float(np.linalg.norm(final_pos - self.target_position))
            # 水平误差
            metrics['FDE_horizontal'] = float(np.linalg.norm(
                final_pos[:2] - self.target_position[:2]))
            # 垂直误差
            metrics['FDE_vertical'] = float(abs(final_pos[2] - self.target_position[2]))
        
        # === 控制平滑性指标 ===
        if len(controls) > 1:
            # delta_u_sum: 控制量变化总和
            delta_controls = np.diff(controls, axis=0)
            metrics['delta_u_sum'] = float(np.sum(np.abs(delta_controls)))
            
            # 控制量均值（反映平均操纵强度）
            metrics['mean_control_effort'] = float(np.mean(np.abs(controls)))
        
        # === 姿态指标 ===
        metrics['max_roll'] = float(np.max(np.abs(euler[:, 0])))
        metrics['max_pitch'] = float(np.max(np.abs(euler[:, 1])))
        
        # 偏航角速率（需要从 yaw 序列计算）
        if len(times) > 1:
            dt = np.diff(times)
            yaw = euler[:, 2]
            # 处理角度跳变
            yaw_diff = np.diff(yaw)
            yaw_diff = np.where(yaw_diff > np.pi, yaw_diff - 2*np.pi, yaw_diff)
            yaw_diff = np.where(yaw_diff < -np.pi, yaw_diff + 2*np.pi, yaw_diff)
            yaw_rate = yaw_diff / dt
            metrics['max_yaw_rate'] = float(np.max(np.abs(yaw_rate)))
        
        # === 饱和指标 ===
        # 定义饱和：控制量接近 0 或 1
        saturated = ((np.abs(controls - 1.0) < 0.01) | (np.abs(controls) < 0.01))
        # 任一通道饱和即计入
        saturation_steps = np.any(saturated, axis=1)
        metrics['saturation_ratio'] = float(np.mean(saturation_steps))
        
        # === 时间指标 ===
        metrics['flight_time'] = float(times[-1] - times[0])
        
        return metrics
    
    def compute_summary_stats(self, metrics_list: List[Dict]) -> Dict[str, Any]:
        """
        计算多次运行的统计量
        
        参数:
            metrics_list: 多次运行的指标列表
        
        返回:
            统计量字典（均值、标准差、最小值、最大值等）
        """
        if not metrics_list:
            return {}
        
        # 收集所有数值型指标的键
        numeric_keys = [k for k, v in metrics_list[0].items() 
                       if isinstance(v, (int, float))]
        
        stats = {}
        for key in numeric_keys:
            values = [m[key] for m in metrics_list if key in m]
            if values:
                stats[key] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'median': float(np.median(values))
                }
        
        return stats
