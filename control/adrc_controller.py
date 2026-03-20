"""
自抗扰控制器 (ADRC) 模块

ADRC 核心组件:
- TD (Tracking Differentiator): 跟踪微分器，安排过渡过程
- ESO (Extended State Observer): 扩展状态观测器，估计状态和扰动
- NLSEF (Nonlinear State Error Feedback): 非线性状态误差反馈控制律

应用于翼伞无人机的航向和下降率控制
"""

import os
import sys
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass, field

# 添加项目根目录到路径
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


# ============================================================
#                     基础组件
# ============================================================

def fhan(x1: float, x2: float, r: float, h: float) -> float:
    """
    最速控制综合函数 (韩京清)
    
    参数:
        x1: 位置误差
        x2: 速度误差
        r: 快速因子
        h: 滤波因子 (采样周期)
    
    返回:
        最速控制输出
    """
    d = r * h
    d0 = h * d
    y = x1 + h * x2
    a0 = np.sqrt(d**2 + 8 * r * abs(y))
    
    if abs(y) > d0:
        a = x2 + (a0 - d) / 2 * np.sign(y)
    else:
        a = x2 + y / h
    
    if abs(a) > d:
        return -r * np.sign(a)
    else:
        return -r * a / d


def fal(e: float, alpha: float, delta: float) -> float:
    """
    非线性函数 fal (韩京清)
    
    参数:
        e: 误差输入
        alpha: 非线性因子 (0 < alpha < 1 增强小误差增益)
        delta: 线性区间宽度
    
    返回:
        非线性输出
    """
    if abs(e) <= delta:
        return e / (delta ** (1 - alpha))
    else:
        return abs(e) ** alpha * np.sign(e)


# ============================================================
#                     跟踪微分器 (TD)
# ============================================================

class TD:
    """
    跟踪微分器 (Tracking Differentiator)
    
    功能: 
    - 安排过渡过程，避免阶跃给定带来的超调
    - 提取输入信号的微分信号
    """
    
    def __init__(self, r: float = 100.0, h: float = 0.01):
        """
        参数:
            r: 快速因子，越大跟踪越快
            h: 滤波因子，通常取采样周期
        """
        self.r = r
        self.h = h
        
        # 状态: v1跟踪输入, v2是v1的微分
        self.v1 = 0.0
        self.v2 = 0.0
    
    def reset(self, v0: float = 0.0):
        """重置状态"""
        self.v1 = v0
        self.v2 = 0.0
    
    def update(self, v: float, dt: float) -> Tuple[float, float]:
        """
        更新跟踪微分器
        
        参数:
            v: 输入信号 (目标值，对于航向角需要归一化)
            dt: 时间步长
        
        返回:
            (v1, v2): 跟踪值和其微分
        """
        # 归一化输入（如果是航向角）
        v = TD._wrap_angle(v)
        
        # 计算误差（归一化到 [-π, π]）
        error = TD._wrap_angle(self.v1 - v)
        
        fh = fhan(error, self.v2, self.r, self.h)
        
        self.v1 = self.v1 + dt * self.v2
        self.v2 = self.v2 + dt * fh
        
        # 归一化v1（航向角）
        self.v1 = TD._wrap_angle(self.v1)
        
        return self.v1, self.v2
    
    @staticmethod
    def _wrap_angle(angle: float) -> float:
        """将角度归一化到 [-pi, pi]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle


# ============================================================
#                     扩展状态观测器 (ESO)
# ============================================================

class ESO:
    """
    扩展状态观测器 (Extended State Observer)
    
    功能:
    - 实时估计系统状态
    - 估计系统总扰动 (内部不确定性 + 外部扰动)
    """
    
    def __init__(self, order: int = 2, beta: np.ndarray = None, 
                 alpha: np.ndarray = None, delta: float = 0.01):
        """
        参数:
            order: 观测器阶数 (2 或 3)
            beta: 观测器增益向量 [beta1, beta2, beta3]
            alpha: 非线性因子向量
            delta: 线性区间宽度
        """
        self.order = order
        self.delta = delta
        
        # 默认参数 (二阶系统)
        if beta is None:
            if order == 2:
                self.beta = np.array([100.0, 300.0, 1000.0])
            else:
                self.beta = np.array([100.0, 300.0, 1000.0])
        else:
            self.beta = np.array(beta)
        
        if alpha is None:
            self.alpha = np.array([0.5, 0.25, 0.125])
        else:
            self.alpha = np.array(alpha)
        
        # 状态: z1估计输出, z2估计速度, z3估计扰动
        self.z = np.zeros(3)
    
    def reset(self, z0: np.ndarray = None):
        """重置状态"""
        if z0 is not None:
            self.z = np.array(z0)
        else:
            self.z = np.zeros(3)
    
    def update(self, y: float, u: float, b0: float, dt: float) -> np.ndarray:
        """
        更新扩展状态观测器
        
        参数:
            y: 系统输出 (测量值)
            u: 控制输入
            b0: 控制增益估计值
            dt: 时间步长
        
        返回:
            z: 状态估计 [z1, z2, z3] = [y估计, y'估计, 扰动估计]
        """
        e = self.z[0] - y  # 估计误差
        
        # 非线性ESO
        fe1 = fal(e, self.alpha[0], self.delta)
        fe2 = fal(e, self.alpha[1], self.delta)
        fe3 = fal(e, self.alpha[2], self.delta)
        
        # 状态更新
        dz1 = self.z[1] - self.beta[0] * fe1
        dz2 = self.z[2] - self.beta[1] * fe2 + b0 * u
        dz3 = -self.beta[2] * fe3
        
        self.z[0] += dt * dz1
        self.z[1] += dt * dz2
        self.z[2] += dt * dz3
        
        return self.z.copy()


class LinearESO:
    """
    线性扩展状态观测器 (LESO)
    
    相比非线性ESO更易调参，适合初步调试
    """
    
    def __init__(self, omega_o: float = 50.0, order: int = 2):
        """
        参数:
            omega_o: 观测器带宽
            order: 系统阶数 (1 或 2)
        """
        self.omega_o = omega_o
        self.order = order
        
        # 根据带宽计算增益 (极点配置)
        if order == 1:
            self.beta = np.array([2*omega_o, omega_o**2])
            self.z = np.zeros(2)
        else:
            self.beta = np.array([3*omega_o, 3*omega_o**2, omega_o**3])
            self.z = np.zeros(3)
    
    def reset(self, z0: np.ndarray = None):
        """重置状态"""
        if z0 is not None:
            self.z = np.array(z0)
        else:
            self.z = np.zeros(len(self.beta))
    
    def update(self, y: float, u: float, b0: float, dt: float) -> np.ndarray:
        """更新LESO"""
        # 归一化输入y和状态z[0]（如果是航向角）
        # 注意：这里假设y是航向角，对于其他状态可能不需要归一化
        y_normalized = LinearESO._wrap_angle(y)
        z0_normalized = LinearESO._wrap_angle(self.z[0])
        
        # 计算归一化后的误差
        e = z0_normalized - y_normalized
        # 归一化误差到 [-π, π]
        e = LinearESO._wrap_angle(e)
        
        if self.order == 1:
            dz1 = self.z[1] - self.beta[0] * e + b0 * u
            dz2 = -self.beta[1] * e
            
            self.z[0] += dt * dz1
            self.z[1] += dt * dz2
        else:
            dz1 = self.z[1] - self.beta[0] * e
            dz2 = self.z[2] - self.beta[1] * e + b0 * u
            dz3 = -self.beta[2] * e
            
            self.z[0] += dt * dz1
            self.z[1] += dt * dz2
            self.z[2] += dt * dz3
        
        # 归一化z[0]（航向角估计）
        self.z[0] = LinearESO._wrap_angle(self.z[0])
        
        return self.z.copy()
    
    @staticmethod
    def _wrap_angle(angle: float) -> float:
        """将角度归一化到 [-pi, pi]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle


# ============================================================
#                     非线性状态误差反馈 (NLSEF)
# ============================================================

class NLSEF:
    """
    非线性状态误差反馈控制律
    """
    
    def __init__(self, kp: float = 10.0, kd: float = 5.0, 
                 alpha1: float = 0.75, alpha2: float = 1.25, 
                 delta: float = 0.01):
        """
        参数:
            kp: 比例增益
            kd: 微分增益
            alpha1: 位置误差非线性因子
            alpha2: 速度误差非线性因子
            delta: 线性区间宽度
        """
        self.kp = kp
        self.kd = kd
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.delta = delta
    
    def compute(self, e1: float, e2: float) -> float:
        """
        计算控制量
        
        参数:
            e1: 位置误差 (参考 - 估计)
            e2: 速度误差 (参考微分 - 估计微分)
        
        返回:
            u0: 控制量 (不含扰动补偿)
        """
        u0 = self.kp * fal(e1, self.alpha1, self.delta) + \
             self.kd * fal(e2, self.alpha2, self.delta)
        return u0


class LinearSEF:
    """
    线性状态误差反馈 (PD控制)
    """
    
    def __init__(self, kp: float = 10.0, kd: float = 5.0):
        self.kp = kp
        self.kd = kd
    
    def compute(self, e1: float, e2: float) -> float:
        return self.kp * e1 + self.kd * e2


# ============================================================
#                     完整ADRC控制器
# ============================================================

class ADRC:
    """
    自抗扰控制器 (完整版)
    
    结构: TD + ESO + NLSEF + 扰动补偿
    """
    
    def __init__(self, 
                 # TD参数
                 td_r: float = 100.0,
                 td_h: float = 0.01,
                 # ESO参数
                 eso_omega: float = 50.0,
                 eso_order: int = 2,
                 use_linear_eso: bool = True,
                 # NLSEF参数
                 kp: float = 10.0,
                 kd: float = 5.0,
                 use_linear_sef: bool = False,
                 # 系统参数
                 b0: float = 1.0,
                 # 输出限幅
                 u_min: float = -1.0,
                 u_max: float = 1.0):
        """
        参数:
            td_r: TD快速因子
            td_h: TD滤波因子
            eso_omega: ESO带宽
            eso_order: ESO阶数
            use_linear_eso: 是否使用线性ESO
            kp, kd: 控制增益
            use_linear_sef: 是否使用线性SEF
            b0: 控制增益估计
            u_min, u_max: 输出限幅
        """
        # 跟踪微分器
        self.td = TD(r=td_r, h=td_h)
        
        # 扩展状态观测器
        if use_linear_eso:
            self.eso = LinearESO(omega_o=eso_omega, order=eso_order)
        else:
            self.eso = ESO(order=eso_order)
        
        # 状态误差反馈
        if use_linear_sef:
            self.sef = LinearSEF(kp=kp, kd=kd)
        else:
            self.sef = NLSEF(kp=kp, kd=kd)
        
        self.b0 = b0
        self.u_min = u_min
        self.u_max = u_max
        
        # 上一次控制输出
        self.u_last = 0.0
    
    def reset(self, y0: float = 0.0):
        """重置控制器状态"""
        # 归一化初始值
        y0 = self._wrap_angle_static(y0)
        self.td.reset(y0)
        self.eso.reset()
        self.u_last = 0.0
    
    @staticmethod
    def _wrap_angle_static(angle: float) -> float:
        """将角度归一化到 [-pi, pi]（静态方法）"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
    
    def update(self, ref: float, y: float, dt: float) -> float:
        """
        更新控制器
        
        参数:
            ref: 参考值（航向角，rad）
            y: 系统输出（当前航向角，rad）
            dt: 时间步长
        
        返回:
            u: 控制输出
        """
        # 归一化航向角到 [-π, π]（避免数值爆炸）
        ref = self._wrap_angle_static(ref)
        y = self._wrap_angle_static(y)
        
        # 1. 跟踪微分器: 安排过渡过程
        v1, v2 = self.td.update(ref, dt)
        
        # 归一化TD输出（v1是航向角，也需要归一化）
        v1 = self._wrap_angle_static(v1)
        
        # 2. 扩展状态观测器: 估计状态和扰动
        z = self.eso.update(y, self.u_last, self.b0, dt)
        z1, z2 = z[0], z[1]
        z3 = z[2] if len(z) > 2 else 0.0
        
        # 归一化ESO状态（z1是航向角估计）
        z1_normalized = self._wrap_angle_static(z1)
        
        # 更新ESO内部状态，避免累积异常值
        if abs(z1 - z1_normalized) > 0.01:  # 如果归一化后有显著变化
            self.eso.z[0] = z1_normalized
            z1 = z1_normalized
        
        # 3. 计算误差（使用归一化后的值）
        e1 = v1 - z1  # 位置误差
        e2 = v2 - z2  # 速度误差
        
        # 归一化误差 e1（航向误差）
        e1 = self._wrap_angle_static(e1)
        
        # 4. 非线性状态误差反馈
        u0 = self.sef.compute(e1, e2)
        
        # 5. 扰动补偿
        u = (u0 - z3) / self.b0
        
        # 6. 限幅
        u = np.clip(u, self.u_min, self.u_max)
        
        self.u_last = u
        return u


# ============================================================
#                     翼伞ADRC控制器
# ============================================================

@dataclass
class ControlOutput:
    """控制输出数据结构"""
    delta_left: float = 0.0        # 左操纵绳偏转 [0, 1]
    delta_right: float = 0.0       # 右操纵绳偏转 [0, 1]
    delta_symmetric: float = 0.0   # 对称偏转 (下降率控制) [0, 1]
    delta_asymmetric: float = 0.0  # 非对称偏转 (航向控制) [-1, 1]
    heading_error: float = 0.0     # 航向误差 (rad)
    cross_track_error: float = 0.0 # 横向误差 (m)
    along_track_error: float = 0.0 # 纵向误差 (m)
    altitude_error: float = 0.0    # 高度误差 (m)
    glide_ratio_required: float = 0.0  # 所需滑翔比
    glide_ratio_current: float = 0.0   # 当前滑翔比
    ref_heading: float = 0.0       # 参考航向 (rad)
    ref_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    ref_position_closest: np.ndarray = field(default_factory=lambda: np.zeros(3))


@dataclass
class GuidancePoint:
    """Interpolated trajectory point used by the shared guidance layer."""
    index: int = 0
    s: float = 0.0
    alpha: float = 0.0
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    heading: float = 0.0
    curvature: float = 0.0


class ParafoilADRCController:
    """
    翼伞ADRC轨迹跟踪控制器

    控制通道:
    - 航向控制: 通过差动操纵绳控制偏航 (非对称偏转 d_a)
    - 下降率控制: 通过对称操纵绳控制下降率 (对称偏转 d_s)
    """

    def __init__(self,
                 # 航向控制参数
                 heading_kp: float = 2.0,
                 heading_kd: float = 0.5,
                 heading_eso_omega: float = 20.0,
                 heading_td_r: float = 30.0,
                 heading_b0: float = 0.5,
                 # 横向误差控制参数
                 lateral_kp: float = 0.01,
                 lateral_kd: float = 0.005,
                 # 下降率控制参数
                 glide_ratio_natural: float = 6.48,   # 自然滑翔比 (无对称偏转)
                 glide_ratio_min: float = 2.47,       # 最小滑翔比 (最大对称偏转)
                 descent_kp: float = 0.5,             # 下降率控制增益
                 descent_margin: float = 1.2,         # 滑翔比余量系数
                 # 系统参数
                 reference_speed: float = 12.0,
                 min_turn_radius: float = 50.0,
                 lookahead_distance: float = 50.0,
                 lookahead_min_distance: float = 35.0,
                 lookahead_max_distance: float = 120.0,
                 lookahead_error_scale: float = 0.012,
                 closest_search_window: int = 80,
                 closest_reacquire_distance: float = 45.0,
                 closest_reacquire_window: int = 600,
                 closest_backtrack_window: int = 6,
                 cross_track_softening: float = 6.0,
                 max_cross_track_heading_correction: float = np.radians(30.0),
                 # 输出限制
                 max_deflection: float = 1.0,
                 dt: float = 0.01):
        """
        参数:
            heading_kp: 航向比例增益 (越大响应越快，太大震荡)
            heading_kd: 航向微分增益 (增加可抑制震荡)
            heading_eso_omega: ESO带宽 (越大扰动估计越快，太大噪声敏感)
            heading_td_r: TD快速因子 (越小参考信号过渡越平滑)
            heading_b0: 控制效能估计值 (翼伞实际偏航加速度灵敏度，过大→控制效能不足)
            lateral_kp: 横向误差增益 (越大路径跟踪越紧)
            lateral_kd: 横向误差微分增益
            glide_ratio_natural: 自然滑翔比，无对称偏转时的L/D
            glide_ratio_min: 最小滑翔比，最大对称偏转时的L/D
            descent_kp: 下降率控制增益
            descent_margin: 滑翔比余量系数 (>1表示保守，提前拉绳)
            reference_speed: 参考飞行速度 (m/s)
            min_turn_radius: 最小转弯半径 (m)
            lookahead_distance: 前视距离 (m，越大转弯越平滑)
            max_deflection: 最大操纵绳偏转量 [0,1]
            dt: 控制周期 (s)
        """
        self.reference_speed = reference_speed
        self.min_turn_radius = min_turn_radius
        self.lookahead_distance = lookahead_distance
        self.lookahead_min_distance = min(lookahead_min_distance, lookahead_max_distance)
        self.lookahead_max_distance = max(lookahead_min_distance, lookahead_max_distance)
        self.lookahead_error_scale = max(0.0, lookahead_error_scale)
        self.closest_search_window = max(10, int(closest_search_window))
        self.closest_reacquire_distance = max(1.0, closest_reacquire_distance)
        self.closest_reacquire_window = max(
            self.closest_search_window,
            int(closest_reacquire_window)
        )
        self.closest_backtrack_window = max(0, int(closest_backtrack_window))
        self.cross_track_softening = max(0.1, cross_track_softening)
        self.max_cross_track_heading_correction = abs(
            max_cross_track_heading_correction
        )
        self.max_deflection = max_deflection
        self.dt = dt

        # 下降率控制参数
        self.glide_ratio_natural = glide_ratio_natural
        self.glide_ratio_min = glide_ratio_min
        self.descent_kp = descent_kp
        self.descent_margin = descent_margin
        
        # 航向ADRC控制器
        # b0: 控制效能估计值，对应翼伞实际偏航角加速度对操纵绳偏转的灵敏度
        # 翼伞最大偏航率仅 ~6.5°/s，Cnda=-0.02（小），Cnr=-0.14（强阻尼）
        # 实际 b0 ≈ 0.3~0.8，过大会导致有效控制增益 kp/b0 不足，跟踪迟缓
        self.heading_adrc = ADRC(
            td_r=heading_td_r,    # TD快速因子：越小过渡越平滑
            td_h=dt,
            eso_omega=heading_eso_omega,
            eso_order=2,
            use_linear_eso=True,
            kp=heading_kp,
            kd=heading_kd,
            use_linear_sef=True,  # 线性SEF，响应更直接
            b0=heading_b0,
            u_min=-max_deflection,
            u_max=max_deflection
        )
        
        # 横向误差控制器 (简单PD)
        self.lateral_kp = lateral_kp
        self.lateral_kd = lateral_kd
        self.lateral_error_last = 0.0
        
        # 轨迹跟踪状态
        self.trajectory = None
        self.current_index = 0
        self.current_progress_s = 0.0
        self.last_heading_ref = None
        self._trajectory_positions = np.zeros((0, 3))
        self._trajectory_xy = np.zeros((0, 2))
        self._trajectory_headings = np.zeros(0)
        self._trajectory_curvatures = np.zeros(0)
        self._trajectory_segment_lengths = np.zeros(0)
        self._trajectory_arc_lengths = np.zeros(0)
        
        # 调试模式
        self.debug = False
        self.debug_counter = 0
    
    def set_trajectory(self, trajectory):
        """
        设置要跟踪的轨迹
        
        参数:
            trajectory: Trajectory 对象 (来自 planning.trajectory)
        """
        self.trajectory = trajectory
        self._build_trajectory_cache()
        self.reset()
    
    def reset(self):
        """重置控制器"""
        self.heading_adrc.reset()
        self.lateral_error_last = 0.0
        self.current_index = 0
        self.current_progress_s = 0.0
        self.last_heading_ref = None
        self.debug_counter = 0
    
    def set_debug(self, enabled: bool = True):
        """启用/禁用调试模式"""
        self.debug = enabled

    def _build_trajectory_cache(self):
        """Pre-compute arrays used by the shared guidance layer."""
        if self.trajectory is None or len(self.trajectory) == 0:
            self._trajectory_positions = np.zeros((0, 3))
            self._trajectory_xy = np.zeros((0, 2))
            self._trajectory_headings = np.zeros(0)
            self._trajectory_curvatures = np.zeros(0)
            self._trajectory_segment_lengths = np.zeros(0)
            self._trajectory_arc_lengths = np.zeros(0)
            return

        self._trajectory_positions = np.array(
            [pt.position for pt in self.trajectory],
            dtype=float,
        )
        self._trajectory_xy = self._trajectory_positions[:, :2]
        self._trajectory_headings = np.array(
            [pt.heading for pt in self.trajectory],
            dtype=float,
        )
        self._trajectory_curvatures = np.array(
            [pt.curvature for pt in self.trajectory],
            dtype=float,
        )

        if len(self.trajectory) > 1:
            diffs = np.diff(self._trajectory_xy, axis=0)
            self._trajectory_segment_lengths = np.linalg.norm(diffs, axis=1)
            self._trajectory_arc_lengths = np.concatenate((
                [0.0],
                np.cumsum(self._trajectory_segment_lengths),
            ))
        else:
            self._trajectory_segment_lengths = np.zeros(0)
            self._trajectory_arc_lengths = np.zeros(1)

    def _sample_guidance_point(self, s: float) -> GuidancePoint:
        """Sample an interpolated point along the trajectory arc length."""
        if self.trajectory is None or len(self.trajectory) == 0:
            return GuidancePoint()

        if len(self.trajectory) == 1:
            pt = self.trajectory[0]
            return GuidancePoint(
                index=0,
                s=0.0,
                alpha=0.0,
                position=pt.position.copy(),
                heading=pt.heading,
                curvature=pt.curvature,
            )

        s = float(np.clip(s, 0.0, self._trajectory_arc_lengths[-1]))
        if s >= self._trajectory_arc_lengths[-1]:
            pt = self.trajectory[-1]
            return GuidancePoint(
                index=len(self.trajectory) - 1,
                s=self._trajectory_arc_lengths[-1],
                alpha=0.0,
                position=pt.position.copy(),
                heading=pt.heading,
                curvature=pt.curvature,
            )

        idx = int(np.searchsorted(self._trajectory_arc_lengths, s, side='right') - 1)
        idx = int(np.clip(idx, 0, len(self.trajectory) - 2))
        s0 = self._trajectory_arc_lengths[idx]
        seg_len = self._trajectory_segment_lengths[idx]
        alpha = 0.0 if seg_len <= 1e-6 else float(np.clip((s - s0) / seg_len, 0.0, 1.0))

        position = (
            self._trajectory_positions[idx]
            + alpha * (self._trajectory_positions[idx + 1] - self._trajectory_positions[idx])
        )
        heading = self._wrap_angle(
            self._trajectory_headings[idx]
            + alpha * self._wrap_angle(
                self._trajectory_headings[idx + 1] - self._trajectory_headings[idx]
            )
        )
        curvature = (
            self._trajectory_curvatures[idx]
            + alpha * (self._trajectory_curvatures[idx + 1] - self._trajectory_curvatures[idx])
        )
        return GuidancePoint(
            index=idx,
            s=s,
            alpha=alpha,
            position=position,
            heading=heading,
            curvature=curvature,
        )

    def _search_projection_window(
        self,
        current_xy: np.ndarray,
        start_idx: int,
        end_idx: int,
    ) -> Tuple[GuidancePoint, float]:
        """Find the closest projected point on a local trajectory window."""
        if self.trajectory is None or len(self.trajectory) == 0:
            return GuidancePoint(), np.inf

        n_points = len(self.trajectory)
        if n_points == 1:
            point = self._sample_guidance_point(0.0)
            return point, float(np.linalg.norm(current_xy - point.position[:2]))

        start_idx = int(np.clip(start_idx, 0, n_points - 1))
        end_idx = int(np.clip(end_idx, 0, n_points - 1))
        if end_idx < start_idx:
            end_idx = start_idx

        best_point = self._sample_guidance_point(self.current_progress_s)
        best_dist = float(np.linalg.norm(current_xy - best_point.position[:2]))

        for idx in {start_idx, end_idx}:
            point = self._sample_guidance_point(self._trajectory_arc_lengths[idx])
            dist = float(np.linalg.norm(current_xy - point.position[:2]))
            if dist < best_dist:
                best_dist = dist
                best_point = point

        for idx in range(start_idx, end_idx):
            p0 = self._trajectory_xy[idx]
            p1 = self._trajectory_xy[idx + 1]
            seg = p1 - p0
            seg_len_sq = float(np.dot(seg, seg))
            if seg_len_sq <= 1e-10:
                alpha = 0.0
                proj_xy = p0
            else:
                alpha = float(np.clip(np.dot(current_xy - p0, seg) / seg_len_sq, 0.0, 1.0))
                proj_xy = p0 + alpha * seg

            dist = float(np.linalg.norm(current_xy - proj_xy))
            if dist < best_dist:
                s = self._trajectory_arc_lengths[idx] + alpha * self._trajectory_segment_lengths[idx]
                best_dist = dist
                best_point = self._sample_guidance_point(s)

        return best_point, best_dist

    def compute_symmetric_deflection(self,
                                      current_pos: np.ndarray,
                                      current_vel: np.ndarray,
                                      target_pos: np.ndarray) -> Tuple[float, float, float]:
        """
        计算对称偏转量用于下降率控制

        基于当前位置和目标位置（前视点），计算所需的滑翔比，
        然后映射到对称偏转量。

        参数:
            current_pos: 当前位置 [x, y, z]
            current_vel: 当前速度 [vx, vy, vz]
            target_pos: 目标位置（前视点） [x, y, z]

        返回:
            (delta_s, glide_ratio_required, glide_ratio_current):
                delta_s: 对称偏转量 [0, max_deflection]
                glide_ratio_required: 到达目标所需的滑翔比
                glide_ratio_current: 当前实际滑翔比
        """
        if self.trajectory is None or len(self.trajectory) == 0:
            return 0.0, 0.0, 0.0

        # 计算当前滑翔比 (水平速度 / 下降速度)
        v_horizontal = np.sqrt(current_vel[0]**2 + current_vel[1]**2)
        v_vertical = -current_vel[2]  # 下降为正
        if v_vertical > 0.1:
            glide_ratio_current = v_horizontal / v_vertical
        else:
            glide_ratio_current = self.glide_ratio_natural

        # 新策略：基于高度误差控制下降率，而不是追求某个滑翔比
        # 计算高度误差：当前高度 - 前视点高度
        altitude_error = current_pos[2] - target_pos[2]

        # 将高度误差转换为期望的下降率调整
        # 高度误差为正（飞得太高）→ 增加下降率 → 降低滑翔比
        # 高度误差为负（飞得太低）→ 减少下降率 → 增加滑翔比

        # 基准滑翔比（不拉绳时的自然状态）
        base_glide_ratio = self.glide_ratio_natural

        # 根据高度误差调整目标滑翔比
        # 每6m高度误差，调整滑翔比±1 (24m误差即请求最大下降)
        glide_ratio_adjustment = -altitude_error / 6.0
        glide_ratio_required = base_glide_ratio + glide_ratio_adjustment

        # 限制在合理范围内
        glide_ratio_required = np.clip(glide_ratio_required,
                                       self.glide_ratio_min,
                                       self.glide_ratio_natural * 1.2)

        # 应用余量系数 (更保守的下降)
        glide_ratio_target = glide_ratio_required / self.descent_margin

        # 计算需要的对称偏转量
        # 滑翔比从 glide_ratio_natural (d_s=0) 到 glide_ratio_min (d_s=max)
        # 线性映射: d_s = (glide_ratio_natural - glide_ratio_target) / (glide_ratio_natural - glide_ratio_min) * max_deflection

        if glide_ratio_target >= self.glide_ratio_natural:
            # 当前滑翔比已经够用或过低，不需要拉绳
            delta_s = 0.0
        elif glide_ratio_target <= self.glide_ratio_min:
            # 需要最大下降率
            delta_s = self.max_deflection
        else:
            # 线性插值
            ratio = (self.glide_ratio_natural - glide_ratio_target) / \
                    (self.glide_ratio_natural - self.glide_ratio_min)
            delta_s = ratio * self.max_deflection

        # 增加基于当前滑翔比误差的反馈控制
        glide_error = glide_ratio_current - glide_ratio_target
        delta_s_feedback = self.descent_kp * glide_error * 0.1  # 缩放系数

        delta_s = np.clip(delta_s + delta_s_feedback, 0, self.max_deflection)

        return delta_s, glide_ratio_required, glide_ratio_current
    
    def _find_closest_point_legacy_unused(self, current_pos: np.ndarray) -> GuidancePoint:
        """
        找到轨迹上距离当前位置最近的点索引 (2D水平面距离)

        两阶段搜索:
        1. 小窗口 (±20索引) 围绕期望位置搜索 → 快速、防止螺旋段跳跃
        2. 若距离 >30m，扩大搜索范围 → 处理初始偏差和大扰动
        """
        if self.trajectory is None or len(self.trajectory) == 0:
            return GuidancePoint()

        # Phase 1: 小窗口搜索（基于速度的期望前进量）
        expected_next = self.current_index + 1
        search_start = max(0, expected_next - 20)
        search_end = min(len(self.trajectory), expected_next + 20)

        min_dist = np.inf
        best_idx = self.current_index

        for i in range(search_start, search_end):
            dist = np.linalg.norm(current_pos[:2] - self.trajectory[i].position[:2])
            if dist < min_dist:
                min_dist = dist
                best_idx = i

        # Phase 2: 若匹配差（>30m），扩大搜索范围（仅向前搜索，禁止回退）
        if min_dist > 30.0:
            search_start2 = max(self.current_index, 0)
            search_end2 = min(len(self.trajectory), self.current_index + 500)
            for i in range(search_start2, search_end2):
                dist = np.linalg.norm(current_pos[:2] - self.trajectory[i].position[:2])
                if dist < min_dist:
                    min_dist = dist
                    best_idx = i

        # 强制单调递增：禁止轨迹索引回退
        # 在螺旋消高段，空间上距离最近的点可能是已经飞过的点，
        # 回退会导致控制器原地画圈，cross-track 持续增大直到触发 divergence
        best_idx = max(best_idx, self.current_index)

        return best_idx

    def _find_closest_point(self, current_pos: np.ndarray) -> GuidancePoint:
        """Project the vehicle onto nearby trajectory segments."""
        if self.trajectory is None or len(self.trajectory) == 0:
            return GuidancePoint()

        current_xy = current_pos[:2]
        search_start = max(0, self.current_index - self.closest_backtrack_window)
        search_end = min(
            len(self.trajectory) - 1,
            self.current_index + self.closest_search_window,
        )
        best_point, best_dist = self._search_projection_window(
            current_xy,
            search_start,
            search_end,
        )

        if best_dist > self.closest_reacquire_distance:
            reacquire_end = min(
                len(self.trajectory) - 1,
                self.current_index + self.closest_reacquire_window,
            )
            candidate_point, candidate_dist = self._search_projection_window(
                current_xy,
                search_start,
                reacquire_end,
            )
            if candidate_dist < best_dist:
                best_point = candidate_point
                best_dist = candidate_dist

        if best_dist > self.closest_reacquire_distance * 1.5:
            candidate_point, candidate_dist = self._search_projection_window(
                current_xy,
                0,
                len(self.trajectory) - 1,
            )
            if candidate_dist + 1e-6 < best_dist:
                best_point = candidate_point

        if self._trajectory_arc_lengths.size > 0:
            min_idx = max(0, self.current_index - self.closest_backtrack_window)
            min_s = self._trajectory_arc_lengths[min_idx]
            if best_point.s < min_s:
                best_point = self._sample_guidance_point(min_s)

        self.current_progress_s = best_point.s
        if self._trajectory_arc_lengths.size > 0:
            self.current_index = int(np.searchsorted(
                self._trajectory_arc_lengths,
                best_point.s,
                side='right',
            ) - 1)
            self.current_index = int(np.clip(
                self.current_index,
                0,
                len(self.trajectory) - 1,
            ))
        else:
            self.current_index = 0

        return best_point
    
    def _find_lookahead_point(
        self,
        closest_point: GuidancePoint,
        cross_track_error: float = 0.0
    ) -> GuidancePoint:
        """
        找到前视点 (Pure Pursuit风格，自适应前视距离)

        弯道自动缩短前视距离以提高跟踪精度，
        直线段使用标称前视距离以保持平滑性。
        """
        if self.trajectory is None or len(self.trajectory) == 0:
            return GuidancePoint()

        curvature = abs(closest_point.curvature)
        if curvature > 1e-3:
            curve_scale = max(0.45, 1.0 - min(curvature * self.min_turn_radius, 0.55))
        else:
            curve_scale = 1.0

        error_scale = 1.0 / (1.0 + self.lookahead_error_scale * abs(cross_track_error))
        effective_la = self.lookahead_distance * curve_scale * error_scale
        effective_la = float(np.clip(
            effective_la,
            self.lookahead_min_distance,
            self.lookahead_max_distance
        ))

        return self._sample_guidance_point(closest_point.s + effective_la)

        # 自适应前视距离：基于最近点的曲率
        # Pure Pursuit 理论：弯道时应缩短前视距离以减少切弯误差
        # 稳态横向误差 ≈ L_a² / (2R)，缩短 L_a 可显著降低弯道误差
        # 直线段可适当增大前视距离以保持平滑
        curvature = self.trajectory[closest_idx].curvature
        if curvature > 0.001:
            # 弯道：缩短前视距离，防止 Pure Pursuit 切弯
            # scale 在 [0.4, 1.0]，曲率越大（弯越紧）前视越短
            scale = max(0.4, 1.0 - min(curvature * self.min_turn_radius, 0.6))
            effective_la = self.lookahead_distance * scale
        else:
            effective_la = self.lookahead_distance

        # 从最近点开始，找到距离超过前视距离的点
        accumulated_dist = 0.0

        for i in range(closest_idx, len(self.trajectory) - 1):
            segment_length = np.linalg.norm(
                self.trajectory[i + 1].position - self.trajectory[i].position
            )
            accumulated_dist += segment_length

            if accumulated_dist >= effective_la:
                return self.trajectory[i + 1]

        # 如果没找到，返回轨迹终点
        return self.trajectory[-1]
    
    def compute_control(self, 
                        current_pos: np.ndarray,
                        current_heading: float,
                        current_heading_rate: float = 0.0,
                        current_speed: Optional[float] = None) -> ControlOutput:
        """
        计算控制输出 (基于已设置的轨迹)
        
        参数:
            current_pos: 当前位置 [x, y, z]
            current_heading: 当前航向 (rad)
            current_heading_rate: 当前航向角速度 (rad/s)
        
        返回:
            ControlOutput: 控制输出
        """
        output = ControlOutput()
        
        if self.trajectory is None or len(self.trajectory) == 0:
            return output

        closest_point = self._find_closest_point(current_pos)
        ref_dir = np.array([np.cos(closest_point.heading), np.sin(closest_point.heading)])
        to_vehicle = current_pos[:2] - closest_point.position[:2]
        cross_track_error = ref_dir[0] * to_vehicle[1] - ref_dir[1] * to_vehicle[0]
        along_track_error = np.dot(to_vehicle, ref_dir)

        lookahead_point = self._find_lookahead_point(
            closest_point,
            cross_track_error=cross_track_error
        )
        dx = lookahead_point.position[0] - current_pos[0]
        dy = lookahead_point.position[1] - current_pos[1]
        dist_to_lookahead = np.sqrt(dx**2 + dy**2)

        if dist_to_lookahead > 1.0:
            los_heading = np.arctan2(dy, dx)
        else:
            los_heading = closest_point.heading

        speed_for_guidance = max(
            current_speed if current_speed is not None else self.reference_speed,
            0.1
        )
        cross_track_rate = (cross_track_error - self.lateral_error_last) / max(self.dt, 1e-6)
        self.lateral_error_last = cross_track_error

        heading_correction = -np.arctan2(
            self.lateral_kp * cross_track_error,
            speed_for_guidance + self.cross_track_softening
        )
        heading_correction -= self.lateral_kd * cross_track_rate
        heading_correction = float(np.clip(
            heading_correction,
            -self.max_cross_track_heading_correction,
            self.max_cross_track_heading_correction
        ))
        target_heading = self._wrap_angle(los_heading + heading_correction)

        output.ref_position = closest_point.position.copy()
        output.ref_position_closest = closest_point.position.copy()
        output.ref_heading = target_heading
        output.heading_error = self._wrap_angle(target_heading - current_heading)
        output.altitude_error = closest_point.position[2] - current_pos[2]
        output.cross_track_error = cross_track_error
        output.along_track_error = along_track_error

        adjusted_heading_ref = target_heading
        adjusted_error = self._wrap_angle(adjusted_heading_ref - current_heading)
        if abs(adjusted_error) > np.radians(60):
            adjusted_heading_ref = (
                current_heading
                + np.sign(adjusted_error) * np.radians(60)
            )
        adjusted_heading_ref = self._wrap_angle(adjusted_heading_ref)

        delta_diff = self.heading_adrc.update(adjusted_heading_ref, current_heading, self.dt)
        output.delta_asymmetric = delta_diff

        if delta_diff > 0:
            delta_right_heading = delta_diff
            delta_left_heading = 0.0
        elif delta_diff < 0:
            delta_left_heading = -delta_diff
            delta_right_heading = 0.0
        else:
            delta_left_heading = 0.0
            delta_right_heading = 0.0

        delta_left = np.clip(delta_left_heading, 0, self.max_deflection)
        delta_right = np.clip(delta_right_heading, 0, self.max_deflection)
        output.delta_left = delta_left
        output.delta_right = delta_right

        if self.debug and self.debug_counter % 100 == 0:
            print(f"[鎺у埗鍣ㄨ皟璇昡 step={self.debug_counter}")
            print(f"  褰撳墠浣嶇疆: ({current_pos[0]:.1f}, {current_pos[1]:.1f}, {current_pos[2]:.1f})")
            print(f"  褰撳墠鑸悜: {np.degrees(current_heading):.1f}掳")
            print(f"  瀵煎紩鑸悜: {np.degrees(target_heading):.1f}掳")
            print(f"  鑸悜璇樊: {np.degrees(output.heading_error):.1f}掳")
            print(f"  妯悜璇樊: {cross_track_error:.2f}m")
            print(f"  妯悜淇: {np.degrees(heading_correction):.1f}掳")
            print(f"  鎺у埗杈撳嚭: {delta_diff:.3f}")
            print(f"  宸︾怀: {delta_left:.3f}, 鍙崇怀: {delta_right:.3f}")

        self.debug_counter += 1
        return output
        
        # ========== 1. 找到轨迹上的参考点 ==========
        self.current_index = self._find_closest_point(current_pos)
        closest_point = self.trajectory[self.current_index]

        # Pure Pursuit 前视点
        lookahead_point = self._find_lookahead_point(current_pos, self.current_index)
        if lookahead_point is None:
            lookahead_point = closest_point

        dx = lookahead_point.position[0] - current_pos[0]
        dy = lookahead_point.position[1] - current_pos[1]
        dist_to_lookahead = np.sqrt(dx**2 + dy**2)

        if dist_to_lookahead > 1.0:
            target_heading = np.arctan2(dy, dx)
        else:
            target_heading = closest_point.heading

        # ref_position 使用最近点（用于 ADE 等指标计算）
        output.ref_position = closest_point.position.copy()
        output.ref_position_closest = closest_point.position.copy()
        output.ref_heading = target_heading
        
        # ========== 2. 计算跟踪误差 ==========
        heading_error = self._wrap_angle(target_heading - current_heading)
        output.heading_error = heading_error

        # 高度误差
        output.altitude_error = closest_point.position[2] - current_pos[2]

        # 横向误差 (Cross-track error) — 仅用于指标记录和失败检测
        ref_dir = np.array([np.cos(closest_point.heading), np.sin(closest_point.heading)])
        to_vehicle = current_pos[:2] - closest_point.position[:2]
        cross_track_error = ref_dir[0] * to_vehicle[1] - ref_dir[1] * to_vehicle[0]
        output.cross_track_error = cross_track_error

        along_track_error = np.dot(to_vehicle, ref_dir)
        output.along_track_error = along_track_error

        # ========== 3. 航向控制 (ADRC + Pure Pursuit) ==========
        # Pure Pursuit 已通过几何关系隐式包含横向误差修正，
        # 无需额外的 lateral_kp/lateral_kd 补偿
        adjusted_heading_ref = target_heading

        # 限制航向参考与当前航向的偏差，防止控制器饱和
        adjusted_error = self._wrap_angle(adjusted_heading_ref - current_heading)
        if abs(adjusted_error) > np.radians(60):
            adjusted_heading_ref = current_heading + np.sign(adjusted_error) * np.radians(60)

        adjusted_heading_ref = self._wrap_angle(adjusted_heading_ref)
        
        # ADRC航向控制（确保输入都在 [-π, π] 范围内）
        delta_diff = self.heading_adrc.update(adjusted_heading_ref, current_heading, self.dt)
        
        # ========== 4. 转换为左右操纵绳偏转 ==========
        # delta_diff: ADRC输出，范围 [-max_deflection, max_deflection]
        # delta_diff > 0: 需要右转 → 拉右绳
        # delta_diff < 0: 需要左转 → 拉左绳
        # delta_diff = 0: 直飞 → 两绳都不拉

        # 翼伞控制逻辑:
        # - 对称偏转 d_s = min(left, right): 控制下降率
        # - 非对称偏转 d_a = left - right: 控制航向

        # 保存非对称偏转量
        output.delta_asymmetric = delta_diff

        # 非对称控制：只拉一侧（暂不考虑对称偏转）
        # 右转时: delta_right = delta_diff (正值), delta_left = 0
        # 左转时: delta_left = -delta_diff (正值), delta_right = 0
        # 直飞时: delta_left = delta_right = 0

        if delta_diff > 0:
            # 右转：只拉右绳
            delta_right_heading = delta_diff
            delta_left_heading = 0.0
        elif delta_diff < 0:
            # 左转：只拉左绳
            delta_left_heading = -delta_diff  # delta_diff是负值，所以用减号
            delta_right_heading = 0.0
        else:
            # 直飞：两绳都不拉
            delta_left_heading = 0.0
            delta_right_heading = 0.0

        # 合成最终偏转量
        # 对称偏转加到两侧（通过update方法设置，这里先设为0）
        # 实际的对称偏转在update方法中计算
        delta_left = delta_left_heading
        delta_right = delta_right_heading

        # 限幅到 [0, max_deflection]
        delta_left = np.clip(delta_left, 0, self.max_deflection)
        delta_right = np.clip(delta_right, 0, self.max_deflection)

        output.delta_left = delta_left
        output.delta_right = delta_right
        
        # 调试输出
        if self.debug and self.debug_counter % 100 == 0:
            print(f"[控制器调试] step={self.debug_counter}")
            print(f"  当前位置: ({current_pos[0]:.1f}, {current_pos[1]:.1f}, {current_pos[2]:.1f})")
            print(f"  当前航向: {np.degrees(current_heading):.1f}°")
            print(f"  Pure Pursuit航向: {np.degrees(target_heading):.1f}°")
            print(f"  航向误差: {np.degrees(heading_error):.1f}°")
            print(f"  横向误差: {cross_track_error:.2f}m")
            print(f"  ADRC输出: {delta_diff:.3f}")
            print(f"  左绳: {delta_left:.3f}, 右绳: {delta_right:.3f}")
            if abs(delta_diff) >= self.max_deflection * 0.95:
                print(f"  [警告] ADRC输出饱和!")
        
        self.debug_counter += 1
        
        return output
    
    def update(self,
               current_pos: np.ndarray,
               current_vel: np.ndarray,
               current_heading: float,
               t: float = None) -> ControlOutput:
        """
        轨迹跟踪更新 (主接口)

        参数:
            current_pos: 当前位置 [x, y, z]
            current_vel: 当前速度 [vx, vy, vz]
            current_heading: 当前航向 (rad)
            t: 当前时间 (可选，用于基于时间的跟踪)

        返回:
            ControlOutput: 控制输出
        """
        # 计算航向角速度 (如果有速度信息)
        current_speed = np.linalg.norm(current_vel[:2])
        heading_rate = 0.0

        # 1. 计算航向控制 (非对称偏转)
        output = self.compute_control(
            current_pos=current_pos,
            current_heading=current_heading,
            current_heading_rate=heading_rate,
            current_speed=current_speed
        )

        # 2. 计算下降率控制 (对称偏转)
        target_pos = output.ref_position_closest
        delta_s, glide_required, glide_current = self.compute_symmetric_deflection(
            current_pos=current_pos,
            current_vel=current_vel,
            target_pos=target_pos
        )


        # 3. 合成最终控制量
        # 对称偏转 d_s = min(left, right): 控制下降率
        # 非对称偏转 d_a = left - right: 控制航向
        # 
        # 关键：优先保证航向控制！
        # 对称偏转需要让出空间给非对称偏转

        delta_a = output.delta_asymmetric  # 航向控制的非对称分量
        delta_a_abs = abs(delta_a)

        # ====== 进度自适应对称/非对称偏转预算分配 ======
        # 根据飞行进度动态调整对称偏转(下降控制)的最低预算：
        #   前半程 (<0.5): 10% → 优先航向修正，下降有规划保障
        #   中段 (0.5~0.8): 20% → 适度平衡，航向仍是主要任务
        #   末段 (>0.8): 35% → 高度收敛逐渐优先
        #   末端下降 (>0.95): 55% → 全力下降收敛
        # [调参] 整体下调预算比例，给航向控制释放更多操纵余量
        progress = self.get_progress()
        if progress > 0.95:
            budget_ratio = 0.55
        elif progress > 0.8:
            budget_ratio = 0.35
        elif progress > 0.5:
            budget_ratio = 0.20
        else:
            budget_ratio = 0.10
        min_symmetric_budget = budget_ratio * self.max_deflection
        max_asymmetric = self.max_deflection - min_symmetric_budget
        delta_a_abs = min(delta_a_abs, max_asymmetric)      # 限制非对称上限
        delta_a = np.sign(delta_a) * delta_a_abs if delta_a != 0 else 0.0
        output.delta_asymmetric = delta_a                    # 更新实际使用值

        max_symmetric = self.max_deflection - delta_a_abs
        delta_s_limited = np.clip(delta_s, 0, max(0, max_symmetric))

        # 更新实际使用的对称偏转量
        output.delta_symmetric = delta_s_limited

        if delta_a >= 0:
            # 右转或直飞: d_a >= 0
            # 右绳拉更多
            delta_left = delta_s_limited
            delta_right = delta_s_limited + delta_a
        else:
            # 左转: d_a < 0
            # 左绳拉更多
            delta_left = delta_s_limited - delta_a  # delta_a是负的，所以用减号
            delta_right = delta_s_limited

        # 限幅（理论上不会超出，但保险起见）
        delta_left = np.clip(delta_left, 0, self.max_deflection)
        delta_right = np.clip(delta_right, 0, self.max_deflection)

        output.delta_left = delta_left
        output.delta_right = delta_right

        # 调试输出
        if self.debug and self.debug_counter % 100 == 1:  # 刚打印完航向信息后
            print(f"  [下降率控制] d_s_请求={delta_s:.3f}, d_s_实际={delta_s_limited:.3f}")
            print(f"    所需滑翔比: {glide_required:.1f}, 当前: {glide_current:.1f}")
            print(f"    d_a={delta_a:.3f}, 最终: L={delta_left:.3f}, R={delta_right:.3f}")

        return output
    
    def is_finished(self, current_pos: np.ndarray, threshold: float = 20.0) -> bool:
        """
        检查是否到达轨迹终点
        
        参数:
            current_pos: 当前位置
            threshold: 到达阈值 (m)
        
        返回:
            是否完成
        """
        if self.trajectory is None or len(self.trajectory) == 0:
            return True
        
        final_pos = self.trajectory[-1].position
        dist = np.linalg.norm(current_pos - final_pos)
        return dist < threshold
    
    def get_progress(self) -> float:
        """
        获取轨迹跟踪进度 [0, 1]
        """
        if self.trajectory is None or len(self.trajectory) == 0:
            return 1.0
        return self.current_index / (len(self.trajectory) - 1)
    
    @staticmethod
    def _wrap_angle(angle: float) -> float:
        """将角度归一化到 [-pi, pi]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
def _compute_symmetric_deflection_trackable(self,
                                            current_pos: np.ndarray,
                                            current_vel: np.ndarray,
                                            target_pos: np.ndarray) -> Tuple[float, float, float]:
    if self.trajectory is None or len(self.trajectory) == 0:
        return 0.0, 0.0, 0.0

    v_horizontal = np.sqrt(current_vel[0]**2 + current_vel[1]**2)
    v_vertical = -current_vel[2]
    if v_vertical > 0.1:
        glide_ratio_current = v_horizontal / v_vertical
    else:
        glide_ratio_current = self.glide_ratio_natural

    altitude_error = current_pos[2] - target_pos[2]
    remaining_xy = float(np.linalg.norm(self.trajectory[-1].position[:2] - current_pos[:2]))
    altitude_deadband = float(np.clip(0.03 * remaining_xy, 5.0, 22.0))
    effective_altitude_error = altitude_error - altitude_deadband

    if effective_altitude_error <= 0.0:
        glide_ratio_required = min(
            self.glide_ratio_natural * 1.05,
            self.glide_ratio_natural + abs(effective_altitude_error) / 18.0
        )
    else:
        glide_ratio_required = self.glide_ratio_natural - effective_altitude_error / 10.0

    glide_ratio_required = float(np.clip(
        glide_ratio_required,
        self.glide_ratio_min,
        self.glide_ratio_natural * 1.1
    ))
    glide_ratio_target = glide_ratio_required / max(self.descent_margin, 1e-6)

    if glide_ratio_target >= self.glide_ratio_natural:
        delta_s = 0.0
    elif glide_ratio_target <= self.glide_ratio_min:
        delta_s = self.max_deflection
    else:
        ratio = (self.glide_ratio_natural - glide_ratio_target) / \
                (self.glide_ratio_natural - self.glide_ratio_min)
        delta_s = ratio * self.max_deflection

    glide_error = glide_ratio_current - glide_ratio_target
    delta_s_feedback = self.descent_kp * glide_error * 0.05

    progress = self.get_progress()
    if progress > 0.95:
        delta_s_cap = 0.55 * self.max_deflection
    elif progress > 0.80:
        delta_s_cap = 0.32 * self.max_deflection
    elif progress > 0.55:
        delta_s_cap = 0.18 * self.max_deflection
    else:
        delta_s_cap = 0.08 * self.max_deflection

    if effective_altitude_error <= 0.0:
        delta_s = 0.0
    delta_s = np.clip(delta_s + delta_s_feedback, 0, min(self.max_deflection, delta_s_cap))

    return float(delta_s), glide_ratio_required, glide_ratio_current


def _update_trackable(self,
                      current_pos: np.ndarray,
                      current_vel: np.ndarray,
                      current_heading: float,
                      t: float = None) -> ControlOutput:
    current_speed = np.linalg.norm(current_vel[:2])
    heading_rate = 0.0

    output = self.compute_control(
        current_pos=current_pos,
        current_heading=current_heading,
        current_heading_rate=heading_rate,
        current_speed=current_speed
    )

    target_pos = output.ref_position_closest
    delta_s, glide_required, glide_current = self.compute_symmetric_deflection(
        current_pos=current_pos,
        current_vel=current_vel,
        target_pos=target_pos
    )

    delta_a = output.delta_asymmetric
    delta_a_abs = abs(delta_a)

    progress = self.get_progress()
    if progress > 0.95:
        budget_ratio = 0.38
    elif progress > 0.8:
        budget_ratio = 0.24
    elif progress > 0.5:
        budget_ratio = 0.12
    else:
        budget_ratio = 0.04

    min_symmetric_budget = budget_ratio * self.max_deflection
    max_asymmetric = self.max_deflection - min_symmetric_budget
    delta_a_abs = min(delta_a_abs, max_asymmetric)
    delta_a = np.sign(delta_a) * delta_a_abs if delta_a != 0 else 0.0
    output.delta_asymmetric = delta_a

    max_symmetric = self.max_deflection - delta_a_abs
    delta_s_limited = np.clip(delta_s, 0, max(0, max_symmetric))
    output.delta_symmetric = delta_s_limited

    if delta_a >= 0:
        delta_left = delta_s_limited
        delta_right = delta_s_limited + delta_a
    else:
        delta_left = delta_s_limited - delta_a
        delta_right = delta_s_limited

    delta_left = np.clip(delta_left, 0, self.max_deflection)
    delta_right = np.clip(delta_right, 0, self.max_deflection)
    output.delta_left = delta_left
    output.delta_right = delta_right

    if self.debug and self.debug_counter % 100 == 1:
        print(f"  [trackable-descent] d_s_req={delta_s:.3f}, d_s_used={delta_s_limited:.3f}")
        print(f"    glide req: {glide_required:.1f}, current: {glide_current:.1f}")
        print(f"    d_a={delta_a:.3f}, final L={delta_left:.3f}, R={delta_right:.3f}")

    return output


ParafoilADRCController.compute_symmetric_deflection = _compute_symmetric_deflection_trackable
ParafoilADRCController.update = _update_trackable


def _compute_symmetric_deflection_energy_guided(self,
                                                current_pos: np.ndarray,
                                                current_vel: np.ndarray,
                                                target_pos: np.ndarray) -> Tuple[float, float, float]:
    if self.trajectory is None or len(self.trajectory) == 0:
        return 0.0, 0.0, 0.0

    v_horizontal = np.sqrt(current_vel[0]**2 + current_vel[1]**2)
    v_vertical = -current_vel[2]
    if v_vertical > 0.1:
        glide_ratio_current = v_horizontal / v_vertical
    else:
        glide_ratio_current = self.glide_ratio_natural

    final_pos = self.trajectory[-1].position
    total_arc = float(self._trajectory_arc_lengths[-1]) if self._trajectory_arc_lengths.size > 0 else 0.0
    remaining_arc = max(total_arc - float(self.current_progress_s), 1.0)
    remaining_drop = max(float(current_pos[2] - final_pos[2]), 1.0)
    dist_to_goal = float(np.linalg.norm(current_pos[:2] - final_pos[:2]))
    if remaining_arc < 1.0 and dist_to_goal < 0.85 * self.min_turn_radius and remaining_drop > 25.0:
        remaining_arc = max(remaining_arc, 2.0 * np.pi * 1.1 * self.min_turn_radius)
    remaining_glide_required = remaining_arc / remaining_drop

    local_altitude_error = float(current_pos[2] - target_pos[2])
    local_deadband = float(np.clip(0.015 * remaining_arc, 4.0, 18.0))
    local_glide_required = self.glide_ratio_natural - (local_altitude_error - local_deadband) / 8.5

    progress = self.get_progress()
    blend = float(np.clip(0.40 + 0.35 * progress, 0.40, 0.78))
    glide_ratio_required = (
        (1.0 - blend) * local_glide_required
        + blend * remaining_glide_required
    )

    if progress > 0.82 and local_altitude_error > 6.0:
        glide_ratio_required = min(
            glide_ratio_required,
            self.glide_ratio_natural - local_altitude_error / 7.0
        )

    glide_ratio_required = np.clip(
        glide_ratio_required,
        self.glide_ratio_min + 0.02,
        self.glide_ratio_natural * 1.08
    )
    glide_ratio_target = glide_ratio_required / max(self.descent_margin, 1e-6)

    if glide_ratio_target >= self.glide_ratio_natural:
        delta_s = 0.0
    elif glide_ratio_target <= self.glide_ratio_min:
        delta_s = self.max_deflection
    else:
        ratio = (self.glide_ratio_natural - glide_ratio_target) / \
            (self.glide_ratio_natural - self.glide_ratio_min)
        delta_s = ratio * self.max_deflection

    glide_error = glide_ratio_current - glide_ratio_target
    delta_s_feedback = self.descent_kp * glide_error * 0.10
    if local_altitude_error > 0.0:
        delta_s_feedback += min(0.10, 0.0035 * local_altitude_error)
    elif local_altitude_error < -8.0:
        delta_s_feedback -= min(0.08, 0.0025 * abs(local_altitude_error))

    if progress > 0.95:
        delta_s_cap = 0.58 * self.max_deflection
    elif progress > 0.82:
        delta_s_cap = 0.40 * self.max_deflection
    elif progress > 0.55:
        delta_s_cap = 0.24 * self.max_deflection
    else:
        delta_s_cap = 0.12 * self.max_deflection

    if glide_ratio_target >= self.glide_ratio_natural and local_altitude_error <= 0.0:
        delta_s = 0.0
    delta_s = np.clip(delta_s + delta_s_feedback, 0, min(self.max_deflection, delta_s_cap))

    return float(delta_s), float(glide_ratio_required), float(glide_ratio_current)


def _update_energy_guided(self,
                          current_pos: np.ndarray,
                          current_vel: np.ndarray,
                          current_heading: float,
                          t: float = None) -> ControlOutput:
    current_speed = np.linalg.norm(current_vel[:2])
    heading_rate = 0.0

    output = self.compute_control(
        current_pos=current_pos,
        current_heading=current_heading,
        current_heading_rate=heading_rate,
        current_speed=current_speed
    )

    target_pos = output.ref_position_closest
    delta_s, glide_required, glide_current = self.compute_symmetric_deflection(
        current_pos=current_pos,
        current_vel=current_vel,
        target_pos=target_pos
    )

    delta_a = output.delta_asymmetric
    delta_a_abs = abs(delta_a)

    progress = self.get_progress()
    if progress > 0.95:
        budget_ratio = 0.52
    elif progress > 0.82:
        budget_ratio = 0.34
    elif progress > 0.55:
        budget_ratio = 0.18
    else:
        budget_ratio = 0.08

    min_symmetric_budget = budget_ratio * self.max_deflection
    max_asymmetric = self.max_deflection - min_symmetric_budget
    delta_a_abs = min(delta_a_abs, max_asymmetric)
    delta_a = np.sign(delta_a) * delta_a_abs if delta_a != 0 else 0.0
    output.delta_asymmetric = delta_a

    max_symmetric = self.max_deflection - delta_a_abs
    delta_s_limited = np.clip(delta_s, 0, max(0, max_symmetric))
    output.delta_symmetric = delta_s_limited

    if delta_a >= 0:
        delta_left = delta_s_limited
        delta_right = delta_s_limited + delta_a
    else:
        delta_left = delta_s_limited - delta_a
        delta_right = delta_s_limited

    delta_left = np.clip(delta_left, 0, self.max_deflection)
    delta_right = np.clip(delta_right, 0, self.max_deflection)
    output.delta_left = delta_left
    output.delta_right = delta_right

    if self.debug and self.debug_counter % 100 == 1:
        print(f"  [energy-guided-descent] d_s_req={delta_s:.3f}, d_s_used={delta_s_limited:.3f}")
        print(f"    glide req: {glide_required:.1f}, current: {glide_current:.1f}")
        print(f"    d_a={delta_a:.3f}, final L={delta_left:.3f}, R={delta_right:.3f}")

    return output


ParafoilADRCController.compute_symmetric_deflection = _compute_symmetric_deflection_energy_guided
ParafoilADRCController.update = _update_energy_guided


_ORIGINAL_COMPUTE_CONTROL = ParafoilADRCController.compute_control


def _compute_control_with_terminal_loiter(self,
                                          current_pos: np.ndarray,
                                          current_heading: float,
                                          current_heading_rate: float = 0.0,
                                          current_speed: Optional[float] = None) -> ControlOutput:
    if self.trajectory is None or len(self.trajectory) == 0:
        return ControlOutput()

    final_pos = self.trajectory[-1].position
    goal_xy = final_pos[:2]
    dist_to_goal = float(np.linalg.norm(current_pos[:2] - goal_xy))
    alt_above_goal = float(current_pos[2] - final_pos[2])
    loiter_radius = max(1.05 * self.min_turn_radius, 95.0)
    loiter_entry_altitude = max(35.0, 0.55 * loiter_radius)

    if dist_to_goal < 0.70 * loiter_radius and alt_above_goal > loiter_entry_altitude:
        output = ControlOutput()

        radial = current_pos[:2] - goal_xy
        radial_norm = float(np.linalg.norm(radial))
        final_heading = self.trajectory[-1].heading
        if radial_norm < 1e-6:
            radial_angle = final_heading - np.pi / 2.0
            radial = loiter_radius * np.array([np.cos(radial_angle), np.sin(radial_angle)])
            radial_norm = loiter_radius
        else:
            radial_angle = float(np.arctan2(radial[1], radial[0]))

        tangent_ccw = self._wrap_angle(radial_angle + np.pi / 2.0)
        tangent_cw = self._wrap_angle(radial_angle - np.pi / 2.0)
        err_ccw = abs(self._wrap_angle(final_heading - tangent_ccw))
        err_cw = abs(self._wrap_angle(final_heading - tangent_cw))
        direction = 1.0 if err_ccw <= err_cw else -1.0

        tangent_heading = self._wrap_angle(radial_angle + direction * np.pi / 2.0)
        radial_error = radial_norm - loiter_radius
        speed_for_guidance = max(
            current_speed if current_speed is not None else self.reference_speed,
            0.1
        )
        heading_correction = -direction * np.arctan2(0.9 * radial_error, speed_for_guidance + 6.0)
        heading_correction = float(np.clip(
            heading_correction,
            -self.max_cross_track_heading_correction,
            self.max_cross_track_heading_correction
        ))
        target_heading = self._wrap_angle(tangent_heading + heading_correction)

        output.ref_position = np.array([goal_xy[0], goal_xy[1], final_pos[2]], dtype=float)
        output.ref_position_closest = final_pos.copy()
        output.ref_heading = target_heading
        output.heading_error = self._wrap_angle(target_heading - current_heading)
        output.altitude_error = final_pos[2] - current_pos[2]
        output.cross_track_error = radial_error
        output.along_track_error = 0.0

        adjusted_heading_ref = target_heading
        adjusted_error = self._wrap_angle(adjusted_heading_ref - current_heading)
        if abs(adjusted_error) > np.radians(60):
            adjusted_heading_ref = current_heading + np.sign(adjusted_error) * np.radians(60)
        adjusted_heading_ref = self._wrap_angle(adjusted_heading_ref)

        delta_diff = self.heading_adrc.update(adjusted_heading_ref, current_heading, self.dt)
        output.delta_asymmetric = delta_diff
        if delta_diff > 0:
            output.delta_left = 0.0
            output.delta_right = np.clip(delta_diff, 0, self.max_deflection)
        else:
            output.delta_left = np.clip(-delta_diff, 0, self.max_deflection)
            output.delta_right = 0.0
        return output

    return _ORIGINAL_COMPUTE_CONTROL(
        self,
        current_pos=current_pos,
        current_heading=current_heading,
        current_heading_rate=current_heading_rate,
        current_speed=current_speed,
    )


ParafoilADRCController.compute_control = _compute_control_with_terminal_loiter
