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
            v: 输入信号 (目标值)
            dt: 时间步长
        
        返回:
            (v1, v2): 跟踪值和其微分
        """
        fh = fhan(self.v1 - v, self.v2, self.r, self.h)
        
        self.v1 = self.v1 + dt * self.v2
        self.v2 = self.v2 + dt * fh
        
        return self.v1, self.v2


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
        e = self.z[0] - y
        
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
        
        return self.z.copy()


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
        self.td.reset(y0)
        self.eso.reset()
        self.u_last = 0.0
    
    def update(self, ref: float, y: float, dt: float) -> float:
        """
        更新控制器
        
        参数:
            ref: 参考值
            y: 系统输出 (测量值)
            dt: 时间步长
        
        返回:
            u: 控制输出
        """
        # 1. 跟踪微分器: 安排过渡过程
        v1, v2 = self.td.update(ref, dt)
        
        # 2. 扩展状态观测器: 估计状态和扰动
        z = self.eso.update(y, self.u_last, self.b0, dt)
        z1, z2 = z[0], z[1]
        z3 = z[2] if len(z) > 2 else 0.0
        
        # 3. 计算误差
        e1 = v1 - z1  # 位置误差
        e2 = v2 - z2  # 速度误差
        
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
    heading_error: float = 0.0     # 航向误差 (rad)
    cross_track_error: float = 0.0 # 横向误差 (m)
    along_track_error: float = 0.0 # 纵向误差 (m)
    altitude_error: float = 0.0    # 高度误差 (m)
    ref_heading: float = 0.0       # 参考航向 (rad)
    ref_position: np.ndarray = field(default_factory=lambda: np.zeros(3))


class ParafoilADRCController:
    """
    翼伞ADRC轨迹跟踪控制器
    
    控制通道:
    - 航向控制: 通过差动操纵绳控制偏航
    - 下降率控制: 通过对称操纵绳控制下降率
    """
    
    def __init__(self, 
                 # 航向控制参数
                 heading_kp: float = 2.0,
                 heading_kd: float = 0.5,
                 heading_eso_omega: float = 20.0,
                 heading_td_r: float = 30.0,
                 # 横向误差控制参数
                 lateral_kp: float = 0.01,
                 lateral_kd: float = 0.005,
                 # 系统参数
                 reference_speed: float = 12.0,
                 min_turn_radius: float = 50.0,
                 lookahead_distance: float = 50.0,
                 # 输出限制
                 max_deflection: float = 1.0,
                 dt: float = 0.01):
        """
        参数:
            heading_kp: 航向比例增益 (越大响应越快，太大震荡)
            heading_kd: 航向微分增益 (增加可抑制震荡)
            heading_eso_omega: ESO带宽 (越大扰动估计越快，太大噪声敏感)
            heading_td_r: TD快速因子 (越小参考信号过渡越平滑)
            lateral_kp: 横向误差增益 (越大路径跟踪越紧)
            lateral_kd: 横向误差微分增益
            reference_speed: 参考飞行速度 (m/s)
            min_turn_radius: 最小转弯半径 (m)
            lookahead_distance: 前视距离 (m，越大转弯越平滑)
            max_deflection: 最大操纵绳偏转量 [0,1]
            dt: 控制周期 (s)
        """
        self.reference_speed = reference_speed
        self.min_turn_radius = min_turn_radius
        self.lookahead_distance = lookahead_distance
        self.max_deflection = max_deflection
        self.dt = dt
        
        # 航向ADRC控制器
        self.heading_adrc = ADRC(
            td_r=heading_td_r,    # TD快速因子：越小过渡越平滑
            td_h=dt,
            eso_omega=heading_eso_omega,
            eso_order=2,
            use_linear_eso=True,
            kp=heading_kp,
            kd=heading_kd,
            use_linear_sef=False,
            b0=1.0,
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
        self.last_heading_ref = None
    
    def set_trajectory(self, trajectory):
        """
        设置要跟踪的轨迹
        
        参数:
            trajectory: Trajectory 对象 (来自 planning.trajectory)
        """
        self.trajectory = trajectory
        self.current_index = 0
        self.reset()
    
    def reset(self):
        """重置控制器"""
        self.heading_adrc.reset()
        self.lateral_error_last = 0.0
        self.current_index = 0
        self.last_heading_ref = None
    
    def _find_closest_point(self, current_pos: np.ndarray) -> int:
        """
        找到轨迹上距离当前位置最近的点索引
        
        参数:
            current_pos: 当前位置 [x, y, z]
        
        返回:
            最近点索引
        """
        if self.trajectory is None or len(self.trajectory) == 0:
            return 0
        
        # 从当前索引开始向前搜索
        min_dist = np.inf
        best_idx = self.current_index
        
        # 搜索范围: 当前索引前后一定范围
        search_start = max(0, self.current_index - 10)
        search_end = min(len(self.trajectory), self.current_index + 100)
        
        for i in range(search_start, search_end):
            dist = np.linalg.norm(current_pos - self.trajectory[i].position)
            if dist < min_dist:
                min_dist = dist
                best_idx = i
        
        # 只允许索引向前移动，避免回退
        return max(best_idx, self.current_index)
    
    def _find_lookahead_point(self, current_pos: np.ndarray, closest_idx: int):
        """
        找到前视点 (Pure Pursuit风格)
        
        参数:
            current_pos: 当前位置
            closest_idx: 最近点索引
        
        返回:
            前视点 (TrajectoryPoint)
        """
        if self.trajectory is None or len(self.trajectory) == 0:
            return None
        
        # 从最近点开始，找到距离超过前视距离的点
        accumulated_dist = 0.0
        
        for i in range(closest_idx, len(self.trajectory) - 1):
            segment_length = np.linalg.norm(
                self.trajectory[i + 1].position - self.trajectory[i].position
            )
            accumulated_dist += segment_length
            
            if accumulated_dist >= self.lookahead_distance:
                return self.trajectory[i + 1]
        
        # 如果没找到，返回轨迹终点
        return self.trajectory[-1]
    
    def compute_control(self, 
                        current_pos: np.ndarray,
                        current_heading: float,
                        current_heading_rate: float = 0.0) -> ControlOutput:
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
        
        # ========== 1. 找到轨迹上的参考点 ==========
        self.current_index = self._find_closest_point(current_pos)
        closest_point = self.trajectory[self.current_index]
        lookahead_point = self._find_lookahead_point(current_pos, self.current_index)
        
        if lookahead_point is None:
            lookahead_point = closest_point
        
        target_pos = lookahead_point.position
        target_heading = lookahead_point.heading
        
        output.ref_position = target_pos.copy()
        output.ref_heading = target_heading
        
        # ========== 2. 计算跟踪误差 ==========
        # 2.1 航向误差
        heading_error = self._wrap_angle(target_heading - current_heading)
        output.heading_error = heading_error
        
        # 2.2 高度误差
        output.altitude_error = closest_point.position[2] - current_pos[2]
        
        # 2.3 横向误差 (Cross-track error)
        # 使用最近点的切线方向计算
        ref_dir = np.array([np.cos(closest_point.heading), np.sin(closest_point.heading)])
        to_vehicle = current_pos[:2] - closest_point.position[:2]
        
        # 横向误差: 正值表示飞机在轨迹右侧
        cross_track_error = ref_dir[0] * to_vehicle[1] - ref_dir[1] * to_vehicle[0]
        output.cross_track_error = cross_track_error
        
        # 2.4 纵向误差 (Along-track error)
        along_track_error = np.dot(to_vehicle, ref_dir)
        output.along_track_error = along_track_error
        
        # ========== 3. 航向控制 (ADRC + 横向误差补偿) ==========
        # 根据横向误差调整目标航向
        heading_correction = self.lateral_kp * cross_track_error
        heading_correction = np.clip(heading_correction, -0.5, 0.5)
        
        adjusted_heading_ref = target_heading - heading_correction  # 负号:右偏需左转
        
        # ADRC航向控制
        delta_diff = self.heading_adrc.update(adjusted_heading_ref, current_heading, self.dt)
        
        # ========== 4. 转换为左右操纵绳偏转 ==========
        # delta_diff > 0 表示需要右转
        delta_base = 0.0  # 可用于下降率控制
        
        delta_left = delta_base - delta_diff / 2
        delta_right = delta_base + delta_diff / 2
        
        # 限幅到 [0, max_deflection]
        delta_left = np.clip(delta_left, 0, self.max_deflection)
        delta_right = np.clip(delta_right, 0, self.max_deflection)
        
        output.delta_left = delta_left
        output.delta_right = delta_right
        
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
        
        return self.compute_control(
            current_pos=current_pos,
            current_heading=current_heading,
            current_heading_rate=heading_rate
        )
    
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
