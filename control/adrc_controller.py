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
        self.max_deflection = max_deflection
        self.dt = dt

        # 下降率控制参数
        self.glide_ratio_natural = glide_ratio_natural
        self.glide_ratio_min = glide_ratio_min
        self.descent_kp = descent_kp
        self.descent_margin = descent_margin
        
        # 航向ADRC控制器
        # b0: 控制增益估计值，需要根据系统特性调整
        # 翼伞通过滚转转弯，航向响应较慢，b0应该较大以避免过度控制
        b0_heading = 2.0  # 适中的b0，避免控制输出过大

        self.heading_adrc = ADRC(
            td_r=heading_td_r,    # TD快速因子：越小过渡越平滑
            td_h=dt,
            eso_omega=heading_eso_omega,
            eso_order=2,
            use_linear_eso=True,
            kp=heading_kp,
            kd=heading_kd,
            use_linear_sef=True,  # 线性SEF，响应更直接
            b0=b0_heading,
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
        self.current_index = 0
        self.reset()
    
    def reset(self):
        """重置控制器"""
        self.heading_adrc.reset()
        self.lateral_error_last = 0.0
        self.current_index = 0
        self.last_heading_ref = None
        self.debug_counter = 0
    
    def set_debug(self, enabled: bool = True):
        """启用/禁用调试模式"""
        self.debug = enabled

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
        # 每10m高度误差，调整滑翔比±1
        glide_ratio_adjustment = -altitude_error / 10.0
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
        
        # 计算横向误差（用于判断是否需要优先回到轨迹）
        ref_dir_temp = np.array([np.cos(closest_point.heading), np.sin(closest_point.heading)])
        to_vehicle_temp = current_pos[:2] - closest_point.position[:2]
        cross_track_error_temp = ref_dir_temp[0] * to_vehicle_temp[1] - ref_dir_temp[1] * to_vehicle_temp[0]
        
        # 如果横向误差过大（>50m），优先回到轨迹，使用最近点而非前视点
        if abs(cross_track_error_temp) > 50.0:
            # 使用最近点，直接指向轨迹
            target_pos = closest_point.position
            target_heading = closest_point.heading
        else:
            # 正常情况：使用前视点
            lookahead_point = self._find_lookahead_point(current_pos, self.current_index)
            if lookahead_point is None:
                lookahead_point = closest_point
            target_pos = lookahead_point.position
            target_heading = lookahead_point.heading
        
        output.ref_position = target_pos.copy()
        output.ref_heading = target_heading
        
        # ========== 2. 计算跟踪误差 ==========
        # 2.1 航向误差 (处理 ±π 跳变，选择最短路径)
        raw_heading_error = target_heading - current_heading
        heading_error = self._wrap_angle(raw_heading_error)
        output.heading_error = heading_error
        
        # 检查航向误差是否过大（可能导致控制器饱和）
        if abs(heading_error) > np.radians(90):
            # 如果误差超过90度，可能需要检查控制方向
            pass
        
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
        # 横向误差为正：在轨迹右侧，需要左转（减小航向）
        # 横向误差为负：在轨迹左侧，需要右转（增大航向）
        
        # 限制横向误差的影响范围，避免过度修正
        # 当横向误差很大时，使用饱和函数而非线性增益
        max_cross_track = 100.0  # 最大有效横向误差 (m)
        normalized_cross_track = np.clip(cross_track_error / max_cross_track, -1.0, 1.0)
        
        # 使用饱和函数，避免大误差时的过度修正
        heading_correction = self.lateral_kp * max_cross_track * normalized_cross_track
        heading_correction = np.clip(heading_correction, -np.radians(30), np.radians(30))  # 限制在30度以内
        
        adjusted_heading_ref = target_heading - heading_correction  # 负号:右偏需左转
        
        # 如果航向误差过大，先限制调整后的目标航向，避免控制器饱和
        # 当误差超过45度时，逐步调整目标航向（更激进的限制）
        adjusted_error = self._wrap_angle(adjusted_heading_ref - current_heading)
        if abs(adjusted_error) > np.radians(45):
            # 限制调整速度：每次最多调整45度
            max_adjustment = np.radians(45)
            adjusted_heading_ref = current_heading + np.sign(adjusted_error) * max_adjustment
        
        # 归一化调整后的目标航向到 [-π, π]
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
        if self.debug and self.debug_counter % 100 == 0:  # 每100步打印一次
            print(f"[控制器调试] step={self.debug_counter}")
            print(f"  当前位置: ({current_pos[0]:.1f}, {current_pos[1]:.1f}, {current_pos[2]:.1f})")
            print(f"  当前航向: {np.degrees(current_heading):.1f}°")
            print(f"  目标航向: {np.degrees(target_heading):.1f}°")
            print(f"  航向误差: {np.degrees(heading_error):.1f}°")
            print(f"  横向误差: {cross_track_error:.2f}m (修正: {np.degrees(heading_correction):.2f}°)")
            print(f"  调整后目标航向: {np.degrees(adjusted_heading_ref):.1f}°")
            print(f"  调整后航向误差: {np.degrees(self._wrap_angle(adjusted_heading_ref - current_heading)):.1f}°")
            print(f"  ADRC输出: {delta_diff:.3f} (范围: [-{self.max_deflection:.1f}, {self.max_deflection:.1f}])")
            print(f"  左绳: {delta_left:.3f}, 右绳: {delta_right:.3f}")
            if abs(delta_diff) >= self.max_deflection * 0.95:
                print(f"  [警告] ADRC输出饱和! 航向误差: {np.degrees(heading_error):.1f}°")
                # 检查ADRC内部状态
                if hasattr(self.heading_adrc, 'td') and hasattr(self.heading_adrc, 'eso'):
                    td_v1, td_v2 = self.heading_adrc.td.v1, self.heading_adrc.td.v2
                    eso_z = self.heading_adrc.eso.z
                    print(f"    TD输出: v1={np.degrees(td_v1):.1f}°, v2={np.degrees(td_v2):.1f}°/s")
                    print(f"    ESO状态: z1={np.degrees(eso_z[0]):.1f}°, z2={np.degrees(eso_z[1]):.1f}°/s, z3={eso_z[2]:.3f}")
        
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
            current_heading_rate=heading_rate
        )

        # 2. 计算下降率控制 (对称偏转)
        # 使用航向控制中已经计算好的参考位置（前视点），而不是终点
        target_pos = output.ref_position
        delta_s, glide_required, glide_current = self.compute_symmetric_deflection(
            current_pos=current_pos,
            current_vel=current_vel,
            target_pos=target_pos  # 传入前视点
        )

        output.delta_symmetric = delta_s
        output.glide_ratio_required = glide_required
        output.glide_ratio_current = glide_current

        # 3. 合成最终控制量
        # 对称偏转 d_s = min(left, right): 控制下降率
        # 非对称偏转 d_a = left - right: 控制航向
        # 
        # 关键：优先保证航向控制！
        # 对称偏转需要让出空间给非对称偏转

        delta_a = output.delta_asymmetric  # 航向控制的非对称分量
        delta_a_abs = abs(delta_a)

        # 限制对称偏转，为非对称偏转留出空间
        # max_deflection = d_s + |d_a|，所以 d_s <= max_deflection - |d_a|
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
