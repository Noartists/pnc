"""
翼伞无人机8自由度动力学模型

状态向量 y (20维):
    [0:3]   - 位置 (x, y, z) 惯性坐标系
    [3:6]   - 伞体姿态角 (phi, theta, psi)
    [6:8]   - 相对角 (theta_r, psi_r)
    [8:11]  - 伞体速度 (u, v, w)
    [11:14] - 伞体角速度 (p, q, r)
    [14:17] - 负载速度
    [17:20] - 负载角速度
"""

import math
import argparse
import yaml
import numpy as np
from scipy.integrate import odeint


# ============================================================
#                       工具函数
# ============================================================

def skew(omega):
    """
    计算向量的反对称矩阵 (叉乘矩阵)
    omega x v = skew(omega) @ v
    """
    ox, oy, oz = omega.flatten()
    return np.array([
        [0, -oz, oy],
        [oz, 0, -ox],
        [-oy, ox, 0]
    ])


def euler_to_dcm(phi, theta, psi):
    """
    欧拉角转方向余弦矩阵 (ZYX顺序)
    从惯性坐标系到体坐标系的转换矩阵
    """
    cp, sp = np.cos(phi), np.sin(phi)
    ct, st = np.cos(theta), np.sin(theta)
    cs, ss = np.cos(psi), np.sin(psi)

    return np.array([
        [ct*cs, ct*ss, -st],
        [sp*st*cs - cp*ss, sp*st*ss + cp*cs, sp*ct],
        [cp*st*cs + sp*ss, cp*st*ss - sp*cs, cp*ct]
    ])


def omega_to_euler_rate(phi, theta):
    """
    角速度到欧拉角变化率的转换矩阵
    d(euler)/dt = T @ omega
    """
    cp, sp = np.cos(phi), np.sin(phi)
    ct, tt = np.cos(theta), np.tan(theta)

    return np.array([
        [1, sp*tt, cp*tt],
        [0, cp, -sp],
        [0, sp/ct, cp/ct]
    ])


def air_density(h):
    """
    根据高度计算空气密度 (国际标准大气模型)
    h: 高度 (m)
    返回: 空气密度 (kg/m^3)
    """
    rho0, T0 = 1.225, 288.15

    if h <= 11000:
        T = T0 - 0.0065 * h
        return rho0 * (T / T0) ** 4.25588
    elif h <= 20000:
        return 0.36392 * math.exp(-(h - 11000) / 6341.62)
    else:
        T = 216.65 + 0.001 * (h - 20000)
        return 0.088035 * (T / 216.65) ** (-35.1632)


# ============================================================
#                     参数加载与处理
# ============================================================

class ParafoilParams:
    """翼伞参数容器类"""

    def __init__(self):
        # 默认初始化
        self.Rho = 1.225

    @classmethod
    def from_yaml(cls, yaml_path):
        """从YAML文件加载参数"""
        with open(yaml_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)

        para = cls()

        # 质量参数
        para.mc = cfg['mass']['canopy']
        para.mp = cfg['mass']['payload']

        # 几何参数
        para.b = cfg['geometry']['span']
        para.Ac = cfg['geometry']['area']
        para.c = para.Ac / para.b  # 弦长
        para.ca = np.radians(cfg['geometry']['arc_angle'])
        para.r = para.b / para.ca  # 曲率半径
        para.t = cfg['geometry']['thickness_ratio'] * para.c
        para.As = cfg['geometry']['ref_area']
        para.Ap = cfg['geometry']['payload_area']
        para.miu = np.radians(cfg['geometry']['rigging_angle'])
        para.sloc = 2 * para.r * np.sin(para.ca / 2) / para.ca

        # 位置向量
        line_length = cfg['geometry']['line_length']
        para.rcOc = np.array([[0, 0, line_length - para.r + para.sloc]]).T
        para.rcOp = np.array([[0, 0, cfg['geometry']['payload_cg_offset']]]).T

        # 铰链参数
        para.k_psi = cfg['hinge']['yaw_stiffness']
        para.k_r = cfg['hinge']['yaw_damping']
        para.k_f = cfg['hinge']['pitch_stiffness']
        para.c_f = cfg['hinge']['pitch_damping']

        # 环境参数
        para.gn = cfg['environment']['gravity']
        para.Rho = cfg['environment']['air_density']

        # 气动参数
        aero = cfg['aerodynamics']
        para.CD0 = aero['CD0']
        para.CDa2 = aero['CDa2']
        para.CDds = aero['CDds']
        para.CDp = aero['CDp']
        para.CL0 = aero['CL0']
        para.CLa = aero['CLa']
        para.CLds = aero['CLds']
        para.CYbeta = aero['CYbeta']
        para.Cm0 = aero['Cm0']
        para.Cma = aero['Cma']
        para.Cmq = aero['Cmq']
        para.Clp = aero['Clp']
        para.Clbeta = aero['Clbeta']
        para.Clr = aero['Clr']
        para.Clda = aero['Clda']
        para.Cnbeta = aero['Cnbeta']
        para.Cnp = aero['Cnp']
        para.Cnr = aero['Cnr']
        para.Cnda = aero['Cnda']

        # 控制输入
        ctrl = cfg['control']
        para.left = ctrl['left_deflection']
        para.right = ctrl['right_deflection']
        para.thrust = np.array([ctrl['thrust']]).T
        para.vw = np.array([ctrl['wind']]).T

        return para

    def update_density(self, altitude):
        """根据高度更新空气密度"""
        self.Rho = air_density(altitude)


# ============================================================
#                     惯量计算
# ============================================================

def compute_canopy_inertia(para):
    """
    计算伞体转动惯量
    包含伞衣质量和伞内空气质量
    """
    r, c, ca = para.r, para.c, para.ca
    sloc = para.sloc

    # 伞内空气质量
    m_air = para.Rho * para.b * c * para.t / 2
    m_total = para.mc + m_air

    # 惯量分量
    sin_half_ca = np.sin(ca / 2)
    sin_ca = np.sin(ca)

    Js11 = r**2 - sloc**2
    Js22 = r**2 * (sin_ca + ca) / (2*ca) + 7*c**2/48 - sloc**2
    Js33 = r**2 * (ca - sin_ca) / (2*ca) + 7*c**2/48

    rho1 = 3 / (r * ca * c)
    rho2 = 1 / (3 * r * ca * c)
    Js13 = (rho1 - 9*rho2) * r**2 * c**2 * sin_half_ca / 16

    return m_total * np.array([
        [Js11, 0, Js13],
        [0, Js22, 0],
        [Js13, 0, Js33]
    ])


def compute_apparent_mass(para):
    """
    计算附加质量矩阵 (6x6)
    考虑空气动力学附加质量效应
    """
    r, c, ca, b, t = para.r, para.c, para.ca, para.b, para.t
    Rho = para.Rho

    # 几何参数
    b_eff = 2 * b / ca * np.sin(ca / 2)  # 有效翼展
    youh = r * (1 - np.cos(ca / 2)) / b_eff
    AR = b / c  # 展弦比

    # 附加质量系数
    Ka, Kb = 0.848, 1.0
    Kc = AR / (1 + AR)
    Ka_s = 0.84 * Kc
    Kb_s = 1.64 * Kc
    Kc_s = 0.848

    # 平动附加质量
    af11 = Rho * Ka * np.pi * t**2 * b_eff / 4
    af22 = Rho * Kb * np.pi * t**2 * c / 4
    af33 = Rho * Kc * np.pi * c**2 * b_eff / 4

    # 转动附加惯量
    af44 = Rho * Ka_s * np.pi * b_eff**3 * c**2 / 48
    af55 = Rho * Kb_s * c**4 * b_eff / (12 * np.pi)
    af66 = Rho * Kc_s * b_eff**3 * t**2 * np.pi / 48

    # 参考点位置
    jyuan = ca / 2
    zpc = r * np.sin(jyuan) / jyuan
    zrc = zpc * af22 / (af22 + af44 / r**2)
    zpr = zrc - zpc

    # 修正后的附加质量
    ma11 = (1 + 8/3 * youh**2) * af11
    ma22 = (r**2 * af22 + af44) / zpc**2
    ma33 = af33 * np.sqrt(1 + 2 * youh**2 * (1 - (t/c)**2))

    ma = np.diag([ma11, ma22, ma33])

    # 修正后的附加惯量
    Ia11 = (zpr/zpc)**2 * r**2 * af22 + (zrc/zpc)**2 * af44
    Ia22 = af55 * (1 + np.pi/6 * (1+AR) * AR * youh**2 * (t/c)**2)
    Ia33 = (1 + 8 * youh**2) * af66

    # 位置偏移矩阵
    zor = para.sloc - zrc
    lor = np.array([[0, -zor, 0], [zor, 0, 0], [0, 0, 0]])
    lrp = np.array([[0, -zpr, 0], [zpr, 0, 0], [0, 0, 0]])
    B2 = np.diag([0, 1, 0])

    # 耦合项
    D = lor + lrp @ B2
    mal = -ma @ D
    Tmal = (B2 @ lrp + lor) @ ma

    # 总附加惯量
    Ja = np.diag([Ia11, Ia22, Ia33])
    Q = B2 @ lrp @ ma @ lor
    Jao = Ja - lor @ ma @ lor - lrp @ ma @ lrp @ B2 - Q - Q.T

    # 组装6x6矩阵
    return np.block([
        [ma, mal],
        [Tmal, Jao]
    ])


# ============================================================
#                     动力学模型
# ============================================================

def parafoil_dynamics(y, t, para):
    """
    翼伞8自由度动力学方程

    参数:
        y: 状态向量 (20,)
        t: 时间
        para: 参数对象

    返回:
        dydt: 状态导数 (20,)
    """
    # -------- 状态解包 --------
    pos = y[0:3].reshape(3, 1)
    euler = y[3:6].reshape(3, 1)         # phi, theta, psi
    theta_r, psi_r = y[6], y[7]          # 相对俯仰角、偏航角
    v_c = y[8:11].reshape(3, 1)          # 伞体速度
    w_c = y[11:14].reshape(3, 1)         # 伞体角速度
    v_p = y[14:17].reshape(3, 1)         # 负载速度
    w_p = y[17:20].reshape(3, 1)         # 负载角速度

    phi, theta, psi = euler.flatten()

    # -------- 惯量矩阵 --------
    Ic = compute_canopy_inertia(para)
    Ia = compute_apparent_mass(para)

    m_air = para.Rho * para.b * para.c * para.t / 2
    m_total = para.mc + m_air

    # 伞体惯量矩阵 (6x6)
    Ic_r = np.block([
        [m_total * np.eye(3), np.zeros((3, 3))],
        [np.zeros((3, 3)), Ic]
    ])

    # 负载惯量
    Ip = para.mp / 12 * 0.5 * np.eye(3)

    # -------- 坐标变换矩阵 --------
    R_cr = euler_to_dcm(0, para.miu, 0)       # 伞体→安装坐标系
    R_cp = euler_to_dcm(psi_r, theta_r, 0)    # 伞体→负载坐标系
    R_nb = euler_to_dcm(psi, theta, phi)      # 惯性→伞体坐标系

    # 安装坐标系中的总惯量
    T_cr = np.block([
        [R_cr.T, np.zeros((3, 3))],
        [np.zeros((3, 3)), R_cr.T]
    ])
    Ir = T_cr @ (Ic_r + Ia) @ T_cr.T

    A1, A2 = Ir[:3, :3], Ir[:3, 3:6]
    A3, A4 = Ir[3:6, :3], Ir[3:6, 3:6]

    # -------- 角速度约束 --------
    T_ang = np.array([
        [-1, 0, -np.sin(theta_r)],
        [0, 1, 0],
        [0, 0, np.cos(theta_r)]
    ])
    rhs = np.array([[0], [w_p[1, 0]], [w_p[2, 0]]]) - R_cp @ w_c
    sol = np.linalg.solve(T_ang, rhs)
    d_theta_r = sol[1, 0]
    d_psi_r = sol[2, 0]

    # -------- 控制输入 --------
    d_s = min(para.left, para.right)  # 对称偏转
    d_a = para.left - para.right       # 非对称偏转

    # -------- 重力 --------
    F_cg = R_nb @ np.array([[0, 0, para.mc * para.gn]]).T
    F_pg = R_cp @ R_nb @ np.array([[0, 0, para.mp * para.gn]]).T

    # -------- 铰链力矩 --------
    M_cz = para.k_r * d_psi_r
    M_pf = -np.array([[0], [para.k_f * theta_r + para.c_f * d_theta_r], [0]])
    M_cf = -R_cp.T @ M_pf

    # -------- 气动力计算 --------
    v_air = R_cr @ (v_c - R_nb @ para.vw)
    V = np.sqrt((v_air.T @ v_air).item())

    if V > 1e-6:
        alpha = np.arctan2(v_air[2, 0], v_air[0, 0])
        beta = np.arcsin(np.clip(v_air[1, 0] / V, -1, 1))
    else:
        alpha, beta = 0.0, 0.0

    q_bar = 0.5 * para.Rho * V**2  # 动压

    # 气动系数
    CD = para.CD0 + para.CDa2 * alpha**2 + para.CDds * d_s
    CY = para.CYbeta * beta
    CL = para.CL0 + para.CLa * alpha + para.CLds * d_s

    # 风轴→体轴变换
    R_wb = euler_to_dcm(beta, alpha - np.pi, 0)
    F_aero = q_bar * para.As * R_cr.T @ R_wb @ np.array([[CD, CY, CL]]).T

    # 气动力矩
    p, q, r = w_c.flatten()
    V_inv = 1.0 / max(V, 1e-6)

    Cl = (para.Clbeta * beta +
          para.b * V_inv / 2 * (para.Clp * p + para.Clr * r) +
          para.Clda * d_a)
    Cm = para.Cm0 + para.Cma * alpha + para.c * V_inv / 2 * para.Cmq * q
    Cn = (para.Cnbeta * beta +
          para.b * V_inv / 2 * (para.Cnp * p + para.Cnr * r) +
          para.Cnda * d_a)

    M_aero = q_bar * para.As * R_cr.T @ np.array([
        [para.b * Cl],
        [para.c * Cm],
        [para.b * Cn]
    ])

    # -------- 负载气动力 --------
    v_p_air = v_p - R_cp @ R_nb @ para.vw
    V_p = np.sqrt((v_p_air.T @ v_p_air).item())
    F_pa = -para.CDp * 0.5 * para.Rho * para.Ap * V_p * v_p_air

    # -------- 约束方程组装 --------
    E_t = np.array([[1, 0], [0, 0], [0, 1]])

    # 构建 K 矩阵
    K1 = np.array([[np.cos(theta_r), 0, np.sin(theta_r)]])
    K2 = np.array([[np.cos(psi_r), np.sin(psi_r), 0]])

    # dps 计算
    dps = (d_theta_r * np.array([[np.sin(theta_r), 0, -np.cos(theta_r)]]) @ w_p +
           d_psi_r * np.array([[-np.sin(psi_r), np.cos(psi_r), 0]]) @ w_c)

    # 组装大矩阵 E (17x17)
    E = np.zeros((17, 17))
    B = np.zeros((17, 1))

    # 第1行: 伞体平动方程
    E[0:3, 0:3] = A1
    E[0:3, 3:6] = A2
    E[0:3, 12:15] = R_cp.T

    # 第2行: 伞体转动方程
    E[3:6, 0:3] = A3
    E[3:6, 3:6] = A4
    E[3:6, 12:15] = skew(para.rcOc) @ R_cp.T
    E[3:6, 15:17] = R_cp.T @ E_t

    # 第3行: 负载平动方程
    E[6:9, 6:9] = para.mp * np.eye(3)
    E[6:9, 12:15] = -np.eye(3)

    # 第4行: 负载转动方程
    E[9:12, 9:12] = Ip
    E[9:12, 12:15] = -skew(para.rcOp)
    E[9:12, 15:17] = -E_t

    # 第5行: 速度约束
    E[12:15, 0:3] = np.eye(3)
    E[12:15, 3:6] = -skew(para.rcOc)
    E[12:15, 6:9] = -R_cp.T
    E[12:15, 9:12] = R_cp.T @ skew(para.rcOp)

    # 第6行: 角速度约束
    E[15, 3:6] = -K2.flatten()
    E[15, 9:12] = K1.flatten()

    # 第7行: 偏航力矩约束
    E7_1 = (np.array([[0, 0, 1]]) @ skew(para.rcOc) + para.k_psi * psi_r) @ R_cp.T
    E7_2 = -np.array([[0, 0, 1]]) @ R_cp.T @ E_t
    E[16, 12:15] = E7_1.flatten()
    E[16, 15:17] = E7_2.flatten()

    # 右端项
    B1_tmp = A1 @ v_c + A2 @ w_c
    B[0:3] = F_aero + F_cg - skew(w_c) @ B1_tmp

    B2_1 = skew(w_c) @ (Ir[3:6, 0:3] @ v_c + Ir[3:6, 3:6] @ w_c)
    B[3:6] = M_aero - B2_1 + M_cf

    B[6:9] = F_pa + F_pg - para.mp * skew(w_p) @ v_p + para.thrust

    B[9:12] = -skew(w_p) @ Ip @ w_p + M_pf

    B5_1 = R_cp.T @ skew(w_p) @ (v_p - skew(para.rcOp) @ w_p)
    B5_2 = skew(w_c) @ (v_c - skew(para.rcOc) @ w_c)
    B[12:15] = B5_1 - B5_2

    B[15] = dps.flatten()[0]
    B[16] = M_cz

    # -------- 求解 --------
    dVW = np.linalg.solve(E, B)

    # -------- 输出状态导数 --------
    d_euler = omega_to_euler_rate(phi, theta) @ w_c

    # 惯性坐标系中的速度
    # 坐标系约定: x向前, y向左, z向上 (右手系)
    v_inertial = R_nb.T @ v_c
    v_inertial[1, 0] = -v_inertial[1, 0]  # y取反，向左为正
    v_inertial[2, 0] = -v_inertial[2, 0]  # z取反，向上为正

    dydt = np.zeros(20)
    dydt[0:3] = v_inertial.flatten()      # 位置导数 (z为高度，向上为正)
    dydt[3:6] = d_euler.flatten()         # 姿态角导数
    dydt[6] = d_theta_r                   # 相对俯仰角导数
    dydt[7] = d_psi_r                     # 相对偏航角导数
    dydt[8:20] = dVW[0:12].flatten()      # 速度、角速度导数

    return dydt


# ============================================================
#                     仿真接口
# ============================================================

def simulate(y0, t_span, para, dt=0.01, update_density=True):
    """
    运行翼伞动力学仿真

    参数:
        y0: 初始状态 (20,)
        t_span: (t_start, t_end) 仿真时间范围
        para: 参数对象
        dt: 积分步长
        update_density: 是否根据高度更新空气密度

    返回:
        t: 时间数组
        y: 状态数组 (n, 20)
    """
    t = np.arange(t_span[0], t_span[1], dt)

    if update_density:
        # 分段积分，每段更新空气密度
        y = [y0]
        segment_length = max(1, int(1.0 / dt))  # 每秒更新一次密度

        for i in range(0, len(t) - 1, segment_length):
            t_seg = t[i:min(i + segment_length + 1, len(t))]
            para.update_density(y[-1][2])  # z为高度，向上为正

            y_seg = odeint(parafoil_dynamics, y[-1], t_seg, args=(para,))
            y.extend(y_seg[1:].tolist())

        y = np.array(y)
    else:
        y = odeint(parafoil_dynamics, y0, t, args=(para,))

    return t, y


