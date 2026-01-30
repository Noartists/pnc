"""
闭环仿真模块

整合: 规划 + 控制 + 动力学模型

流程:
1. 加载地图配置，运行 RRT* 全局规划
2. 路径平滑，生成参考轨迹
3. 初始化翼伞动力学模型
4. 闭环控制仿真:
   - 每个控制周期: 获取状态 → 控制器计算 → 更新控制输入 → 动力学积分
5. 可视化结果
"""

import os
import sys
import numpy as np
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
from scipy.integrate import odeint
import time
import yaml

# 添加项目根目录到路径
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from planning.map_manager import MapManager
from planning.global_planner import RRTStarPlanner
from planning.path_smoother import PathSmoother
from planning.trajectory import Trajectory
from control.adrc_controller import ParafoilADRCController, ControlOutput
from models.parafoil_model import ParafoilParams, parafoil_dynamics


# ============================================================
#                     仿真数据记录
# ============================================================

@dataclass
class SimulationLog:
    """仿真数据记录"""
    t: List[float]                    # 时间
    state: List[np.ndarray]           # 20维状态
    position: List[np.ndarray]        # 位置 [x, y, z]
    velocity: List[np.ndarray]        # 速度 [vx, vy, vz]
    euler: List[np.ndarray]           # 姿态角 [phi, theta, psi]
    control: List[ControlOutput]      # 控制输出
    ref_position: List[np.ndarray]    # 参考位置
    ref_heading: List[float]          # 参考航向
    
    def __init__(self):
        self.t = []
        self.state = []
        self.position = []
        self.velocity = []
        self.euler = []
        self.control = []
        self.ref_position = []
        self.ref_heading = []
    
    def append(self, t: float, state: np.ndarray, ctrl: ControlOutput):
        """添加一条记录"""
        self.t.append(t)
        self.state.append(state.copy())
        self.position.append(state[0:3].copy())
        self.velocity.append(state[8:11].copy())
        self.euler.append(state[3:6].copy())
        self.control.append(ctrl)
        self.ref_position.append(ctrl.ref_position.copy())
        self.ref_heading.append(ctrl.ref_heading)
    
    def to_arrays(self) -> Dict[str, np.ndarray]:
        """转换为数组字典"""
        return {
            't': np.array(self.t),
            'state': np.array(self.state),
            'position': np.array(self.position),
            'velocity': np.array(self.velocity),
            'euler': np.array(self.euler),
            'ref_position': np.array(self.ref_position),
            'ref_heading': np.array(self.ref_heading),
            'delta_left': np.array([c.delta_left for c in self.control]),
            'delta_right': np.array([c.delta_right for c in self.control]),
            'delta_symmetric': np.array([c.delta_symmetric for c in self.control]),
            'delta_asymmetric': np.array([c.delta_asymmetric for c in self.control]),
            'cross_track_error': np.array([c.cross_track_error for c in self.control]),
            'along_track_error': np.array([c.along_track_error for c in self.control]),
            'heading_error': np.array([c.heading_error for c in self.control]),
            'glide_ratio_required': np.array([c.glide_ratio_required for c in self.control]),
            'glide_ratio_current': np.array([c.glide_ratio_current for c in self.control]),
        }


# ============================================================
#                     闭环仿真器
# ============================================================

class ClosedLoopSimulator:
    """
    闭环仿真器
    
    整合规划、控制、动力学模型
    """
    
    def __init__(self, 
                 map_config_path: str,
                 model_config_path: str,
                 control_dt: float = 0.01,
                 dynamics_dt: float = 0.001):
        """
        参数:
            map_config_path: 地图配置文件路径
            model_config_path: 动力学模型配置文件路径
            control_dt: 控制周期 (s)
            dynamics_dt: 动力学积分步长 (s)
        """
        self.control_dt = control_dt
        self.dynamics_dt = dynamics_dt
        
        # 加载地图
        print("[1/4] 加载地图配置...")
        self.map_manager = MapManager.from_yaml(map_config_path)

        # 读取轨迹参数（参考速度）
        with open(map_config_path, 'r', encoding='utf-8') as f:
            map_cfg = yaml.safe_load(f)
        traj_cfg = map_cfg.get('trajectory', {}) if isinstance(map_cfg, dict) else {}
        reference_speed = traj_cfg.get('reference_speed', 12.0)
        self.reference_speed = reference_speed
        
        # 检查可达性
        self.map_manager.print_reachability_report()
        
        # 加载动力学模型参数
        print("[2/4] 加载动力学模型...")
        self.para = ParafoilParams.from_yaml(model_config_path)
        
        # 创建规划器
        self.planner = RRTStarPlanner(self.map_manager)
        self.smoother = PathSmoother(
            turn_radius=self.map_manager.constraints.min_turn_radius,
            reference_speed=reference_speed,
            control_frequency=1.0 / control_dt
        )
        
        # 创建控制器
        # ============ 控制器超参数说明 ============
        # heading_kp:        航向比例增益，越大响应越快，太大会震荡
        # heading_kd:        航向微分增益，增加可抑制震荡，太大会迟钝
        # heading_eso_omega: ESO带宽，越大扰动估计越快，太大会放大噪声
        # heading_td_r:      TD快速因子，越小参考信号过渡越平滑
        # lateral_kp:        横向误差→航向修正增益，越大路径跟踪越紧，太大会震荡
        # lookahead_distance: 前视距离，越大转弯越平滑，太大会切弯
        # max_deflection:    最大操纵绳偏转量 [0,1]
        # glide_ratio_natural: 自然滑翔比，无对称偏转时的L/D (约11.0)
        # glide_ratio_min:   最小滑翔比，最大对称偏转时的L/D (约5.0)
        # descent_kp:        下降率控制增益
        # descent_margin:    滑翔比余量系数，>1表示保守
        # ==========================================
        # 翼伞转弯是通过滚转实现的，响应较慢，需要保守的控制参数
        self.controller = ParafoilADRCController(
            heading_kp=0.8,           # 保守：避免过度响应导致滚转失控
            heading_kd=0.4,           # 适中：阻尼
            heading_eso_omega=3.0,    # 低：减少噪声和过度补偿
            heading_td_r=10.0,        # 低：平滑的参考过渡
            lateral_kp=0.002,         # 低：避免横向误差过度修正
            glide_ratio_natural=6.48, # 自然滑翔比 (无对称偏转)
            glide_ratio_min=2.47,     # 最小滑翔比 (最大对称偏转)
            descent_kp=0.5,           # 下降率控制增益
            descent_margin=1.2,       # 滑翔比余量系数
            reference_speed=reference_speed,
            lookahead_distance=100.0, # 大：更平滑的转弯
            max_deflection=0.5,       # 增加：允许更大偏转用于下降率控制
            dt=control_dt
        )
        
        # 调试模式（可通过参数控制）
        self.controller_debug = True  # 设为 True 启用控制器调试输出
        if self.controller_debug:
            self.controller.set_debug(True)
        
        # 轨迹和仿真状态
        self.trajectory: Optional[Trajectory] = None
        self.state: Optional[np.ndarray] = None
        self.log: Optional[SimulationLog] = None
    
    def plan(self, max_time: float = 30.0) -> bool:
        """
        运行路径规划
        
        参数:
            max_time: 最大规划时间 (s)
        
        返回:
            是否成功
        """
        print("[3/4] 运行 RRT* 路径规划（含画圆消高）...")
        path, info = self.planner.plan_with_loiter(max_time=max_time)

        if path is None:
            print("    路径规划失败!")
            return False

        # 显示画圆信息
        if info.get('loiter_loops', 0) > 0:
            print(f"    规划完成: {len(path)} 航点, 长度 {info['path_length']:.1f}m")
            print(f"    画圆消高: {info['loiter_loops']}圈, 半径 {info['loiter_radius']:.1f}m, {info['loiter_direction'].upper()}")
        else:
            print(f"    规划完成: {len(path)} 航点, 长度 {info['path_length']:.1f}m (无需画圆)")
        
        # 路径平滑
        # 注意：不传递map_manager，因为画圆消高已经在plan_with_loiter中处理了
        # PathSmoother只负责平滑，不再重复添加画圆
        print("[4/4] 路径平滑...")
        end_heading = self.map_manager.target.approach_heading if self.map_manager.target else None
        self.trajectory = self.smoother.smooth(
            path,
            end_heading=end_heading,
            waypoint_density=15
        )
        
        print(f"    生成轨迹: {len(self.trajectory)} 点, 时长 {self.trajectory.duration:.1f}s")
        self.controller.set_trajectory(self.trajectory)
        
        return True
    
    def init_state(self, 
                   position: np.ndarray = None,
                   heading: float = None,
                   velocity: float = 10.0,
                   position_noise: np.ndarray = None,
                   heading_noise: float = 0.0) -> np.ndarray:
        """
        初始化仿真状态
        
        参数:
            position: 初始位置 [x, y, z]，默认使用轨迹起点
            heading: 初始航向 (rad)，默认使用轨迹起点
            velocity: 初始前向速度 (m/s)
            position_noise: 位置噪声 [dx, dy, dz]
            heading_noise: 航向噪声 (rad)
        
        返回:
            初始状态向量 (20,)
        """
        if self.trajectory is None or len(self.trajectory) == 0:
            raise ValueError("请先运行规划 (plan)")
        
        # 默认从轨迹起点开始
        init_point = self.trajectory[0]
        
        if position is None:
            position = init_point.position.copy()
        
        if heading is None:
            heading = init_point.heading
        
        # 添加噪声
        if position_noise is not None:
            position = position + np.array(position_noise)
        heading = heading + heading_noise
        
        # 构建20维状态向量
        # 翼伞典型滑翔状态: 前向速度约10-12m/s, 下沉速度约4-6m/s
        # 攻角约 5-10°, 俯仰角约 5-15°
        self.state = np.zeros(20)
        self.state[0:3] = position                    # 位置
        self.state[3] = 0.0                           # phi (滚转角)
        self.state[4] = np.radians(8)                 # theta (俯仰角，典型滑翔约8°)
        self.state[5] = heading                       # psi (航向角)
        self.state[8] = velocity                      # u (前向速度)
        self.state[9] = 0.0                           # v (侧向速度)
        self.state[10] = 5.0                          # w (下沉速度，体坐标系，典型约5m/s)
        
        return self.state.copy()
    
    def step(self, state: np.ndarray, t: float) -> Tuple[np.ndarray, ControlOutput]:
        """
        仿真单步
        
        参数:
            state: 当前状态 (20,)
            t: 当前时间
        
        返回:
            (next_state, control_output)
        """
        # 1. 提取状态信息
        position = state[0:3]
        euler = state[3:6]
        velocity_body = state[8:11]
        heading_raw = euler[2]  # psi (可能超出 [-π, π])
        
        # 归一化航向角到 [-π, π]
        heading = heading_raw
        while heading > np.pi:
            heading -= 2 * np.pi
        while heading < -np.pi:
            heading += 2 * np.pi
        
        # 体坐标系速度转惯性系
        psi = heading
        cos_psi, sin_psi = np.cos(psi), np.sin(psi)
        vx = velocity_body[0] * cos_psi - velocity_body[1] * sin_psi
        vy = velocity_body[0] * sin_psi + velocity_body[1] * cos_psi
        vz = -velocity_body[2]  # 体坐标系 w 向下为正，惯性系 z 向上为正
        velocity = np.array([vx, vy, vz])
        
        # 2. 控制器更新（使用归一化后的航向角）
        ctrl = self.controller.update(
            current_pos=position,
            current_vel=velocity,
            current_heading=heading,  # 已归一化
            t=t
        )
        
        # 3. 设置控制输入
        # 控制器输出是归一化值 [0,1]，需要转换为实际偏转量 (米)
        # 翼伞操纵绳最大偏转约 0.4m
        MAX_DEFLECTION_METERS = 0.4
        self.para.left = ctrl.delta_left * MAX_DEFLECTION_METERS
        self.para.right = ctrl.delta_right * MAX_DEFLECTION_METERS
        
        # 调试: 检查控制输入（每100步打印一次）
        if self.controller_debug and int(t / self.control_dt) % 100 == 0:
            d_s = min(self.para.left, self.para.right)  # 对称偏转
            d_a = self.para.left - self.para.right      # 非对称偏转
            print(f"  [控制输入] left={self.para.left:.3f}m, right={self.para.right:.3f}m")
            print(f"  [偏转] d_s={d_s:.3f}m (对称/下降), d_a={d_a:.3f}m (非对称/航向)")
            print(f"  [滑翔比] 所需={ctrl.glide_ratio_required:.1f}, 当前={ctrl.glide_ratio_current:.1f}")
        
        # 4. 根据高度更新空气密度
        self.para.update_density(position[2])
        
        # 5. 动力学积分 (一个控制周期)
        n_substeps = max(1, int(self.control_dt / self.dynamics_dt))
        actual_dt = self.control_dt / n_substeps
        
        next_state = state.copy()
        for i in range(n_substeps):
            try:
                dydt = parafoil_dynamics(next_state, t, self.para)
            except Exception as e:
                print(f"[错误] 动力学模型计算失败! t={t:.3f}s, step={i}/{n_substeps}")
                print(f"  状态[5] (heading) = {np.degrees(next_state[5]):.1f}°")
                print(f"  错误: {e}")
                raise
            
            # 检查是否有数值异常
            if np.any(np.isnan(dydt)) or np.any(np.isinf(dydt)):
                print(f"[错误] 动力学模型返回NaN/Inf! t={t:.3f}s, step={i}/{n_substeps}")
                print(f"  state[5] (heading) = {np.degrees(next_state[5]):.1f}°")
                print(f"  dydt[5] (d_heading/dt) = {np.degrees(dydt[5]):.1f}°/s")
                print(f"  dydt范围: [{np.min(dydt):.2e}, {np.max(dydt):.2e}]")
                raise ValueError("动力学模型数值异常")
            
            # 检查导数是否过大（可能导致数值不稳定）
            max_derivative = np.max(np.abs(dydt))
            if max_derivative > 1e6:
                print(f"[警告] 导数过大! t={t:.3f}s, step={i}/{n_substeps}, max={max_derivative:.2e}")
            
            next_state = next_state + dydt * actual_dt
            
            # 归一化航向角（避免累积超出范围）
            next_state[5] = self._wrap_angle(next_state[5])
            
            # 限制俯仰角和滚转角，避免欧拉角奇点（theta = ±90°）
            # 限制 theta 在 [-85°, 85°] 范围内
            theta_max = np.radians(85)
            if abs(next_state[4]) > theta_max:
                next_state[4] = np.sign(next_state[4]) * theta_max
                if self.controller_debug and i == 0:
                    print(f"  [警告] 俯仰角超出范围，已限制: {np.degrees(next_state[4]):.1f}°")
            
            # 限制滚转角在 [-85°, 85°] 范围内
            if abs(next_state[3]) > theta_max:
                next_state[3] = np.sign(next_state[3]) * theta_max
                if self.controller_debug and i == 0:
                    print(f"  [警告] 滚转角超出范围，已限制: {np.degrees(next_state[3]):.1f}°")
            
            # 检查状态是否异常
            if np.any(np.isnan(next_state)) or np.any(np.isinf(next_state)):
                print(f"[错误] 状态向量出现NaN/Inf! t={t:.3f}s, step={i}/{n_substeps}")
                print(f"  next_state[3] (roll) = {np.degrees(next_state[3]):.1f}°")
                print(f"  next_state[4] (pitch) = {np.degrees(next_state[4]):.1f}°")
                print(f"  next_state[5] (heading) = {np.degrees(next_state[5]):.1f}°")
                raise ValueError("状态向量数值异常")
        
        return next_state, ctrl
    
    @staticmethod
    def _wrap_angle(angle: float) -> float:
        """将角度归一化到 [-pi, pi]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
    
    def run(self, 
            max_time: float = None,
            stop_on_ground: bool = True,
            stop_on_target: bool = True,
            target_threshold: float = 30.0,
            verbose: bool = True) -> SimulationLog:
        """
        运行闭环仿真
        
        参数:
            max_time: 最大仿真时间，默认为轨迹时长 + 30s
            stop_on_ground: 落地时停止
            stop_on_target: 到达目标时停止
            target_threshold: 到达目标阈值 (m)
            verbose: 是否打印进度
        
        返回:
            SimulationLog: 仿真记录
        """
        if self.state is None:
            raise ValueError("请先初始化状态 (init_state)")
        
        if max_time is None:
            max_time = self.trajectory.duration + 30.0
        
        # 初始化记录
        self.log = SimulationLog()
        
        state = self.state.copy()
        t = 0.0
        
        if verbose:
            print("\n" + "=" * 60)
            print("  闭环仿真开始")
            print("=" * 60)
            print(f"  最大时间: {max_time:.1f}s")
            print(f"  控制周期: {self.control_dt*1000:.1f}ms")
            print(f"  积分步长: {self.dynamics_dt*1000:.2f}ms")
        
        start_time = time.time()
        last_print = 0
        
        while t < max_time:
            # 仿真单步
            next_state, ctrl = self.step(state, t)
            
            # 记录
            self.log.append(t, state, ctrl)
            
            # 更新状态
            state = next_state
            t += self.control_dt
            
            # 进度打印
            if verbose and t - last_print >= 5.0:
                pos = state[0:3]
                progress = self.controller.get_progress() * 100
                print(f"  t={t:6.1f}s | pos=({pos[0]:7.1f}, {pos[1]:7.1f}, {pos[2]:6.1f}) | progress={progress:5.1f}%")
                last_print = t
            
            # 停止条件: 落地
            if stop_on_ground and state[2] < 0:
                if verbose:
                    print(f"\n  [落地] t={t:.1f}s, 位置=({state[0]:.1f}, {state[1]:.1f}, {state[2]:.1f})")
                break
            
            # 停止条件: 到达目标
            if stop_on_target and self.controller.is_finished(state[0:3], threshold=target_threshold):
                if verbose:
                    print(f"\n  [到达目标] t={t:.1f}s")
                break
        
        elapsed = time.time() - start_time
        
        if verbose:
            print("=" * 60)
            print(f"  仿真完成")
            print(f"  仿真时长: {t:.1f}s, 计算耗时: {elapsed:.2f}s")
            print(f"  轨迹跟踪进度: {self.controller.get_progress()*100:.1f}%")
            final_pos = state[0:3]
            target_pos = self.trajectory[-1].position if self.trajectory else np.zeros(3)
            final_error = np.linalg.norm(final_pos - target_pos)
            print(f"  最终位置误差: {final_error:.1f}m")
            print("=" * 60)
        
        return self.log
    
    def visualize(self, save_path: str = None):
        """
        可视化仿真结果
        
        参数:
            save_path: 保存路径 (可选)
        """
        import matplotlib.pyplot as plt
        
        if self.log is None or len(self.log.t) == 0:
            print("没有仿真数据可视化")
            return
        
        data = self.log.to_arrays()
        ref_traj = self.trajectory.to_position_array()
        
        fig = plt.figure(figsize=(16, 12))
        
        # 1. 3D轨迹
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        ax1.plot(ref_traj[:, 0], ref_traj[:, 1], ref_traj[:, 2],
                 'b-', linewidth=2, alpha=0.6, label='Reference')
        ax1.plot(data['position'][:, 0], data['position'][:, 1], data['position'][:, 2],
                 'r-', linewidth=2, alpha=0.9, label='Actual')
        ax1.scatter(*ref_traj[0], c='green', s=100, marker='o', label='Start')
        ax1.scatter(*ref_traj[-1], c='red', s=100, marker='*', label='Goal')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title('3D Trajectory')
        ax1.legend()
        
        # 2. XY平面
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.plot(ref_traj[:, 0], ref_traj[:, 1], 'b-', linewidth=2, alpha=0.6, label='Reference')
        ax2.plot(data['position'][:, 0], data['position'][:, 1], 'r-', linewidth=2, alpha=0.9, label='Actual')
        ax2.scatter(ref_traj[0, 0], ref_traj[0, 1], c='green', s=100, marker='o')
        ax2.scatter(ref_traj[-1, 0], ref_traj[-1, 1], c='red', s=100, marker='*')
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_title('XY Plane')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axis('equal')
        
        # 3. 跟踪误差
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.plot(data['t'], data['cross_track_error'], 'b-', linewidth=1.5, label='Cross-track')
        ax3.plot(data['t'], data['along_track_error'], 'g-', linewidth=1.5, alpha=0.7, label='Along-track')
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Error (m)')
        ax3.set_title('Tracking Errors')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 控制输入
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.plot(data['t'], data['delta_left'], 'g-', linewidth=1.5, label='Left')
        ax4.plot(data['t'], data['delta_right'], 'm-', linewidth=1.5, label='Right')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Deflection')
        ax4.set_title('Control Inputs')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"图像已保存到: {save_path}")
        
        # 额外: 高度和速度曲线
        fig2, axes = plt.subplots(2, 1, figsize=(12, 6))
        
        # 高度
        axes[0].plot(data['t'], data['position'][:, 2], 'b-', linewidth=2, label='Altitude')
        axes[0].plot(data['t'], data['ref_position'][:, 2], 'r--', linewidth=1.5, alpha=0.7, label='Reference')
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Altitude (m)')
        axes[0].set_title('Altitude Profile')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 速度
        speed = np.linalg.norm(data['velocity'], axis=1)
        axes[1].plot(data['t'], speed, 'b-', linewidth=2)
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Speed (m/s)')
        axes[1].set_title('Speed Profile')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plt.show()


# ============================================================
#                     主程序
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="翼伞闭环仿真")
    parser.add_argument("--map-config", type=str, default="cfg/map_config.yaml",
                        help="地图配置文件")
    parser.add_argument("--model-config", type=str, default="cfg/config.yaml",
                        help="动力学模型配置文件")
    parser.add_argument("--max-time", type=float, default=None,
                        help="最大仿真时间 (s)")
    parser.add_argument("--control-dt", type=float, default=0.01,
                        help="控制周期 (s)")
    parser.add_argument("--dynamics-dt", type=float, default=0.002,
                        help="动力学积分步长 (s)")
    parser.add_argument("--position-noise", type=float, nargs=3, default=[10, -15, 0],
                        help="初始位置噪声 [dx, dy, dz]")
    parser.add_argument("--heading-noise", type=float, default=0.1,
                        help="初始航向噪声 (rad)")
    args = parser.parse_args()
    
    # 创建仿真器
    sim = ClosedLoopSimulator(
        map_config_path=args.map_config,
        model_config_path=args.model_config,
        control_dt=args.control_dt,
        dynamics_dt=args.dynamics_dt
    )
    
    # 规划
    if not sim.plan(max_time=30.0):
        exit(1)
    
    # 初始化状态 (加一点初始偏差)
    sim.init_state(
        position_noise=args.position_noise,
        heading_noise=args.heading_noise
    )
    
    # 运行仿真
    log = sim.run(max_time=args.max_time)
    
    # 可视化
    sim.visualize()
