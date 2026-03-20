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
from typing import Tuple, Optional, Dict, List, Any
from dataclasses import dataclass
from scipy.integrate import odeint
import time
import yaml
from datetime import datetime

# 添加项目根目录到路径
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from planning.map_manager import MapManager
from planning.trajectory import Trajectory, TrajectoryPoint
from planning.kinodynamic_rrt import KinodynamicRRTStar
from planning.trajectory_postprocess import TrajectoryPostprocessor, validate_trajectory
from control.adrc_controller import ParafoilADRCController, ControlOutput
from control.pid_controller import ParafoilPIDController
from models.parafoil_model import ParafoilParams, parafoil_dynamics

# Benchmark 框架
from benchmark.rng_manager import RNGManager, get_config_hash, get_git_commit
from benchmark.metrics import (
    MetricsCalculator, FailureDetector, FailureThresholds, TerminationReason
)
from benchmark.outputs import MetricsOutput, CaseOutput, create_output_dir


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
    支持 Benchmark 框架的标准化输出
    """
    
    def __init__(self, 
                 map_config_path: str,
                 model_config_path: str,
                 control_dt: float = 0.01,
                 dynamics_dt: float = 0.001,
                 seed: int = None,
                 scene_name: str = "default",
                 controller_type: str = "adrc",
                 controller_kwargs: Optional[Dict[str, Any]] = None,
                 quiet: bool = False):
        """
        参数:
            map_config_path: 地图配置文件路径
            model_config_path: 动力学模型配置文件路径
            control_dt: 控制周期 (s)
            dynamics_dt: 动力学积分步长 (s)
            seed: 随机数种子（用于 benchmark，None 表示不使用）
            scene_name: 场景名称（用于 benchmark 输出）
            quiet: 静默模式（不打印详细信息）
        """
        self.control_dt = control_dt
        self.dynamics_dt = dynamics_dt
        self.scene_name = scene_name
        self.quiet = quiet
        self.controller_type = controller_type.lower()
        self.controller_kwargs = dict(controller_kwargs or {})
        
        # 保存配置路径
        self.map_config_path = map_config_path
        self.model_config_path = model_config_path
        
        # RNG 管理器（用于可复现性）
        self.seed = seed
        self.rng: Optional[RNGManager] = None
        if seed is not None:
            self.rng = RNGManager(seed)
        
        # 加载地图
        if not quiet:
            print("[1/4] 加载地图配置...")
        # 如果有 seed，传给 map_manager 用于场景随机化
        if self.rng is not None:
            self.map_manager = MapManager.from_yaml(
                map_config_path, 
                rng=self.rng.get_env_rng(),
                quiet=quiet
            )
        else:
            self.map_manager = MapManager.from_yaml(map_config_path, quiet=quiet)

        # 读取轨迹参数（参考速度）
        with open(map_config_path, 'r', encoding='utf-8') as f:
            map_cfg = yaml.safe_load(f)
        traj_cfg = map_cfg.get('trajectory', {}) if isinstance(map_cfg, dict) else {}
        reference_speed = traj_cfg.get('reference_speed', 12.0)
        self.reference_speed = reference_speed
        
        # 检查可达性
        if not quiet:
            self.map_manager.print_reachability_report()
        
        # 加载动力学模型参数
        if not quiet:
            print("[2/4] 加载动力学模型...")
        self.para = ParafoilParams.from_yaml(model_config_path)
        
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
        min_turn_r = self.map_manager.constraints.min_turn_radius  # 从地图配置读取
        self.controller = ParafoilADRCController(
            heading_kp=6.0,           # [调参] 提高响应增益：原1.5→6.0
            heading_kd=2.0,           # [调参] 提高阻尼：原0.6→2.0
            heading_eso_omega=40.0,   # [调参] 提高ESO带宽：原6.0→40.0
            heading_td_r=20.0,        # 加快参考跟踪
            heading_b0=0.5,           # [调参] 修正控制效能估计：原3.0→0.5，kp/b0从0.5提升到12
            lateral_kp=0.01,          # [调参] 提高横向修正强度：原0.006→0.01
            lateral_kd=0.003,         # 抑制来回摆动
            glide_ratio_natural=6.48, # 自然滑翔比 (无对称偏转)
            glide_ratio_min=2.47,     # 最小滑翔比 (最大对称偏转)
            descent_kp=0.5,           # 下降率控制增益
            descent_margin=1.15,      # 15%余量：偏向更陡下降，预防预算不足造成的高度延迟
            reference_speed=reference_speed,
            min_turn_radius=min_turn_r,  # 与规划器保持一致
            lookahead_distance=100.0, # [调参] 增大前视距离：翼伞响应慢，需要更长预见时间
            max_deflection=1.0,       # 打满：允许使用全部滑翔比范围[2.47, 6.48]
            dt=control_dt
        )
        
        # 调试模式（可通过参数控制）
        # 静默模式下禁用调试输出
        self.controller_debug = not self.quiet  # 静默模式下关闭控制器调试
        if self.controller_type == "pid":
            self.controller = ParafoilPIDController(
                heading_kp=1.8,
                heading_ki=0.10,
                heading_kd=0.60,
                heading_integral_limit=0.9,
                heading_derivative_alpha=0.88,
                heading_eso_omega=40.0,
                heading_td_r=20.0,
                heading_b0=0.5,
                lateral_kp=0.01,
                lateral_kd=0.003,
                glide_ratio_natural=6.48,
                glide_ratio_min=2.47,
                descent_kp=0.5,
                descent_margin=1.15,
                reference_speed=reference_speed,
                min_turn_radius=min_turn_r,
                lookahead_distance=100.0,
                max_deflection=1.0,
                dt=control_dt
            )
        elif self.controller_type != "adrc":
            raise ValueError(f"Unsupported controller_type: {controller_type}")

        shared_controller_config = {
            'heading_eso_omega': 40.0,
            'heading_td_r': 20.0,
            'heading_b0': 0.5,
            'lateral_kp': 0.14,
            'lateral_kd': 0.03,
            'glide_ratio_natural': 6.48,
            'glide_ratio_min': 2.47,
            'descent_kp': 0.22,
            'descent_margin': 1.0,
            'reference_speed': reference_speed,
            'min_turn_radius': min_turn_r,
            'lookahead_distance': 85.0,
            'lookahead_min_distance': 30.0,
            'lookahead_max_distance': 110.0,
            'lookahead_error_scale': 0.025,
            'closest_search_window': 120,
            'closest_reacquire_distance': 35.0,
            'closest_reacquire_window': 800,
            'closest_backtrack_window': 4,
            'cross_track_softening': 3.0,
            'max_cross_track_heading_correction': np.radians(32.0),
            'max_deflection': 1.0,
            'dt': control_dt,
        }

        if self.controller_type == "pid":
            self.controller_config = dict(shared_controller_config)
            self.controller_config.update({
                'heading_kp': 1.8,
                'heading_ki': 0.10,
                'heading_kd': 0.60,
                'heading_integral_limit': 0.9,
                'heading_derivative_alpha': 0.88,
            })
            self.controller_config.update(self.controller_kwargs)
            self.controller = ParafoilPIDController(**self.controller_config)
        else:
            self.controller_config = dict(shared_controller_config)
            self.controller_config.update({
                'heading_kp': 6.0,
                'heading_kd': 2.0,
            })
            self.controller_config.update(self.controller_kwargs)
            self.controller = ParafoilADRCController(**self.controller_config)

        if self.controller_debug:
            self.controller.set_debug(True)
        
        # 轨迹和仿真状态
        self.trajectory: Optional[Trajectory] = None
        self.state: Optional[np.ndarray] = None
        self.log: Optional[SimulationLog] = None
        
        # === Benchmark 相关 ===
        # 失败检测器
        target_pos = None
        if self.map_manager.target:
            target_pos = self.map_manager.target.position
        
        # 收集禁飞区信息
        nfz_list = []
        for obs in self.map_manager.obstacles:
            obs_type = obs.__class__.__name__.lower()
            if obs_type == 'cylinder':
                nfz_list.append({
                    'type': 'cylinder',
                    'center': [obs.center[0], obs.center[1]],
                    'radius': obs.radius,
                    'z_min': obs.z_min,
                    'z_max': obs.z_max
                })
            elif obs_type == 'prism':
                nfz_list.append({
                    'type': 'polygon',
                    'vertices': obs.polygon.vertices,
                    'z_min': obs.z_min,
                    'z_max': obs.z_max
                })
        
        self.failure_detector = FailureDetector(
            thresholds=FailureThresholds(
                landing_radius=50.0,
                landing_altitude=20.0,
                safety_margin=self.map_manager.constraints.safety_margin
            ),
            no_fly_zones=nfz_list,
            target_position=target_pos
        )
        self.failure_detector.set_dt(control_dt)
        
        # 指标计算器
        self.metrics_calculator = MetricsCalculator(target_position=target_pos)
        
        # 规划耗时记录
        self.planning_time: float = 0.0
        
        # Benchmark 输出
        self.metrics_output: Optional[MetricsOutput] = None
        self.case_output: Optional[CaseOutput] = None
    
    def plan(self, max_time: float = 30.0, smooth: bool = True, 
             progress_callback: callable = None) -> bool:
        """
        使用Kinodynamic RRT*规划路径

        特点：
        1. 使用Dubins曲线扩展，生成平滑路径
        2. 在规划阶段考虑滑翔比约束
        3. 使用 TrajectoryPostprocessor 进行平滑处理
        4. 自动注入螺旋下降段消高

        参数:
            max_time: 最大规划时间 (s)
            smooth: 是否进行轨迹平滑（默认True）
            progress_callback: 规划进度回调 callback(iteration, max_iterations)

        返回:
            是否成功
        """
        if not self.quiet:
            print("[3/4] 运行 Kinodynamic RRT* 路径规划...")

        # 固定 RRT* 的随机数种子，确保相同 seed 产生相同规划结果
        # 这样切换控制器时，参考轨迹保持不变
        if self.seed is not None:
            np.random.seed(self.seed)

        # 记录规划开始时间
        plan_start_time = time.time()

        # 创建Kinodynamic RRT*规划器
        kino_planner = KinodynamicRRTStar(self.map_manager, quiet=self.quiet)
        path, info = kino_planner.plan(max_time=max_time, progress_callback=progress_callback)

        # 记录规划耗时
        self.planning_time = time.time() - plan_start_time

        if path is None:
            if not self.quiet:
                print("    路径规划失败!")
            return False

        if not self.quiet:
            print(f"    规划完成: {len(path)} 航点, 耗时: {self.planning_time:.2f}s")
            if 'path_length' in info:
                print(f"    路径长度: {info['path_length']:.1f}m")

        # 提取 edge_paths / node_chain（供 Dubins 感知后处理器使用）
        edge_paths = info.get('edge_paths', None)
        node_chain = info.get('node_chain', None)

        # 使用 TrajectoryPostprocessor 进行平滑处理
        if not self.quiet:
            print("[4/4] 轨迹后处理（平滑 + 时间参数化）...")
        postprocessor = TrajectoryPostprocessor(
            reference_speed=self.reference_speed,
            control_frequency=1.0 / self.control_dt,
            min_turn_radius=self.map_manager.constraints.min_turn_radius,
            max_glide_ratio=self.map_manager.constraints.glide_ratio,
            min_glide_ratio=self.map_manager.constraints.min_glide_ratio,
            map_manager=self.map_manager,
            quiet=self.quiet
        )

        end_heading = self.map_manager.target.approach_heading if self.map_manager.target else None
        self.trajectory = postprocessor.process(
            path, smooth=smooth, end_heading=end_heading,
            edge_paths=edge_paths,
            node_chain=node_chain
        )

        if len(self.trajectory) == 0:
            if not self.quiet:
                print("    轨迹生成失败!")
            return False

        # 验证轨迹
        if not self.quiet:
            self._validate_trajectory_descent_rate()

        self.controller.set_trajectory(self.trajectory)

        return True
    
    def _path_to_trajectory(self, path: List[np.ndarray]) -> Trajectory:
        """
        将路径点列表转换为Trajectory对象
        
        参数:
            path: 路径点列表 [np.array([x,y,z]), ...]
        
        返回:
            Trajectory对象
        """
        trajectory = Trajectory()
        
        if len(path) < 2:
            return trajectory
        
        # 计算每个点的信息
        dt = 1.0 / self.smoother.control_frequency
        t = 0.0
        
        for i, pos in enumerate(path):
            # 计算航向
            if i < len(path) - 1:
                dx = path[i+1][0] - pos[0]
                dy = path[i+1][1] - pos[1]
                heading = np.arctan2(dy, dx)
            else:
                # 最后一点使用进场航向
                heading = self.map_manager.target.approach_heading
            
            # 计算速度向量
            velocity = self.reference_speed * np.array([
                np.cos(heading), np.sin(heading), 0
            ])
            
            # 计算下降分量
            if i < len(path) - 1:
                dz = pos[2] - path[i+1][2]
                dxy = np.linalg.norm(path[i+1][:2] - pos[:2])
                if dxy > 0:
                    descent_rate = self.reference_speed * dz / dxy
                    velocity[2] = -descent_rate
            
            # 估算曲率
            curvature = 0.0
            if 0 < i < len(path) - 1:
                v1 = path[i][:2] - path[i-1][:2]
                v2 = path[i+1][:2] - path[i][:2]
                cross = v1[0]*v2[1] - v1[1]*v2[0]
                d1, d2 = np.linalg.norm(v1), np.linalg.norm(v2)
                if d1 > 0 and d2 > 0:
                    curvature = 2 * abs(cross) / (d1 * d2 * (d1 + d2))
            
            point = TrajectoryPoint(
                t=t,
                position=pos.copy(),
                velocity=velocity,
                heading=heading,
                curvature=curvature
            )
            trajectory.append(point)
            
            # 计算到下一点的时间
            if i < len(path) - 1:
                dist = np.linalg.norm(path[i+1] - pos)
                t += dist / self.reference_speed
        
        return trajectory
    
    def _validate_trajectory_descent_rate(self):
        """
        验证轨迹的下降率是否在翼伞物理可行范围内
        
        翼伞是欠驱动系统，下沉率有物理限制:
        - 最小下沉率 = 速度 / 最大滑翔比
        - 最大下沉率 = 速度 / 最小滑翔比
        """
        if self.trajectory is None or len(self.trajectory) < 2:
            return
        
        # 从配置获取滑翔比
        glide_max = self.map_manager.constraints.glide_ratio  # 6.48
        glide_min = self.map_manager.constraints.min_glide_ratio  # 2.47
        v = self.reference_speed  # 9 m/s
        
        # 物理极限下沉率
        sink_rate_min = v / glide_max  # 最小下沉率 ≈ 1.4 m/s
        sink_rate_max = v / glide_min  # 最大下沉率 ≈ 3.6 m/s
        
        if not self.quiet:
            print(f"\n    === 轨迹下降率验证 ===")
            print(f"    物理可行下沉率: {sink_rate_min:.2f} ~ {sink_rate_max:.2f} m/s")
        
        # 分析轨迹的下降率
        positions = self.trajectory.to_position_array()
        n_points = len(positions)
        
        # 计算每段的下降率
        climb_segments = 0  # 爬升段（不可能实现）
        too_slow_segments = 0  # 下降太慢（可能难以实现）
        too_fast_segments = 0  # 下降太快（会超前于轨迹）
        ok_segments = 0
        
        window = 10  # 用10个点的窗口计算平均下降率
        for i in range(0, n_points - window, window):
            dz = positions[i, 2] - positions[i + window, 2]  # 高度变化（下降为正）
            dxy = np.linalg.norm(positions[i + window, :2] - positions[i, :2])  # 水平距离
            
            if dxy < 1e-6:
                continue
                
            # 实际下降率 = 下降高度 / 水平距离 × 速度
            # 或者直接用 dz / dt，但我们用等效的滑翔比
            actual_glide = dxy / max(dz, 0.01) if dz > 0 else float('inf')
            
            if dz < -1.0:  # 爬升超过1m
                climb_segments += 1
            elif actual_glide > glide_max * 1.2:  # 下降太慢
                too_slow_segments += 1
            elif actual_glide < glide_min * 0.8:  # 下降太快
                too_fast_segments += 1
            else:
                ok_segments += 1
        
        total = climb_segments + too_slow_segments + too_fast_segments + ok_segments
        if not self.quiet and total > 0:
            ok_ratio = ok_segments / total * 100
            print(f"    轨迹段统计 (共{total}段):")
            print(f"      - 可跟踪:   {ok_segments} ({ok_ratio:.1f}%)")
            if climb_segments > 0:
                print(f"      - 爬升段:   {climb_segments} (⚠️ 物理不可能!)")
            if too_slow_segments > 0:
                print(f"      - 下降过慢: {too_slow_segments} (⚠️ 可能难以跟踪)")
            if too_fast_segments > 0:
                print(f"      - 下降过快: {too_fast_segments} (会超前于轨迹)")
        
        # 整体下降率
        total_dz = positions[0, 2] - positions[-1, 2]
        total_dxy = 0
        for i in range(n_points - 1):
            total_dxy += np.linalg.norm(positions[i+1, :2] - positions[i, :2])
        
        if not self.quiet and total_dxy > 0:
            avg_glide = total_dxy / max(total_dz, 0.01)
            print(f"    整体平均滑翔比: {avg_glide:.2f} (目标范围: {glide_min:.2f} ~ {glide_max:.2f})")
    
    def init_state(self,
                   position: np.ndarray = None,
                   heading: float = None,
                   velocity: float = None,
                   position_noise: np.ndarray = None,
                   heading_noise: float = 0.0,
                   use_rng: bool = True) -> np.ndarray:
        """
        初始化仿真状态
        
        参数:
            position: 初始位置 [x, y, z]，默认使用轨迹起点
            heading: 初始航向 (rad)，默认使用轨迹起点
            velocity: 初始前向速度 (m/s)
            position_noise: 位置噪声 [dx, dy, dz]，如果有 RNG 且 use_rng=True 则自动采样
            heading_noise: 航向噪声 (rad)，如果有 RNG 且 use_rng=True 则自动采样
            use_rng: 是否使用 RNG 采样噪声（仅当 self.rng 存在时生效）
        
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
        
        # 如果有 RNG，使用 RNG 采样噪声
        if self.rng is not None and use_rng:
            if position_noise is None:
                # 默认噪声范围
                position_noise = self.rng.sample_position_noise(
                    x_range=(-20, 20),
                    y_range=(-20, 20),
                    z_range=(-10, 10)
                )
            if heading_noise == 0.0:
                heading_noise = self.rng.sample_heading_noise(range_rad=0.2)
        
        # 添加噪声
        if position_noise is not None:
            position = position + np.array(position_noise)
        heading = heading + heading_noise

        # 默认使用参考速度，消除初始速度偏差
        if velocity is None:
            velocity = self.reference_speed

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
        # w (下沉速度, 体坐标系)
        # 惯性下降率 v_D = -sin(θ)*u + cos(θ)*w  (NED 约定)
        # 自然滑翔比 6.48, u=velocity → 惯性下降率 = u/GR
        # u=8: 8/6.48=1.235 → w = (1.235+sin(8°)*8)/cos(8°) ≈ 2.37
        # u=10: 10/6.48=1.54 → w ≈ 2.96
        theta = np.radians(8)
        descent_rate = velocity / 6.48  # 匹配自然滑翔比
        w_init = (descent_rate + np.sin(theta) * velocity) / np.cos(theta)
        self.state[10] = w_init                       # w (下沉速度，匹配自然滑翔比平衡态)
        
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
        
        # 体坐标系速度转惯性系 —— 使用完整 DCM（含 roll, pitch, yaw）
        # 注意: 之前只用 yaw 做 2D 旋转，忽略了 pitch 分量
        #   u*sin(theta) ≈ 10*sin(8°) ≈ 1.4 m/s 垂直分量被漏掉
        #   导致传给控制器的下降率偏大，对称偏转控制失调
        from models.parafoil_model import euler_to_dcm
        phi, theta_e, psi_e = euler[0], euler[1], heading
        R_nb = euler_to_dcm(phi, theta_e, psi_e)   # 惯性→体
        v_body_col = velocity_body.reshape(3, 1)
        v_inertial = R_nb.T @ v_body_col            # 体→惯性
        v_inertial[2, 0] = -v_inertial[2, 0]        # NED→NEU (z 向上为正)
        velocity = v_inertial.flatten()
        
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
                if not self.quiet:
                    print(f"[错误] 动力学模型计算失败! t={t:.3f}s, step={i}/{n_substeps}")
                    print(f"  状态[5] (heading) = {np.degrees(next_state[5]):.1f}°")
                    print(f"  错误: {e}")
                raise
            
            # 检查是否有数值异常
            if np.any(np.isnan(dydt)) or np.any(np.isinf(dydt)):
                if not self.quiet:
                    print(f"[错误] 动力学模型返回NaN/Inf! t={t:.3f}s, step={i}/{n_substeps}")
                    print(f"  state[5] (heading) = {np.degrees(next_state[5]):.1f}°")
                    print(f"  dydt[5] (d_heading/dt) = {np.degrees(dydt[5]):.1f}°/s")
                    print(f"  dydt范围: [{np.min(dydt):.2e}, {np.max(dydt):.2e}]")
                raise ValueError("动力学模型数值异常")
            
            # 检查导数是否过大（可能导致数值不稳定）
            max_derivative = np.max(np.abs(dydt))
            if max_derivative > 1e6 and not self.quiet:
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
                if not self.quiet:
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
            enable_failure_detection: bool = True,
            verbose: bool = True,
            progress_callback: callable = None) -> SimulationLog:
        """
        运行闭环仿真
        
        参数:
            max_time: 最大仿真时间，默认为轨迹时长 + 30s
            stop_on_ground: 落地时停止
            stop_on_target: 到达目标时停止
            target_threshold: 到达目标阈值 (m)
            enable_failure_detection: 是否启用失败检测（Benchmark 模式）
            verbose: 是否打印进度
            progress_callback: 进度回调函数 callback(t, progress)
        
        返回:
            SimulationLog: 仿真记录
        """
        if self.state is None:
            raise ValueError("请先初始化状态 (init_state)")
        
        if max_time is None:
            max_time = self.trajectory.duration + 30.0
        
        # 初始化记录
        self.log = SimulationLog()
        
        # 重置失败检测器
        if enable_failure_detection:
            self.failure_detector.reset()
        
        state = self.state.copy()
        t = 0.0
        termination_reason: Optional[TerminationReason] = None
        
        if verbose:
            print("\n" + "=" * 60)
            print("  闭环仿真开始")
            print("=" * 60)
            print(f"  最大时间: {max_time:.1f}s")
            print(f"  控制周期: {self.control_dt*1000:.1f}ms")
            print(f"  积分步长: {self.dynamics_dt*1000:.2f}ms")
            if self.seed is not None:
                print(f"  Seed: {self.seed}")
        
        start_time = time.time()
        last_print = 0
        prev_euler = None
        
        while t < max_time:
            # 仿真单步
            try:
                next_state, ctrl = self.step(state, t)
            except ValueError as e:
                # 数值异常 - H3 失败
                if enable_failure_detection:
                    termination_reason = TerminationReason.H3_NUMERICAL_EXPLOSION
                    self.failure_detector.detection_results['H3_numerical_explosion'] = True
                    self.failure_detector.hard_fail_detected = True
                    self.failure_detector.termination_reason = termination_reason
                    self.failure_detector.termination_time = t
                if verbose:
                    print(f"\n  [数值异常] t={t:.1f}s: {e}")
                break
            
            # 记录
            self.log.append(t, state, ctrl)
            
            # === 失败检测 ===
            if enable_failure_detection:
                # 提取状态
                position = state[0:3]
                velocity_body = state[8:11]
                euler = state[3:6]
                
                # 计算惯性系速度
                psi = euler[2]
                vx = velocity_body[0] * np.cos(psi) - velocity_body[1] * np.sin(psi)
                vy = velocity_body[0] * np.sin(psi) + velocity_body[1] * np.cos(psi)
                vz = -velocity_body[2]
                velocity = np.array([vx, vy, vz])
                
                # 计算姿态角速率（简单差分）
                if prev_euler is not None:
                    euler_rate = (euler - prev_euler) / self.control_dt
                    # 处理角度跳变
                    for i in range(3):
                        if euler_rate[i] > np.pi / self.control_dt:
                            euler_rate[i] -= 2 * np.pi / self.control_dt
                        elif euler_rate[i] < -np.pi / self.control_dt:
                            euler_rate[i] += 2 * np.pi / self.control_dt
                else:
                    euler_rate = np.zeros(3)
                prev_euler = euler.copy()
                
                # 检测失败
                failure = self.failure_detector.check_step(
                    t=t,
                    position=position,
                    velocity=velocity,
                    euler=euler,
                    euler_rate=euler_rate,
                    control=(ctrl.delta_left, ctrl.delta_right),
                    cross_track_error=ctrl.cross_track_error,
                    max_time=max_time
                )
                
                if failure is not None:
                    termination_reason = failure
                    if verbose:
                        print(f"\n  [失败检测] t={t:.1f}s: {failure.value}")
                    break
            
            # 更新状态
            state = next_state
            t += self.control_dt
            
            # 进度回调（每秒更新一次）
            if progress_callback and t - last_print >= 1.0:
                progress = self.controller.get_progress()
                progress_callback(t, progress)
                last_print = t
            # 进度打印（verbose 模式，每5秒打印一次）
            elif verbose and t - last_print >= 5.0:
                pos = state[0:3]
                progress = self.controller.get_progress() * 100
                print(f"  t={t:6.1f}s | pos=({pos[0]:7.1f}, {pos[1]:7.1f}, {pos[2]:6.1f}) | progress={progress:5.1f}%")
                last_print = t
            
            # 停止条件: 落地
            if stop_on_ground and state[2] < 0:
                if verbose:
                    print(f"\n  [落地] t={t:.1f}s, 位置=({state[0]:.1f}, {state[1]:.1f}, {state[2]:.1f})")
                termination_reason = TerminationReason.GROUND_CONTACT
                break
            
            # 停止条件: 到达目标
            terminal_altitude_gate = (
                self.failure_detector.thresholds.landing_altitude
                if enable_failure_detection else 20.0
            )
            if (stop_on_target and
                    state[2] <= terminal_altitude_gate and
                    self.controller.is_finished(state[0:3], threshold=target_threshold)):
                if verbose:
                    print(f"\n  [到达目标] t={t:.1f}s")
                # 检查是否真正成功
                if enable_failure_detection:
                    if self.failure_detector.check_success(state[0:3]):
                        termination_reason = TerminationReason.SUCCESS
                    else:
                        termination_reason = TerminationReason.GROUND_CONTACT
                else:
                    termination_reason = TerminationReason.SUCCESS
                break
        
        elapsed = time.time() - start_time
        
        # 如果循环结束但没有明确的终止原因，标记为超时
        if termination_reason is None and t >= max_time:
            termination_reason = TerminationReason.S4_TIMEOUT
            if enable_failure_detection and hasattr(self, 'failure_detector'):
                self.failure_detector._trigger_failure(TerminationReason.S4_TIMEOUT, t)
                self.failure_detector.detection_results['S4_timeout'] = True
        
        # 存储终止信息
        self._last_termination_reason = termination_reason
        self._last_wall_time = elapsed
        self._last_flight_time = t
        self._last_final_state = state.copy()
        
        if verbose:
            print("=" * 60)
            print(f"  仿真完成")
            print(f"  仿真时长: {t:.1f}s, 计算耗时: {elapsed:.2f}s")
            if termination_reason:
                print(f"  终止原因: {termination_reason.value}")
            
            # 计算跟踪指标
            current_pos = state[0:3]
            final_ref = self.trajectory[-1].position
            
            # 索引进度
            index_progress = self.controller.get_progress()
            
            # 到目标终点的距离
            dist_to_goal = np.linalg.norm(current_pos - final_ref)
            horizontal_to_goal = np.linalg.norm(current_pos[:2] - final_ref[:2])
            altitude_to_goal = current_pos[2] - final_ref[2]
            
            print(f"\n  === 跟踪质量指标 ===")
            print(f"  索引进度:     {index_progress*100:.1f}% (轨迹点遍历)")
            print(f"  ")
            print(f"  到目标终点:   {dist_to_goal:.1f}m")
            print(f"    - 水平距离: {horizontal_to_goal:.1f}m")
            print(f"    - 高度差:   {altitude_to_goal:+.1f}m")
            
            # 判定成功/失败
            if enable_failure_detection:
                success = self.failure_detector.check_success(current_pos)
                print(f"\n  === 最终判定 ===")
                print(f"  成功: {'✓ YES' if success else '✗ NO'}")
                if not success and termination_reason:
                    print(f"  原因: {termination_reason.value}")
            
            print("=" * 60)
        
        return self.log
    
    def compute_metrics(self) -> Dict[str, Any]:
        """
        计算 Quality 指标
        
        返回:
            指标字典
        """
        if self.log is None or len(self.log.t) == 0:
            return {}
        
        data = self.log.to_arrays()
        
        # 构建控制量数组
        controls = np.column_stack([
            data['delta_left'],
            data['delta_right']
        ])
        
        # 计算指标
        metrics = self.metrics_calculator.compute_all(
            positions=data['position'],
            ref_positions=data['ref_position'],
            controls=controls,
            euler=data['euler'],
            times=data['t'],
            cross_track_errors=data['cross_track_error']
        )
        
        return metrics
    
    def export_benchmark_results(self, output_dir: str) -> Tuple[str, str]:
        """
        导出 Benchmark 标准化结果
        
        参数:
            output_dir: 输出目录（例如 benchmark/outputs/exp_xxx/scene1/seed_001）
        
        返回:
            (metrics_path, case_path) 两个文件的路径
        """
        import json
        
        if self.log is None or len(self.log.t) == 0:
            raise ValueError("没有仿真数据可导出")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # === 1. 构建 metrics.json ===
        metrics_output = MetricsOutput()
        
        # 复现信息
        metrics_output.seed = self.seed if self.seed is not None else -1
        metrics_output.config_hash = self._compute_config_hash()
        metrics_output.git_commit = get_git_commit()
        
        # 场景信息
        metrics_output.scene = self.scene_name
        metrics_output.wind_speed = 0.0  # 目前无风
        metrics_output.controller = self.controller_type
        
        # 成功判定
        final_pos = self._last_final_state[0:3] if hasattr(self, '_last_final_state') else self.log.position[-1]
        metrics_output.success = self.failure_detector.check_success(final_pos)
        metrics_output.termination_reason = (
            self._last_termination_reason.value 
            if hasattr(self, '_last_termination_reason') and self._last_termination_reason 
            else "unknown"
        )
        metrics_output.termination_time = (
            self._last_flight_time if hasattr(self, '_last_flight_time') else self.log.t[-1]
        )
        
        # 失败检测结果
        det = self.failure_detector.detection_results
        metrics_output.hard_fail.H1_nfz_violation = det.get('H1_nfz_violation', False)
        metrics_output.hard_fail.H1_clearance_violation = det.get('H1_clearance_violation', False)
        metrics_output.hard_fail.H2_attitude_violation = det.get('H2_attitude_violation', False)
        metrics_output.hard_fail.H3_numerical_explosion = det.get('H3_numerical_explosion', False)
        
        metrics_output.soft_fail.S1_tracking_divergence = det.get('S1_tracking_divergence', False)
        metrics_output.soft_fail.S3_saturation_divergence = det.get('S3_saturation_divergence', False)
        metrics_output.soft_fail.S4_timeout = det.get('S4_timeout', False)
        
        # Quality 指标
        quality = self.compute_metrics()
        metrics_output.quality.ADE = quality.get('ADE', 0.0)
        metrics_output.quality.RMSE = quality.get('RMSE', 0.0)
        metrics_output.quality.FDE = quality.get('FDE', 0.0)
        metrics_output.quality.FDE_horizontal = quality.get('FDE_horizontal', 0.0)
        metrics_output.quality.FDE_vertical = quality.get('FDE_vertical', 0.0)
        metrics_output.quality.mean_cross_track_error = quality.get('mean_cross_track_error', 0.0)
        metrics_output.quality.max_cross_track_error = quality.get('max_cross_track_error', 0.0)
        metrics_output.quality.delta_u_sum = quality.get('delta_u_sum', 0.0)
        metrics_output.quality.mean_control_effort = quality.get('mean_control_effort', 0.0)
        metrics_output.quality.max_roll = quality.get('max_roll', 0.0)
        metrics_output.quality.max_pitch = quality.get('max_pitch', 0.0)
        metrics_output.quality.max_yaw_rate = quality.get('max_yaw_rate', 0.0)
        metrics_output.quality.saturation_ratio = quality.get('saturation_ratio', 0.0)
        
        # 时间信息
        metrics_output.timing.planning_time = self.planning_time
        metrics_output.timing.flight_time = quality.get('flight_time', 0.0)
        metrics_output.timing.wall_time = self._last_wall_time if hasattr(self, '_last_wall_time') else 0.0
        
        # 时间戳
        metrics_output.timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # 保存
        metrics_path = os.path.join(output_dir, 'metrics.json')
        metrics_output.save(metrics_path)
        
        # === 2. 构建 case.json ===
        case_output = CaseOutput()
        
        # 配置快照
        case_output.config = {
            'map_config_path': self.map_config_path,
            'model_config_path': self.model_config_path,
            'control_dt': self.control_dt,
            'dynamics_dt': self.dynamics_dt,
            'seed': self.seed
        }
        
        # 障碍物/禁飞区
        for obs in self.map_manager.obstacles:
            obs_type = obs.__class__.__name__.lower()
            if obs_type == 'cylinder':
                case_output.no_fly_zones.append({
                    'type': 'cylinder',
                    'center': [float(obs.center[0]), float(obs.center[1])],
                    'radius': float(obs.radius),
                    'z_min': float(obs.z_min),
                    'z_max': float(obs.z_max)
                })
            elif obs_type == 'prism':
                case_output.no_fly_zones.append({
                    'type': 'polygon',
                    'vertices': [[float(v[0]), float(v[1])] for v in obs.polygon.vertices],
                    'z_min': float(obs.z_min),
                    'z_max': float(obs.z_max)
                })
        
        # 起终点
        if self.map_manager.start:
            case_output.start_position = [
                float(self.map_manager.start.x),
                float(self.map_manager.start.y),
                float(self.map_manager.start.z)
            ]
        if self.map_manager.target:
            case_output.target_position = [
                float(self.map_manager.target.position[0]),
                float(self.map_manager.target.position[1]),
                float(self.map_manager.target.position[2])
            ]
        
        # 参考轨迹
        if self.trajectory:
            case_output.set_reference_trajectory(self.trajectory)
        
        # 实际轨迹和控制数据
        data = self.log.to_arrays()
        for i in range(len(data['t'])):
            case_output.add_trajectory_point(
                t=data['t'][i],
                position=data['position'][i],
                velocity=data['velocity'][i],
                euler=data['euler'][i],
                ref_position=data['ref_position'][i],
                ref_heading=data['ref_heading'][i]
            )
            case_output.add_control_point(
                t=data['t'][i],
                delta_left=data['delta_left'][i],
                delta_right=data['delta_right'][i],
                cross_track_error=data['cross_track_error'][i],
                heading_error=data['heading_error'][i]
            )
        
        # 添加终止事件
        if hasattr(self, '_last_termination_reason') and self._last_termination_reason:
            case_output.add_event(
                t=self._last_flight_time if hasattr(self, '_last_flight_time') else data['t'][-1],
                event_type=self._last_termination_reason.value,
                details={
                    'final_position': final_pos.tolist() if isinstance(final_pos, np.ndarray) else final_pos
                }
            )
        
        # 保存
        case_path = os.path.join(output_dir, 'case.json')
        case_output.save(case_path)
        
        return metrics_path, case_path
    
    def _compute_config_hash(self) -> str:
        """计算配置哈希"""
        configs = {}
        
        # 加载配置文件内容
        try:
            with open(self.map_config_path, 'r', encoding='utf-8') as f:
                configs['map_config'] = yaml.safe_load(f)
        except:
            configs['map_config'] = self.map_config_path
        
        try:
            with open(self.model_config_path, 'r', encoding='utf-8') as f:
                configs['model_config'] = yaml.safe_load(f)
        except:
            configs['model_config'] = self.model_config_path
        
        # 添加仿真参数
        configs['sim_config'] = {
            'control_dt': self.control_dt,
            'dynamics_dt': self.dynamics_dt
        }
        
        # 添加控制器参数（这里简化处理）
        configs['controller_config'] = {
            'type': self.controller_type,
            'params': getattr(self, 'controller_config', {}),
            'overrides': self.controller_kwargs
        }
        
        # 风场配置
        configs['disturbance_config'] = {
            'wind': 'none'
        }
        
        return get_config_hash(configs)
    
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

    def export_to_json(self, output_path: str) -> str:
        """
        导出仿真数据到 JSON 文件（用于 Web 可视化）

        参数:
            output_path: 输出文件路径

        返回:
            实际保存的文件路径
        """
        import json
        from datetime import datetime

        if self.log is None or len(self.log.t) == 0:
            print("没有仿真数据可导出")
            return None

        data = self.log.to_arrays()

        # 构建导出数据
        export_data = {
            'type': 'closed_loop_simulation',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),

            # 参考轨迹（平滑后的轨迹）
            'reference_trajectory': [],

            # 实际轨迹
            'actual_trajectory': [],

            # 控制数据
            'control_data': [],

            # 仿真信息
            'info': {
                'duration': float(data['t'][-1]),
                'n_steps': len(data['t']),
                'control_dt': self.control_dt,
                'reference_speed': self.reference_speed,
            }
        }

        # 添加起点终点
        if self.map_manager.start:
            export_data['start'] = {
                'x': float(self.map_manager.start.x),
                'y': float(self.map_manager.start.y),
                'z': float(self.map_manager.start.z)
            }
        if self.map_manager.target:
            export_data['goal'] = {
                'x': float(self.map_manager.target.position[0]),
                'y': float(self.map_manager.target.position[1]),
                'z': float(self.map_manager.target.position[2])
            }

        # 添加障碍物
        export_data['obstacles'] = []
        for obs in self.map_manager.obstacles:
            # 根据类名判断类型
            obs_type = obs.__class__.__name__.lower()

            if obs_type == 'cylinder':
                obs_data = {
                    'type': 'cylinder',
                    'center': [float(obs.center[0]), float(obs.center[1])],
                    'radius': float(obs.radius),
                    'z_min': float(obs.z_min),
                    'z_max': float(obs.z_max)
                }
            elif obs_type == 'prism':
                # 计算多边形中心（顶点平均值）
                verts = obs.polygon.vertices
                center_x = float(np.mean([v[0] for v in verts]))
                center_y = float(np.mean([v[1] for v in verts]))
                obs_data = {
                    'type': 'prism',
                    'center': [center_x, center_y],
                    'vertices': [[float(v[0]), float(v[1])] for v in verts],
                    'z_min': float(obs.z_min),
                    'z_max': float(obs.z_max)
                }
            else:
                continue  # 跳过未知类型

            export_data['obstacles'].append(obs_data)

        # 参考轨迹（从 Trajectory 对象提取）
        if self.trajectory:
            for pt in self.trajectory.points:
                export_data['reference_trajectory'].append({
                    't': float(pt.t),
                    'x': float(pt.position[0]),
                    'y': float(pt.position[1]),
                    'z': float(pt.position[2]),
                    'heading': float(pt.heading),
                    'curvature': float(pt.curvature)
                })

        # 实际轨迹和控制数据（降采样，避免文件过大）
        # 每10步保存一次（100Hz -> 10Hz）
        step = max(1, len(data['t']) // 1000)  # 最多1000个点

        for i in range(0, len(data['t']), step):
            # 实际轨迹
            export_data['actual_trajectory'].append({
                't': float(data['t'][i]),
                'x': float(data['position'][i, 0]),
                'y': float(data['position'][i, 1]),
                'z': float(data['position'][i, 2]),
                'vx': float(data['velocity'][i, 0]),
                'vy': float(data['velocity'][i, 1]),
                'vz': float(data['velocity'][i, 2]),
                'roll': float(data['euler'][i, 0]),
                'pitch': float(data['euler'][i, 1]),
                'yaw': float(data['euler'][i, 2])
            })

            # 控制数据
            export_data['control_data'].append({
                't': float(data['t'][i]),
                'delta_left': float(data['delta_left'][i]),
                'delta_right': float(data['delta_right'][i]),
                'cross_track_error': float(data['cross_track_error'][i]),
                'heading_error': float(data['heading_error'][i]),
                'glide_ratio_required': float(data['glide_ratio_required'][i]),
                'glide_ratio_current': float(data['glide_ratio_current'][i])
            })

        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # 保存
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2)

        print(f"\n仿真数据已导出到: {output_path}")
        print(f"  参考轨迹: {len(export_data['reference_trajectory'])} 点")
        print(f"  实际轨迹: {len(export_data['actual_trajectory'])} 点")

        return output_path


# ============================================================
#                     主程序
# ============================================================

def _build_raw_edge_trajectory(self,
                               postprocessor: TrajectoryPostprocessor,
                               candidate_path: List[np.ndarray],
                               end_heading: Optional[float]) -> Trajectory:
    deduped_points = []
    for pt in candidate_path:
        pt = np.asarray(pt, dtype=np.float64).copy()
        if not deduped_points or np.linalg.norm(pt - deduped_points[-1]) > 1e-6:
            deduped_points.append(pt)

    if len(deduped_points) < 2:
        return Trajectory(dt=self.control_dt)
    return postprocessor._create_trajectory(deduped_points, end_heading)


def _build_junction_smoothed_trajectory(self,
                                        postprocessor: TrajectoryPostprocessor,
                                        candidate_path: List[np.ndarray],
                                        candidate_info: Dict[str, Any],
                                        end_heading: Optional[float]) -> Trajectory:
    waypoints = [np.asarray(wp, dtype=np.float64) for wp in candidate_path]
    if len(waypoints) < 2:
        return Trajectory(dt=self.control_dt)

    z_start = waypoints[0][2]
    z_goal = (
        self.map_manager.target.position[2]
        if self.map_manager is not None and self.map_manager.target is not None
        else waypoints[-1][2]
    )
    points_2d = [wp[:2].copy() for wp in waypoints]
    total_2d = sum(np.linalg.norm(points_2d[i + 1] - points_2d[i]) for i in range(len(points_2d) - 1))
    required_glide = total_2d / max(z_start - z_goal, 1e-6) if z_start > z_goal else float("inf")
    trackable_min_glide = 5.0
    if required_glide < trackable_min_glide - 0.05:
        extra_length = (z_start - z_goal) * trackable_min_glide - total_2d
        points_2d = _append_terminal_loiter(
            points_2d,
            extra_length,
            postprocessor.min_turn_radius,
            end_heading,
        )
    elif required_glide < postprocessor.min_glide + 0.10:
        points_2d = postprocessor._inject_spiral_dubins(points_2d, z_start, z_goal, end_heading)

    deduped_points = []
    for pt in points_2d:
        pt = np.asarray(pt, dtype=np.float64)
        if not deduped_points or np.linalg.norm(pt - deduped_points[-1]) > 1e-6:
            deduped_points.append(pt)

    points_2d = postprocessor._resample_arc_length(deduped_points)
    points_3d = _redistribute_altitude_with_clearance(
        postprocessor,
        points_2d,
        z_start,
        z_goal,
        profile="balanced",
        end_heading=end_heading,
    )
    return postprocessor._create_trajectory(points_3d, end_heading)


def _build_profiled_trajectory(self,
                               postprocessor: TrajectoryPostprocessor,
                               candidate_path: List[np.ndarray],
                               candidate_info: Dict[str, Any],
                               end_heading: Optional[float],
                               profile: str = "front_loaded") -> Trajectory:
    waypoints = [np.asarray(wp, dtype=np.float64) for wp in candidate_path]
    if len(waypoints) < 2:
        return Trajectory(dt=self.control_dt)

    z_start = waypoints[0][2]
    z_goal = (
        self.map_manager.target.position[2]
        if self.map_manager is not None and self.map_manager.target is not None
        else waypoints[-1][2]
    )

    points_2d = [wp[:2].copy() for wp in waypoints]
    total_2d = sum(np.linalg.norm(points_2d[i + 1] - points_2d[i]) for i in range(len(points_2d) - 1))
    required_glide = total_2d / max(z_start - z_goal, 1e-6) if z_start > z_goal else float("inf")
    trackable_min_glide = 5.0
    if required_glide < trackable_min_glide - 0.05:
        extra_length = (z_start - z_goal) * trackable_min_glide - total_2d
        points_2d = _append_terminal_loiter(
            points_2d,
            extra_length,
            postprocessor.min_turn_radius,
            end_heading,
        )
    elif required_glide < postprocessor.min_glide + 0.10:
        points_2d = postprocessor._inject_spiral_dubins(points_2d, z_start, z_goal, end_heading)

    deduped_points = []
    for pt in points_2d:
        pt = np.asarray(pt, dtype=np.float64)
        if not deduped_points or np.linalg.norm(pt - deduped_points[-1]) > 1e-6:
            deduped_points.append(pt)

    if len(deduped_points) < 2:
        return Trajectory(dt=self.control_dt)

    points_2d = postprocessor._resample_arc_length(deduped_points)
    points_3d = _redistribute_altitude_with_clearance(
        postprocessor,
        points_2d,
        z_start,
        z_goal,
        profile=profile,
        end_heading=end_heading,
    )
    return postprocessor._create_trajectory(points_3d, end_heading)


def _planner_obstacle_clearance_floor(self, xy: np.ndarray) -> float:
    floor_z = float(self.min_altitude)
    margin = float(self.map.constraints.safety_margin)
    probe = np.array([xy[0], xy[1], 0.0], dtype=np.float64)

    for obs in self.map.obstacles:
        if hasattr(obs, "center") and hasattr(obs, "radius"):
            xy_dist = np.linalg.norm(
                np.asarray(xy, dtype=np.float64) - np.asarray(obs.center[:2], dtype=np.float64)
            )
            if xy_dist < float(obs.radius) + margin:
                floor_z = max(floor_z, float(obs.z_max) + margin)
        elif hasattr(obs, "polygon"):
            xy_dist = float(obs.polygon.distance(probe))
            if xy_dist < margin:
                floor_z = max(floor_z, float(obs.z_max) + margin)

    return floor_z


def _planner_reassign_altitude_clearance_aware(self,
                                               path: List[np.ndarray],
                                               z_start: float,
                                               z_goal: float,
                                               goal_heading: float) -> Tuple[Optional[List[np.ndarray]], float]:
    if len(path) < 2:
        return path, 0.0

    original_path = [pt.copy() for pt in path]

    delta_z = z_start - z_goal
    if delta_z <= 0:
        return path, float("inf")

    total_len_2d = self._path_length_2d(path)
    if total_len_2d < 1.0:
        return path, 0.0

    required_glide = total_len_2d / delta_z
    if required_glide > self.max_glide * 1.001:
        return None, required_glide

    path = [pt.copy() for pt in path]
    n_points = len(path)
    seg_len_2d = np.array([
        np.linalg.norm(path[i + 1][:2] - path[i][:2])
        for i in range(n_points - 1)
    ], dtype=np.float64)

    dz_floor = seg_len_2d / max(self.max_glide, 1e-6)
    dz_ceil = seg_len_2d / max(self.min_glide, 1e-6)
    if float(np.sum(dz_floor)) > delta_z + 0.5:
        return original_path, required_glide
    if float(np.sum(dz_ceil)) < delta_z - 0.5:
        return original_path, required_glide

    clearance_floor = np.array([
        min(self._obstacle_clearance_floor(pt[:2]), z_start)
        for pt in path
    ], dtype=np.float64)
    clearance_floor[0] = z_start
    clearance_floor[-1] = z_goal
    max_drop_prefix = np.clip(z_start - clearance_floor, 0.0, delta_z)

    cum_min = np.concatenate(([0.0], np.cumsum(dz_floor)))
    if np.any(cum_min - max_drop_prefix > 0.5):
        return original_path, required_glide

    # Start from the shallowest feasible descent and push extra drop as late as possible.
    dz = dz_floor.copy()
    residual = float(delta_z - np.sum(dz))

    for seg_idx in range(len(dz) - 1, -1, -1):
        if residual <= 1e-6:
            break

        cum_drop = np.concatenate(([0.0], np.cumsum(dz)))
        slack = max_drop_prefix - cum_drop
        suffix_slack = (
            float(np.min(slack[seg_idx + 1:]))
            if seg_idx + 1 < len(slack) else float("inf")
        )
        if suffix_slack <= 1e-9:
            continue

        seg_capacity = float(dz_ceil[seg_idx] - dz[seg_idx])
        if seg_capacity <= 1e-9:
            continue

        add = min(residual, seg_capacity, suffix_slack)
        if add <= 1e-9:
            continue

        dz[seg_idx] += add
        residual -= add

    if residual > 0.5:
        return original_path, required_glide

    cum_drop = 0.0
    path[0][2] = z_start
    for i in range(1, n_points):
        cum_drop += dz[i - 1]
        path[i][2] = z_start - cum_drop
    path[-1][2] = z_goal

    for i, pt in enumerate(path):
        if pt[2] + 1e-6 < clearance_floor[i]:
            return original_path, required_glide

    if self._has_collision_path(path):
        return original_path, required_glide

    return path, required_glide


def _postprocessor_obstacle_clearance_floor(postprocessor: TrajectoryPostprocessor,
                                            xy: np.ndarray,
                                            z_start: float,
                                            z_goal: float) -> float:
    map_manager = postprocessor.map_manager
    if map_manager is None:
        return float(z_goal)

    target_xy = (
        np.asarray(map_manager.target.position[:2], dtype=np.float64)
        if map_manager.target is not None else None
    )
    terminal_taper_radius = float(max(120.0, 1.25 * postprocessor.min_turn_radius))
    baseline_floor = float(max(map_manager.constraints.min_altitude, z_goal))
    if target_xy is not None:
        dist_to_goal = float(np.linalg.norm(np.asarray(xy, dtype=np.float64) - target_xy))
        if dist_to_goal < terminal_taper_radius:
            alpha = dist_to_goal / max(terminal_taper_radius, 1e-6)
            baseline_floor = z_goal + alpha * (baseline_floor - z_goal)

    floor_z = baseline_floor
    margin = float(map_manager.constraints.safety_margin)
    probe = np.array([xy[0], xy[1], 0.0], dtype=np.float64)

    for obs in map_manager.obstacles:
        if hasattr(obs, "center") and hasattr(obs, "radius"):
            xy_dist = np.linalg.norm(
                np.asarray(xy, dtype=np.float64) - np.asarray(obs.center[:2], dtype=np.float64)
            )
            if xy_dist < float(obs.radius) + margin:
                floor_z = max(floor_z, float(obs.z_max) + margin)
        elif hasattr(obs, "polygon"):
            xy_dist = float(obs.polygon.distance(probe))
            if xy_dist < margin:
                floor_z = max(floor_z, float(obs.z_max) + margin)

    return min(floor_z, z_start)


def _append_terminal_loiter(points_2d: List[np.ndarray],
                            extra_length: float,
                            min_turn_radius: float,
                            end_heading: Optional[float]) -> List[np.ndarray]:
    if len(points_2d) < 2 or extra_length <= 1.0:
        return [pt.copy() for pt in points_2d]

    goal_xy = np.asarray(points_2d[-1], dtype=np.float64)
    prev_xy = np.asarray(points_2d[-2], dtype=np.float64)
    if end_heading is not None:
        approach_heading = float(end_heading)
    else:
        approach_heading = float(np.arctan2(goal_xy[1] - prev_xy[1], goal_xy[0] - prev_xy[0]))

    radius = float(max(min_turn_radius * 2.0, 1.0))
    base_length = float(np.linalg.norm(goal_xy - prev_xy))

    def build_arc(direction: float, arc_angle: float) -> Tuple[float, List[np.ndarray]]:
        theta_goal = approach_heading - direction * (np.pi / 2.0)
        theta_start = theta_goal - direction * arc_angle
        center = goal_xy - radius * np.array([np.cos(theta_goal), np.sin(theta_goal)])
        start_xy = center + radius * np.array([np.cos(theta_start), np.sin(theta_start)])
        total_len = np.linalg.norm(prev_xy - start_xy) + radius * arc_angle
        added_len = total_len - base_length

        n_arc = max(8, int(np.ceil(arc_angle / (np.pi / 18.0))) + 1)
        arc_points = []
        for i in range(n_arc):
            frac = i / max(n_arc - 1, 1)
            theta = theta_start + direction * frac * arc_angle
            arc_points.append(center + radius * np.array([np.cos(theta), np.sin(theta)]))
        return added_len, arc_points

    best_points = None
    best_err = float("inf")
    for direction in (-1.0, 1.0):
        low = 0.02
        high = max(extra_length / radius, 0.02)
        while True:
            added_len, _ = build_arc(direction, high)
            if added_len >= extra_length or high >= 6.0 * np.pi:
                break
            high *= 1.5

        for _ in range(18):
            mid = 0.5 * (low + high)
            added_len, _ = build_arc(direction, mid)
            if added_len >= extra_length:
                high = mid
            else:
                low = mid

        added_len, arc_points = build_arc(direction, high)
        err = abs(added_len - extra_length)
        if err < best_err:
            best_err = err
            best_points = arc_points

    if not best_points:
        return [pt.copy() for pt in points_2d]

    result = [np.asarray(pt, dtype=np.float64).copy() for pt in points_2d[:-1]]
    if result and np.linalg.norm(best_points[0] - result[-1]) < 1e-6:
        result.extend(best_points[1:])
    else:
        result.extend(best_points)
    return result


def _redistribute_altitude_with_clearance(postprocessor: TrajectoryPostprocessor,
                                          points_2d: List[np.ndarray],
                                          z_start: float,
                                          z_goal: float,
                                          profile: str = "balanced",
                                          end_heading: Optional[float] = None,
                                          allow_terminal_extension: bool = True,
                                          anchor_floors: Optional[List[Tuple[np.ndarray, float]]] = None) -> List[np.ndarray]:
    n_points = len(points_2d)
    if n_points < 2:
        return [np.array([points_2d[0][0], points_2d[0][1], z_start], dtype=np.float64)]

    delta_z = z_start - z_goal
    if abs(delta_z) < 1e-6:
        return [np.array([pt[0], pt[1], z_start], dtype=np.float64) for pt in points_2d]

    seg_len_2d = np.array([
        np.linalg.norm(points_2d[i + 1] - points_2d[i])
        for i in range(n_points - 1)
    ], dtype=np.float64)
    total_2d = float(np.sum(seg_len_2d))
    if total_2d < 1e-6:
        return [np.array([pt[0], pt[1], z_start], dtype=np.float64) for pt in points_2d]

    g_max = float(postprocessor.max_glide)
    g_min = float(postprocessor.min_glide)
    min_required_drop = total_2d / max(g_max, 1e-6)
    max_available_drop = total_2d / max(g_min, 1e-6)
    if min_required_drop > delta_z + 0.5:
        raise ValueError(
            f"path too long for available altitude: required_glide={total_2d / max(delta_z, 1e-6):.2f} > max={g_max:.2f}"
        )
    if max_available_drop < delta_z - 0.5:
        raise ValueError(
            f"path too short to dissipate altitude: required_glide={total_2d / max(delta_z, 1e-6):.2f} < min={g_min:.2f}"
        )

    clearance_floor = np.array([
        _postprocessor_obstacle_clearance_floor(postprocessor, pt, z_start, z_goal)
        for pt in points_2d
    ], dtype=np.float64)
    clearance_floor[0] = z_start
    clearance_floor[-1] = z_goal
    if anchor_floors:
        for anchor_xy, anchor_z in anchor_floors:
            if anchor_xy is None:
                continue
            anchor_xy = np.asarray(anchor_xy, dtype=np.float64)
            anchor_idx = int(np.argmin([
                np.linalg.norm(np.asarray(pt, dtype=np.float64) - anchor_xy)
                for pt in points_2d
            ]))
            clearance_floor[anchor_idx] = max(
                clearance_floor[anchor_idx],
                min(float(anchor_z), z_start)
            )
    max_drop_prefix = np.clip(z_start - clearance_floor, 0.0, delta_z)

    dz_floor = seg_len_2d / max(g_max, 1e-6)
    dz_ceil = seg_len_2d / max(g_min, 1e-6)
    cum_min = np.concatenate(([0.0], np.cumsum(dz_floor)))
    if np.any(cum_min - max_drop_prefix > 0.5):
        raise ValueError("clearance floor conflicts with available altitude budget")

    profile_name = str(profile or "balanced").lower()
    curvatures = np.zeros(n_points, dtype=np.float64)
    for idx in range(1, n_points - 1):
        curvatures[idx] = postprocessor._compute_curvature_2d(
            points_2d[idx - 1], points_2d[idx], points_2d[idx + 1]
        )

    if profile_name == "front_loaded":
        frontload_gain = 0.70
        early_fraction = 0.16
        late_fraction = 0.58
        transition_progress = 0.72
        conservative_margin = 0.22
    elif profile_name == "approach":
        frontload_gain = 0.55
        early_fraction = 0.20
        late_fraction = 0.68
        transition_progress = 0.76
        conservative_margin = 0.28
    else:
        frontload_gain = 0.45
        early_fraction = 0.24
        late_fraction = 0.74
        transition_progress = 0.80
        conservative_margin = 0.38

    required_glide = total_2d / max(delta_z, 1e-6)
    conservative_glide_cap = min(
        g_max * 0.94,
        max(required_glide + conservative_margin, g_min + 0.65),
    )
    conservative_glide_cap = float(np.clip(conservative_glide_cap, g_min + 0.2, g_max))

    weighted_dz = np.zeros(n_points - 1, dtype=np.float64)
    cum_dist = np.cumsum(seg_len_2d)
    r_min = float(postprocessor.min_turn_radius)
    for idx in range(n_points - 1):
        progress = cum_dist[idx] / max(total_2d, 1e-6)
        kappa = 0.5 * (curvatures[idx] + curvatures[min(idx + 1, n_points - 1)])
        f_sym = max(0.0, 1.0 - kappa * r_min)
        g_eff_min = g_max - f_sym * (g_max - g_min)

        blend_fraction = early_fraction
        if progress > transition_progress:
            blend_fraction = late_fraction
        elif progress > 0.55:
            alpha = (progress - 0.55) / max(transition_progress - 0.55, 1e-6)
            blend_fraction = early_fraction + alpha * (late_fraction - early_fraction)

        g_target = g_eff_min + blend_fraction * (conservative_glide_cap - g_eff_min)
        g_target = float(np.clip(g_target, g_eff_min, conservative_glide_cap))
        frontload_weight = 1.0 + frontload_gain * (1.0 - progress)
        weighted_dz[idx] = frontload_weight * seg_len_2d[idx] / max(g_target, 1e-6)

    def late_allocate() -> Tuple[np.ndarray, float]:
        dz_late = dz_floor.copy()
        residual_late = float(delta_z - np.sum(dz_late))
        for seg_idx in range(len(dz_late) - 1, -1, -1):
            if residual_late <= 1e-6:
                break

            cum_drop_late = np.concatenate(([0.0], np.cumsum(dz_late)))
            slack_late = max_drop_prefix - cum_drop_late
            suffix_slack_late = (
                float(np.min(slack_late[seg_idx + 1:]))
                if seg_idx + 1 < len(slack_late) else float("inf")
            )
            if suffix_slack_late <= 1e-9:
                continue

            seg_capacity_late = float(dz_ceil[seg_idx] - dz_late[seg_idx])
            if seg_capacity_late <= 1e-9:
                continue

            add_late = min(residual_late, seg_capacity_late, suffix_slack_late)
            if add_late <= 1e-9:
                continue

            dz_late[seg_idx] += add_late
            residual_late -= add_late
        return dz_late, residual_late

    dz = dz_floor.copy()
    residual = float(delta_z - np.sum(dz))
    allocation_priority = np.maximum(weighted_dz - dz_floor, 1e-6)

    for _ in range(48):
        if residual <= 1e-6:
            break

        cum_drop = np.concatenate(([0.0], np.cumsum(dz)))
        slack = max_drop_prefix - cum_drop
        available = np.zeros_like(dz)

        for seg_idx in range(len(dz)):
            seg_capacity = float(dz_ceil[seg_idx] - dz[seg_idx])
            if seg_capacity <= 1e-9:
                continue
            suffix_slack = (
                float(np.min(slack[seg_idx + 1:]))
                if seg_idx + 1 < len(slack) else float("inf")
            )
            available[seg_idx] = max(0.0, min(seg_capacity, suffix_slack))

        if float(np.sum(available)) <= 1e-9:
            break

        weights = available * allocation_priority
        if float(np.sum(weights)) <= 1e-9:
            weights = available

        delta = residual * (weights / max(float(np.sum(weights)), 1e-9))
        delta = np.minimum(delta, available)
        used = float(np.sum(delta))
        if used <= 1e-9:
            break
        dz += delta
        residual -= used

    cum_drop = np.concatenate(([0.0], np.cumsum(dz)))
    if residual > 0.5 or np.any(cum_drop - max_drop_prefix > 0.5):
        dz, residual = late_allocate()

    if residual > 0.5:
        if allow_terminal_extension:
            extra_length = max(residual * max(g_min, 1.0) * 1.15, 15.0)
            extended_points_2d = _append_terminal_loiter(
                points_2d,
                extra_length,
                postprocessor.min_turn_radius,
                end_heading,
            )
            return _redistribute_altitude_with_clearance(
                postprocessor,
                extended_points_2d,
                z_start,
                z_goal,
                profile=profile,
                end_heading=end_heading,
                allow_terminal_extension=False,
                anchor_floors=anchor_floors,
            )
        raise ValueError("unable to allocate remaining altitude drop under clearance floors")

    result = []
    cum_drop = 0.0
    for i in range(n_points):
        z = z_start - cum_drop
        result.append(np.array([points_2d[i][0], points_2d[i][1], z], dtype=np.float64))
        if i < n_points - 1:
            cum_drop += dz[i]
    result[-1][2] = z_goal

    for idx, point in enumerate(result):
        if point[2] + 0.5 < clearance_floor[idx]:
            raise ValueError("altitude redistribution violated obstacle clearance floor")

    return result


def _select_virtual_approach(self,
                             desired_glide: float = 4.4,
                             max_length: float = 240.0,
                             min_length: float = 120.0) -> Optional[Dict[str, Any]]:
    if self.map_manager is None or self.map_manager.target is None or self.map_manager.start is None:
        return None

    target = self.map_manager.target
    target_pos = np.asarray(target.position, dtype=np.float64).copy()
    available_altitude = max(float(self.map_manager.start.z - target_pos[2]), 1.0)
    max_length = float(np.clip(max_length, 120.0, min(360.0, available_altitude * 0.90 * self.map_manager.constraints.glide_ratio)))
    min_length = float(np.clip(min_length, 80.0, max_length))
    approach_altitude = target_pos[2] + np.clip(
        max_length / max(desired_glide, 1e-6),
        max(45.0, self.map_manager.constraints.terminal_altitude),
        min(140.0, available_altitude - 20.0)
    )
    if approach_altitude <= target_pos[2] + 5.0:
        return None

    approach_point, approach_heading, approach_length = self.map_manager.find_safe_approach_point(
        target_pos=target_pos.copy(),
        altitude=approach_altitude,
        desired_heading=target.approach_heading,
        max_length=max_length,
        min_length=min_length,
        heading_tolerance=target.approach_heading_tolerance,
    )
    if approach_point is None:
        return None

    return {
        'point': np.asarray(approach_point, dtype=np.float64).copy(),
        'heading': float(approach_heading),
        'length': float(approach_length),
        'target_position': target_pos.copy(),
    }


def _build_safe_approach_trajectory(self,
                                    postprocessor: TrajectoryPostprocessor,
                                    candidate_path: List[np.ndarray],
                                    candidate_info: Dict[str, Any],
                                    end_heading: Optional[float],
                                    profile: str = "approach") -> Trajectory:
    approach_info = candidate_info.get("virtual_approach")
    if not approach_info:
        raise ValueError("virtual approach info missing")

    waypoints = [np.asarray(wp, dtype=np.float64) for wp in candidate_path]
    if len(waypoints) < 2:
        return Trajectory(dt=self.control_dt)

    z_start = float(waypoints[0][2])
    z_goal = float(approach_info["target_position"][2])
    approach_xy = np.asarray(approach_info["point"][:2], dtype=np.float64)
    target_xy = np.asarray(approach_info["target_position"][:2], dtype=np.float64)

    points_2d = [wp[:2].copy() for wp in waypoints]
    if np.linalg.norm(points_2d[-1] - approach_xy) > 1e-6:
        points_2d.append(approach_xy.copy())

    final_vec = target_xy - approach_xy
    final_len = float(np.linalg.norm(final_vec))
    if final_len > 1e-6:
        n_final = max(4, int(np.ceil(final_len / 10.0)))
        for k in range(1, n_final + 1):
            frac = k / n_final
            points_2d.append(approach_xy + frac * final_vec)

    total_2d = sum(
        np.linalg.norm(points_2d[i + 1] - points_2d[i])
        for i in range(len(points_2d) - 1)
    )
    required_glide = total_2d / max(z_start - z_goal, 1e-6) if z_start > z_goal else float("inf")
    trackable_min_glide = 5.0
    if required_glide < trackable_min_glide - 0.05:
        extra_length = (z_start - z_goal) * trackable_min_glide - total_2d
        points_2d = _append_terminal_loiter(
            points_2d,
            extra_length,
            postprocessor.min_turn_radius,
            end_heading,
        )

    deduped_points = []
    for pt in points_2d:
        pt = np.asarray(pt, dtype=np.float64)
        if not deduped_points or np.linalg.norm(pt - deduped_points[-1]) > 1e-6:
            deduped_points.append(pt)

    if len(deduped_points) < 2:
        return Trajectory(dt=self.control_dt)

    points_2d = postprocessor._resample_arc_length(deduped_points)
    points_3d = _redistribute_altitude_with_clearance(
        postprocessor,
        points_2d,
        z_start,
        z_goal,
        profile=profile,
        end_heading=end_heading,
        anchor_floors=[(approach_xy, float(approach_info["point"][2]))],
    )
    return postprocessor._create_trajectory(points_3d, end_heading)


def _build_safe_tail_trajectory(self,
                                postprocessor: TrajectoryPostprocessor,
                                candidate_path: List[np.ndarray],
                                candidate_info: Dict[str, Any],
                                end_heading: Optional[float],
                                profile: str = "approach") -> Trajectory:
    approach_info = _select_virtual_approach(
        self,
        desired_glide=4.6,
        max_length=260.0,
        min_length=130.0,
    )
    if not approach_info:
        raise ValueError("no safe approach available")

    waypoints = [np.asarray(wp, dtype=np.float64) for wp in candidate_path]
    if len(waypoints) < 3:
        return Trajectory(dt=self.control_dt)

    target_xy = np.asarray(approach_info["target_position"][:2], dtype=np.float64)
    approach_xy = np.asarray(approach_info["point"][:2], dtype=np.float64)
    approach_heading = float(approach_info["heading"])
    r_min = float(postprocessor.min_turn_radius)

    d_goal = np.array([
        np.linalg.norm(wp[:2] - target_xy)
        for wp in waypoints
    ], dtype=np.float64)
    desired_anchor_distance = float(approach_info["length"] + 0.75 * r_min)
    candidate_indices = np.where(d_goal >= desired_anchor_distance)[0]
    anchor_idx = int(candidate_indices[-1]) if len(candidate_indices) > 0 else max(0, len(waypoints) - 3)
    anchor_idx = int(np.clip(anchor_idx, 0, len(waypoints) - 2))

    anchor_xy = waypoints[anchor_idx][:2].copy()
    if anchor_idx < len(waypoints) - 1:
        anchor_dir = waypoints[anchor_idx + 1][:2] - waypoints[anchor_idx][:2]
    else:
        anchor_dir = waypoints[anchor_idx][:2] - waypoints[anchor_idx - 1][:2]
    if np.linalg.norm(anchor_dir) < 1e-6:
        anchor_heading = approach_heading
    else:
        anchor_heading = float(np.arctan2(anchor_dir[1], anchor_dir[0]))

    transition_points = [anchor_xy.copy()]
    dubins_path = postprocessor.dubins.compute(
        (anchor_xy[0], anchor_xy[1], anchor_heading),
        (approach_xy[0], approach_xy[1], approach_heading),
    )
    if dubins_path is not None:
        sample_spacing = max(3.0, min(5.0, postprocessor.min_turn_radius / 30.0))
        n_pts = int(np.clip(np.ceil(dubins_path['length'] / sample_spacing) + 1, 6, 240))
        sampled = postprocessor.dubins.sample(dubins_path, num_points=n_pts)
        transition_points = [np.array([pt[0], pt[1]], dtype=np.float64) for pt in sampled]
    else:
        connect_len = float(np.linalg.norm(approach_xy - anchor_xy))
        n_connect = max(3, int(np.ceil(connect_len / 10.0)))
        transition_points = [
            anchor_xy + (k / n_connect) * (approach_xy - anchor_xy)
            for k in range(n_connect + 1)
        ]

    final_vec = target_xy - approach_xy
    final_len = float(np.linalg.norm(final_vec))
    final_points = [approach_xy.copy()]
    if final_len > 1e-6:
        n_final = max(4, int(np.ceil(final_len / 10.0)))
        final_points = [
            approach_xy + (k / n_final) * final_vec
            for k in range(n_final + 1)
        ]

    points_2d = [wp[:2].copy() for wp in waypoints[:anchor_idx + 1]]
    points_2d.extend(transition_points[1:] if len(transition_points) > 1 else transition_points)
    points_2d.extend(final_points[1:] if len(final_points) > 1 else final_points)

    total_2d = sum(
        np.linalg.norm(points_2d[i + 1] - points_2d[i])
        for i in range(len(points_2d) - 1)
    )
    z_start = float(waypoints[0][2])
    z_goal = float(self.map_manager.target.position[2]) if self.map_manager and self.map_manager.target else float(waypoints[-1][2])
    required_glide = total_2d / max(z_start - z_goal, 1e-6) if z_start > z_goal else float("inf")
    trackable_min_glide = 5.0
    if required_glide < trackable_min_glide - 0.05:
        extra_length = (z_start - z_goal) * trackable_min_glide - total_2d
        points_2d = _append_terminal_loiter(
            points_2d,
            extra_length,
            postprocessor.min_turn_radius,
            end_heading,
        )

    deduped_points = []
    for pt in points_2d:
        pt = np.asarray(pt, dtype=np.float64)
        if not deduped_points or np.linalg.norm(pt - deduped_points[-1]) > 1e-6:
            deduped_points.append(pt)

    points_2d = postprocessor._resample_arc_length(deduped_points)
    points_3d = _redistribute_altitude_with_clearance(
        postprocessor,
        points_2d,
        z_start,
        z_goal,
        profile=profile,
        end_heading=end_heading,
        anchor_floors=[(approach_xy, float(approach_info["point"][2]))],
    )
    return postprocessor._create_trajectory(points_3d, end_heading)


def _sampled_collision_count(self, trajectory: Trajectory, sample_spacing: float = 5.0) -> int:
    if self.map_manager is None or len(trajectory) == 0:
        return 0

    positions = trajectory.get_positions()
    collisions = 0
    for i in range(len(positions) - 1):
        p0 = positions[i]
        p1 = positions[i + 1]
        seg_len = np.linalg.norm(p1 - p0)
        n = max(1, int(np.ceil(seg_len / max(sample_spacing, 1e-6))))
        for k in range(n + 1):
            frac = k / n
            sample = p0 + frac * (p1 - p0)
            if self.map_manager.is_collision(sample):
                collisions += 1
                if collisions >= 3:
                    return collisions
    return collisions


def _terminal_trackability_metrics(self, trajectory: Trajectory) -> Dict[str, float]:
    if len(trajectory) < 3 or self.map_manager.target is None:
        return {
            'ok': True,
            'last_turn_distance': float('inf'),
            'tail_p05_radius': float('inf'),
            'preferred_straight_tail': 0.0,
        }

    target_xy = self.map_manager.target.position[:2]
    positions = trajectory.get_positions()
    curvatures = trajectory.get_curvatures()
    radii = trajectory.get_turning_radii()
    d_goal = np.linalg.norm(positions[:, :2] - target_xy, axis=1)
    min_turn_radius = self.map_manager.constraints.min_turn_radius

    turn_threshold = 1.0 / max(2.5 * min_turn_radius, 1e-6)
    turn_indices = np.where(np.abs(curvatures) > turn_threshold)[0]
    last_turn_distance = float(d_goal[turn_indices[-1]]) if len(turn_indices) > 0 else float('inf')

    tail_mask = d_goal <= max(150.0, 1.25 * min_turn_radius)
    finite_tail_radii = radii[np.isfinite(radii) & tail_mask]
    tail_p05_radius = (
        float(np.percentile(finite_tail_radii, 5))
        if len(finite_tail_radii) > 0 else float('inf')
    )
    preferred_straight_tail = float(max(120.0, 1.1 * min_turn_radius))

    ok = (
        last_turn_distance >= 0.85 * min_turn_radius and
        tail_p05_radius >= 0.65 * min_turn_radius
    )
    return {
        'ok': ok,
        'last_turn_distance': last_turn_distance,
        'tail_p05_radius': tail_p05_radius,
        'preferred_straight_tail': preferred_straight_tail,
    }


def _vertical_trackability_metrics(self, trajectory: Trajectory) -> Dict[str, float]:
    if len(trajectory) < 3:
        return {'ok': True, 'mid_deficit': 0.0, 'late_deficit': 0.0, 'max_deficit': 0.0}

    positions = trajectory.get_positions()
    seg_lengths = np.linalg.norm(np.diff(positions[:, :2], axis=0), axis=1)
    arc_lengths = np.concatenate(([0.0], np.cumsum(seg_lengths)))
    total_arc = float(arc_lengths[-1])
    total_drop = float(positions[0, 2] - positions[-1, 2])
    if total_arc < 1e-6 or total_drop <= 1.0:
        return {'ok': True, 'mid_deficit': 0.0, 'late_deficit': 0.0, 'max_deficit': 0.0}

    conservative_glide = min(self.map_manager.constraints.glide_ratio * 0.9, 5.7)
    consumed_drop = positions[0, 2] - positions[:, 2]
    expected_min_drop = arc_lengths / max(conservative_glide, 1e-6)
    deficits = np.maximum(expected_min_drop - consumed_drop, 0.0)

    mid_idx = int(np.searchsorted(arc_lengths, 0.50 * total_arc, side='left'))
    late_idx = int(np.searchsorted(arc_lengths, 0.75 * total_arc, side='left'))
    mid_idx = int(np.clip(mid_idx, 0, len(deficits) - 1))
    late_idx = int(np.clip(late_idx, 0, len(deficits) - 1))

    mid_deficit = float(deficits[mid_idx])
    late_deficit = float(deficits[late_idx])
    max_deficit = float(np.max(deficits))
    ok = mid_deficit <= 12.0 and late_deficit <= 20.0 and max_deficit <= 30.0

    return {
        'ok': ok,
        'mid_deficit': mid_deficit,
        'late_deficit': late_deficit,
        'max_deficit': max_deficit,
    }


def _energy_trackability_metrics(self, trajectory: Trajectory) -> Dict[str, float]:
    if len(trajectory) < 3:
        return {
            'ok': True,
            'total_glide': 0.0,
            'late_glide': 0.0,
            'budget': float('inf'),
            'late_glide_floor': 0.0,
            'trackable_floor': 0.0,
        }

    positions = trajectory.get_positions()
    seg_lengths = np.linalg.norm(np.diff(positions[:, :2], axis=0), axis=1)
    arc_lengths = np.concatenate(([0.0], np.cumsum(seg_lengths)))
    total_arc = float(arc_lengths[-1])
    total_drop = float(positions[0, 2] - positions[-1, 2])
    if total_arc < 1e-6 or total_drop <= 1.0:
        return {
            'ok': True,
            'total_glide': 0.0,
            'late_glide': 0.0,
            'budget': float('inf'),
            'late_glide_floor': 0.0,
            'trackable_floor': 0.0,
        }

    trackable_budget = min(
        self.map_manager.constraints.glide_ratio * 0.96,
        self.map_manager.constraints.glide_ratio - 0.12,
    )
    total_glide = total_arc / max(total_drop, 1e-6)

    late_start_idx = int(np.searchsorted(arc_lengths, 0.75 * total_arc, side='left'))
    late_start_idx = int(np.clip(late_start_idx, 0, len(positions) - 1))
    late_arc = float(total_arc - arc_lengths[late_start_idx])
    late_drop = float(positions[late_start_idx, 2] - positions[-1, 2])
    late_glide = late_arc / max(late_drop, 1e-6) if late_arc > 1.0 and late_drop > 0.5 else 0.0
    trackable_floor = 5.0
    late_glide_floor = max(self.map_manager.constraints.min_glide_ratio + 0.45, 3.25)

    ok = total_glide <= trackable_budget + 1e-6
    if late_glide > 0.0:
        ok = ok and late_glide <= trackable_budget + 0.10

    return {
        'ok': ok,
        'total_glide': float(total_glide),
        'late_glide': float(late_glide),
        'budget': float(trackable_budget),
        'late_glide_floor': float(late_glide_floor),
        'trackable_floor': float(trackable_floor),
    }


def _evaluate_trajectory_candidate(self, trajectory: Trajectory) -> Dict[str, Any]:
    validation = validate_trajectory(
        trajectory,
        min_glide_ratio=self.map_manager.constraints.min_glide_ratio,
        max_glide_ratio=self.map_manager.constraints.glide_ratio,
        min_turn_radius=self.map_manager.constraints.min_turn_radius
    )
    collision_count = _sampled_collision_count(self, trajectory)
    terminal_metrics = _terminal_trackability_metrics(self, trajectory)
    vertical_metrics = _vertical_trackability_metrics(self, trajectory)
    energy_metrics = _energy_trackability_metrics(self, trajectory)
    target_z = (
        float(self.map_manager.target.position[2])
        if self.map_manager is not None and self.map_manager.target is not None
        else 0.0
    )
    final_altitude_error = float(abs(trajectory.get_positions()[-1, 2] - target_z))
    strict_valid = (
        len(trajectory) > 0 and
        validation.get('valid', False) and
        collision_count == 0 and
        terminal_metrics['ok'] and
        energy_metrics['ok'] and
        energy_metrics['total_glide'] >= energy_metrics['trackable_floor'] - 0.15 and
        (
            energy_metrics['late_glide'] <= 0.0 or
            energy_metrics['late_glide'] >= energy_metrics['late_glide_floor'] - 0.10
        ) and
        final_altitude_error <= self.failure_detector.thresholds.landing_altitude
    )
    late_glide_penalty = 0.0
    if energy_metrics['late_glide'] > 0.0:
        late_glide_penalty = max(
            0.0,
            energy_metrics['late_glide_floor'] - energy_metrics['late_glide']
        )
    total_glide_floor_penalty = max(
        0.0,
        energy_metrics['trackable_floor'] - energy_metrics['total_glide']
    )
    score = (
        1000 * len(validation.get('errors', [])) +
        100 * collision_count +
        2.0 * max(0.0, self.map_manager.constraints.min_turn_radius - terminal_metrics['tail_p05_radius']) +
        3.0 * max(0.0, terminal_metrics['preferred_straight_tail'] - terminal_metrics['last_turn_distance']) +
        40.0 * final_altitude_error +
        280.0 * total_glide_floor_penalty +
        500.0 * max(0.0, energy_metrics['total_glide'] - energy_metrics['budget']) +
        350.0 * max(0.0, energy_metrics['late_glide'] - (energy_metrics['budget'] + 0.10)) +
        220.0 * late_glide_penalty +
        2.0 * vertical_metrics['mid_deficit'] +
        3.0 * vertical_metrics['late_deficit'] +
        4.0 * vertical_metrics['max_deficit']
    )
    return {
        'strict_valid': strict_valid,
        'validation': validation,
        'collision_count': collision_count,
        'terminal_metrics': terminal_metrics,
        'vertical_metrics': vertical_metrics,
        'energy_metrics': energy_metrics,
        'final_altitude_error': final_altitude_error,
        'score': score,
    }

def _plan_with_dynamic_retries(self,
                               max_time: float = 30.0,
                               smooth: bool = True,
                               progress_callback: callable = None) -> bool:
    """Retry planning until a dynamically feasible trajectory is found or budget is exhausted."""
    if not self.quiet:
        print("[3/4] planning Kinodynamic RRT* trajectory...")

    plan_start_time = time.time()
    planner_attempts = [
        {"step_size": 100.0, "goal_sample_rate": 0.30, "max_iterations": 5000, "weight": 0.18},
        {"step_size": 80.0, "goal_sample_rate": 0.40, "max_iterations": 7000, "weight": 0.18},
        {"step_size": 140.0, "goal_sample_rate": 0.22, "max_iterations": 7000, "weight": 0.22},
        {"step_size": 60.0, "goal_sample_rate": 0.45, "max_iterations": 9000, "weight": 0.18},
        {"step_size": 120.0, "goal_sample_rate": 0.35, "max_iterations": 8000, "weight": 0.16},
        {
            "step_size": 90.0,
            "goal_sample_rate": 0.36,
            "max_iterations": 6500,
            "weight": 0.08,
            "use_virtual_approach": True,
            "approach_glide": 4.2,
            "approach_max_length": 220.0,
            "approach_min_length": 110.0,
        },
    ]
    postprocessor = TrajectoryPostprocessor(
        reference_speed=self.reference_speed,
        control_frequency=1.0 / self.control_dt,
        min_turn_radius=self.map_manager.constraints.min_turn_radius,
        max_glide_ratio=self.map_manager.constraints.glide_ratio,
        min_glide_ratio=self.map_manager.constraints.min_glide_ratio,
        map_manager=self.map_manager,
        quiet=self.quiet
    )

    end_heading = self.map_manager.target.approach_heading if self.map_manager.target else None
    path = None
    info = {}
    self.trajectory = None
    best_grounded_fallback = None
    best_grounded_fallback_score = float('inf')
    best_loose_fallback = None
    best_loose_fallback_score = float('inf')

    for attempt_idx, planner_cfg in enumerate(planner_attempts, start=1):
        remaining_time = max_time - (time.time() - plan_start_time)
        if remaining_time <= 1.0:
            break
        remaining_weight = sum(
            cfg.get("weight", 1.0) for cfg in planner_attempts[attempt_idx - 1:]
        )
        attempt_budget = min(
            remaining_time,
            max(
                5.0,
                remaining_time * planner_cfg.get("weight", 1.0) / max(remaining_weight, 1e-6),
            )
        )

        if self.seed is not None:
            np.random.seed(self.seed + 9973 * (attempt_idx - 1))

        approach_info = None
        if planner_cfg.get("use_virtual_approach"):
            approach_info = _select_virtual_approach(
                self,
                desired_glide=planner_cfg.get("approach_glide", 4.4),
                max_length=planner_cfg.get("approach_max_length", 240.0),
                min_length=planner_cfg.get("approach_min_length", 120.0),
            )

        if not self.quiet:
            attempt_desc = (
                f"    [planning attempt {attempt_idx}/{len(planner_attempts)}] "
                f"step={planner_cfg['step_size']:.0f}m, "
                f"goal_rate={planner_cfg['goal_sample_rate']:.2f}, "
                f"iter={planner_cfg['max_iterations']}, "
                f"budget={attempt_budget:.1f}s"
            )
            if approach_info is not None:
                attempt_desc += f", virtual_approach={approach_info['length']:.0f}m"
            print(attempt_desc)

        original_target = None
        if approach_info is not None:
            original_target = {
                "position": self.map_manager.target.position.copy(),
                "heading": float(self.map_manager.target.desired_approach_heading),
            }
            self.map_manager.target.position = approach_info["point"].copy()
            self.map_manager.target.desired_approach_heading = float(approach_info["heading"])

        try:
            kino_planner = KinodynamicRRTStar(
                self.map_manager,
                quiet=self.quiet,
                **{
                    k: v for k, v in planner_cfg.items()
                    if k not in {
                        "weight",
                        "use_virtual_approach",
                        "approach_glide",
                        "approach_max_length",
                        "approach_min_length",
                    }
                }
            )
            candidate_path, candidate_info = kino_planner.plan(
                max_time=attempt_budget,
                progress_callback=progress_callback
            )
        finally:
            if original_target is not None:
                self.map_manager.target.position = original_target["position"]
                self.map_manager.target.desired_approach_heading = original_target["heading"]

        if candidate_path is None:
            continue
        candidate_info = dict(candidate_info)
        if approach_info is not None:
            candidate_info["virtual_approach"] = approach_info

        if not self.quiet:
            print("[4/4] trajectory postprocess + timing...")

        if approach_info is not None:
            candidate_builders = [
                ("safe_approach", lambda: _build_safe_approach_trajectory(
                    self, postprocessor, candidate_path, candidate_info, end_heading,
                    profile="approach"
                )),
                ("safe_approach_front_loaded", lambda: _build_safe_approach_trajectory(
                    self, postprocessor, candidate_path, candidate_info, end_heading,
                    profile="front_loaded"
                )),
            ]
        else:
            candidate_builders = [
                ("processed", lambda: postprocessor.process(
                    candidate_path, smooth=smooth, end_heading=end_heading
                )),
                ("raw_edge", lambda: _build_raw_edge_trajectory(
                    self, postprocessor, candidate_path, end_heading
                )),
                ("safe_tail", lambda: _build_safe_tail_trajectory(
                    self, postprocessor, candidate_path, candidate_info, end_heading,
                    profile="approach"
                )),
                ("junction_smoothed", lambda: _build_junction_smoothed_trajectory(
                    self, postprocessor, candidate_path, candidate_info, end_heading
                )),
                ("front_loaded", lambda: _build_profiled_trajectory(
                    self, postprocessor, candidate_path, candidate_info, end_heading,
                    profile="front_loaded"
                )),
            ]

        strict_choice = None

        for candidate_name, builder in candidate_builders:
            try:
                candidate_traj = builder()
            except Exception:
                continue

            if len(candidate_traj) == 0:
                continue

            evaluation = _evaluate_trajectory_candidate(self, candidate_traj)
            if evaluation['strict_valid']:
                if strict_choice is None or evaluation['score'] < strict_choice[2]['score']:
                    strict_choice = (candidate_name, candidate_traj, evaluation)

            if evaluation['collision_count'] == 0:
                grounded = evaluation['final_altitude_error'] <= (
                    self.failure_detector.thresholds.landing_altitude + 5.0
                )
                if grounded and evaluation['score'] < best_grounded_fallback_score:
                    best_grounded_fallback = (
                        candidate_path, candidate_info, candidate_traj, candidate_name, evaluation
                    )
                    best_grounded_fallback_score = evaluation['score']
                elif (not grounded) and evaluation['score'] < best_loose_fallback_score:
                    best_loose_fallback = (
                        candidate_path, candidate_info, candidate_traj, candidate_name, evaluation
                    )
                    best_loose_fallback_score = evaluation['score']

        if strict_choice is not None:
            candidate_name, candidate_traj, evaluation = strict_choice
            if not self.quiet:
                print(
                    f"    [select] using {candidate_name} trajectory "
                    f"(vertical_deficit={evaluation['vertical_metrics']['max_deficit']:.1f}m, "
                    f"total_glide={evaluation['energy_metrics']['total_glide']:.2f}, "
                    f"final_alt_err={evaluation['final_altitude_error']:.1f}m)"
                )
            path = candidate_path
            info = candidate_info
            self.trajectory = candidate_traj
            break

        if not self.quiet:
            print("    [replan] no trajectory candidate passed strict 3D/terminal screening")

    self.planning_time = time.time() - plan_start_time

    fallback_choice = best_grounded_fallback if best_grounded_fallback is not None else best_loose_fallback
    if (path is None or self.trajectory is None) and fallback_choice is not None:
        path, info, self.trajectory, candidate_name, evaluation = fallback_choice
        if not self.quiet:
            print(
                f"    [fallback] using {candidate_name} trajectory "
                f"(collisions={evaluation['collision_count']}, "
                f"errors={len(evaluation['validation'].get('errors', []))}, "
                f"final_alt_err={evaluation['final_altitude_error']:.1f}m)"
            )

    if path is None or self.trajectory is None:
        if not self.quiet:
            print("    planning failed!")
        return False

    if not self.quiet:
        print(f"    planning complete: {len(path)} waypoints, time: {self.planning_time:.2f}s")
        if 'path_length' in info:
            print(f"    path length: {info['path_length']:.1f}m")
        self._validate_trajectory_descent_rate()

    self.controller.set_trajectory(self.trajectory)
    return True


KinodynamicRRTStar._obstacle_clearance_floor = _planner_obstacle_clearance_floor
KinodynamicRRTStar._reassign_altitude = _planner_reassign_altitude_clearance_aware
ClosedLoopSimulator.plan = _plan_with_dynamic_retries


if __name__ == "__main__":
    import argparse
    from datetime import datetime

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
    parser.add_argument("--output_dir", type=str, default=None,
                        help="输出目录（保存仿真数据 JSON，用于 Web 可视化），目录不存在会自动创建")
    parser.add_argument("--controller", type=str, default="adrc",
                        choices=["adrc", "pid"],
                        help="controller type")
    parser.add_argument("--no-plot", action="store_true",
                        help="不显示 matplotlib 图表")
    args = parser.parse_args()

    # 创建仿真器
    sim = ClosedLoopSimulator(
        map_config_path=args.map_config,
        model_config_path=args.model_config,
        control_dt=args.control_dt,
        dynamics_dt=args.dynamics_dt,
        controller_type=args.controller
    )

    # 规划
    success = sim.plan(max_time=30.0)

    if not success:
        exit(1)

    # 初始化状态 (加一点初始偏差)
    sim.init_state(
        position_noise=args.position_noise,
        heading_noise=args.heading_noise
    )

    # 运行仿真
    log = sim.run(max_time=args.max_time)

    # 导出数据到 JSON（用于 Web 可视化）
    if args.output_dir:
        # 确保输出目录存在
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir, exist_ok=True)
            print(f"\n创建输出目录: {args.output_dir}")

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(args.output_dir, f'sim_{timestamp}.json')
        sim.export_to_json(output_path)

    # 可视化
    if not args.no_plot:
        sim.visualize()
