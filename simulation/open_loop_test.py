"""
开环仿真测试

测试翼伞动力学模型的开环响应（无控制）
用于验证动力学模型是否正常工作
"""

import os
import sys
import numpy as np
import argparse
import yaml

# 添加项目根目录到路径
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from models.parafoil_model import ParafoilParams, simulate


def run_open_loop_test(
    model_config: str,
    map_config: str = None,
    initial_altitude: float = None,
    initial_velocity: float = 10.0,
    initial_heading: float = 0.0,
    left_deflection: float = 0.0,
    right_deflection: float = 0.0,
    duration: float = 60.0,
    dt: float = 0.02,
    visualize: bool = True
):
    """
    运行开环仿真测试
    
    参数:
        model_config: 动力学模型配置文件
        map_config: 地图配置文件（用于读取起点高度）
        initial_altitude: 初始高度 (m)，如果为None则从map_config读取
        initial_velocity: 初始前向速度 (m/s)
        initial_heading: 初始航向 (rad)
        left_deflection: 左操纵绳偏转 (m)
        right_deflection: 右操纵绳偏转 (m)
        duration: 仿真时长 (s)
        dt: 积分步长 (s)
        visualize: 是否可视化
    
    返回:
        t: 时间数组
        y: 状态数组 (n, 20)
    """
    print("=" * 60)
    print("  翼伞动力学开环测试")
    print("=" * 60)
    
    # 加载动力学参数
    print("\n[1] 加载动力学模型参数...")
    para = ParafoilParams.from_yaml(model_config)
    
    # 确定初始高度
    if initial_altitude is None:
        if map_config is not None:
            print(f"[2] 从 {map_config} 读取起点高度...")
            with open(map_config, 'r', encoding='utf-8') as f:
                map_cfg = yaml.safe_load(f)
            initial_altitude = map_cfg['start']['z']
            print(f"    起点高度: {initial_altitude}m")
        else:
            initial_altitude = 200.0
            print(f"[2] 使用默认高度: {initial_altitude}m")
    else:
        print(f"[2] 使用指定高度: {initial_altitude}m")
    
    # 设置控制输入
    para.left = left_deflection
    para.right = right_deflection
    print(f"\n[3] 控制输入:")
    print(f"    左操纵绳: {left_deflection:.3f}m")
    print(f"    右操纵绳: {right_deflection:.3f}m")
    
    # 构建初始状态
    # 状态向量 y (20维):
    #   [0:3]   - 位置 (x, y, z)
    #   [3:6]   - 伞体姿态角 (phi, theta, psi)
    #   [6:8]   - 相对角 (theta_r, psi_r)
    #   [8:11]  - 伞体速度 (u, v, w)
    #   [11:14] - 伞体角速度 (p, q, r)
    #   [14:17] - 负载速度
    #   [17:20] - 负载角速度
    y0 = np.zeros(20)
    y0[0] = 0.0                           # x
    y0[1] = 0.0                           # y
    y0[2] = initial_altitude              # z (高度)
    y0[3] = 0.0                           # phi (滚转角)
    y0[4] = np.radians(8)                 # theta (俯仰角，典型滑翔约8°)
    y0[5] = initial_heading               # psi (航向角)
    y0[8] = initial_velocity              # u (前向速度)
    y0[9] = 0.0                           # v (侧向速度)
    y0[10] = 5.0                          # w (下沉速度，体坐标系)
    
    print(f"\n[4] 初始状态:")
    print(f"    位置: ({y0[0]:.1f}, {y0[1]:.1f}, {y0[2]:.1f})m")
    print(f"    俯仰角: {np.degrees(y0[4]):.1f}°")
    print(f"    航向: {np.degrees(y0[5]):.1f}°")
    print(f"    前向速度: {y0[8]:.1f}m/s")
    print(f"    下沉速度: {y0[10]:.1f}m/s")
    
    # 运行仿真
    print(f"\n[5] 运行开环仿真 (时长: {duration}s, 步长: {dt}s)...")
    t, y = simulate(y0, (0, duration), para, dt=dt)
    
    # 分析结果
    print("\n" + "=" * 60)
    print("  仿真结果")
    print("=" * 60)
    print(f"  时间步数: {len(t)}")
    print(f"  初始位置: ({y[0,0]:.1f}, {y[0,1]:.1f}, {y[0,2]:.1f})m")
    print(f"  最终位置: ({y[-1,0]:.1f}, {y[-1,1]:.1f}, {y[-1,2]:.1f})m")
    
    # 计算飞行距离和下降高度
    horizontal_dist = np.sqrt(y[-1,0]**2 + y[-1,1]**2)
    altitude_drop = y[0,2] - y[-1,2]
    glide_ratio = horizontal_dist / altitude_drop if altitude_drop > 0 else float('inf')
    
    print(f"\n  水平飞行距离: {horizontal_dist:.1f}m")
    print(f"  高度下降: {altitude_drop:.1f}m")
    print(f"  滑翔比: {glide_ratio:.2f}:1")
    
    # 平均速度
    avg_horizontal_speed = horizontal_dist / duration
    avg_descent_rate = altitude_drop / duration
    print(f"\n  平均水平速度: {avg_horizontal_speed:.2f}m/s")
    print(f"  平均下降率: {avg_descent_rate:.2f}m/s")
    
    # 检查是否落地
    if y[-1,2] <= 0:
        landing_idx = np.argmax(y[:,2] <= 0)
        landing_time = t[landing_idx]
        print(f"\n  [落地] t={landing_time:.1f}s")
    
    print("=" * 60)
    
    # 可视化
    if visualize:
        import matplotlib.pyplot as plt
        
        fig = plt.figure(figsize=(16, 10))
        
        # 1. 3D轨迹
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        ax1.plot(y[:, 0], y[:, 1], y[:, 2], 'b-', linewidth=2)
        ax1.scatter(y[0, 0], y[0, 1], y[0, 2], c='green', s=100, marker='o', label='Start')
        ax1.scatter(y[-1, 0], y[-1, 1], y[-1, 2], c='red', s=100, marker='*', label='End')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title('3D Trajectory')
        ax1.legend()
        
        # 2. XY平面
        ax2 = fig.add_subplot(2, 3, 2)
        ax2.plot(y[:, 0], y[:, 1], 'b-', linewidth=2)
        ax2.scatter(y[0, 0], y[0, 1], c='green', s=100, marker='o')
        ax2.scatter(y[-1, 0], y[-1, 1], c='red', s=100, marker='*')
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_title('XY Plane')
        ax2.grid(True, alpha=0.3)
        ax2.axis('equal')
        
        # 3. 高度剖面
        ax3 = fig.add_subplot(2, 3, 3)
        ax3.plot(t, y[:, 2], 'b-', linewidth=2)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Altitude (m)')
        ax3.set_title('Altitude Profile')
        ax3.grid(True, alpha=0.3)
        
        # 4. 速度
        ax4 = fig.add_subplot(2, 3, 4)
        speed = np.sqrt(y[:, 8]**2 + y[:, 9]**2 + y[:, 10]**2)
        ax4.plot(t, y[:, 8], 'r-', linewidth=1.5, label='u (forward)')
        ax4.plot(t, y[:, 10], 'b-', linewidth=1.5, label='w (down)')
        ax4.plot(t, speed, 'k--', linewidth=1.5, label='total')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Velocity (m/s)')
        ax4.set_title('Body Velocities')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. 姿态角
        ax5 = fig.add_subplot(2, 3, 5)
        ax5.plot(t, np.degrees(y[:, 3]), 'r-', linewidth=1.5, label='phi (roll)')
        ax5.plot(t, np.degrees(y[:, 4]), 'g-', linewidth=1.5, label='theta (pitch)')
        ax5.plot(t, np.degrees(y[:, 5]), 'b-', linewidth=1.5, label='psi (yaw)')
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Angle (deg)')
        ax5.set_title('Euler Angles')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. 下降率
        ax6 = fig.add_subplot(2, 3, 6)
        descent_rate = -np.diff(y[:, 2]) / dt
        ax6.plot(t[:-1], descent_rate, 'b-', linewidth=1.5)
        ax6.axhline(y=np.mean(descent_rate), color='r', linestyle='--', 
                    label=f'Mean: {np.mean(descent_rate):.2f} m/s')
        ax6.set_xlabel('Time (s)')
        ax6.set_ylabel('Descent Rate (m/s)')
        ax6.set_title('Descent Rate')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    return t, y


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="翼伞动力学开环测试")
    parser.add_argument("--model-config", type=str, default="cfg/config.yaml",
                        help="动力学模型配置文件")
    parser.add_argument("--map-config", type=str, default="cfg/map_config.yaml",
                        help="地图配置文件（读取起点高度）")
    parser.add_argument("--altitude", type=float, default=None,
                        help="初始高度 (m)，不指定则从map_config读取")
    parser.add_argument("--velocity", type=float, default=10.0,
                        help="初始前向速度 (m/s)")
    parser.add_argument("--heading", type=float, default=0.0,
                        help="初始航向 (deg)")
    parser.add_argument("--left", type=float, default=0.0,
                        help="左操纵绳偏转 (m)")
    parser.add_argument("--right", type=float, default=0.0,
                        help="右操纵绳偏转 (m)")
    parser.add_argument("--duration", type=float, default=60.0,
                        help="仿真时长 (s)")
    parser.add_argument("--dt", type=float, default=0.02,
                        help="积分步长 (s)")
    parser.add_argument("--no-viz", action="store_true",
                        help="不显示可视化")
    args = parser.parse_args()
    
    run_open_loop_test(
        model_config=args.model_config,
        map_config=args.map_config,
        initial_altitude=args.altitude,
        initial_velocity=args.velocity,
        initial_heading=np.radians(args.heading),
        left_deflection=args.left,
        right_deflection=args.right,
        duration=args.duration,
        dt=args.dt,
        visualize=not args.no_viz
    )
