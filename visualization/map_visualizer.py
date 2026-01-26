"""
地图可视化模块
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon as MplPolygon, FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# 添加项目根目录到路径
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from planning.map_manager import MapManager


class MapVisualizer:
    """地图可视化器"""

    def __init__(self, map_manager: MapManager):
        self.map_mgr = map_manager

        # Set font to Times New Roman
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 11

        self.colors = {
            'obstacle': '#FF6B6B',
            'corridor': '#4ECDC4',
            'start': '#2ECC71',
            'target': '#3498DB',
            'approach': '#9B59B6',
            'path': '#E74C3C'
        }

    def plot_2d(self, ax=None, show_grid: bool = True,
                trajectory: np.ndarray = None) -> plt.Axes:
        """
        绘制2D俯视图

        参数:
            ax: matplotlib轴对象
            show_grid: 是否显示网格
            trajectory: 轨迹数组 [n, 3]
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(12, 10))

        data = self.map_mgr.get_visualization_data()

        # 绘制走廊
        if data['corridor']:
            boundary = data['corridor']['boundary']
            corridor_patch = MplPolygon(
                boundary, closed=True,
                facecolor=self.colors['corridor'], alpha=0.2,
                edgecolor=self.colors['corridor'], linewidth=2,
                label='Corridor'
            )
            ax.add_patch(corridor_patch)

        # 绘制禁飞区
        for i, obs in enumerate(data['obstacles']):
            boundary = obs['boundary']
            z_min, z_max = obs['z_range']

            # 根据高度范围调整透明度
            alpha = 0.6 if z_max < 500 else 0.3

            obs_patch = MplPolygon(
                boundary, closed=True,
                facecolor=self.colors['obstacle'], alpha=alpha,
                edgecolor='darkred', linewidth=1.5,
                label='No-fly Zone' if i == 0 else None
            )
            ax.add_patch(obs_patch)

        # 绘制起点
        if data['start'] is not None:
            start = data['start']
            ax.plot(start[0], start[1], 'o', markersize=15,
                   color=self.colors['start'], label='Start', zorder=10)
            ax.annotate('Start', (start[0], start[1]),
                       xytext=(10, 10), textcoords='offset points', fontsize=10)

        # 绘制目标
        if data['target']:
            target = data['target']
            pos = target['position']

            # 着陆区圆
            landing_circle = Circle(
                (pos[0], pos[1]), target['radius'],
                facecolor=self.colors['target'], alpha=0.5,
                edgecolor='darkblue', linewidth=2,
                label='Landing Zone'
            )
            ax.add_patch(landing_circle)

            # 进场方向箭头
            approach = target['approach_point']
            ax.annotate('', xy=(pos[0], pos[1]), xytext=(approach[0], approach[1]),
                       arrowprops=dict(arrowstyle='->', color=self.colors['approach'],
                                      lw=2, mutation_scale=15))
            ax.plot(approach[0], approach[1], 's', markersize=10,
                   color=self.colors['approach'], label='Approach Point')

        # 绘制轨迹
        if trajectory is not None:
            ax.plot(trajectory[:, 0], trajectory[:, 1], '-',
                   color=self.colors['path'], linewidth=2, label='Trajectory')
            # 标记方向箭头
            n = len(trajectory)
            for i in range(0, n - 1, max(1, n // 10)):
                dx = trajectory[i + 1, 0] - trajectory[i, 0]
                dy = trajectory[i + 1, 1] - trajectory[i, 1]
                ax.arrow(trajectory[i, 0], trajectory[i, 1],
                        dx * 0.3, dy * 0.3,
                        head_width=15, head_length=10,
                        fc=self.colors['path'], ec=self.colors['path'])

        # 设置坐标轴
        bounds = data['bounds']
        margin = 100
        ax.set_xlim(bounds['x_min'] - margin, bounds['x_max'] + margin)
        ax.set_ylim(bounds['y_min'] - margin, bounds['y_max'] + margin)
        ax.set_aspect('equal')
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_title('Parafoil Mission Map (Top View)', fontsize=14)

        if show_grid:
            ax.grid(True, alpha=0.3)

        ax.legend(loc='upper left')

        return ax

    def plot_3d(self, ax=None, trajectory: np.ndarray = None) -> Axes3D:
        """
        绘制3D视图

        参数:
            ax: matplotlib 3D轴对象
            trajectory: 轨迹数组 [n, 3]
        """
        if ax is None:
            fig = plt.figure(figsize=(14, 10))
            ax = fig.add_subplot(111, projection='3d')

        data = self.map_mgr.get_visualization_data()

        # 绘制禁飞区 (简化为圆柱体底面和顶面)
        for obs in data['obstacles']:
            boundary = obs['boundary']
            z_min, z_max = obs['z_range']
            if z_max == np.inf:
                z_max = 500

            # 底面
            verts_bottom = [list(zip(boundary[:, 0], boundary[:, 1],
                                    [z_min] * len(boundary)))]
            ax.add_collection3d(Poly3DCollection(
                verts_bottom, alpha=0.4, facecolor=self.colors['obstacle'],
                edgecolor='darkred', linewidth=1
            ))

            # 顶面
            verts_top = [list(zip(boundary[:, 0], boundary[:, 1],
                                 [z_max] * len(boundary)))]
            ax.add_collection3d(Poly3DCollection(
                verts_top, alpha=0.4, facecolor=self.colors['obstacle'],
                edgecolor='darkred', linewidth=1
            ))

            # 侧面轮廓线
            for i in range(len(boundary)):
                x = [boundary[i, 0], boundary[i, 0]]
                y = [boundary[i, 1], boundary[i, 1]]
                z = [z_min, z_max]
                ax.plot(x, y, z, color='darkred', alpha=0.5, linewidth=0.5)

        # 绘制起点
        if data['start'] is not None:
            start = data['start']
            ax.scatter(start[0], start[1], start[2],
                      s=200, c=self.colors['start'], marker='o',
                      label='Start', zorder=10)

        # 绘制目标
        if data['target']:
            target = data['target']
            pos = target['position']
            ax.scatter(pos[0], pos[1], pos[2],
                      s=200, c=self.colors['target'], marker='*',
                      label='Target', zorder=10)

            # 进场点
            approach = target['approach_point']
            ax.scatter(approach[0], approach[1], approach[2],
                      s=100, c=self.colors['approach'], marker='s',
                      label='Approach Point')

            # 进场路径
            ax.plot([approach[0], pos[0]], [approach[1], pos[1]],
                   [approach[2], pos[2]], '--',
                   color=self.colors['approach'], linewidth=2)

        # 绘制轨迹
        if trajectory is not None:
            ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
                   '-', color=self.colors['path'], linewidth=2, label='Trajectory')

        # 绘制地面
        bounds = data['bounds']
        xx, yy = np.meshgrid(
            np.linspace(bounds['x_min'], bounds['x_max'], 10),
            np.linspace(bounds['y_min'], bounds['y_max'], 10)
        )
        ax.plot_surface(xx, yy, np.zeros_like(xx),
                       alpha=0.1, color='green')

        # 设置坐标轴
        ax.set_xlim(bounds['x_min'], bounds['x_max'])
        ax.set_ylim(bounds['y_min'], bounds['y_max'])
        ax.set_zlim(bounds['z_min'], min(bounds['z_max'], 1500))

        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_zlabel('Z (m)', fontsize=12)
        ax.set_title('Parafoil Mission Map (3D View)', fontsize=14)

        ax.legend(loc='upper left')

        return ax

    def plot_combined(self, trajectory: np.ndarray = None,
                      save_path: str = None) -> plt.Figure:
        """
        绘制组合视图 (2D + 3D)
        """
        fig = plt.figure(figsize=(18, 8))

        # 2D视图
        ax1 = fig.add_subplot(121)
        self.plot_2d(ax1, trajectory=trajectory)

        # 3D视图
        ax2 = fig.add_subplot(122, projection='3d')
        self.plot_3d(ax2, trajectory=trajectory)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"图片已保存: {save_path}")

        return fig


# ============================================================
#                     测试
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="地图可视化测试")
    parser.add_argument("--config", type=str, required=True, help="地图配置文件路径")
    parser.add_argument("--save", type=str, default=None, help="保存图片路径")
    args = parser.parse_args()

    # 加载地图
    map_mgr = MapManager.from_yaml(args.config)

    # 创建可视化器
    visualizer = MapVisualizer(map_mgr)

    # 生成测试轨迹 (简单直线)
    if map_mgr.start and map_mgr.target:
        start = map_mgr.start.to_array()
        target = map_mgr.target.position
        n_points = 100
        t = np.linspace(0, 1, n_points).reshape(-1, 1)
        trajectory = start + t * (target - start)
        # 添加高度下降
        trajectory[:, 2] = start[2] * (1 - t.flatten()) + target[2] * t.flatten()
    else:
        trajectory = None

    # 绘制
    fig = visualizer.plot_combined(trajectory=trajectory, save_path=args.save)
    plt.show()
