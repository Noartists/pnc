"""
Dubins 曲线计算器（独立模块）

从 kinodynamic_rrt.py 提取，供规划器和后处理器共用。
"""

import numpy as np
from typing import List, Tuple, Optional


class DubinsPath:
    """
    Dubins曲线计算器

    连接两个2D位姿 (x, y, heading)，满足最小转弯半径约束。
    """

    def __init__(self, turn_radius: float):
        self.turn_radius = turn_radius

    def compute(self, start: Tuple[float, float, float],
                end: Tuple[float, float, float]) -> Optional[dict]:
        """
        计算Dubins曲线

        参数:
            start: (x, y, heading) 起点
            end: (x, y, heading) 终点

        返回:
            最优Dubins路径参数，或None（不可达）
        """
        x0, y0, h0 = start
        x1, y1, h1 = end
        r = self.turn_radius

        # 转换到局部坐标系
        dx = x1 - x0
        dy = y1 - y0
        D = np.sqrt(dx**2 + dy**2)
        d = D / r  # 归一化距离

        if d < 1e-6:
            return None

        theta = np.arctan2(dy, dx)
        alpha = self._normalize_angle(h0 - theta)
        beta = self._normalize_angle(h1 - theta)

        # 尝试所有6种Dubins路径类型
        paths = [
            self._LSL(alpha, beta, d),
            self._RSR(alpha, beta, d),
            self._LSR(alpha, beta, d),
            self._RSL(alpha, beta, d),
            self._LRL(alpha, beta, d),
            self._RLR(alpha, beta, d),
        ]

        # 选择最短的有效路径
        best = None
        best_len = float('inf')

        for path in paths:
            if path is not None and path['length'] < best_len:
                best = path
                best_len = path['length']

        if best is None:
            return None

        # 转换回世界坐标
        best['start'] = start
        best['end'] = end
        best['turn_radius'] = r
        best['length'] *= r  # 实际长度

        return best

    def sample(self, path: dict, num_points: int = 20) -> List[np.ndarray]:
        """
        沿Dubins曲线采样点

        返回:
            2D点列表 [(x, y), ...]
        """
        if path is None:
            return []

        x0, y0, h0 = path['start']
        r = path['turn_radius']
        segments = path['segments']  # [(length, direction), ...]

        points = []
        x, y, h = x0, y0, h0

        total_len = sum(s[0] for s in segments)
        step = total_len / (num_points - 1) if num_points > 1 else total_len

        accumulated = 0.0
        seg_idx = 0
        seg_progress = 0.0

        for i in range(num_points):
            target_dist = i * step

            # 前进到目标距离
            while accumulated + (segments[seg_idx][0] * r - seg_progress) < target_dist and seg_idx < len(segments) - 1:
                # 完成当前段
                seg_len, direction = segments[seg_idx]
                remaining = seg_len * r - seg_progress

                if direction == 'S':  # 直线
                    x += remaining * np.cos(h)
                    y += remaining * np.sin(h)
                elif direction == 'L':  # 左转
                    dtheta = remaining / r
                    cx = x - r * np.sin(h)
                    cy = y + r * np.cos(h)
                    h += dtheta
                    x = cx + r * np.sin(h)
                    y = cy - r * np.cos(h)
                elif direction == 'R':  # 右转
                    dtheta = remaining / r
                    cx = x + r * np.sin(h)
                    cy = y - r * np.cos(h)
                    h -= dtheta
                    x = cx - r * np.sin(h)
                    y = cy + r * np.cos(h)

                accumulated += remaining
                seg_progress = 0.0
                seg_idx += 1

            # 在当前段内前进
            if seg_idx < len(segments):
                seg_len, direction = segments[seg_idx]
                advance = target_dist - accumulated

                if direction == 'S':
                    px = x + advance * np.cos(h)
                    py = y + advance * np.sin(h)
                elif direction == 'L':
                    dtheta = advance / r
                    cx = x - r * np.sin(h)
                    cy = y + r * np.cos(h)
                    px = cx + r * np.sin(h + dtheta)
                    py = cy - r * np.cos(h + dtheta)
                elif direction == 'R':
                    dtheta = advance / r
                    cx = x + r * np.sin(h)
                    cy = y - r * np.cos(h)
                    px = cx - r * np.sin(h - dtheta)
                    py = cy + r * np.cos(h - dtheta)

                points.append(np.array([px, py]))
            else:
                points.append(np.array([x, y]))

        return points

    @staticmethod
    def end_heading(path_dict: dict) -> float:
        """
        根据 Dubins 路径的最后一段计算出口航向。

        参数:
            path_dict: compute() 返回的路径字典

        返回:
            出口航向 (rad)
        """
        if path_dict is None:
            return 0.0

        x0, y0, h0 = path_dict['start']
        r = path_dict['turn_radius']
        segments = path_dict['segments']

        # 沿路径依次推进航向
        h = h0
        x, y = x0, y0
        for seg_len, direction in segments:
            actual_len = seg_len * r
            if direction == 'S':
                x += actual_len * np.cos(h)
                y += actual_len * np.sin(h)
            elif direction == 'L':
                dtheta = actual_len / r
                cx = x - r * np.sin(h)
                cy = y + r * np.cos(h)
                h += dtheta
                x = cx + r * np.sin(h)
                y = cy - r * np.cos(h)
            elif direction == 'R':
                dtheta = actual_len / r
                cx = x + r * np.sin(h)
                cy = y - r * np.cos(h)
                h -= dtheta
                x = cx - r * np.sin(h)
                y = cy + r * np.cos(h)

        # 归一化到 [-pi, pi]
        while h > np.pi:
            h -= 2 * np.pi
        while h < -np.pi:
            h += 2 * np.pi
        return h

    def _normalize_angle(self, angle: float) -> float:
        """归一化角度到 [-pi, pi]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

    def _LSL(self, alpha: float, beta: float, d: float) -> Optional[dict]:
        """Left-Straight-Left"""
        ca, sa = np.cos(alpha), np.sin(alpha)
        cb, sb = np.cos(beta), np.sin(beta)

        tmp = 2 + d**2 - 2*(ca*cb + sa*sb - d*(sa - sb))
        if tmp < 0:
            return None

        p = np.sqrt(tmp)
        theta = np.arctan2(cb - ca, d + sa - sb)
        t = self._normalize_angle(-alpha + theta)
        q = self._normalize_angle(beta - theta)

        if t < -1e-6 or q < -1e-6:
            return None
        t = max(t, 0.0)
        q = max(q, 0.0)

        return {
            'type': 'LSL',
            'segments': [(t, 'L'), (p, 'S'), (q, 'L')],
            'length': t + p + q
        }

    def _RSR(self, alpha: float, beta: float, d: float) -> Optional[dict]:
        """Right-Straight-Right"""
        ca, sa = np.cos(alpha), np.sin(alpha)
        cb, sb = np.cos(beta), np.sin(beta)

        tmp = 2 + d**2 - 2*(ca*cb + sa*sb - d*(sb - sa))
        if tmp < 0:
            return None

        p = np.sqrt(tmp)
        theta = np.arctan2(ca - cb, d - sa + sb)
        t = self._normalize_angle(alpha - theta)
        q = self._normalize_angle(-beta + theta)

        if t < -1e-6 or q < -1e-6:
            return None
        t = max(t, 0.0)
        q = max(q, 0.0)

        return {
            'type': 'RSR',
            'segments': [(t, 'R'), (p, 'S'), (q, 'R')],
            'length': t + p + q
        }

    def _LSR(self, alpha: float, beta: float, d: float) -> Optional[dict]:
        """Left-Straight-Right"""
        ca, sa = np.cos(alpha), np.sin(alpha)
        cb, sb = np.cos(beta), np.sin(beta)

        tmp = -2 + d**2 + 2*(ca*cb + sa*sb + d*(sa + sb))
        if tmp < 0:
            return None

        p = np.sqrt(tmp)
        theta = np.arctan2(-ca - cb, d + sa + sb) - np.arctan2(-2, p)
        t = self._normalize_angle(-alpha + theta)
        q = self._normalize_angle(-beta + theta)

        if t < -1e-6 or q < -1e-6:
            return None
        t = max(t, 0.0)
        q = max(q, 0.0)

        return {
            'type': 'LSR',
            'segments': [(t, 'L'), (p, 'S'), (q, 'R')],
            'length': t + p + q
        }

    def _RSL(self, alpha: float, beta: float, d: float) -> Optional[dict]:
        """Right-Straight-Left"""
        ca, sa = np.cos(alpha), np.sin(alpha)
        cb, sb = np.cos(beta), np.sin(beta)

        tmp = -2 + d**2 + 2*(ca*cb + sa*sb - d*(sa + sb))
        if tmp < 0:
            return None

        p = np.sqrt(tmp)
        theta = np.arctan2(ca + cb, d - sa - sb) - np.arctan2(2, p)
        t = self._normalize_angle(alpha - theta)
        q = self._normalize_angle(beta - theta)

        if t < -1e-6 or q < -1e-6:
            return None
        t = max(t, 0.0)
        q = max(q, 0.0)

        return {
            'type': 'RSL',
            'segments': [(t, 'R'), (p, 'S'), (q, 'L')],
            'length': t + p + q
        }

    def _LRL(self, alpha: float, beta: float, d: float) -> Optional[dict]:
        """Left-Right-Left"""
        ca, sa = np.cos(alpha), np.sin(alpha)
        cb, sb = np.cos(beta), np.sin(beta)

        tmp = (6 - d**2 + 2*(ca*cb + sa*sb + d*(sa - sb))) / 8
        if abs(tmp) > 1:
            return None

        p = np.arccos(tmp)
        theta = np.arctan2(ca - cb, d + sa - sb)
        t = self._normalize_angle(-alpha + theta + p/2)
        q = self._normalize_angle(beta - theta + p/2)

        if t < -1e-6 or q < -1e-6:
            return None
        t = max(t, 0.0)
        q = max(q, 0.0)

        return {
            'type': 'LRL',
            'segments': [(t, 'L'), (p, 'R'), (q, 'L')],
            'length': t + p + q
        }

    def _RLR(self, alpha: float, beta: float, d: float) -> Optional[dict]:
        """Right-Left-Right"""
        ca, sa = np.cos(alpha), np.sin(alpha)
        cb, sb = np.cos(beta), np.sin(beta)

        tmp = (6 - d**2 + 2*(ca*cb + sa*sb - d*(sa - sb))) / 8
        if abs(tmp) > 1:
            return None

        p = np.arccos(tmp)
        theta = np.arctan2(ca - cb, d - sa + sb)
        t = self._normalize_angle(alpha - theta + p/2)
        q = self._normalize_angle(-beta + theta + p/2)

        if t < -1e-6 or q < -1e-6:
            return None
        t = max(t, 0.0)
        q = max(q, 0.0)

        return {
            'type': 'RLR',
            'segments': [(t, 'R'), (p, 'L'), (q, 'R')],
            'length': t + p + q
        }
