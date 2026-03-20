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
        alpha = self._mod2pi(h0 - theta)
        beta = self._mod2pi(h1 - theta)

        # Prefer classic CSC paths for trackability; fall back to CCC only when needed.
        csc_paths = [
            self._LSL(alpha, beta, d),
            self._RSR(alpha, beta, d),
            self._LSR(alpha, beta, d),
            self._RSL(alpha, beta, d),
        ]
        ccc_paths = [
            self._LRL(alpha, beta, d),
            self._RLR(alpha, beta, d),
        ]

        best = None
        best_len = float('inf')

        for path in csc_paths:
            if path is not None and path['length'] < best_len:
                best = path
                best_len = path['length']

        if best is None:
            for path in ccc_paths:
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

        total_len = path.get('length', sum(seg_len * r for seg_len, _ in segments))
        if num_points <= 1:
            return [np.array([x0, y0], dtype=np.float64)]
        step = total_len / (num_points - 1)

        accumulated = 0.0
        seg_idx = 0
        seg_progress = 0.0

        for i in range(num_points):
            target_dist = i * step

            # 前进到目标距离
            while (
                seg_idx < len(segments) - 1 and
                accumulated + (segments[seg_idx][0] * r - seg_progress) < target_dist
            ):
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
                max_advance = max(seg_len * r - seg_progress, 0.0)
                advance = np.clip(target_dist - accumulated, 0.0, max_advance)

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

        if points:
            points[0] = np.array([x0, y0], dtype=np.float64)
            x1, y1, _ = path['end']
            points[-1] = np.array([x1, y1], dtype=np.float64)

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

    def _mod2pi(self, angle: float) -> float:
        """Wrap angle to [0, 2*pi)."""
        return angle % (2 * np.pi)

    def _LSL(self, alpha: float, beta: float, d: float) -> Optional[dict]:
        """Left-Straight-Left"""
        tmp0 = d + np.sin(alpha) - np.sin(beta)
        tmp = 2 + d**2 - 2 * np.cos(alpha - beta) + 2 * d * (np.sin(alpha) - np.sin(beta))
        if tmp < 0.0:
            return None

        p = np.sqrt(tmp)
        theta = np.arctan2(np.cos(beta) - np.cos(alpha), tmp0)
        t = self._mod2pi(-alpha + theta)
        q = self._mod2pi(beta - theta)

        return {
            'type': 'LSL',
            'segments': [(t, 'L'), (p, 'S'), (q, 'L')],
            'length': t + p + q
        }

    def _RSR(self, alpha: float, beta: float, d: float) -> Optional[dict]:
        """Right-Straight-Right"""
        tmp0 = d - np.sin(alpha) + np.sin(beta)
        tmp = 2 + d**2 - 2 * np.cos(alpha - beta) + 2 * d * (np.sin(beta) - np.sin(alpha))
        if tmp < 0.0:
            return None

        p = np.sqrt(tmp)
        theta = np.arctan2(np.cos(alpha) - np.cos(beta), tmp0)
        t = self._mod2pi(alpha - theta)
        q = self._mod2pi(-beta + theta)

        return {
            'type': 'RSR',
            'segments': [(t, 'R'), (p, 'S'), (q, 'R')],
            'length': t + p + q
        }

    def _LSR(self, alpha: float, beta: float, d: float) -> Optional[dict]:
        """Left-Straight-Right"""
        tmp = -2 + d**2 + 2 * np.cos(alpha - beta) + 2 * d * (np.sin(alpha) + np.sin(beta))
        if tmp < 0.0:
            return None

        p = np.sqrt(tmp)
        theta = np.arctan2(-np.cos(alpha) - np.cos(beta), d + np.sin(alpha) + np.sin(beta))
        theta -= np.arctan2(-2.0, p)
        t = self._mod2pi(-alpha + theta)
        q = self._mod2pi(-beta + theta)

        return {
            'type': 'LSR',
            'segments': [(t, 'L'), (p, 'S'), (q, 'R')],
            'length': t + p + q
        }

    def _RSL(self, alpha: float, beta: float, d: float) -> Optional[dict]:
        """Right-Straight-Left"""
        tmp = -2 + d**2 + 2 * np.cos(alpha - beta) - 2 * d * (np.sin(alpha) + np.sin(beta))
        if tmp < 0.0:
            return None

        p = np.sqrt(tmp)
        theta = np.arctan2(np.cos(alpha) + np.cos(beta), d - np.sin(alpha) - np.sin(beta))
        theta -= np.arctan2(2.0, p)
        t = self._mod2pi(alpha - theta)
        q = self._mod2pi(beta - theta)

        return {
            'type': 'RSL',
            'segments': [(t, 'R'), (p, 'S'), (q, 'L')],
            'length': t + p + q
        }

    def _LRL(self, alpha: float, beta: float, d: float) -> Optional[dict]:
        """Left-Right-Left"""
        tmp = (6 - d**2 + 2 * np.cos(alpha - beta) + 2 * d * (-np.sin(alpha) + np.sin(beta))) / 8
        if abs(tmp) > 1.0:
            return None

        p = self._mod2pi(2 * np.pi - np.arccos(tmp))
        theta = np.arctan2(np.cos(alpha) - np.cos(beta), d + np.sin(alpha) - np.sin(beta))
        t = self._mod2pi(-alpha - theta + p / 2.0)
        q = self._mod2pi(beta - alpha - t + p)

        return {
            'type': 'LRL',
            'segments': [(t, 'L'), (p, 'R'), (q, 'L')],
            'length': t + p + q
        }

    def _RLR(self, alpha: float, beta: float, d: float) -> Optional[dict]:
        """Right-Left-Right"""
        tmp = (6 - d**2 + 2 * np.cos(alpha - beta) + 2 * d * (np.sin(alpha) - np.sin(beta))) / 8
        if abs(tmp) > 1.0:
            return None

        p = self._mod2pi(2 * np.pi - np.arccos(tmp))
        theta = np.arctan2(np.cos(alpha) - np.cos(beta), d - np.sin(alpha) + np.sin(beta))
        t = self._mod2pi(alpha - theta + p / 2.0)
        q = self._mod2pi(alpha - beta - t + p)

        return {
            'type': 'RLR',
            'segments': [(t, 'R'), (p, 'L'), (q, 'R')],
            'length': t + p + q
        }


def _sample_per_segment(self, path: dict, num_points: int = 20) -> List[np.ndarray]:
    """Sample each Dubins segment independently so short terminal turns are not collapsed."""
    if path is None:
        return []

    x0, y0, h0 = path['start']
    r = path['turn_radius']
    segments = path['segments']
    total_len = path.get('length', sum(seg_len * r for seg_len, _ in segments))
    if num_points <= 1:
        return [np.array([x0, y0], dtype=np.float64)]

    spacing = max(total_len / (num_points - 1), 1e-6)
    points = [np.array([x0, y0], dtype=np.float64)]
    x, y, h = x0, y0, h0

    for seg_len_norm, direction in segments:
        seg_len = seg_len_norm * r
        sx, sy, sh = x, y, h
        n_seg = max(int(np.ceil(seg_len / spacing)), 1)
        ph = sh

        for i in range(1, n_seg + 1):
            advance = min(seg_len, seg_len * i / n_seg)

            if direction == 'S':
                px = sx + advance * np.cos(sh)
                py = sy + advance * np.sin(sh)
                ph = sh
            elif direction == 'L':
                dtheta = advance / r
                cx = sx - r * np.sin(sh)
                cy = sy + r * np.cos(sh)
                ph = sh + dtheta
                px = cx + r * np.sin(ph)
                py = cy - r * np.cos(ph)
            else:  # direction == 'R'
                dtheta = advance / r
                cx = sx + r * np.sin(sh)
                cy = sy - r * np.cos(sh)
                ph = sh - dtheta
                px = cx - r * np.sin(ph)
                py = cy + r * np.cos(ph)

            points.append(np.array([px, py], dtype=np.float64))

        x, y, h = points[-1][0], points[-1][1], ph

    x1, y1, _ = path['end']
    points[-1] = np.array([x1, y1], dtype=np.float64)
    return points


DubinsPath.sample = _sample_per_segment
