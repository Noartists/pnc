"""
仿真模块

包含:
- 闭环仿真 (规划 + 控制 + 动力学)
- 开环测试 (纯动力学模型)
"""

from simulation.closed_loop_sim import ClosedLoopSimulator
from simulation.open_loop_test import run_open_loop_test

__all__ = ['ClosedLoopSimulator', 'run_open_loop_test']
