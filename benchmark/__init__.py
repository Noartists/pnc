"""
Benchmark 框架

用于系统性评估闭环控制性能的框架，包括：
- RNGManager: 统一随机数管理，保证可复现性
- MetricsCalculator: 指标计算与失败判定
- FailureDetector: 硬失败/软发散检测
- MetricsOutput/CaseOutput: 输出数据结构
"""

from benchmark.rng_manager import RNGManager, get_config_hash, get_git_commit
from benchmark.metrics import MetricsCalculator, FailureDetector, FailureThresholds, TerminationReason
from benchmark.outputs import MetricsOutput, CaseOutput, create_output_dir, create_experiment_dir

__all__ = [
    'RNGManager', 
    'get_config_hash',
    'get_git_commit',
    'MetricsCalculator', 
    'FailureDetector',
    'FailureThresholds',
    'TerminationReason',
    'MetricsOutput',
    'CaseOutput',
    'create_output_dir',
    'create_experiment_dir'
]
