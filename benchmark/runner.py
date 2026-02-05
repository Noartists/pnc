"""
Benchmark 运行器

支持批量运行仿真、生成汇总报告
"""

import os
import sys
import json
import csv
import argparse
from datetime import datetime
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from tqdm import tqdm

# 添加项目根目录到路径
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from benchmark.outputs import create_experiment_dir


@dataclass
class ExperimentConfig:
    """实验配置"""
    name: str = "default"
    seeds: List[int] = field(default_factory=lambda: list(range(1, 11)))
    scene: str = "default"
    map_config: str = "cfg/map_config.yaml"
    model_config: str = "cfg/config.yaml"
    max_planning_time: float = 30.0
    max_sim_time: float = 300.0
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'ExperimentConfig':
        """从 YAML 文件加载配置"""
        import yaml
        with open(yaml_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        
        config = cls()
        for key, value in cfg.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        # 处理 seeds
        if 'seeds' in cfg:
            seeds_cfg = cfg['seeds']
            if isinstance(seeds_cfg, int):
                config.seeds = list(range(1, seeds_cfg + 1))
            elif isinstance(seeds_cfg, list):
                config.seeds = seeds_cfg
            elif isinstance(seeds_cfg, str):
                if '-' in seeds_cfg:
                    start, end = map(int, seeds_cfg.split('-'))
                    config.seeds = list(range(start, end + 1))
        
        return config


class BenchmarkRunner:
    """Benchmark 运行器"""
    
    def __init__(self, 
                 config: ExperimentConfig,
                 output_base_dir: str = 'benchmark/outputs',
                 verbose: bool = False):  # 默认静默
        """
        参数:
            config: 实验配置
            output_base_dir: 输出基础目录
            verbose: 是否打印详细信息（默认关闭）
        """
        self.config = config
        self.output_base_dir = output_base_dir
        self.verbose = verbose
        
        # 创建实验目录
        self.exp_dir = create_experiment_dir(output_base_dir)
        
        # 结果收集
        self.results: List[Dict[str, Any]] = []
        
        # 统计
        self.stats = {
            'planning_success': 0,
            'planning_failed': 0,
            'control_success': 0,
            'control_failed': 0,
        }
    
    def run_single(self, seed: int, pbar=None) -> Dict[str, Any]:
        """
        运行单个 seed 的仿真
        
        参数:
            seed: 随机数种子
            pbar: tqdm 进度条（用于更新状态）
        
        返回:
            metrics 字典
        """
        from simulation.closed_loop_sim import ClosedLoopSimulator
        
        result = {
            'seed': seed,
            'planning_success': False,
            'control_success': False,
            'success': False,
            'termination_reason': None,
            'error': None
        }
        
        def update_status(status: str):
            """更新进度条状态"""
            if pbar:
                pbar.set_description(f"Seed {seed:03d} {status}")
        
        try:
            update_status("[初始化]")
            
            # 创建仿真器（始终使用 quiet 模式）
            sim = ClosedLoopSimulator(
                map_config_path=self.config.map_config,
                model_config_path=self.config.model_config,
                seed=seed,
                scene_name=self.config.scene,
                quiet=True  # 始终静默
            )
            
            # 规划（使用回调更新进度）
            def plan_callback(iteration: int, max_iterations: int):
                """规划进度回调"""
                pct = iteration / max_iterations * 100 if max_iterations > 0 else 0
                update_status(f"[规划 {pct:4.1f}%]")
            
            planning_success = sim.plan(
                max_time=self.config.max_planning_time,
                progress_callback=plan_callback
            )
            result['planning_success'] = planning_success
            
            if not planning_success:
                update_status("[规划失败]")
                result['termination_reason'] = 'planning_failed'
                result['error'] = '路径规划失败'
                return result
            
            update_status("[规划完成]")
            
            # 初始化状态
            sim.init_state(use_rng=True)
            
            # 运行仿真（使用回调更新进度）
            max_time = self.config.max_sim_time
            
            def progress_callback(t: float, progress: float):
                """仿真进度回调"""
                update_status(f"[仿真 {progress*100:4.1f}%]")
            
            sim.run(
                max_time=max_time,
                enable_failure_detection=True,
                verbose=False,
                progress_callback=progress_callback
            )
            
            update_status("[导出中]")
            
            # 导出结果
            output_dir = os.path.join(self.exp_dir, self.config.scene, f'seed_{seed:03d}')
            metrics_path, case_path = sim.export_benchmark_results(output_dir)
            
            # 读取 metrics
            with open(metrics_path, 'r', encoding='utf-8') as f:
                metrics = json.load(f)
            
            # 更新 result
            result['success'] = metrics.get('success', False)
            result['control_success'] = metrics.get('success', False)
            result['termination_reason'] = metrics.get('termination_reason', 'unknown')
            result.update(metrics)
            
            update_status("[完成]")
            return result
            
        except Exception as e:
            update_status("[异常]")
            result['error'] = str(e)
            result['termination_reason'] = 'exception'
            return result
    
    def run_all(self, resume: bool = False) -> List[Dict[str, Any]]:
        """
        运行所有 seeds
        
        参数:
            resume: 是否跳过已有结果
        
        返回:
            所有 metrics 列表
        """
        n_total = len(self.config.seeds)
        
        # 打印启动信息
        print()
        print("╔" + "═"*58 + "╗")
        print("║" + " BENCHMARK RUNNER ".center(58) + "║")
        print("╠" + "═"*58 + "╣")
        print(f"║  实验名称: {self.config.name:<45}║")
        print(f"║  场景:     {self.config.scene:<45}║")
        print(f"║  Seeds:    {n_total:<45}║")
        print(f"║  输出目录: {os.path.basename(self.exp_dir):<45}║")
        print("╚" + "═"*58 + "╝")
        print()
        
        self.results = []
        self.stats = {
            'planning_success': 0,
            'planning_failed': 0,
            'control_success': 0,
            'control_failed': 0,
        }
        n_skipped = 0
        
        # 使用 tqdm 进度条
        pbar = tqdm(
            self.config.seeds, 
            desc="Running", 
            unit="seed",
            ncols=100,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}'
        )
        
        for seed in pbar:
            # 更新进度条描述
            pbar.set_description(f"Seed {seed:03d}")
            
            # 检查是否已有结果
            if resume:
                output_dir = os.path.join(self.exp_dir, self.config.scene, f'seed_{seed:03d}')
                metrics_path = os.path.join(output_dir, 'metrics.json')
                if os.path.exists(metrics_path):
                    with open(metrics_path, 'r', encoding='utf-8') as f:
                        metrics = json.load(f)
                    
                    result = {
                        'seed': seed,
                        'planning_success': True,  # 如果有文件说明规划成功了
                        'control_success': metrics.get('success', False),
                        'success': metrics.get('success', False),
                        'termination_reason': metrics.get('termination_reason', 'unknown')
                    }
                    result.update(metrics)
                    self.results.append(result)
                    n_skipped += 1
                    self._update_stats(result)
                    pbar.set_postfix_str(self._get_progress_str())
                    continue
            
            # 运行
            result = self.run_single(seed, pbar=pbar)
            self.results.append(result)
            
            # 更新统计
            self._update_stats(result)
            
            # 更新进度条后缀
            pbar.set_postfix_str(self._get_progress_str())
        
        pbar.close()
        
        # 生成汇总报告
        self.generate_summary()
        
        return self.results
    
    def _update_stats(self, result: Dict[str, Any]):
        """更新统计信息"""
        if result.get('planning_success', False):
            self.stats['planning_success'] += 1
            if result.get('control_success', False) or result.get('success', False):
                self.stats['control_success'] += 1
            else:
                self.stats['control_failed'] += 1
        else:
            self.stats['planning_failed'] += 1
    
    def _get_progress_str(self) -> str:
        """获取进度字符串"""
        ps = self.stats['planning_success']
        pf = self.stats['planning_failed']
        cs = self.stats['control_success']
        cf = self.stats['control_failed']
        return f"Plan:{ps}✓/{pf}✗ Ctrl:{cs}✓/{cf}✗"
    
    def generate_summary(self):
        """生成汇总报告"""
        if not self.results:
            print("没有结果可汇总")
            return
        
        # 创建 summary.csv
        summary_path = os.path.join(self.exp_dir, self.config.scene, 'summary.csv')
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        
        # 收集所有可能的字段
        all_keys = set()
        for r in self.results:
            all_keys.update(r.keys())
        
        # 排序字段
        priority_keys = ['seed', 'success', 'planning_success', 'control_success', 
                        'termination_reason', 'termination_time']
        other_keys = sorted([k for k in all_keys if k not in priority_keys and k != 'error'])
        fieldnames = priority_keys + other_keys
        
        with open(summary_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            for r in self.results:
                writer.writerow(r)
        
        # 打印汇总
        n_total = len(self.results)
        n_success = sum(1 for r in self.results if r.get('success', False))
        n_failed = n_total - n_success
        success_rate = n_success / n_total * 100 if n_total > 0 else 0
        
        # 统计失败原因
        failure_reasons = {}
        for r in self.results:
            if not r.get('success', False):
                reason = r.get('termination_reason', 'unknown')
                failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
        
        # 计算成功运行的质量指标
        success_metrics = [r for r in self.results if r.get('success', False)]
        
        # 获取规划和控制统计
        ps = self.stats['planning_success']
        pf = self.stats['planning_failed']
        cs = self.stats['control_success']
        cf = self.stats['control_failed']
        
        # 打印结果
        print()
        print("╔" + "═"*58 + "╗")
        print("║" + " BENCHMARK RESULTS ".center(58) + "║")
        print("╠" + "═"*58 + "╣")
        
        # 成功率条
        bar_len = 40
        filled = int(bar_len * success_rate / 100)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f"║  Success Rate: [{bar}] {success_rate:5.1f}%  ║")
        
        print("╠" + "═"*58 + "╣")
        print(f"║  Total:   {n_total:<47}║")
        print(f"║  Success: {n_success:<4} ✓{' '*42}║")
        print(f"║  Failed:  {n_failed:<4} ✗{' '*42}║")
        
        # 规划/控制统计
        print("╠" + "─"*58 + "╣")
        print("║  Planning & Control Statistics:" + " "*25 + "║")
        plan_rate = ps / (ps + pf) * 100 if (ps + pf) > 0 else 0
        ctrl_rate = cs / (cs + cf) * 100 if (cs + cf) > 0 else 0
        print(f"║    Planning: {ps:>3} ✓ / {pf:>3} ✗  ({plan_rate:5.1f}% success){' '*12}║")
        print(f"║    Control:  {cs:>3} ✓ / {cf:>3} ✗  ({ctrl_rate:5.1f}% success){' '*12}║")
        
        # 失败原因
        if failure_reasons:
            print("╠" + "─"*58 + "╣")
            print("║  Failure Reasons:" + " "*40 + "║")
            for reason, count in sorted(failure_reasons.items(), key=lambda x: -x[1]):
                pct = count / n_failed * 100 if n_failed > 0 else 0
                # 简化原因名称
                short_reason = reason.replace('soft_', 'S:').replace('hard_', 'H:')
                short_reason = short_reason[:25]
                line = f"    {short_reason}: {count} ({pct:.0f}%)"
                print(f"║{line:<57}║")
        
        # 质量指标
        if success_metrics:
            print("╠" + "─"*58 + "╣")
            print("║  Quality Metrics (successful runs):" + " "*21 + "║")
            
            for metric in ['ADE', 'FDE', 'FDE_horizontal']:
                values = [r.get('quality', {}).get(metric, r.get(metric, None)) 
                         for r in success_metrics]
                values = [v for v in values if v is not None]
                if values:
                    mean_val = sum(values) / len(values)
                    line = f"    {metric}: {mean_val:.2f}m (mean)"
                    print(f"║{line:<57}║")
        
        print("╠" + "═"*58 + "╣")
        print(f"║  Summary: {os.path.basename(summary_path):<46}║")
        print("╚" + "═"*58 + "╝")
        print()


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description='Benchmark Runner')
    parser.add_argument('--config', type=str, default=None,
                       help='实验配置文件 (YAML)')
    parser.add_argument('--seeds', type=str, default='10',
                       help='Seed 数量或范围，例如 "100" 或 "1-100" 或 "1,5,10"')
    parser.add_argument('--scene', type=str, default='default',
                       help='场景名称')
    parser.add_argument('--map-config', type=str, default='cfg/map_config.yaml',
                       help='地图配置文件')
    parser.add_argument('--model-config', type=str, default='cfg/config.yaml',
                       help='模型配置文件')
    parser.add_argument('--output-dir', type=str, default='benchmark/outputs',
                       help='输出目录')
    parser.add_argument('--resume', action='store_true',
                       help='跳过已有结果')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='显示详细输出（默认关闭）')
    
    args = parser.parse_args()
    
    # 创建配置
    if args.config:
        config = ExperimentConfig.from_yaml(args.config)
    else:
        config = ExperimentConfig()
        config.scene = args.scene
        config.map_config = args.map_config
        config.model_config = args.model_config
        
        # 解析 seeds
        if '-' in args.seeds:
            # 范围格式: "1-100"
            start, end = map(int, args.seeds.split('-'))
            config.seeds = list(range(start, end + 1))
        elif ',' in args.seeds:
            # 列表格式: "1,5,10"
            config.seeds = [int(s) for s in args.seeds.split(',')]
        else:
            # 纯数字格式: "100" → seeds 1-100
            n = int(args.seeds)
            config.seeds = list(range(1, n + 1))
    
    # 运行
    runner = BenchmarkRunner(
        config=config,
        output_base_dir=args.output_dir,
        verbose=args.verbose  # 默认 False
    )
    
    runner.run_all(resume=args.resume)


if __name__ == '__main__':
    main()
