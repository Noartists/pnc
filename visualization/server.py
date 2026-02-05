"""
RRT/仿真/Benchmark 可视化服务器

功能：
1. 显示 RRT* 规划结果
2. 显示平滑轨迹
3. 播放闭环仿真动画
4. 浏览 Benchmark 结果

运行: python visualization/server.py
访问: http://localhost:8080

参数:
  --dataset-root  数据目录（默认: visualization/data）
  --benchmark-root  Benchmark 输出目录（默认: benchmark/outputs）
  --port          端口号（默认: 8080）
"""

import os
import sys
import json
import glob
from flask import Flask, render_template, jsonify, request

# 添加项目根目录
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

app = Flask(__name__,
            template_folder='templates',
            static_folder='static')

# 数据目录（可通过参数配置）
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
BENCHMARK_DIR = os.path.join(ROOT_DIR, 'benchmark', 'outputs')


@app.route('/')
def index():
    """主页"""
    return render_template('index.html')


@app.route('/api/files')
def list_files():
    """列出所有数据文件"""
    if not os.path.exists(DATA_DIR):
        return jsonify({'files': []})

    files = []

    # RRT 规划数据
    for f in sorted(glob.glob(os.path.join(DATA_DIR, 'rrt_*.json')), reverse=True):
        filename = os.path.basename(f)
        stat = os.stat(f)
        files.append({
            'name': filename,
            'type': 'rrt',
            'size': stat.st_size,
            'mtime': stat.st_mtime
        })

    # 仿真数据
    for f in sorted(glob.glob(os.path.join(DATA_DIR, 'sim_*.json')), reverse=True):
        filename = os.path.basename(f)
        stat = os.stat(f)
        files.append({
            'name': filename,
            'type': 'simulation',
            'size': stat.st_size,
            'mtime': stat.st_mtime
        })

    # 按修改时间排序
    files.sort(key=lambda x: x['mtime'], reverse=True)

    return jsonify({'files': files, 'data_dir': DATA_DIR})


@app.route('/api/data/<filename>')
def get_data(filename):
    """获取指定数据文件"""
    filepath = os.path.join(DATA_DIR, filename)

    if not os.path.exists(filepath):
        return jsonify({'error': f'File not found: {filename}'}), 404

    # 安全检查：防止路径遍历
    if '..' in filename or '/' in filename or '\\' in filename:
        return jsonify({'error': 'Invalid filename'}), 400

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    data['filename'] = filename

    # 判断数据类型
    if 'actual_trajectory' in data:
        data['data_type'] = 'simulation'
    else:
        data['data_type'] = 'rrt'

    return jsonify(data)


@app.route('/api/latest')
def get_latest():
    """获取最新的数据文件"""
    if not os.path.exists(DATA_DIR):
        return jsonify({'error': 'No data directory'}), 404

    # 优先查找仿真数据，其次是 RRT 数据
    sim_files = sorted(glob.glob(os.path.join(DATA_DIR, 'sim_*.json')), reverse=True)
    rrt_files = sorted(glob.glob(os.path.join(DATA_DIR, 'rrt_*.json')), reverse=True)

    files = sim_files + rrt_files

    if not files:
        return jsonify({'error': 'No data files. Run RRT planner or simulation first.'}), 404

    filepath = files[0]
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    data['filename'] = os.path.basename(filepath)

    # 判断数据类型
    if 'actual_trajectory' in data:
        data['data_type'] = 'simulation'
    else:
        data['data_type'] = 'rrt'

    return jsonify(data)


@app.route('/api/latest/rrt')
def get_latest_rrt():
    """获取最新的 RRT 数据"""
    if not os.path.exists(DATA_DIR):
        return jsonify({'error': 'No data directory'}), 404

    files = sorted(glob.glob(os.path.join(DATA_DIR, 'rrt_*.json')), reverse=True)

    if not files:
        return jsonify({'error': 'No RRT data files.'}), 404

    filepath = files[0]
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    data['filename'] = os.path.basename(filepath)
    data['data_type'] = 'rrt'

    return jsonify(data)


@app.route('/api/latest/sim')
def get_latest_sim():
    """获取最新的仿真数据"""
    if not os.path.exists(DATA_DIR):
        return jsonify({'error': 'No data directory'}), 404

    files = sorted(glob.glob(os.path.join(DATA_DIR, 'sim_*.json')), reverse=True)

    if not files:
        return jsonify({'error': 'No simulation data files.'}), 404

    filepath = files[0]
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    data['filename'] = os.path.basename(filepath)
    data['data_type'] = 'simulation'

    return jsonify(data)


# ============================================================
#                   Benchmark API
# ============================================================

@app.route('/api/benchmark/experiments')
def list_experiments():
    """列出所有 Benchmark 实验"""
    if not os.path.exists(BENCHMARK_DIR):
        return jsonify({'experiments': [], 'benchmark_dir': BENCHMARK_DIR})

    experiments = []
    for exp_dir in sorted(glob.glob(os.path.join(BENCHMARK_DIR, 'exp_*')), reverse=True):
        exp_name = os.path.basename(exp_dir)
        
        # 统计场景和种子数
        scenes = []
        for scene_dir in glob.glob(os.path.join(exp_dir, '*')):
            if os.path.isdir(scene_dir):
                scene_name = os.path.basename(scene_dir)
                seed_dirs = glob.glob(os.path.join(scene_dir, 'seed_*'))
                
                # 读取 summary.csv 获取统计信息
                summary_path = os.path.join(scene_dir, 'summary.csv')
                summary_info = None
                if os.path.exists(summary_path):
                    try:
                        import csv
                        with open(summary_path, 'r', encoding='utf-8') as f:
                            reader = csv.DictReader(f)
                            rows = list(reader)
                            if rows:
                                success_count = sum(1 for r in rows if r.get('success', '').lower() == 'true')
                                summary_info = {
                                    'total': len(rows),
                                    'success': success_count,
                                    'success_rate': success_count / len(rows) * 100 if rows else 0
                                }
                    except Exception as e:
                        pass
                
                scenes.append({
                    'name': scene_name,
                    'seed_count': len(seed_dirs),
                    'summary': summary_info
                })
        
        stat = os.stat(exp_dir)
        experiments.append({
            'name': exp_name,
            'scenes': scenes,
            'mtime': stat.st_mtime
        })

    return jsonify({
        'experiments': experiments,
        'benchmark_dir': BENCHMARK_DIR
    })


@app.route('/api/benchmark/experiment/<exp_name>')
def get_experiment(exp_name):
    """获取实验详情"""
    exp_dir = os.path.join(BENCHMARK_DIR, exp_name)
    
    if not os.path.exists(exp_dir):
        return jsonify({'error': f'Experiment not found: {exp_name}'}), 404
    
    scenes = {}
    for scene_dir in glob.glob(os.path.join(exp_dir, '*')):
        if os.path.isdir(scene_dir):
            scene_name = os.path.basename(scene_dir)
            seeds = []
            
            for seed_dir in sorted(glob.glob(os.path.join(scene_dir, 'seed_*'))):
                seed_name = os.path.basename(seed_dir)
                seed_num = int(seed_name.replace('seed_', ''))
                
                # 读取 metrics.json
                metrics_path = os.path.join(seed_dir, 'metrics.json')
                metrics = None
                if os.path.exists(metrics_path):
                    try:
                        with open(metrics_path, 'r', encoding='utf-8') as f:
                            metrics = json.load(f)
                    except:
                        pass
                
                seeds.append({
                    'seed': seed_num,
                    'has_metrics': os.path.exists(metrics_path),
                    'has_case': os.path.exists(os.path.join(seed_dir, 'case.json')),
                    'metrics': metrics
                })
            
            scenes[scene_name] = {
                'seeds': sorted(seeds, key=lambda x: x['seed'])
            }
    
    return jsonify({
        'experiment': exp_name,
        'scenes': scenes
    })


@app.route('/api/benchmark/case/<exp_name>/<scene_name>/<int:seed>')
def get_benchmark_case(exp_name, scene_name, seed):
    """获取 Benchmark 的 case.json 并转换为可视化格式"""
    case_path = os.path.join(BENCHMARK_DIR, exp_name, scene_name, f'seed_{seed:03d}', 'case.json')
    
    if not os.path.exists(case_path):
        return jsonify({'error': f'Case not found: {case_path}'}), 404
    
    with open(case_path, 'r', encoding='utf-8') as f:
        case_data = json.load(f)
    
    # 转换为可视化格式
    viz_data = convert_benchmark_to_viz(case_data)
    viz_data['filename'] = f'{exp_name}/{scene_name}/seed_{seed:03d}'
    viz_data['data_type'] = 'simulation'
    viz_data['source'] = 'benchmark'
    
    return jsonify(viz_data)


def convert_benchmark_to_viz(case_data):
    """将 Benchmark case.json 格式转换为可视化格式"""
    viz = {}
    
    # 起终点
    if 'start_position' in case_data and case_data['start_position']:
        pos = case_data['start_position']
        viz['start'] = {'x': pos[0], 'y': pos[1], 'z': pos[2]}
    
    if 'target_position' in case_data and case_data['target_position']:
        pos = case_data['target_position']
        viz['goal'] = {'x': pos[0], 'y': pos[1], 'z': pos[2]}
    
    # 障碍物（从 no_fly_zones 转换）
    viz['obstacles'] = []
    if 'no_fly_zones' in case_data:
        for nfz in case_data['no_fly_zones']:
            if nfz.get('type') == 'cylinder':
                viz['obstacles'].append({
                    'type': 'cylinder',
                    'center': nfz['center'],
                    'radius': nfz['radius'],
                    'z_min': nfz.get('z_min', 0),
                    'z_max': nfz.get('z_max', 500)
                })
            elif nfz.get('type') == 'polygon':
                viz['obstacles'].append({
                    'type': 'prism',
                    'vertices': nfz['vertices'],
                    'z_min': nfz.get('z_min', 0),
                    'z_max': nfz.get('z_max', 500)
                })
    
    # 参考轨迹
    if 'reference_trajectory' in case_data:
        viz['reference_trajectory'] = case_data['reference_trajectory']
    
    # 实际轨迹
    if 'actual_trajectory' in case_data:
        viz['actual_trajectory'] = case_data['actual_trajectory']
    
    # 控制数据
    if 'control_history' in case_data:
        viz['control_history'] = case_data['control_history']
    
    # 事件
    if 'events' in case_data:
        viz['events'] = case_data['events']
    
    # 配置
    if 'config' in case_data:
        viz['config'] = case_data['config']
    
    # 计算仿真信息
    if viz.get('actual_trajectory'):
        traj = viz['actual_trajectory']
        if traj:
            viz['info'] = {
                'duration': traj[-1].get('t', 0) if traj else 0,
                'n_steps': len(traj)
            }
    
    return viz


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='RRT/仿真/Benchmark 可视化服务器')
    parser.add_argument('--port', type=int, default=8080, help='端口号 (默认: 8080)')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='主机地址 (默认: 127.0.0.1)')
    parser.add_argument('--dataset_root', type=str, default=None,
                        help='数据目录 (默认: visualization/data)，目录不存在会自动创建')
    parser.add_argument('--benchmark_root', type=str, default=None,
                        help='Benchmark 输出目录 (默认: benchmark/outputs)')
    parser.add_argument('--debug', action='store_true', default=True, help='调试模式')
    args = parser.parse_args()

    # 设置数据目录
    if args.dataset_root:
        DATA_DIR = os.path.abspath(args.dataset_root)
        # 确保目录存在
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR, exist_ok=True)
            print(f"创建数据目录: {DATA_DIR}")
    
    if args.benchmark_root:
        BENCHMARK_DIR = os.path.abspath(args.benchmark_root)

    # 确保数据目录存在
    os.makedirs(DATA_DIR, exist_ok=True)

    print("=" * 50)
    print("  RRT/仿真/Benchmark 可视化服务器")
    print("=" * 50)
    print(f"  访问: http://{args.host}:{args.port}")
    print(f"  数据目录: {DATA_DIR}")
    print(f"  Benchmark 目录: {BENCHMARK_DIR}")
    print("=" * 50)

    app.run(host=args.host, port=args.port, debug=args.debug)
