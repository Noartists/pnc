"""
RRT/仿真 可视化服务器

功能：
1. 显示 RRT* 规划结果
2. 显示平滑轨迹
3. 播放闭环仿真动画

运行: python visualization/server.py
访问: http://localhost:8080

参数:
  --dataset-root  数据目录（默认: visualization/data）
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


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='RRT/仿真 可视化服务器')
    parser.add_argument('--port', type=int, default=8080, help='端口号 (默认: 8080)')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='主机地址 (默认: 127.0.0.1)')
    parser.add_argument('--dataset_root', type=str, default=None,
                        help='数据目录 (默认: visualization/data)，目录不存在会自动创建')
    parser.add_argument('--debug', action='store_true', default=True, help='调试模式')
    args = parser.parse_args()

    # 设置数据目录
    if args.dataset_root:
        DATA_DIR = os.path.abspath(args.dataset_root)
        # 确保目录存在
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR, exist_ok=True)
            print(f"创建数据目录: {DATA_DIR}")

    # 确保数据目录存在
    os.makedirs(DATA_DIR, exist_ok=True)

    print("=" * 50)
    print("  RRT/仿真 可视化服务器")
    print("=" * 50)
    print(f"  访问: http://{args.host}:{args.port}")
    print(f"  数据目录: {DATA_DIR}")
    print("=" * 50)

    app.run(host=args.host, port=args.port, debug=args.debug)
