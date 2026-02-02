"""
RRT可视化服务器

运行: python visualization/server.py
访问: http://localhost:8080
"""

import os
import sys
import json
import glob
from flask import Flask, render_template, jsonify

# 添加项目根目录
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

app = Flask(__name__, 
            template_folder='templates',
            static_folder='static')

# 数据目录
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
    for f in sorted(glob.glob(os.path.join(DATA_DIR, 'rrt_*.json')), reverse=True):
        filename = os.path.basename(f)
        stat = os.stat(f)
        files.append({
            'name': filename,
            'size': stat.st_size,
            'mtime': stat.st_mtime
        })
    
    return jsonify({'files': files})


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
    return jsonify(data)


@app.route('/api/latest')
def get_latest():
    """获取最新的数据文件"""
    if not os.path.exists(DATA_DIR):
        return jsonify({'error': 'No data directory'}), 404
    
    files = sorted(glob.glob(os.path.join(DATA_DIR, 'rrt_*.json')), reverse=True)
    
    if not files:
        return jsonify({'error': 'No data files. Run RRT planner first.'}), 404
    
    filepath = files[0]
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    data['filename'] = os.path.basename(filepath)
    return jsonify(data)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='RRT可视化服务器')
    parser.add_argument('--port', type=int, default=8080, help='端口号 (默认: 8080)')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='主机地址 (默认: 127.0.0.1)')
    parser.add_argument('--debug', action='store_true', default=True, help='调试模式')
    args = parser.parse_args()
    
    # 确保数据目录存在
    os.makedirs(DATA_DIR, exist_ok=True)
    
    print("=" * 50)
    print("  RRT 可视化服务器")
    print("=" * 50)
    print(f"  访问: http://{args.host}:{args.port}")
    print(f"  数据目录: {DATA_DIR}")
    print("=" * 50)
    
    app.run(host=args.host, port=args.port, debug=args.debug)
