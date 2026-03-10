import os
import csv
import numpy as np
from flask import Flask, render_template, jsonify, request, send_from_directory
from glob import glob
from pathlib import Path
import json

app = Flask(__name__, template_folder=str(Path(__file__).parent / "templates"))

KITTI_DATA_ROOT = None
SEQUENCES_DIR = None
MAPPING_FILE = None
SATELLITE_JSON = None

# 类别定义（标签1-7，0不使用）
LABEL_NAMES = {
    1: 'main_body',
    2: 'solar_panel',
    3: 'dish_antenna',
    4: 'omni_antenna',
    5: 'payload',
    6: 'thruster',
    7: 'adapter_ring'
}

# 颜色定义 (RGB格式，0-255)
LABEL_COLORS = {
    1: [31, 119, 180],    # blue
    2: [255, 127, 14],    # orange
    3: [44, 160, 44],     # green
    4: [214, 39, 40],     # red
    5: [148, 103, 189],   # purple
    6: [140, 86, 75],     # brown
    7: [227, 119, 194]    # pink/magenta
}

def load_satellite_info():
    """从JSON文件加载卫星详细信息"""
    satellite_info = {}
    if os.path.exists(SATELLITE_JSON):
        try:
            with open(SATELLITE_JSON, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for sat in data['satellites']:
                satellite_info[sat['name']] = {
                    'description': sat.get('description', ''),
                    'max_diameter_meters': sat.get('max_diameter_meters', None)
                }
        except Exception as e:
            print(f"警告: 无法读取卫星信息: {e}")
    return satellite_info


def load_sequence_mapping():
    """加载sequence序号与卫星名称的映射，并附加详细信息"""
    mapping = {}
    satellite_info = load_satellite_info()
    
    if os.path.exists(MAPPING_FILE):
        with open(MAPPING_FILE, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                sat_name = row['satellite_name']
                mapping[row['sequence_id']] = {
                    'name': sat_name,
                    'info': satellite_info.get(sat_name, {})
                }
    return mapping

def read_bin_pointcloud(bin_path):
    """读取.bin格式的点云文件"""
    points = np.fromfile(bin_path, dtype=np.float32)
    if points.size % 4 != 0:
        raise ValueError(f"点云数据长度不是4的倍数: {points.size}")
    points = points.reshape(-1, 4)
    return points

def read_label_file(label_path):
    """读取.label格式的标签文件"""
    if not os.path.exists(label_path):
        return None
    labels = np.fromfile(label_path, dtype=np.uint32)
    return labels

@app.route('/')
def index():
    """主页"""
    return render_template('visualizer.html')

@app.route('/api/satellites')
def get_satellites():
    """获取所有卫星列表（基于sequence mapping），包含详细信息"""
    try:
        mapping = load_sequence_mapping()
        # 返回序号和卫星详细信息的列表
        satellites = []
        for seq_id in sorted(mapping.keys(), key=lambda x: int(x)):
            sat_data = mapping[seq_id]
            satellites.append({
                'id': seq_id,
                'name': sat_data['name'],
                'max_diameter': sat_data['info'].get('max_diameter_meters'),
                'description': sat_data['info'].get('description', '')[:100] + '...' if len(sat_data['info'].get('description', '')) > 100 else sat_data['info'].get('description', '')
            })
        return jsonify({'success': True, 'satellites': satellites})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/frames/<sequence_id>')
def get_frames(sequence_id):
    """获取指定sequence的所有帧"""
    try:
        velodyne_dir = os.path.join(SEQUENCES_DIR, sequence_id, 'velodyne')
        bin_files = sorted(glob(os.path.join(velodyne_dir, "*.bin")))
        frame_ids = [os.path.splitext(os.path.basename(f))[0] for f in bin_files]
        
        # 获取卫星信息
        mapping = load_sequence_mapping()
        sat_data = mapping.get(sequence_id, {'name': f"Sequence {sequence_id}", 'info': {}})
        satellite_name = sat_data['name']
        
        return jsonify({
            'success': True, 
            'frames': frame_ids, 
            'total': len(frame_ids),
            'satellite_name': satellite_name,
            'max_diameter': sat_data['info'].get('max_diameter_meters')
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/pointcloud/<sequence_id>/<frame_id>')
def get_pointcloud(sequence_id, frame_id):
    """获取指定帧的点云数据"""
    try:
        # 读取点云
        bin_path = os.path.join(SEQUENCES_DIR, sequence_id, 'velodyne', f'{frame_id}.bin')
        points = read_bin_pointcloud(bin_path)
        
        # 读取标签
        label_path = os.path.join(SEQUENCES_DIR, sequence_id, 'labels', f'{frame_id}.label')
        labels = read_label_file(label_path)
        
        # 准备数据
        data = {
            'success': True,
            'points': points[:, :3].tolist(),  # 只返回XYZ
            'labels': labels.tolist() if labels is not None else None,
            'point_count': len(points)
        }
        
        # 添加标签统计
        if labels is not None:
            unique_labels = np.unique(labels)
            label_stats = {}
            for lbl in unique_labels:
                count = int(np.sum(labels == lbl))
                label_stats[int(lbl)] = {
                    'name': LABEL_NAMES.get(int(lbl), 'unknown'),
                    'count': count,
                    'color': LABEL_COLORS.get(int(lbl), [128, 128, 128])
                }
            data['label_stats'] = label_stats
        
        return jsonify(data)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/label_info')
def get_label_info():
    """获取标签定义信息"""
    label_info = {}
    for label_id, name in LABEL_NAMES.items():
        label_info[label_id] = {
            'name': name,
            'color': LABEL_COLORS[label_id]
        }
    return jsonify(label_info)

@app.route('/api/image/<sequence_id>/<frame_id>')
def get_image(sequence_id, frame_id):
    """获取指定帧的图像"""
    try:
        image_dir = os.path.join(SEQUENCES_DIR, sequence_id, 'image_2')
        return send_from_directory(image_dir, f'{frame_id}.png')
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 404

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Semantic-KITTI点云可视化Web服务器")
    parser.add_argument('--data-root', type=str, required=True,
                       help='Semantic-KITTI格式数据根目录 (包含 sequences/)')
    parser.add_argument('--satellite-json', type=str, default=None,
                       help='satellite_descriptions.json 路径 (可选)')
    parser.add_argument('--port', type=int, default=5000, help='端口号')
    _args = parser.parse_args()

    KITTI_DATA_ROOT = _args.data_root
    SEQUENCES_DIR = os.path.join(KITTI_DATA_ROOT, 'sequences')
    MAPPING_FILE = os.path.join(KITTI_DATA_ROOT, 'sequence_mapping.csv')
    SATELLITE_JSON = _args.satellite_json

    print("\n=== 点云可视化 Web 服务器 ===")
    print(f"数据根目录: {KITTI_DATA_ROOT}")
    print(f"请在浏览器中打开: http://localhost:{_args.port}")
    print("按 Ctrl+C 停止服务器\n")
    app.run(debug=True, host='0.0.0.0', port=_args.port)

