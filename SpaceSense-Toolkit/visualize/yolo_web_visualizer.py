#!/usr/bin/env python3
"""
YOLO格式数据可视化Web服务器
可视化转换后的YOLO格式标注

使用方法:
  python yolo_web_visualizer.py --data-root /path/to/yolo_data
"""
import os
import sys
import cv2
import numpy as np
from flask import Flask, render_template, jsonify, send_file
from pathlib import Path
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont

app = Flask(__name__, template_folder=str(Path(__file__).parent / "templates"))

YOLO_DATA_ROOT = None

# 类别定义
CLASS_NAMES = ['main_body', 'solar_panel', 'dish_antenna', 'omni_antenna', 
               'payload', 'thruster', 'adapter_ring']

# 颜色定义 (BGR格式用于OpenCV)
CLASS_COLORS = [
    (180, 119, 31),   # main_body - blue
    (14, 127, 255),   # solar_panel - orange
    (44, 160, 44),    # dish_antenna - green
    (40, 39, 214),    # omni_antenna - red
    (189, 103, 148),  # payload - purple
    (75, 86, 140),    # thruster - brown
    (194, 119, 227)   # adapter_ring - pink
]


def extract_satellite_name(filename):
    """从文件名提取卫星名称
    例如: ACE_trajectory1_frame001.png -> ACE
    """
    return filename.split('_')[0]


def get_dataset_info():
    """获取数据集信息"""
    datasets = {}
    
    for split in ['train', 'val']:
        image_dir = YOLO_DATA_ROOT / split / 'images'
        label_dir = YOLO_DATA_ROOT / split / 'labels'
        
        if not image_dir.exists():
            continue
        
        # 获取所有图像
        image_files = sorted(image_dir.glob('*.png'))
        
        # 按卫星分组
        satellites = {}
        for img_file in image_files:
            sat_name = extract_satellite_name(img_file.stem)
            if sat_name not in satellites:
                satellites[sat_name] = []
            satellites[sat_name].append(img_file.stem)
        
        datasets[split] = {
            'total_images': len(image_files),
            'satellites': satellites,
            'satellite_count': len(satellites)
        }
    
    return datasets


def parse_yolo_label(label_path, img_width, img_height):
    """
    解析YOLO标签文件
    
    Args:
        label_path: 标签文件路径
        img_width: 图像宽度
        img_height: 图像高度
    
    Returns:
        list of dict: [{class_id, class_name, bbox, center, size}, ...]
    """
    if not label_path.exists():
        return []
    
    boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            values = line.strip().split()
            if len(values) != 5:
                continue
            
            class_id = int(values[0])
            cx, cy, w, h = map(float, values[1:])
            
            # 转换为像素坐标
            cx_px = cx * img_width
            cy_px = cy * img_height
            w_px = w * img_width
            h_px = h * img_height
            
            # 计算左上角和右下角坐标
            x1 = int(cx_px - w_px / 2)
            y1 = int(cy_px - h_px / 2)
            x2 = int(cx_px + w_px / 2)
            y2 = int(cy_px + h_px / 2)
            
            boxes.append({
                'class_id': class_id,
                'class_name': CLASS_NAMES[class_id],
                'bbox': [x1, y1, x2, y2],
                'center': [int(cx_px), int(cy_px)],
                'size': [int(w_px), int(h_px)],
                'normalized': [cx, cy, w, h]
            })
    
    return boxes


def draw_boxes_on_image(image_path, label_path):
    """
    在图像上绘制边界框
    
    Args:
        image_path: 图像文件路径
        label_path: 标签文件路径
    
    Returns:
        PIL.Image: 绘制了边界框的图像
    """
    # 读取图像
    img = Image.open(image_path)
    img_width, img_height = img.size
    draw = ImageDraw.Draw(img)
    
    # 解析标签
    boxes = parse_yolo_label(label_path, img_width, img_height)
    
    # 尝试加载字体
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    # 绘制每个边界框
    for box in boxes:
        class_id = box['class_id']
        class_name = box['class_name']
        x1, y1, x2, y2 = box['bbox']
        
        # BGR转RGB
        color = CLASS_COLORS[class_id]
        color_rgb = (color[2], color[1], color[0])
        
        # 绘制边界框
        draw.rectangle([x1, y1, x2, y2], outline=color_rgb, width=3)
        
        # 绘制标签背景
        text = f"{class_name}"
        bbox = draw.textbbox((x1, y1), text, font=font)
        draw.rectangle([bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2], fill=color_rgb)
        
        # 绘制文字
        draw.text((x1, y1), text, fill=(255, 255, 255), font=font)
    
    return img, boxes


@app.route('/')
def index():
    """主页"""
    return render_template('yolo_visualizer.html')


@app.route('/api/dataset_info')
def get_dataset_info_api():
    """获取数据集信息"""
    try:
        datasets = get_dataset_info()
        return jsonify({'success': True, 'datasets': datasets})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/satellites/<split>')
def get_satellites(split):
    """获取指定数据集的卫星列表"""
    try:
        datasets = get_dataset_info()
        
        if split not in datasets:
            return jsonify({'success': False, 'error': f'数据集 {split} 不存在'})
        
        satellites = datasets[split]['satellites']
        satellite_list = []
        
        for sat_name in sorted(satellites.keys()):
            satellite_list.append({
                'name': sat_name,
                'image_count': len(satellites[sat_name])
            })
        
        return jsonify({
            'success': True, 
            'satellites': satellite_list,
            'total': len(satellite_list)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/images/<split>/<satellite_name>')
def get_images(split, satellite_name):
    """获取指定卫星的所有图像"""
    try:
        datasets = get_dataset_info()
        
        if split not in datasets:
            return jsonify({'success': False, 'error': f'数据集 {split} 不存在'})
        
        satellites = datasets[split]['satellites']
        
        if satellite_name not in satellites:
            return jsonify({'success': False, 'error': f'卫星 {satellite_name} 不存在'})
        
        images = satellites[satellite_name]
        
        return jsonify({
            'success': True,
            'images': images,
            'total': len(images)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/image_with_boxes/<split>/<image_name>')
def get_image_with_boxes(split, image_name):
    """获取带有边界框的图像"""
    try:
        image_path = YOLO_DATA_ROOT / split / 'images' / f'{image_name}.png'
        label_path = YOLO_DATA_ROOT / split / 'labels' / f'{image_name}.txt'
        
        if not image_path.exists():
            return jsonify({'success': False, 'error': '图像不存在'}), 404
        
        # 绘制边界框
        img, boxes = draw_boxes_on_image(image_path, label_path)
        
        # 转换为字节流
        img_io = BytesIO()
        img.save(img_io, 'PNG')
        img_io.seek(0)
        
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/annotation/<split>/<image_name>')
def get_annotation(split, image_name):
    """获取指定图像的标注信息"""
    try:
        image_path = YOLO_DATA_ROOT / split / 'images' / f'{image_name}.png'
        label_path = YOLO_DATA_ROOT / split / 'labels' / f'{image_name}.txt'
        
        if not image_path.exists():
            return jsonify({'success': False, 'error': '图像不存在'})
        
        # 获取图像尺寸
        img = Image.open(image_path)
        img_width, img_height = img.size
        
        # 解析标签
        boxes = parse_yolo_label(label_path, img_width, img_height)
        
        # 统计类别
        class_stats = {}
        for box in boxes:
            class_name = box['class_name']
            if class_name not in class_stats:
                class_stats[class_name] = {
                    'count': 0,
                    'class_id': box['class_id'],
                    'color': CLASS_COLORS[box['class_id']]
                }
            class_stats[class_name]['count'] += 1
        
        return jsonify({
            'success': True,
            'boxes': boxes,
            'total_boxes': len(boxes),
            'class_stats': class_stats,
            'image_size': [img_width, img_height]
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/class_info')
def get_class_info():
    """获取类别信息"""
    class_info = []
    for i, name in enumerate(CLASS_NAMES):
        color = CLASS_COLORS[i]
        class_info.append({
            'id': i,
            'name': name,
            'color': [color[2], color[1], color[0]]  # 转为RGB
        })
    return jsonify(class_info)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="YOLO数据可视化Web服务器")
    parser.add_argument('--data-root', type=str, required=True,
                       help='YOLO格式数据根目录')
    parser.add_argument('--port', type=int, default=5001, help='端口号')
    _args = parser.parse_args()

    YOLO_DATA_ROOT = Path(_args.data_root)

    print("\n" + "="*70)
    print("YOLO数据可视化Web服务器")
    print("="*70)
    print(f"数据根目录: {YOLO_DATA_ROOT}")
    print(f"\n请在浏览器中打开: http://localhost:{_args.port}")
    print("="*70 + "\n")

    app.run(debug=True, host='0.0.0.0', port=_args.port)

