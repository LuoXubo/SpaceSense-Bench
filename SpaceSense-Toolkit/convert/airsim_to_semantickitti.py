#!/usr/bin/env python3
"""
将AirSim采集的卫星数据完整转换为Semantic-KITTI格式
合并了点云语义标注和数据组织两个步骤
默认使用多进程并行加速，使用CPU核心数的2/3（避免电脑卡死）

使用方法:
  python airsim_to_semantickitti.py                 # 默认并行，使用2/3核心数
  python airsim_to_semantickitti.py --workers 4     # 指定使用4个进程
  python airsim_to_semantickitti.py --serial        # 使用串行处理（不推荐）
"""
import os
import sys

# 限制numpy线程数，避免多进程冲突（必须在导入numpy之前设置）
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import csv
import json
import math
import shutil
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed


def read_asc_pointcloud(filepath):
    """读取.asc格式的点云文件"""
    points = []
    
    with open(filepath, 'r') as f:
        for line in f:
            values = [x.strip() for x in line.strip().split(',') if x.strip()]
            
            if len(values) >= 3:
                try:
                    x, y, z = map(float, values[:3])
                    points.append([x, y, z])
                except ValueError as e:
                    print(f"格式错误: {values} -> {e}")
    
    return np.array(points, dtype=np.float32)


def transform_lidar_to_camera_frame(point):
    """将激光雷达坐标系下的点转换到相机坐标系"""
    translation = np.array([0, 0, 0])
    return point + translation


def project_point_to_image(point, image_width, image_height):
    """将3D点投影到图像平面上"""
    fov_rad = math.radians(50)
    focal_length = (image_width / 2) / math.tan(fov_rad / 2)
    
    if point[0] <= 0:
        return None
    
    cx = image_width / 2
    cy = image_height / 2
    
    u = int(focal_length * point[1] / point[0] + cx)
    v = int(focal_length * point[2] / point[0] + cy)
    
    if 0 <= u < image_width and 0 <= v < image_height:
        return (u, v)
    return None


def get_label_from_segmentation(seg_image, u, v):
    """从语义分割图像中获取标签"""
    pixel_value = seg_image[v, u]
    
    if len(seg_image.shape) == 3:
        pixel_rgb = (pixel_value[2], pixel_value[1], pixel_value[0])
        
        color_to_label = {
            # main_body - 类别1
            (156, 198, 23): 1, (68, 218, 116): 1, (11, 236, 9): 1, (0, 53, 65): 1,
            # solar_panel - 类别2
            (146, 52, 70): 2, (194, 39, 7): 2, (211, 80, 208): 2, (189, 135, 188): 2,
            # dish_antenna - 类别3
            (124, 21, 123): 3, (90, 162, 242): 3, (35, 196, 244): 3, (220, 163, 49): 3,
            # omni_antenna - 类别4
            (86, 254, 214): 4, (125, 75, 48): 4, (85, 152, 34): 4, (173, 69, 31): 4,
            # payload - 类别5
            (37, 128, 125): 5, (58, 19, 33): 5, (218, 124, 115): 5, (202, 97, 155): 5,
            # thruster - 类别6
            (133, 244, 133): 6, (1, 222, 192): 6, (65, 54, 217): 6, (216, 78, 75): 6,
            # adapter_ring - 类别7
            (158, 114, 88): 7, (181, 213, 93): 7,
        }
        
        return color_to_label.get(pixel_rgb, None)
    else:
        return int(pixel_value)


def convert_trajectory_to_kitti(pointcloud_dir, seg_dir, img_dir, 
                                output_bin_dir, output_label_dir, output_img_dir):
    """将单个轨迹的Airsim点云数据转换为Semantic KITTI格式"""
    os.makedirs(output_bin_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)
    os.makedirs(output_img_dir, exist_ok=True)
    
    pointcloud_files = sorted(glob(os.path.join(pointcloud_dir, "*.asc")))
    
    for pc_file in tqdm(pointcloud_files, desc="  Converting frames", leave=False):
        file_id = os.path.splitext(os.path.basename(pc_file))[0]
        
        seg_file = os.path.join(seg_dir, f"{file_id}.png")
        img_file = os.path.join(img_dir, f"{file_id}.png")
        
        if not os.path.exists(seg_file):
            continue
        
        # 复制图像
        shutil.copy(img_file, os.path.join(output_img_dir, f"{file_id}.png"))
        
        # 读取点云和分割图像
        points = read_asc_pointcloud(pc_file)
        seg_image = cv2.imread(seg_file)
        height, width = seg_image.shape[:2]
        
        labels = np.zeros(len(points), dtype=np.uint32)
        valid_indices = []
        
        # 投影并获取标签
        for i, point in enumerate(points):
            camera_point = transform_lidar_to_camera_frame(point)
            projection = project_point_to_image(camera_point, width, height)
            
            label = None
            if projection:
                u, v = projection
                label = get_label_from_segmentation(seg_image, u, v)
            
            if label is not None:
                labels[i] = label
                valid_indices.append(i)
        
        # 筛选有效点
        points = points[valid_indices]
        labels = labels[valid_indices]
        
        # 检查有效点数，如果为0则跳过该帧
        if len(points) == 0:
            print(f"    警告: {file_id} 没有有效点，跳过该帧")
            continue
        
        # 保存点云（XYZI格式）
        xyz = np.zeros((len(points), 4), dtype=np.float32)
        xyz[:, :3] = points
        xyz[:, 3] = 0.0  # 强度值
        
        output_bin_file = os.path.join(output_bin_dir, f"{file_id}.bin")
        xyz.tofile(output_bin_file)
        
        # 保存标签
        output_label_file = os.path.join(output_label_dir, f"{file_id}.label")
        labels.tofile(output_label_file)


def create_default_calib():
    """创建calib.txt文件"""
    return """P2: 1097.98754330 0 512.0 0 0 1097.98754330 512.0 0 0 0 1 0
Tr: 0 1 0 0 0 0 1 0 1 0 0 0
"""


def extract_satellite_name(folder_name):
    """从文件夹名称中提取卫星名称
    例如: 20260114230632_ACE -> ACE
    """
    if '_' in folder_name:
        return folder_name.split('_', 1)[1]
    return folder_name


def load_satellite_order(json_path):
    """
    从satellite_descriptions.json加载卫星顺序
    
    Args:
        json_path: JSON文件路径
    
    Returns:
        list: 卫星名称列表，按JSON中的顺序
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return [sat['name'] for sat in data['satellites']]
    except Exception as e:
        print(f"警告: 无法读取 {json_path}: {e}")
        return []


def sort_satellites_by_json_order(satellite_folders, json_path):
    """
    按照JSON文件中的顺序对卫星文件夹进行排序
    
    Args:
        satellite_folders: 卫星文件夹路径列表
        json_path: satellite_descriptions.json路径
    
    Returns:
        排序后的卫星文件夹列表
    """
    # 加载JSON中的卫星顺序
    json_order = load_satellite_order(json_path)
    
    if not json_order:
        print("警告: 未能加载JSON顺序，将按文件夹名称排序")
        return sorted(satellite_folders)
    
    # 创建名称到索引的映射
    name_to_index = {name: idx for idx, name in enumerate(json_order)}
    
    # 为每个文件夹创建排序键
    def sort_key(folder):
        sat_name = extract_satellite_name(folder.name)
        # 如果在JSON中找到，返回其索引；否则返回一个很大的数（放在最后）
        return name_to_index.get(sat_name, len(json_order) + 1000)
    
    sorted_folders = sorted(satellite_folders, key=sort_key)
    
    # 输出排序信息
    print("\n卫星处理顺序（按satellite_descriptions.json）:")
    for idx, folder in enumerate(sorted_folders):
        sat_name = extract_satellite_name(folder.name)
        if sat_name in name_to_index:
            print(f"  {idx:3d}. {sat_name} (JSON中第 {name_to_index[sat_name]} 个)")
        else:
            print(f"  {idx:3d}. {sat_name} (未在JSON中)")
    
    return sorted_folders


def process_single_satellite(satellite_folder, raw_data_root, sequences_dir, seq_id):
    """
    处理单颗卫星的数据（用于并行执行）
    
    Args:
        satellite_folder: 卫星数据文件夹
        raw_data_root: 原始数据根目录
        sequences_dir: sequences输出目录
        seq_id: 序列ID
    
    Returns:
        (seq_id, satellite_name, frame_count, status)
    """
    try:
        satellite_name = extract_satellite_name(satellite_folder.name)
        
        # 创建sequence目录结构
        seq_dir = sequences_dir / seq_id
        velodyne_dir = seq_dir / "velodyne"
        labels_dir = seq_dir / "labels"
        image_dir = seq_dir / "image_2"
        
        velodyne_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        image_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建calib.txt
        calib_path = seq_dir / "calib.txt"
        with open(calib_path, 'w') as f:
            f.write(create_default_calib())
        
        # 获取该卫星下的所有轨迹文件夹
        trajectory_dirs = sorted([
            d for d in satellite_folder.iterdir() 
            if d.is_dir() and not d.name.startswith('trajectory')
        ])
        
        if len(trajectory_dirs) == 0:
            return (seq_id, satellite_name, 0, "未找到轨迹文件夹")
        
        # 创建临时目录（每个进程独立的临时目录）
        temp_dir = sequences_dir.parent / f"temp_{seq_id}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        frame_counter = 0
        
        # 遍历每条轨迹进行转换
        for traj_dir in trajectory_dirs:
            src_pointcloud = traj_dir / "lidar"
            src_seg = traj_dir / "seg"
            src_img = traj_dir / "image"
            
            # 检查必要文件夹
            if not all([src_pointcloud.exists(), src_seg.exists(), src_img.exists()]):
                continue
            
            # 转换为KITTI格式（保存到临时目录）
            temp_traj_dir = temp_dir / traj_dir.name
            temp_velodyne = temp_traj_dir / "velodyne"
            temp_labels = temp_traj_dir / "labels"
            temp_images = temp_traj_dir / "image_2"
            
            try:
                convert_trajectory_to_kitti(
                    str(src_pointcloud), str(src_seg), str(src_img),
                    str(temp_velodyne), str(temp_labels), str(temp_images)
                )
                
                # 获取转换后的文件并重命名复制
                bin_files = sorted(temp_velodyne.glob("*.bin"))
                
                for bin_file in bin_files:
                    new_name = f"{frame_counter:06d}"
                    timestamp = bin_file.stem
                    
                    # 复制velodyne文件
                    shutil.copy2(bin_file, velodyne_dir / f"{new_name}.bin")
                    
                    # 复制label文件
                    label_file = temp_labels / f"{timestamp}.label"
                    if label_file.exists():
                        shutil.copy2(label_file, labels_dir / f"{new_name}.label")
                    
                    # 复制image文件
                    image_file = temp_images / f"{timestamp}.png"
                    if image_file.exists():
                        shutil.copy2(image_file, image_dir / f"{new_name}.png")
                    
                    frame_counter += 1
                
            except Exception as e:
                print(f"  [Seq {seq_id}] 错误: {e}")
                continue
        
        # 清理临时文件
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        
        return (seq_id, satellite_name, frame_counter, "成功")
        
    except Exception as e:
        return (seq_id, extract_satellite_name(satellite_folder.name), 0, f"错误: {str(e)}")


def convert_airsim_to_kitti_sequences(raw_data_root, output_root, json_path=None):
    """
    将AirSim采集的所有卫星数据转换为Semantic-KITTI格式（串行版本）
    每颗卫星对应一个sequence，合并该卫星的所有轨迹
    
    Args:
        raw_data_root: 原始数据根目录 (data_collect/raw_data)
        output_root: 输出根目录
        json_path: satellite_descriptions.json路径（用于确定顺序）
    
    Returns:
        sequence_mapping: 字典，{sequence_id: satellite_name}
    """
    raw_data_root = Path(raw_data_root)
    output_root = Path(output_root)
    
    # 创建sequences目录
    sequences_dir = output_root / "sequences"
    sequences_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取所有卫星数据文件夹
    # 过滤掉trajectory开头的文件夹
    satellite_folders = [
        d for d in raw_data_root.iterdir() 
        if d.is_dir() and not d.name.startswith('trajectory')
    ]
    
    # 按照JSON文件顺序排序
    if json_path and Path(json_path).exists():
        satellite_folders = sort_satellites_by_json_order(satellite_folders, json_path)
    else:
        print("警告: 未指定JSON文件或文件不存在，将按文件夹名称排序")
        satellite_folders = sorted(satellite_folders)
    
    print(f"\n找到 {len(satellite_folders)} 个卫星数据文件夹")
    
    sequence_mapping = {}
    
    # 为每颗卫星创建一个sequence
    for seq_idx, satellite_folder in enumerate(satellite_folders):
        seq_id = f"{seq_idx:02d}"
        satellite_name = extract_satellite_name(satellite_folder.name)
        sequence_mapping[seq_id] = satellite_name
        
        print(f"\n{'='*60}")
        print(f"处理 Sequence {seq_id}: {satellite_name}")
        print(f"{'='*60}")
        
        # 创建sequence目录结构
        seq_dir = sequences_dir / seq_id
        velodyne_dir = seq_dir / "velodyne"
        labels_dir = seq_dir / "labels"
        image_dir = seq_dir / "image_2"
        
        velodyne_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        image_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建calib.txt
        calib_path = seq_dir / "calib.txt"
        with open(calib_path, 'w') as f:
            f.write(create_default_calib())
        
        # 获取该卫星下的所有轨迹文件夹
        trajectory_dirs = sorted([
            d for d in satellite_folder.iterdir() 
            if d.is_dir() and not d.name.startswith('trajectory')
        ])
        
        if len(trajectory_dirs) == 0:
            print(f"  警告: 未找到轨迹文件夹，跳过")
            continue
        
        print(f"  找到 {len(trajectory_dirs)} 条轨迹")
        
        # 创建临时目录用于存储转换后的数据
        temp_dir = output_root / "temp" / satellite_name
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        frame_counter = 0
        
        # 遍历每条轨迹进行转换
        for traj_dir in trajectory_dirs:
            print(f"\n  处理轨迹: {traj_dir.name}")
            
            src_pointcloud = traj_dir / "lidar"
            src_seg = traj_dir / "seg"
            src_img = traj_dir / "image"
            
            # 检查必要文件夹
            if not all([src_pointcloud.exists(), src_seg.exists(), src_img.exists()]):
                print(f"    跳过: 缺少必要的数据文件夹")
                continue
            
            # 转换为KITTI格式（保存到临时目录）
            temp_traj_dir = temp_dir / traj_dir.name
            temp_velodyne = temp_traj_dir / "velodyne"
            temp_labels = temp_traj_dir / "labels"
            temp_images = temp_traj_dir / "image_2"
            
            try:
                convert_trajectory_to_kitti(
                    str(src_pointcloud), str(src_seg), str(src_img),
                    str(temp_velodyne), str(temp_labels), str(temp_images)
                )
                
                # 获取转换后的文件并重命名复制
                bin_files = sorted(temp_velodyne.glob("*.bin"))
                print(f"    转换了 {len(bin_files)} 帧")
                
                for bin_file in bin_files:
                    new_name = f"{frame_counter:06d}"
                    timestamp = bin_file.stem
                    
                    # 复制velodyne文件
                    shutil.copy2(bin_file, velodyne_dir / f"{new_name}.bin")
                    
                    # 复制label文件
                    label_file = temp_labels / f"{timestamp}.label"
                    if label_file.exists():
                        shutil.copy2(label_file, labels_dir / f"{new_name}.label")
                    
                    # 复制image文件
                    image_file = temp_images / f"{timestamp}.png"
                    if image_file.exists():
                        shutil.copy2(image_file, image_dir / f"{new_name}.png")
                    
                    frame_counter += 1
                
            except Exception as e:
                print(f"    错误: {e}")
                continue
        
        # 清理临时文件
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        
        print(f"\n  完成: Sequence {seq_id} 共 {frame_counter} 帧数据")
    
    return sequence_mapping


def convert_airsim_to_kitti_sequences_parallel(raw_data_root, output_root, max_workers=None, json_path=None):
    """
    并行版本：将AirSim采集的所有卫星数据转换为Semantic-KITTI格式
    每颗卫星对应一个sequence，合并该卫星的所有轨迹
    
    Args:
        raw_data_root: 原始数据根目录 (data_collect/raw_data)
        output_root: 输出根目录
        max_workers: 最大并行进程数，None表示使用CPU核心数的2/3
        json_path: satellite_descriptions.json路径（用于确定顺序）
    
    Returns:
        sequence_mapping: 字典，{sequence_id: satellite_name}
        results: 处理结果列表
    """
    raw_data_root = Path(raw_data_root)
    output_root = Path(output_root)
    
    # 创建sequences目录
    sequences_dir = output_root / "sequences"
    sequences_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取所有卫星数据文件夹
    satellite_folders = [
        d for d in raw_data_root.iterdir() 
        if d.is_dir() and not d.name.startswith('trajectory')
    ]
    
    # 按照JSON文件顺序排序
    if json_path and Path(json_path).exists():
        satellite_folders = sort_satellites_by_json_order(satellite_folders, json_path)
    else:
        print("警告: 未指定JSON文件或文件不存在，将按文件夹名称排序")
        satellite_folders = sorted(satellite_folders)
    
    total_satellites = len(satellite_folders)
    print(f"\n找到 {total_satellites} 个卫星数据文件夹")
    
    # 自动设置并行进程数：默认使用CPU核心数的2/3，避免电脑卡死
    if max_workers is None:
        max_workers = max(1, min(mp.cpu_count() * 2 // 3, total_satellites))
    
    cpu_count = mp.cpu_count()
    print(f"CPU核心数: {cpu_count}, 使用 {max_workers} 个并行进程 (核心数的2/3)")
    print("="*60)
    
    sequence_mapping = {}
    results = []
    
    # 使用ProcessPoolExecutor进行并行处理
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        futures = {}
        for seq_idx, satellite_folder in enumerate(satellite_folders):
            seq_id = f"{seq_idx:02d}"
            future = executor.submit(
                process_single_satellite,
                satellite_folder, raw_data_root, sequences_dir, seq_id
            )
            futures[future] = seq_id
        
        # 等待完成并收集结果
        completed = 0
        for future in as_completed(futures):
            seq_id, satellite_name, frame_count, status = future.result()
            sequence_mapping[seq_id] = satellite_name
            results.append((seq_id, satellite_name, frame_count, status))
            
            completed += 1
            print(f"\n[{completed}/{total_satellites}] Sequence {seq_id} ({satellite_name}): {frame_count} 帧 - {status}")
    
    print("\n" + "="*60)
    print("所有卫星处理完成")
    
    # 按seq_id排序结果
    results.sort(key=lambda x: x[0])
    
    return sequence_mapping, results


def save_sequence_mapping(sequence_mapping, output_path):
    """保存序号与卫星名称的映射到CSV文件"""
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['sequence_id', 'satellite_name'])
        
        for seq_id in sorted(sequence_mapping.keys()):
            writer.writerow([seq_id, sequence_mapping[seq_id]])
    
    print(f"\n序号映射已保存到: {output_path}")


if __name__ == "__main__":
    # Windows多进程支持
    mp.freeze_support()
    
    import argparse
    import time
    
    # 命令行参数
    parser = argparse.ArgumentParser(description="AirSim到Semantic-KITTI格式转换工具（默认并行）")
    parser.add_argument("--raw-data", type=str, required=True,
                       help="原始数据根目录 (包含各卫星子文件夹)")
    parser.add_argument("--output", type=str, default="./converted_data",
                       help="输出目录 (默认: ./converted_data)")
    parser.add_argument("--satellite-json", type=str, default=None,
                       help="satellite_descriptions.json 路径 (用于控制卫星排序)")
    parser.add_argument("--workers", type=int, default=None,
                       help="并行进程数（默认使用CPU核心数的2/3，避免电脑卡死）")
    parser.add_argument("--serial", action="store_true",
                       help="使用串行处理（不推荐，速度较慢）")
    args = parser.parse_args()
    
    raw_data_root = args.raw_data
    output_root = args.output
    json_path = args.satellite_json
    
    print("="*60)
    if args.serial:
        print("AirSim 到 Semantic-KITTI 完整转换工具 (串行版本)")
    else:
        print("AirSim 到 Semantic-KITTI 完整转换工具 (并行版本)")
    print("="*60)
    print(f"源目录: {raw_data_root}")
    print(f"目标目录: {output_root}")
    print(f"顺序参考: {json_path}")
    print("="*60)
    
    # 记录开始时间
    start_time = time.time()
    
    # 执行转换（默认使用并行）
    if args.serial:
        # 串行处理
        sequence_mapping = convert_airsim_to_kitti_sequences(raw_data_root, output_root, json_path)
        results = [(seq_id, name, 0, "成功") for seq_id, name in sequence_mapping.items()]
    else:
        # 并行处理（默认）
        sequence_mapping, results = convert_airsim_to_kitti_sequences_parallel(
            raw_data_root, output_root, max_workers=args.workers, json_path=json_path
        )
    
    # 统计信息
    if not args.serial:
        total_frames = sum(r[2] for r in results)
        successful = sum(1 for r in results if r[3] == "成功")
        failed = len(results) - successful
        
        print("\n" + "="*60)
        print("处理统计:")
        print(f"  总卫星数: {len(results)}")
        print(f"  成功: {successful}")
        print(f"  失败: {failed}")
        print(f"  总帧数: {total_frames}")
        
        if failed > 0:
            print("\n失败的卫星:")
            for seq_id, sat_name, _, status in results:
                if status != "成功":
                    print(f"  [{seq_id}] {sat_name}: {status}")
    
    # 计算耗时
    elapsed_time = time.time() - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    
    # 保存映射文件
    mapping_file = Path(output_root) / "sequence_mapping.csv"
    save_sequence_mapping(sequence_mapping, mapping_file)
    
    # 计算实际使用的进程数
    actual_workers = args.workers if args.workers else (mp.cpu_count() * 2 // 3)
    
    print("\n" + "="*60)
    print("转换完成!")
    print(f"共创建了 {len(sequence_mapping)} 个sequences")
    print(f"数据保存在: {output_root}/sequences")
    print(f"映射文件: {mapping_file}")
    print(f"总耗时: {minutes}分{seconds}秒")
    if not args.serial:
        print(f"使用了 {actual_workers} 个并行进程")
    print("="*60)

