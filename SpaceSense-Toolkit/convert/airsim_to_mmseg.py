#!/usr/bin/env python3
"""
将AirSim采集的卫星数据转换为MMSegmentation格式
- 处理图像数据（RGB + 语义分割标注）
- 支持多进程并行转换
- 数据集划分遵循Semantic-KITTI标准
"""
import os
import shutil
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse

RAW_DATA_ROOT = None
OUTPUT_ROOT = None

### 最终的数据格式
"""
spacesense_136_sparse_mmseg
├── img_dir
│   ├── train
│   │   ├── xxx.png
│   ├── val
│   │   ├── xxx.png
│   └── test
│       ├── xxx.png
│
└── ann_dir
    ├── train
    │   ├── xxx.png
    ├── val
    │   ├── xxx.png
    └── test
        ├── xxx.png
"""
### 数据集划分：
### train: 117个卫星（序列01-09, 11-129，排除10的倍数）
### val: 5个卫星（序列131-135）
### test: 14个卫星（序列00,10,20,...,130）

# 语义标签颜色到类别ID的映射（RGB格式）
COLOR_TO_CLASS = {
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

CLASS_NAMES = ['background', 'main_body', 'solar_panel', 'dish_antenna', 
               'omni_antenna', 'payload', 'thruster', 'adapter_ring']

# MMSeg调色板：8个类别（背景+7个部件类），每个类别用RGB表示
MMSEG_PALETTE = np.array([
    [0, 0, 0],       # 0: background (黑色)
    [128, 0, 0],     # 1: main_body (深红色)
    [0, 128, 0],     # 2: solar_panel (深绿色)
    [128, 128, 0],   # 3: dish_antenna (深黄色)
    [0, 0, 128],     # 4: omni_antenna (深蓝色)
    [128, 0, 128],   # 5: payload (深紫色)
    [0, 128, 128],   # 6: thruster (深青色)
    [128, 128, 128], # 7: adapter_ring (灰色)
], dtype=np.uint8).flatten()  # 展平为1维数组 [R1,G1,B1, R2,G2,B2, ...]


def extract_satellite_name(folder_name):
    """从文件夹名称中提取卫星名称"""
    if '_' in folder_name:
        return folder_name.split('_', 1)[1]
    return folder_name


def convert_seg_to_mmseg(seg_image):
    """
    将RGB分割图转换为单通道语义标签图
    
    Args:
        seg_image: 分割图像（BGR格式，OpenCV读取）
    
    Returns:
        单通道标签图（numpy数组，dtype=uint8）
    """
    # 转换为RGB
    seg_rgb = cv2.cvtColor(seg_image, cv2.COLOR_BGR2RGB)
    height, width = seg_rgb.shape[:2]
    
    # 创建单通道标签图（初始化为0，即背景）
    label_map = np.zeros((height, width), dtype=np.uint8)
    
    # 遍历所有颜色映射，将RGB颜色转换为类别ID
    for color, class_id in COLOR_TO_CLASS.items():
        mask = np.all(seg_rgb == color, axis=-1)
        label_map[mask] = class_id
    
    return label_map


def save_mmseg_annotation(label_map, output_path):
    """
    保存MMSeg格式的标注文件（调色板模式的PNG）
    
    Args:
        label_map: 单通道标签图（numpy数组）
        output_path: 输出文件路径
    """
    # 转换为PIL Image（调色板模式）
    img_p = Image.fromarray(label_map, mode='P')
    img_p.putpalette(MMSEG_PALETTE)
    img_p.save(output_path)


def convert_satellite_data(satellite_folder, output_images, output_labels, dataset_name):
    """
    转换单颗卫星的数据
    
    Args:
        satellite_folder: 卫星数据文件夹
        output_images: 输出图像目录
        output_labels: 输出标签目录
        dataset_name: 数据集名称（train/val/test）
    
    Returns:
        (satellite_name, frame_count, error_message)
    """
    satellite_folder = Path(satellite_folder)
    output_images = Path(output_images)
    output_labels = Path(output_labels)
    
    satellite_name = extract_satellite_name(satellite_folder.name)
    
    try:
        # 获取所有轨迹文件夹
        trajectory_dirs = sorted([
            d for d in satellite_folder.iterdir()
            if d.is_dir() and not d.name.startswith('trajectory')
        ])
        
        if not trajectory_dirs:
            return (satellite_name, 0, None)
        
        frame_count = 0
        
        # 遍历每条轨迹
        for traj_dir in trajectory_dirs:
            image_dir = traj_dir / 'image'
            seg_dir = traj_dir / 'seg'
            
            if not image_dir.exists() or not seg_dir.exists():
                continue
            
            # 获取所有图像文件
            image_files = sorted(image_dir.glob('*.png'))
            
            for img_file in image_files:
                frame_id = img_file.stem
                seg_file = seg_dir / f'{frame_id}.png'
                
                if not seg_file.exists():
                    continue
                
                # 读取分割图像
                seg_image = cv2.imread(str(seg_file))
                if seg_image is None:
                    continue
                
                # 转换为MMSeg格式的标签图
                label_map = convert_seg_to_mmseg(seg_image)
                
                # 生成唯一的文件名：卫星名_轨迹名_帧ID
                unique_name = f"{satellite_name}_{traj_dir.name}_{frame_id}"
                
                # 复制原始图像
                output_img_path = output_images / f"{unique_name}.png"
                shutil.copy2(img_file, output_img_path)
                
                # 保存MMSeg标注
                output_label_path = output_labels / f"{unique_name}.png"
                save_mmseg_annotation(label_map, output_label_path)
                
                frame_count += 1
        
        return (satellite_name, frame_count, None)
    
    except Exception as e:
        return (satellite_name, 0, str(e))


# =============================================================================
# 数据集划分配置（遵循Semantic-KITTI标准）
# =============================================================================

# 测试集：序列 00, 10, 20, 30, ..., 130（共14个卫星）
TEST_SATELLITES = {
    'ACE', 'CALIPSO', 'Dawn', 'ExoMars_TGO', 'GRAIL', 'Integral', 'LADEE',
    'Lunar_Reconnaissance_Orbiter', 'Mercury_Magnetospheric_Orbiter',
    'OSIRIS_REX', 'Proba_2', 'SOHO', 'Suomi_NPP', 'Ulysses'
}

# 验证集：序列 131-135（共5个卫星）
VAL_SATELLITES = {
    'Van_Allen_Probe', 'Venus_Express', 'Voyager', 'WIND', 'XMM_newton'
}

# 训练集：其余117个卫星（自动通过排除法确定）


def split_satellites(satellite_folders):
    """
    将卫星数据划分为训练集、验证集和测试集
    - train: 117个卫星（其余）
    - val: 5个卫星（序列131-135）
    - test: 14个卫星（序列00,10,20,...,130）
    
    Args:
        satellite_folders: 所有卫星文件夹列表
    
    Returns:
        (train_folders, val_folders, test_folders)
    """
    train_folders = []
    val_folders = []
    test_folders = []
    
    for folder in satellite_folders:
        sat_name = extract_satellite_name(folder.name)
        
        if sat_name in TEST_SATELLITES:
            test_folders.append(folder)
        elif sat_name in VAL_SATELLITES:
            val_folders.append(folder)
        else:
            train_folders.append(folder)
    
    return train_folders, val_folders, test_folders


def process_single_satellite(args):
    """
    并行处理单个卫星的包装函数
    
    Args:
        args: (satellite_folder, output_images, output_labels, dataset_name)
    
    Returns:
        (satellite_name, frame_count, error_message)
    """
    satellite_folder, output_images, output_labels, dataset_name = args
    return convert_satellite_data(satellite_folder, output_images, output_labels, dataset_name)


def convert_parallel(satellite_folders, output_images, output_labels, dataset_name, max_workers=None):
    """
    并行转换多个卫星的数据
    
    Args:
        satellite_folders: 卫星文件夹列表
        output_images: 输出图像目录
        output_labels: 输出标签目录
        dataset_name: 数据集名称（train/val/test）
        max_workers: 最大并行进程数
    
    Returns:
        (total_frames, results)
    """
    if max_workers is None:
        max_workers = max(1, mp.cpu_count() * 2 // 3)
    
    tasks = [(folder, output_images, output_labels, dataset_name) for folder in satellite_folders]
    
    results = []
    total_frames = 0
    
    if max_workers == 1:
        # 串行处理
        for task in tqdm(tasks, desc=f"转换{dataset_name}"):
            result = process_single_satellite(task)
            results.append(result)
            total_frames += result[1]
    else:
        # 并行处理
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_sat = {
                executor.submit(process_single_satellite, task): extract_satellite_name(task[0].name)
                for task in tasks
            }
            
            with tqdm(total=len(tasks), desc=f"转换{dataset_name}") as pbar:
                for future in as_completed(future_to_sat):
                    sat_name = future_to_sat[future]
                    try:
                        result = future.result()
                        results.append(result)
                        total_frames += result[1]
                        
                        if result[2]:
                            tqdm.write(f"  ❌ {result[0]}: {result[2]}")
                        
                    except Exception as e:
                        tqdm.write(f"  ❌ {sat_name}: {e}")
                        results.append((sat_name, 0, str(e)))
                    
                    pbar.update(1)
    
    return total_frames, results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='AirSim到MMSegmentation格式转换工具（支持多进程并行）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python airsim_to_mmseg.py --raw-data /path/to/raw_data --output ./mmseg_output
  python airsim_to_mmseg.py --raw-data /path/to/raw_data --serial
  python airsim_to_mmseg.py --raw-data /path/to/raw_data --workers 8
        """
    )
    parser.add_argument('--raw-data', type=str, required=True,
                       help='原始数据根目录 (包含各卫星子文件夹)')
    parser.add_argument('--output', type=str, default='./spacesense_mmseg',
                       help='输出目录 (默认: ./spacesense_mmseg)')
    parser.add_argument('--serial', action='store_true',
                       help='使用串行处理（默认为并行）')
    parser.add_argument('--workers', type=int, default=None,
                       help='并行进程数（默认为CPU核心数的2/3）')
    
    args = parser.parse_args()
    
    global RAW_DATA_ROOT, OUTPUT_ROOT
    RAW_DATA_ROOT = Path(args.raw_data)
    OUTPUT_ROOT = Path(args.output)
    
    print("="*70)
    print("AirSim 到 MMSegmentation 格式转换工具（支持并行）")
    print("="*70)
    
    if not RAW_DATA_ROOT.exists():
        print(f"❌ 数据目录不存在: {RAW_DATA_ROOT}")
        return
    
    # 创建输出目录结构
    train_images = OUTPUT_ROOT / "img_dir" / "train"
    train_labels = OUTPUT_ROOT / "ann_dir" / "train"
    val_images = OUTPUT_ROOT / "img_dir" / "val"
    val_labels = OUTPUT_ROOT / "ann_dir" / "val"
    test_images = OUTPUT_ROOT / "img_dir" / "test"
    test_labels = OUTPUT_ROOT / "ann_dir" / "test"
    
    for dir_path in [train_images, train_labels, val_images, val_labels, test_images, test_labels]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print(f"✓ 输出目录已创建: {OUTPUT_ROOT}")
    
    # 获取所有卫星文件夹
    satellite_folders = [
        d for d in RAW_DATA_ROOT.iterdir()
        if d.is_dir() and not d.name.startswith('trajectory')
    ]
    
    print(f"✓ 找到 {len(satellite_folders)} 个卫星数据文件夹")
    
    # 划分训练集、验证集和测试集
    train_folders, val_folders, test_folders = split_satellites(satellite_folders)
    
    print(f"\n数据划分（遵循Semantic-KITTI标准）:")
    print(f"  - 训练集: {len(train_folders)} 个卫星")
    print(f"  - 验证集: {len(val_folders)} 个卫星（序列131-135）")
    print(f"  - 测试集: {len(test_folders)} 个卫星（序列00,10,20,...,130）")
    
    # 确定并行进程数
    if args.serial:
        max_workers = 1
        print("\n✓ 使用串行处理")
    elif args.workers is not None:
        max_workers = args.workers
        print(f"\n✓ 使用 {max_workers} 个并行进程（用户指定）")
    else:
        total_satellites = len(train_folders) + len(val_folders) + len(test_folders)
        max_workers = max(1, min(mp.cpu_count() * 2 // 3, total_satellites))
        print(f"\n✓ CPU核心数: {mp.cpu_count()}, 使用 {max_workers} 个并行进程（核心数的2/3）")
    
    # 转换训练集
    print("\n" + "="*70)
    print("转换训练集...")
    print("="*70)
    train_frame_count, train_results = convert_parallel(
        train_folders, train_images, train_labels, "训练集", max_workers
    )
    
    # 转换验证集
    print("\n" + "="*70)
    print("转换验证集...")
    print("="*70)
    val_frame_count, val_results = convert_parallel(
        val_folders, val_images, val_labels, "验证集", max_workers
    )
    
    # 转换测试集
    print("\n" + "="*70)
    print("转换测试集...")
    print("="*70)
    test_frame_count, test_results = convert_parallel(
        test_folders, test_images, test_labels, "测试集", max_workers
    )
    
    # 统计信息
    print("\n" + "="*70)
    print("转换完成!")
    print("="*70)
    print(f"训练集: {train_frame_count} 帧 ({len(train_folders)} 个卫星)")
    print(f"验证集: {val_frame_count} 帧 ({len(val_folders)} 个卫星)")
    print(f"测试集: {test_frame_count} 帧 ({len(test_folders)} 个卫星)")
    print(f"总计: {train_frame_count + val_frame_count + test_frame_count} 帧")
    print(f"\n数据保存在: {OUTPUT_ROOT}")
    print(f"  - 训练图像: {train_images}")
    print(f"  - 训练标签: {train_labels}")
    print(f"  - 验证图像: {val_images}")
    print(f"  - 验证标签: {val_labels}")
    print(f"  - 测试图像: {test_images}")
    print(f"  - 测试标签: {test_labels}")
    
    # 显示失败的卫星
    all_results = train_results + val_results + test_results
    failed = [r for r in all_results if r[2] is not None]
    if failed:
        print(f"\n⚠️  转换失败的卫星 ({len(failed)}):")
        for sat_name, frame_count, error in failed:
            print(f"  - {sat_name}: {error}")
    
    print("="*70)
    
    # 显示类别统计
    print("\n类别信息:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {i}: {name}")
    print()


if __name__ == "__main__":
    # Windows多进程需要freeze_support
    mp.freeze_support()
    
    main()