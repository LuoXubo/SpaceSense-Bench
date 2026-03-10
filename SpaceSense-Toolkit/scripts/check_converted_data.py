#!/usr/bin/env python3
"""
检查转换后的数据质量

使用方法:
  python check_converted_data.py --data-root /path/to/converted_data
  python check_converted_data.py --data-root /path/to/converted_data --satellite-json /path/to/satellite_descriptions.json
"""
import os
import csv
import json
import argparse
from pathlib import Path
from glob import glob

parser = argparse.ArgumentParser(description="检查转换后的Semantic-KITTI数据质量")
parser.add_argument("--data-root", type=str, required=True,
                   help="转换后的数据根目录 (包含 sequences/ 和 sequence_mapping.csv)")
parser.add_argument("--satellite-json", type=str, default=None,
                   help="satellite_descriptions.json 路径 (可选)")
_args = parser.parse_args()

converted_data_root = Path(_args.data_root)
sequences_dir = converted_data_root / "sequences"
mapping_file = converted_data_root / "sequence_mapping.csv"
json_path = Path(_args.satellite_json) if _args.satellite_json else None

print("="*70)
print("转换数据质量检查")
print("="*70)

# 1. 读取sequence_mapping.csv
sequence_mapping = {}
if mapping_file.exists():
    with open(mapping_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sequence_mapping[row['sequence_id']] = row['satellite_name']
    print(f"\n✓ 找到 sequence_mapping.csv，共 {len(sequence_mapping)} 个序列")
else:
    print(f"\n❌ 未找到 sequence_mapping.csv")
    exit(1)

# 2. 读取JSON中的卫星顺序
json_satellites = []
if json_path and json_path.exists():
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    json_satellites = [sat['name'] for sat in data['satellites']]
    print(f"✓ 找到 satellite_descriptions.json，共 {len(json_satellites)} 个卫星")
else:
    print(f"⚠️  未找到 satellite_descriptions.json")

# 3. 检查每个sequence的数据完整性
print("\n" + "="*70)
print("检查每个序列的数据完整性:")
print("="*70)

problems = []
empty_sequences = []
frame_stats = []

for seq_id in sorted(sequence_mapping.keys(), key=lambda x: int(x)):
    sat_name = sequence_mapping[seq_id]
    seq_dir = sequences_dir / seq_id
    
    # 检查目录是否存在
    if not seq_dir.exists():
        problems.append(f"[{seq_id}] {sat_name}: 序列目录不存在")
        continue
    
    # 检查必要的子目录
    velodyne_dir = seq_dir / "velodyne"
    labels_dir = seq_dir / "labels"
    image_dir = seq_dir / "image_2"
    calib_file = seq_dir / "calib.txt"
    
    missing_parts = []
    if not velodyne_dir.exists():
        missing_parts.append("velodyne")
    if not labels_dir.exists():
        missing_parts.append("labels")
    if not image_dir.exists():
        missing_parts.append("image_2")
    if not calib_file.exists():
        missing_parts.append("calib.txt")
    
    if missing_parts:
        problems.append(f"[{seq_id}] {sat_name}: 缺少 {', '.join(missing_parts)}")
        continue
    
    # 统计帧数
    bin_files = sorted(glob(str(velodyne_dir / "*.bin")))
    label_files = sorted(glob(str(labels_dir / "*.label")))
    image_files = sorted(glob(str(image_dir / "*.png")))
    
    bin_count = len(bin_files)
    label_count = len(label_files)
    image_count = len(image_files)
    
    # 检查数量是否一致
    if bin_count == 0:
        empty_sequences.append(f"[{seq_id}] {sat_name}")
    elif not (bin_count == label_count == image_count):
        problems.append(f"[{seq_id}] {sat_name}: 帧数不一致 (bin:{bin_count}, label:{label_count}, img:{image_count})")
    
    frame_stats.append((seq_id, sat_name, bin_count, label_count, image_count))
    
    # 简短输出
    status = "✓" if bin_count > 0 and bin_count == label_count == image_count else "❌"
    print(f"{status} [{seq_id}] {sat_name:30s}: {bin_count:3d} 帧")

# 4. 检查顺序是否与JSON一致
print("\n" + "="*70)
print("检查序列顺序是否与JSON一致:")
print("="*70)

if json_satellites:
    order_problems = []
    for seq_id in sorted(sequence_mapping.keys(), key=lambda x: int(x)):
        sat_name = sequence_mapping[seq_id]
        seq_idx = int(seq_id)
        
        if seq_idx < len(json_satellites):
            expected_name = json_satellites[seq_idx]
            if sat_name != expected_name:
                order_problems.append(f"[{seq_id}] 期望: {expected_name}, 实际: {sat_name}")
        else:
            order_problems.append(f"[{seq_id}] {sat_name} 超出JSON范围 (JSON只有{len(json_satellites)}个)")
    
    if order_problems:
        print("❌ 顺序不一致:")
        for prob in order_problems[:10]:  # 只显示前10个
            print(f"  {prob}")
        if len(order_problems) > 10:
            print(f"  ... 还有 {len(order_problems) - 10} 个问题")
    else:
        print("✓ 所有序列顺序与JSON一致")

# 5. 检查是否有卫星未转换
print("\n" + "="*70)
print("检查缺失的卫星:")
print("="*70)

converted_names = set(sequence_mapping.values())
missing_satellites = [name for name in json_satellites if name not in converted_names]

if missing_satellites:
    print(f"❌ 有 {len(missing_satellites)} 个卫星未转换:")
    for i, name in enumerate(missing_satellites, 1):
        json_idx = json_satellites.index(name)
        print(f"  {i}. {name} (JSON第 {json_idx} 个)")
else:
    print("✓ JSON中的所有卫星都已转换")

# 6. 统计摘要
print("\n" + "="*70)
print("统计摘要:")
print("="*70)

total_sequences = len(sequence_mapping)
total_frames = sum(stat[2] for stat in frame_stats)
non_empty = len([s for s in frame_stats if s[2] > 0])
avg_frames = total_frames / non_empty if non_empty > 0 else 0

print(f"  总序列数: {total_sequences}")
print(f"  非空序列: {non_empty}")
print(f"  空序列: {len(empty_sequences)}")
print(f"  总帧数: {total_frames}")
print(f"  平均帧数: {avg_frames:.1f}")
print(f"  问题数: {len(problems)}")

# 7. 显示问题列表
if problems:
    print("\n" + "="*70)
    print("发现的问题:")
    print("="*70)
    for prob in problems:
        print(f"  ❌ {prob}")

if empty_sequences:
    print("\n" + "="*70)
    print("空序列 (0帧):")
    print("="*70)
    for seq in empty_sequences:
        print(f"  ⚠️  {seq}")

# 8. 帧数分布
print("\n" + "="*70)
print("帧数分布:")
print("="*70)

frame_counts = [stat[2] for stat in frame_stats]
if frame_counts:
    print(f"  最小帧数: {min(frame_counts)}")
    print(f"  最大帧数: {max(frame_counts)}")
    print(f"  中位数: {sorted(frame_counts)[len(frame_counts)//2]}")

# 按帧数排序显示前5和后5
print("\n  帧数最多的5个序列:")
sorted_stats = sorted(frame_stats, key=lambda x: x[2], reverse=True)
for seq_id, sat_name, bin_count, _, _ in sorted_stats[:5]:
    print(f"    [{seq_id}] {sat_name:30s}: {bin_count} 帧")

print("\n  帧数最少的5个序列 (不含0帧):")
non_zero_stats = [s for s in sorted_stats if s[2] > 0]
for seq_id, sat_name, bin_count, _, _ in non_zero_stats[-5:]:
    print(f"    [{seq_id}] {sat_name:30s}: {bin_count} 帧")

print("\n" + "="*70)
if not problems and not empty_sequences and not missing_satellites:
    print("✓ 转换数据质量检查通过！")
else:
    print("⚠️  发现一些问题，请检查上面的详细信息")
print("="*70)

