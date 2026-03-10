# SpaceSense-Toolkit

**SpaceSense-136 数据集的格式转换与可视化工具**

SpaceSense-136 是一个基于高保真仿真的多模态航天器视觉数据集，包含 136 颗卫星的 RGB 图像、深度图、语义分割标注和激光雷达点云。本仓库提供将原始采集数据转换为主流深度学习框架格式的工具。

## 数据集概况

| 项目 | 内容 |
|------|------|
| 卫星数量 | 136 颗 |
| 数据模态 | RGB、深度图、语义分割、LiDAR 点云、位姿标注 |
| 语义类别 | 7 类（main_body, solar_panel, dish_antenna, omni_antenna, payload, thruster, adapter_ring） |
| 图像分辨率 | 1024 x 1024 |

### 数据集划分 (Zero-shot / OOD)

训练集与验证集的卫星完全不重叠，验证集反映模型对未见卫星的泛化能力。

| 集合 | 卫星数量 | 说明 |
|------|------:|------|
| 训练集 | 117 | 排除验证集和排除集后的所有卫星 |
| 验证集 | 14 | 按序号每隔 10 个取一个 (00, 10, 20, ..., 130) |
| 排除集 | 5 | 序列 131-135，保留用于未来测试 |

## 安装

```bash
pip install -r requirements.txt
```

## 目录结构

```
SpaceSense-Toolkit/
├── convert/                 # 数据格式转换
│   ├── airsim_to_semantickitti.py
│   ├── airsim_to_yolo.py
│   ├── airsim_to_mmseg.py
│   ├── project_lidar2img.py
│   └── mmseg_output_to_jpg.py
├── visualize/               # Web 可视化工具
│   ├── semantickitti_web_visualizer.py
│   ├── yolo_web_visualizer.py
│   └── templates/
├── configs/
│   └── satellite_descriptions.json
├── scripts/
│   ├── check_converted_data.py
│   └── upload_to_huggingface.py
├── requirements.txt
└── README.md
```

## 数据格式转换

### 转换为 Semantic-KITTI 格式 (3D 点云语义分割)

```bash
python convert/airsim_to_semantickitti.py \
    --raw-data /path/to/raw_data \
    --output ./converted_data \
    --satellite-json configs/satellite_descriptions.json \
    --workers 8
```

输出结构:

```
converted_data/
├── sequences/
│   ├── 00/        # 每颗卫星一个 sequence
│   │   ├── velodyne/       # .bin 点云
│   │   ├── labels/         # .label 语义标签
│   │   ├── image_2/        # .png RGB 图像
│   │   ├── calib.txt
│   │   ├── poses.txt
│   │   └── times.txt
│   ├── 01/
│   └── ...
└── sequence_mapping.csv
```

### 转换为 YOLO 格式 (2D 目标检测)

```bash
python convert/airsim_to_yolo.py \
    --raw-data /path/to/raw_data \
    --output ./spacesense_yolo \
    --workers 8
```

### 转换为 MMSegmentation 格式 (2D 语义分割)

```bash
python convert/airsim_to_mmseg.py \
    --raw-data /path/to/raw_data \
    --output ./spacesense_mmseg \
    --workers 8
```

### 点云投影到图像

```bash
python convert/project_lidar2img.py \
    --img-path /path/to/raw_data/satellite/trajectory/image \
    --lidar-path /path/to/raw_data/satellite/trajectory/lidar
```

## 可视化

### Semantic-KITTI 点云可视化

```bash
python visualize/semantickitti_web_visualizer.py \
    --data-root ./converted_data \
    --satellite-json configs/satellite_descriptions.json
# 浏览器打开 http://localhost:5000
```

### YOLO 标注可视化

```bash
python visualize/yolo_web_visualizer.py \
    --data-root ./spacesense_yolo
# 浏览器打开 http://localhost:5001
```

## 上传数据到 HuggingFace

### 方式一：先本地打包，再网页手动上传（推荐网络不稳定时使用）

```bash
# 按卫星打包为 tar.gz
python scripts/upload_to_huggingface.py \
    --raw-data /path/to/raw_data \
    --pack-only \
    --pack-dir ./packed

# 然后在 HuggingFace 网页拖拽上传 packed/ 下的 .tar.gz 文件
```

### 方式二：命令行自动上传

```bash
pip install huggingface_hub
huggingface-cli login

python scripts/upload_to_huggingface.py \
    --raw-data /path/to/raw_data \
    --repo-id your-username/SpaceSense-136
```

## 原始数据目录结构

采集的原始数据按如下结构组织：

```
raw_data/
└── <timestamp>_<satellite_name>/
    ├── approach_back/
    │   ├── rgb/          # RGB 图像 (.png)
    │   ├── depth/        # 深度图 (.png, int32, 毫米)
    │   ├── segmentation/ # 语义分割图 (.png, uint8)
    │   ├── lidar/        # 激光雷达点云 (.asc)
    │   └── poses.csv     # 位姿标注 (x,y,z,qw,qx,qy,qz)
    ├── approach_front/
    ├── orbit_xy/
    └── ...
```

## 数据质量检查

```bash
python scripts/check_converted_data.py \
    --data-root ./converted_data \
    --satellite-json configs/satellite_descriptions.json
```
