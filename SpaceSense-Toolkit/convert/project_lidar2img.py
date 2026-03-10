import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from airsim2semantickitti import *

class InteractiveVisualizer:
    """交互式点云投影可视化"""
    def __init__(self, img_path, lidar_path, step=10):
        self.img_path = img_path
        self.lidar_path = lidar_path
        self.step = step
        self.current_idx = 0
        
        # 获取所有 .asc 文件并排序
        asc_files = sorted([f for f in os.listdir(lidar_path) if f.endswith('.asc')])
        png_files = [f for f in os.listdir(img_path) if f.endswith('.png')]
        
        # 每隔step个取一个文件
        self.file_pairs = []
        for i in range(0, len(asc_files), step):
            asc_file = asc_files[i]
            corresponding_png = asc_file.replace('.asc', '.png')
            if corresponding_png in png_files:
                self.file_pairs.append((asc_file, corresponding_png))
        
        print(f"总共找到 {len(asc_files)} 个文件，每隔{step}个处理一帧，共处理 {len(self.file_pairs)} 帧")
        
        # 创建图形
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(15, 7.5))
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
    def visualize_current(self):
        """可视化当前帧"""
        if self.current_idx >= len(self.file_pairs):
            print("已经是最后一帧")
            return
            
        asc_file, png_file = self.file_pairs[self.current_idx]
        pointcloud_path = os.path.join(self.lidar_path, asc_file)
        image_path = os.path.join(self.img_path, png_file)
        
        print(f"显示第 {self.current_idx + 1}/{len(self.file_pairs)} 帧: {png_file}")
        
        # 读取RGB图像
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        
        # 读取点云数据
        points = read_asc_pointcloud(pointcloud_path)
        
        # 清空之前的内容
        self.ax1.clear()
        self.ax2.clear()
        
        # 显示原始图像
        self.ax1.imshow(img)
        self.ax1.set_title('Original Image')
        
        # 显示带投影点的图像
        projected_img = img.copy()
        
        # 投影所有有效点
        valid_points = []
        for point in points:
            # 坐标系转换
            transformed_point = transform_lidar_to_camera_frame(point)
            # 投影到图像
            uv = project_point_to_image(transformed_point, w, h)
            
            if uv:
                valid_points.append(uv)
                # 在图像上画点（红色，更小的点）
                cv2.circle(projected_img, uv, radius=1, color=(255, 0, 0), thickness=2)
        
        # 显示投影结果
        self.ax2.imshow(projected_img)
        self.ax2.set_title(f'Projected Points ({len(valid_points)} valid) - Press any key for next')
        
        self.fig.canvas.draw()
    
    def on_key_press(self, event):
        """按键事件处理"""
        self.current_idx += 1
        if self.current_idx < len(self.file_pairs):
            self.visualize_current()
        else:
            print("所有帧已显示完毕")
            plt.close(self.fig)
    
    def start(self):
        """开始可视化"""
        if len(self.file_pairs) == 0:
            print("没有找到匹配的文件对")
            return
        self.visualize_current()
        plt.show()

def process_files_in_folder(img_path, lidar_path, step=10):
    """处理文件夹内文件，每隔step个处理一帧"""
    visualizer = InteractiveVisualizer(img_path, lidar_path, step)
    visualizer.start()

# 使用示例 -----------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="交互式点云投影到图像可视化")
    parser.add_argument("--img-path", type=str, required=True,
                       help="图像文件夹路径 (如 raw_data/xxx/approach_back/image)")
    parser.add_argument("--lidar-path", type=str, required=True,
                       help="点云文件夹路径 (如 raw_data/xxx/approach_back/lidar)")
    parser.add_argument("--step", type=int, default=10, help="每隔step帧处理一帧")
    _args = parser.parse_args()

    process_files_in_folder(_args.img_path, _args.lidar_path, _args.step)
