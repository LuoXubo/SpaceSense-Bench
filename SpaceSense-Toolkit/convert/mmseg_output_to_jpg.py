# 将单通道的黑乎乎的png输出，转换为三通道的jpg，和大赛提供的mask真知一致，以便提交
import numpy as np
from PIL import Image
import os
from tqdm import tqdm

# 定义颜色映射关系
color_mapping = {
    0: (0, 0, 0),    # 0 映射到 (0, 0, 0)
    1: (255, 0, 0),  # 1 映射到 (255, 0, 0)
    2: (0, 0, 255)   # 2 映射到 (0, 0, 255)
}

# 指定输入和输出文件夹路径
input_folder = 'work_dirs/output'
output_folder = 'work_dirs/output_jpg'

# 遍历文件夹中的PNG图像文件
for filename in tqdm(os.listdir(input_folder)):
    if filename.endswith('.png'):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + '.jpg')

        # 打开PNG图像并转换为NumPy数组
        image = np.array(Image.open(input_path).convert('L'))

        # 创建新的RGB图像数组
        rgb_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

        # 遍历每个像素，并根据颜色映射关系设置RGB值
        for value, rgb_value in color_mapping.items():
            mask = (image == value)
            rgb_image[mask] = rgb_value

        # 创建RGB图像对象
        rgb_image = Image.fromarray(rgb_image)

        # 保存为JPG图像
        rgb_image.save(output_path, 'JPEG')
        