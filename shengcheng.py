import os
import cv2
import numpy as np


def add_poisson_noise(image, scale=1.0):
    """
    给图像添加泊松噪声
    :param image: 输入图像，numpy数组
    :param scale: 噪声强度，默认为1.0
    :return: 添加噪声后的图像
    """
    # 将图像转换为浮点数类型
    img_float = image.astype(np.float32) / 255.0

    # 生成泊松噪声
    noise = np.random.poisson(img_float * scale) / scale

    # 将噪声添加到图像中
    noisy_image = img_float + noise

    # 将图像值裁剪到[0, 1]范围内
    noisy_image = np.clip(noisy_image, 0, 1)

    # 将图像转换回8位无符号整数类型
    noisy_image = (noisy_image * 255).astype(np.uint8)

    return noisy_image


def process_images_in_directory(input_dir, output_dir, scale=1.0):
    """
    批量处理目录中的所有图片，添加泊松噪声并保存到输出目录
    :param input_dir: 输入目录路径
    :param output_dir: 输出目录路径
    :param scale: 噪声强度，默认为1.0
    """
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍历输入目录中的所有文件
    for filename in os.listdir(input_dir):
        # 检查文件是否为图片文件
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            # 读取图像
            image_path = os.path.join(input_dir, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            # 添加泊松噪声
            noisy_image = add_poisson_noise(image, scale)

            # 保存添加噪声后的图像
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, noisy_image)
            print(f"Processed and saved {output_path}")


# 输入和输出目录路径
input_directory = "D:\workspace\python\input"
output_directory = "D:\workspace\python\output"


# 批量处理图片
process_images_in_directory(input_directory, output_directory, scale=20.0)