import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import cv2
from tools.add_noise import add_random_noise


class DenoisingDataset(Dataset):
    def __init__(self, root_dir, transform=None,
                 gaussian_var=10,  # 高斯噪声方差
                 salt_pepper_ratio=0.1,  # 椒盐噪声总比例
                 uniform_range=0.1,  # 均匀噪声范围 [-range, range]
                 periodic_amp=50,  # 周期性噪声振幅
                 resize=(128, 128)):
        self.root_dir = root_dir
        self.gaussian_var = gaussian_var
        self.salt_pepper_ratio = salt_pepper_ratio
        self.uniform_range = uniform_range
        self.periodic_amp = periodic_amp
        self.resize = resize

        # 默认预处理流水线
        self.transform = transform or Compose([
            ToTensor(),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        # 收集图像文件
        self.image_files = [
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ]

    def __len__(self):
        return len(self.image_files)  # 每种图片生成1种噪声版本

    def __getitem__(self, idx):
        # 加载图像
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        image = ResizeWithPadding(image, self.resize)

        # 转换为numpy数组并调整到[0,255]范围
        image_np = np.array(image)
        clean_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)  # PIL是RGB，OpenCV需要BGR

        # 生成噪声类型映射

        # 添加噪声
        noisy_image = add_random_noise(
            clean_image,
            var=self.gaussian_var,
            salt_prob=self.salt_pepper_ratio / 2,
            pepper_prob=self.salt_pepper_ratio / 2,
            amplitude=self.periodic_amp,
            low=-self.uniform_range * 255,
            high=self.uniform_range * 255
        )

        # 转换回PIL并应用预处理
        noisy_pil = Image.fromarray(cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB))
        clean_pil = Image.fromarray(cv2.cvtColor(clean_image, cv2.COLOR_BGR2RGB))

        # 转换为张量
        noisy_tensor = self.transform(noisy_pil)
        clean_tensor = self.transform(clean_pil)

        return noisy_tensor, clean_tensor


def ResizeWithPadding(img, target_size):
    img_np = np.array(img)
    h, w = img_np.shape[:2]
    target_h, target_w = target_size

    # 计算缩放比例
    scale = min(target_h / h, target_w / w)
    new_h = int(h * scale)
    new_w = int(w * scale)

    # 缩放图像（确保使用正确的插值方法）
    resized_img = cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 计算填充
    delta_w = target_w - new_w
    delta_h = target_h - new_h
    top = delta_h // 2
    bottom = delta_h - top
    left = delta_w // 2
    right = delta_w - left

    # 添加黑色填充（确保填充值为0）
    padded_img = cv2.copyMakeBorder(
        resized_img, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )

    # 记录原始尺寸和填充参数
    padded_pil = Image.fromarray(padded_img)
    padded_pil.original_size = img.size  # 原始尺寸 (width, height)
    padded_pil.padding = (left, top, right, bottom)
    return padded_pil
