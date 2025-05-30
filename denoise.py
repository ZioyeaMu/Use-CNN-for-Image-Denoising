import os
import time

import torch
from model.denoising_model import DenoisingModel
from torchvision.transforms import Compose, ToTensor, Resize, Normalize, ToPILImage
from PIL import Image
import argparse
from tools.create_sequential_folder import create_sequential_folder


# 定义去噪函数
def denoise_image(noisy_image_path, model_weights_path, output_path):
    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DenoisingModel().to(device)
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model.eval()  # 将模型设置为评估模式

    # 数据预处理：定义图像转换操作，将图像转换为模型可处理的格式
    transform = Compose([
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 加载并预处理图像
    image = Image.open(noisy_image_path).convert('RGB')  # 打开图像文件并转换为RGB格式
    noisy_image = transform(image).unsqueeze(0).to(device)  # 对图像应用转换操作，添加批次维度，并移动到GPU或CPU上

    # 前向传播去噪
    with torch.no_grad():  # 禁用梯度计算，以提高推理速度并减少内存占用
        denoised_image = model(noisy_image)  # 使用模型对噪声图像进行去噪处理

    # 后处理
    denoised_image = denoised_image.squeeze(0).cpu()  # 移除批次维度（squeeze(0)）并将去噪后的图像从GPU（如果使用）移动到CPU。
    denoised_image = denoised_image * 0.5 + 0.5  # 反归一化处理，将图像像素值从归一化范围（-1到1）转换回原始范围（0到1）
    denoised_image = ToPILImage()(denoised_image.clamp(0, 1))  # 将张量转换为PIL图像

    # 保存去噪后的图像
    denoised_image.save(output_path)


# 批量去噪函数
def denoise_folder(noisy_folder_path, model_weights_path, output_folder_path):
    # 创建输出文件夹（如果不存在）
    os.makedirs(output_folder_path, exist_ok=True)

    # 获取输入文件夹中的所有文件
    image_files = [f for f in os.listdir(noisy_folder_path) if
                   f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    total_images = len(image_files)

    # 对每个图像文件进行去噪
    for idx, image_file in enumerate(image_files):
        noisy_image_path = os.path.join(noisy_folder_path, image_file)
        output_image_path = os.path.join(output_folder_path, f"denoise_{image_file}")
        denoise_image(noisy_image_path, model_weights_path, output_image_path)

        # 计算进度和ETA
        progress = (idx + 1) / total_images * 100
        elapsed_time = time.time() - start_time
        eta = elapsed_time / (idx + 1) * (total_images - idx - 1)
        eta_str = time.strftime("%H:%M:%S", time.gmtime(eta))

        # 构建进度条
        progress_bar = "[" + "=" * int(progress // 2) + ">" + "-" * (50 - int(progress // 2)) + "]"
        print(f"\rDenoising [{idx + 1}/{total_images}]\t{progress_bar} {progress:.1f}%\tETA: {eta_str}", end="")
    print("\r", end="")


parser = argparse.ArgumentParser(description='Denoise an image using a pre-trained model.')
parser.add_argument('--noisy_image_path', type=str, default="runs/addnoise005",  # 替换为你的有噪声图像路径
                    help='要去噪的图像的路径')
parser.add_argument('--model_weights_path', type=str, default="weights/model_weight2.pth",  # 替换为你的模型权重路径
                    help='要使用的去噪模型')
parser.add_argument('--output_path', type=str, default=None, help='去噪图片保存路径')
parser.add_argument('--silence', action='store_true', help='是否开启静默模式，不再询问用户意见')
args = parser.parse_args()


def get_user_confirmation():
    while True:
        print("=" * 50)
        for arg in vars(args):
            print(f"{arg}: {getattr(args, arg)}")
        print("=" * 50)
        confirmation = input("请确认以上参数是否正确(Y/N): ")
        if confirmation.lower() == 'y':
            break
        elif confirmation.lower() == 'n':
            modify_parameters()
        else:
            print("无效输入，请输入Y或N。")


# 不用看，定义一个函数允许用户修改参数
def modify_parameters():
    while True:
        key = input("请输入要修改的参数键名，或者输入Q退出修改: ")
        if key.lower() == 'q':
            break
        if hasattr(args, key):
            value = input(f"请输入新的{key}值: ")
            # 根据参数类型进行转换
            if isinstance(getattr(args, key), int):
                value = int(value)
            elif isinstance(getattr(args, key), float):
                value = float(value)
            elif isinstance(getattr(args, key), list):
                value = list(map(int, value.split()))
            elif isinstance(getattr(args, key), bool):
                value = True if value.lower() == 'true' else False
            setattr(args, key, value)
        else:
            print(f"无效的参数键名: {key}")


# 主函数
if __name__ == "__main__":
    # args.silence = True
    if not args.silence:
        get_user_confirmation()  # 询问用户的确认
    if not args.noisy_image_path:
        raise ValueError("必须填写噪声图片路径（noisy_image_path）")
    if not args.model_weights_path:
        raise ValueError("必须填写权重路径（model_weights_path）")
    if not args.output_path:
        args.output_path = create_sequential_folder("runs", "denoise")

    start_time = time.time()  # 记录开始时间
    # 去噪处理
    if os.path.isdir(args.noisy_image_path):  # 如果输入是一个文件夹
        denoise_folder(args.noisy_image_path, args.model_weights_path, args.output_path)
    else:  # 如果输入是一个文件
        _, input_filename = os.path.split(args.noisy_image_path)
        filename, _ = os.path.splitext(input_filename)
        denoise_image(args.noisy_image_path, args.model_weights_path, os.path.join(args.output_path, f"denoise_{filename}.png"))

    print(f"去噪完成，结果已保存到 '{args.output_path}'")
