import numpy as np
import random
import os
import argparse
import cv2
import time  # 添加时间模块
if __name__ == "__main__":
    from create_sequential_folder import create_sequential_folder
else:
    from tools.create_sequential_folder import create_sequential_folder


def add_random_noise(image, noise_type=None, **kwargs):
    """
    为图像随机添加一种噪声并返回加噪图像。

    参数:
        image: 输入图像(numpy数组, uint8类型)
        noise_type: 指定的噪声类型，可选类型有['gaussian', 'salt_pepper', 'periodic', 'uniform', 'mixed']，默认为随机选择
        **kwargs: 可选噪声参数:
            - 高斯噪声: mean(均值，默认0), var(方差，默认0.01)
            - 椒盐噪声: salt_prob(盐概率，默认0.05), pepper_prob(椒概率，默认0.05)
            - 周期性噪声: amplitude(振幅，默认50), frequency(频率，默认10)
            - 均匀噪声: low(最小值，默认-10), high(最大值，默认10)

    返回:
        加噪后的图像(numpy数组, uint8类型)
    """
    # 输入校验和类型转换
    if not isinstance(image, np.ndarray):
        raise TypeError("输入必须是numpy数组")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    # 默认随机选择噪声类型
    if noise_type is None:
        noise_types = ['gaussian', 'salt_pepper', 'periodic', 'uniform', 'mixed']
        selected_noise = random.choice(noise_types)
    else:
        selected_noise = noise_type

    # 高斯噪声
    if selected_noise == 'gaussian':
        mean = kwargs.get('mean', 0)
        var = kwargs.get('var', 0.01)
        noise = np.random.normal(mean, var, image.shape)
        return np.clip(image + noise, 0, 255).astype(np.uint8)

    # 椒盐噪声
    elif selected_noise == 'salt_pepper':
        salt_prob = kwargs.get('salt_prob', 0.05)
        pepper_prob = kwargs.get('pepper_prob', 0.05)
        prob = salt_prob + pepper_prob
        if prob >= 1:
            raise ValueError("椒盐概率总和需小于1")

        noisy = image.copy()
        mask = np.random.random(image.shape[:2])
        # 胡椒噪声（黑点）
        noisy[mask < pepper_prob] = 0
        # 盐噪声（白点）
        noisy[(mask >= pepper_prob) & (mask < prob)] = 255
        return noisy

    # 周期性噪声
    elif selected_noise == 'periodic':
        h, w = image.shape[:2]
        amplitude = kwargs.get('amplitude', 50)
        frequency = kwargs.get('frequency', 10)
        noise = amplitude * np.sin(2 * np.pi * frequency * np.arange(w) / w)
        if len(image.shape) == 3:
            noise = noise[np.newaxis, :, np.newaxis]
        return np.clip(image + noise, 0, 255).astype(np.uint8)

    # 均匀噪声
    elif selected_noise == 'uniform':
        low = kwargs.get('low', -25)
        high = kwargs.get('high', 25)
        noise = np.random.uniform(low, high, image.shape)
        return np.clip(image + noise, 0, 255).astype(np.uint8)

    # 混合噪声（高斯+椒盐）
    elif selected_noise == 'mixed':
        # 先加高斯
        noisy = add_random_noise(image, noise_type='gaussian',
                                 mean=kwargs.get('mean', 0),
                                 var=kwargs.get('var', 0.05))
        # 再加椒盐
        return add_random_noise(noisy, noise_type='salt_pepper',
                                salt_prob=kwargs.get('salt_prob', 0.05),
                                pepper_prob=kwargs.get('pepper_prob', 0.05))
    else:
        raise ValueError(f"不支持的噪声类型: {selected_noise}")


def process_image(image_path, output_path, noise_type=None, **kwargs):
    """
    对单张图像进行加噪处理并保存。

    参数:
        image_path: 输入图像路径
        output_path: 加噪后图像的保存路径
        noise_type: 指定的噪声类型
        **kwargs: 加噪参数
    """
    import cv2

    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像文件: {image_path}")
        return

    # 加噪
    noisy_image = add_random_noise(image, noise_type=noise_type, **kwargs)

    # 保存加噪后的图像
    cv2.imwrite(output_path, noisy_image)


def process_folder(input_folder, output_folder, noise_type=None, **kwargs):
    """
    对文件夹中的所有图像进行加噪处理并保存。

    参数:
        input_folder: 输入图像文件夹路径
        output_folder: 加噪后图像的保存文件夹路径
        noise_type: 指定的噪声类型
        **kwargs: 加噪参数
    """

    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 获取输入文件夹中的所有文件
    image_files = [f for f in os.listdir(input_folder) if
                   f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    total_images = len(image_files)

    start_time = time.time()  # 记录开始时间

    # 对每个图像文件进行加噪
    for idx, image_file in enumerate(image_files):
        input_path = os.path.join(input_folder, image_file)
        output_path = os.path.join(output_folder, image_file)

        # 跳过非图像文件
        if not os.path.isfile(input_path):
            continue

        # 读取图像
        image = cv2.imread(input_path)
        if image is None:
            print(f"无法读取图像文件: {input_path}")
            continue

        # 加噪
        noisy_image = add_random_noise(image, noise_type=noise_type, **kwargs)

        # 保存加噪后的图像
        cv2.imwrite(output_path, noisy_image)

        # 计算进度和ETA
        progress = (idx + 1) / total_images * 100
        elapsed_time = time.time() - start_time
        eta = elapsed_time / (idx + 1) * (total_images - idx - 1)
        eta_str = time.strftime("%H:%M:%S", time.gmtime(eta))

        # 构建进度条
        progress_bar = "[" + "=" * int(progress // 2) + ">" + "-" * (50 - int(progress // 2)) + "]"
        print(f"\rAdding Noise [{idx + 1}/{total_images}]\t{progress_bar} {progress:.1f}%\tETA: {eta_str}", end="")
    print("\r", end="")


parser = argparse.ArgumentParser(description='Random addnoise an image.')
parser.add_argument('--image_path', type=str, default="../val_dataset",  # 替换为你的有干净图像路径
                    help='要加噪的图像的路径')
parser.add_argument('--output_path', type=str, default=None, help='加噪图片保存路径')
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
    if not args.image_path:
        raise ValueError("必须填写图片路径（image_path）")
    if not args.output_path:
        args.output_path = create_sequential_folder("../runs", "addnoise")

    # 加噪处理
    if os.path.isdir(args.image_path):  # 如果输入是一个文件夹
        process_folder(args.image_path, args.output_path)
    else:  # 如果输入是一个文件
        _, input_filename = os.path.split(args.image_path)
        filename, _ = os.path.splitext(input_filename)
        process_image(args.image_path, os.path.join(args.output_path, f"noisy_{filename}.png"))

    print(f"加噪完成，结果已保存到 '{args.output_path}'")