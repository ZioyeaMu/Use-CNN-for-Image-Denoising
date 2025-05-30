import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from model.denoising_model import DenoisingModel
from dataset.denoising_dataset import DenoisingDataset
import matplotlib.pyplot as plt
from torchvision import models
from tools.create_sequential_folder import create_sequential_folder


# 定义了SSIM损失函数，不是重点，不想看的话直接跳过，外教问到SSIM如何实现的就直接承认不会，在网上找的代码
# 但是如果问到了SSIM的功能一定要记住：这是用来检测图像边缘的
class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 3
        self.window = self.create_window(window_size, self.channel)

    def create_window(self, window_size, channel):
        window = torch.ones(window_size, window_size,
                            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')) / (
                         window_size * window_size)
        window = window.unsqueeze(0).unsqueeze(0).expand(channel, 1, window_size, window_size).contiguous()
        return window

    def forward(self, img1, img2):
        mu1 = F.conv2d(img1, self.window, padding=self.window_size // 2, groups=self.channel)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size // 2, groups=self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, self.window, padding=self.window_size // 2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, self.window, padding=self.window_size // 2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, self.window, padding=self.window_size // 2, groups=self.channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if self.size_average:
            return 1 - ssim_map.mean()
        else:
            return 1 - ssim_map.mean(1).mean(1).mean(1)


# VGG结构感知的辅助类，用于提取图像的特征，不是重点
class VGGFeatureExtractor(nn.Module):
    def __init__(self):
        super(VGGFeatureExtractor, self).__init__()
        vgg = models.vgg19(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg.features.children())[:35])  # 提取到 conv5_1

    def forward(self, x):
        return self.feature_extractor(x)


# 这个是定义了VGG结构感知损失函数，同样不是重点
# 功能是比较两张图片的结构相似性
class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.feature_extractor = VGGFeatureExtractor().to(
            torch.device('cuda' if torch.cuda.is_available() else 'cpu')).eval()
        self.mse_loss = nn.MSELoss()

    def forward(self, output, target):
        with torch.no_grad():
            target_features = self.feature_extractor(target)
        output_features = self.feature_extractor(output)
        return self.mse_loss(output_features, target_features)


# 重点：这个是将上面定义的损失函数混合在一起
# s为SSIM的混合比例，m为MSE的混合比例，v为VGG的混合比例
class MixedLoss(nn.Module):
    def __init__(self, s=0.6, m=1, v=0.6):
        super(MixedLoss, self).__init__()
        # 初始化比例
        self.s = s
        self.m = m
        self.v = v
        # 初始化各种损失函数
        self.vgg_loss = PerceptualLoss()
        self.SSIM_loss = SSIMLoss()
        self.mse_loss = nn.MSELoss()  # MSE损失函数，功能是比较两张图片的色彩是否相同

    def forward(self, output, target):
        # 将图像分别应用到损失函数上
        vgg_loss = self.vgg_loss(output, target)
        SSIM_loss = self.SSIM_loss(output, target)
        mse_loss = self.mse_loss(output, target)
        # 然后按比例混合损失函数
        total_loss = self.s * SSIM_loss + self.m * mse_loss + self.v * vgg_loss
        return total_loss


# 重点：如何训练模型
def train_model(opt):
    # 评估模型并保存评估图像，不用看
    def evaluate_model():
        model.eval()
        with torch.no_grad():
            for noisy_images, clean_images in val_loader:
                noisy_images = noisy_images.to(device)
                clean_images = clean_images.to(device)

                outputs = model(noisy_images)
                break  # 只取第一批数据进行评估

        # 保存评估图像
        noisy_image = (noisy_images[0].cpu().numpy().transpose(1, 2, 0) + 1) / 2  # 假设数据范围是 [-1, 1]
        clean_image = (clean_images[0].cpu().numpy().transpose(1, 2, 0) + 1) / 2
        output_image = ((outputs[0].cpu().numpy().transpose(1, 2, 0) + 1) / 2).clip(0, 1)

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(clean_image)
        ax[0].set_title('Clean Image')
        ax[1].imshow(noisy_image)
        ax[1].set_title('Noisy Image')
        ax[2].imshow(output_image)
        ax[2].set_title('Denoised Image')
        for a in ax:
            a.axis('off')

        save_path = os.path.join(step_dir, f'eval_img.png')
        plt.savefig(save_path)
        plt.close()

        # 绘制并保存损失曲线
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(losses) + 1), losses, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.legend()
        plt.grid(True)
        save_path = os.path.join(step_dir, 'loss_curve.png')
        plt.savefig(save_path)
        plt.close()

    step = 0  # 训练步数
    train_dir = create_sequential_folder("runs", "train")
    best_loss = float('inf')  # 初始化最佳损失为无穷大
    best_model_weights = None  # 用于存储最佳模型的 state_dict
    # 创建数据集和数据加载器
    train_dataset = DenoisingDataset(opt.dataset_path, resize=opt.resize)  # 将路径下的图像制作成数据集（为他们添加噪声）并重新裁剪
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)  # 将数据集转化成张量
    # 创建验证集数据加载器，供评估图像时使用
    val_dataset = DenoisingDataset(opt.valset_path, resize=opt.resize)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 自动选择使用cpu或者gpu（cuda）训练
    model = DenoisingModel().to(device)  # 创建模型结构并移到gpu或cpu上
    criterion = MixedLoss()  # 设置损失函数为我们自定义的混合损失函数
    optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate)  # 选择优化器为Adam优化器
    losses = []  # 损失函数列表，用于记录每一步的损失函数，最终绘制图像用

    # 尝试加载之前的模型并接着之前的模型训练
    if opt.model_weights_path:
        print(f"[提示] 已加载模型:{opt.model_weights_path}。")
        model.load_state_dict(torch.load(opt.model_weights_path, map_location=device))
    else:
        print("[提示] 未指定模型路径，将会使用新模型训练。")

    start_time = time.time()  # 记录整个训练开始时间
    for epoch in range(opt.num_epochs):  # 训练num_epochs指定的步数
        step += 1  # 训练步数+1
        model.train()  # 设置模型为训练模式（另一个模式叫做评估模式），在训练模式下，模型会记录梯度信息，以便进行反向传播和参数更新
        running_loss = 0.0  # 记录单步损失
        batch_num = 0
        epoch_start_time = time.time()  # 记录当前epoch的开始时间
        for noisy_images, clean_images in train_loader:  # 在数据加载器中获取噪声图像和干净图像
            noisy_images = noisy_images.to(device)  # 加载到gpu/cpu里
            clean_images = clean_images.to(device)

            # 前向传播
            outputs = model(noisy_images)  # 模型处理噪声图像，生成去噪图像
            loss = criterion(outputs, clean_images)  # 使用损失函数计算去噪图像与干净图像之间的损失
            # 反向传播和优化
            optimizer.zero_grad()  # 清零优化器的梯度。在每次反向传播之前，需要清零梯度，否则梯度会累加。
            loss.backward()  # 计算损失对模型参数的梯度
            optimizer.step()  # 根据计算出的梯度，更新模型的参数。

            running_loss += loss.item() * noisy_images.size(0)  # 将当前批次的损失值累加
            batch_num += 1
            # 计算进度百分比
            progress = int((batch_num / len(train_loader)) * 100)
            # 构建进度条，长度为50
            progress_bar = "[" + "=" * (progress // 2) + ">" + "-" * (50 - (progress // 2)) + "]"
            # 计算当前批次的ETA
            current_batch_time = time.time() - epoch_start_time
            avg_batch_time = current_batch_time / batch_num if batch_num > 0 else 0
            remaining_batches = len(train_loader) - batch_num
            eta_batch_seconds = avg_batch_time * remaining_batches
            eta_batch_str = time.strftime("%H:%M:%S", time.gmtime(eta_batch_seconds))
            # 计算整个训练的ETA
            total_elapsed_time = time.time() - start_time
            avg_epoch_time = total_elapsed_time / (epoch + 1) if epoch > 0 else 0
            remaining_epochs = opt.num_epochs - epoch - 1
            eta_total_seconds = avg_epoch_time * (
                    remaining_epochs + (len(train_loader) - batch_num) / len(train_loader))
            eta_total_str = time.strftime("%H:%M:%S", time.gmtime(eta_total_seconds))
            print(
                f"\rBatch [{batch_num}/{len(train_loader)}]\tloss: {(running_loss / len(train_dataset)):.4f}\t{progress_bar} {progress}%\tBatch ETA: {eta_batch_str}\tTotal ETA: {eta_total_str}",
                end="")

        print("\r", end="")
        epoch_loss = running_loss / len(train_dataset)  # 计算当前轮次的平均损失
        losses.append(epoch_loss)  # 记录每个epoch的损失值
        print(f'Epoch [{epoch + 1}/{opt.num_epochs}]\tLoss: {epoch_loss:.4f}')
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_weights = model.state_dict()
        # 每x步保存一次模型
        if step % opt.autosave_epochs == 0 and step != opt.num_epochs:
            step_dir = os.path.join(train_dir, f"step{step}")
            os.makedirs(step_dir, exist_ok=True)
            torch.save(best_model_weights, os.path.join(step_dir, "best.pth"))
            torch.save(model.state_dict(), os.path.join(step_dir, "last.pth"))
            evaluate_model()
            print(f"[提示] 已自动保存训练文件和评估结果到：{step_dir}")

    # 保存模型权重
    torch.save(best_model_weights, os.path.join(train_dir, "best.pth"))
    torch.save(model.state_dict(), os.path.join(train_dir, "last.pth"))
    step_dir = train_dir
    evaluate_model()
    print(f'训练完成，模型权重已保存到：{train_dir}')
    # plot_loss_curve(opt.num_epochs, losses)


# 定义绘制损失曲线的函数，用于可视化训练过程
def plot_loss_curve(num_epochs, losses):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.show()


# 创建ArgumentParser对象
parser = argparse.ArgumentParser(description='Train a denoising model.')
# 添加命令行参数
parser.add_argument('--batch_size', type=int, default=1,
                    help='设置批次大小')  # 设置批次，批次越大训练越快也越占内存（显存），如果你运行时候报错内存不足试着减小一下这个
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='设置学习率')  # 设置学习率，学习率越高模型收敛（学习）越快，但是学不精，学习率越慢反之。推荐先用0.001直到模型收敛，再改为0.0001.
parser.add_argument('--num_epochs', type=int, default=100, help='设置训练的轮数')  # 步数，学习多少次后停止
parser.add_argument('--autosave_epochs', type=int, default=25, help='设置每x轮自动保存')  # 每x步自动评估并保存模型
parser.add_argument('--resize', type=int, nargs=2, default=(512, 512),
                    help='设置图像裁剪大小')  # 重新裁剪数据集的图像大小，太大的图片训练慢，而且用批次加速训练要求统一大小的图像
parser.add_argument('--model_weights_path', type=str, default=None, help='模型权重文件路径')
parser.add_argument('--dataset_path', type=str, default='./datasets/denoise/train_dataset', help='训练数据集路径')
parser.add_argument('--valset_path', type=str, default='./datasets/denoise/val_dataset', help='验证数据集路径')
parser.add_argument('--silence', action='store_true', help='是否开启静默模式，不再询问用户意见')
# 解析命令行参数
args = parser.parse_args()


# 不用看，定义一个函数来获取用户确认
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


if __name__ == "__main__":
    # args.silence = True
    if not args.silence:
        get_user_confirmation()  # 询问用户的确认
    train_model(args)  # 训练
