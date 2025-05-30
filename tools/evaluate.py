import argparse
import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
from PIL import Image
import torchvision.transforms as transforms


def calculate_mse(original: torch.Tensor, denoised: torch.Tensor) -> torch.Tensor:
    """Calculate Mean Squared Error between two images."""
    return torch.mean((original - denoised) ** 2)


def calculate_psnr(original: torch.Tensor, denoised: torch.Tensor) -> float:
    """Calculate Peak Signal-to-Noise Ratio between two images."""
    mse = calculate_mse(original, denoised)
    max_pixel = 1.0
    return 20 * torch.log10(max_pixel / torch.sqrt(mse)).item()


def calculate_ssim(original: np.ndarray, denoised: np.ndarray) -> float:
    """Calculate Structural Similarity Index between two images."""
    return ssim(original, denoised, channel_axis=2)


def calculate_ncc(original: torch.Tensor, denoised: torch.Tensor) -> float:
    """Calculate Normalized Cross-Correlation between two images."""
    original = original - torch.mean(original)
    denoised = denoised - torch.mean(denoised)
    ncc = torch.sum(original * denoised) / (
            torch.sqrt(torch.sum(original ** 2)) * torch.sqrt(torch.sum(denoised ** 2))
    )
    return ncc.item()


def calculate_npcr(original: np.ndarray, denoised: np.ndarray) -> float:
    """Calculate Noise Peak Criteria Ratio between two images."""
    original_gray = np.mean(original, axis=2).astype(np.int32)
    denoised_gray = np.mean(denoised, axis=2).astype(np.int32)
    diff = np.abs(original_gray - denoised_gray)
    npcr = np.sum(diff > 0) / diff.size
    return npcr


def evaluate_image_quality(original_path: str, other: str) -> dict:
    """Evaluate the quality of a denoised image against the original."""
    # Load images
    original_image = Image.open(original_path).convert('RGB')
    denoised_image = Image.open(other).convert('RGB')

    # Ensure images are the same size
    if original_image.size != denoised_image.size:
        # Resize the denoised image to match the original image size
        denoised_image = denoised_image.resize(original_image.size)

    # Transform images to tensors and numpy arrays
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    original_tensor = transform(original_image)
    denoised_tensor = transform(denoised_image)

    original_np = np.array(original_image)
    denoised_np = np.array(denoised_image)

    # Calculate metrics
    metrics = {
        'MSE': calculate_mse(original_tensor, denoised_tensor).item(),
        'PSNR': calculate_psnr(original_tensor, denoised_tensor),
        'SSIM': calculate_ssim(original_np, denoised_np),
        'NCC': calculate_ncc(original_tensor, denoised_tensor),
        'NPCR': calculate_npcr(original_np, denoised_np)
    }

    return metrics


parser = argparse.ArgumentParser(description='Evaluate the quality of a denoised image.')
parser.add_argument('--original', type=str, default='F:\\PyCharm\\dncnn_pytorch\\data\\Test\\Set68\\test051.png',
                    help='Path to the original image.')
parser.add_argument('--other', type=str, default='F:\\PyCharm\\dncnn_pytorch\\results\\Set68\\test051_denoised.png',
                    help='Path to the denoised image.')
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


if __name__ == "__main__":
    args.silence = True
    if not args.silence:
        get_user_confirmation()  # 询问用户的确认
    if not args.original:
        raise ValueError("必须填写原图片路径（original）")
    if not args.other:
        raise ValueError("必须填写对比图片路径（other）")

    metrics = evaluate_image_quality(args.original, args.other)

    print(f"MSE: {metrics['MSE']:.4f}")
    print(f"PSNR: {metrics['PSNR']:.4f} dB")
    print(f"SSIM: {metrics['SSIM']:.4f}")
    print(f"NCC: {metrics['NCC']:.4f}")
    print(f"NPCR: {metrics['NPCR']:.4f}")