import torch.nn as nn


class UpsamplingLayer(nn.Module):
    def __init__(self, in_channels, scale_factor=2):
        super(UpsamplingLayer, self).__init__()
        # 输入通道数为 in_channels，输出通道数为 in_channels * (scale_factor ** 2)
        self.conv = nn.Conv2d(in_channels, in_channels * (scale_factor ** 2), kernel_size=3, padding=1)
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.conv(x)
        # 使用 PixelShuffle 进行上采样
        x = nn.PixelShuffle(self.scale_factor)(x)
        return x
