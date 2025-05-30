import torch.nn as nn


class OutputLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(OutputLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        # self.sigmoid = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv.forward(x)
        # x = self.sigmoid.forward(x)
        return x



