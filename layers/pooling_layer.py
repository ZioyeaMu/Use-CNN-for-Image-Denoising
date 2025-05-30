import torch.nn as nn

class PoolingLayer(nn.Module):
    def __init__(self, kernel_size=2, stride=2):
        super(PoolingLayer, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size, stride)

    def forward(self, x):
        return self.pool(x)