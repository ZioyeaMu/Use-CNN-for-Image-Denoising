from layers.convolutional_layer import ConvolutionalLayer
from layers.pooling_layer import PoolingLayer
from layers.activation_layer import ActivationLayer
from layers.upsampling_layer import UpsamplingLayer
from layers.output_layer import OutputLayer
import torch.nn as nn
import torch.nn.init as init


class DenoisingModel(nn.Module):
    def __init__(self):
        super(DenoisingModel, self).__init__()
        middle_depth = 4
        middle_channels = 64
        layers = []

        # 输入层
        layers.append(ConvolutionalLayer(3, middle_channels))
        layers.append(ActivationLayer('leaky_relu'))
        layers.append(PoolingLayer())

        # 中间层
        for _ in range(middle_depth):
            layers.append(ConvolutionalLayer(middle_channels, middle_channels))
            layers.append(ActivationLayer('leaky_relu'))

        # 输出层
        layers.append(UpsamplingLayer(middle_channels, scale_factor=2))
        layers.append(OutputLayer(middle_channels, 3))

        self.model = nn.Sequential(*layers)     # 把列表转化成pytorch网络
        self.init_weights()

    def forward(self, x):
        return self.model(x)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)