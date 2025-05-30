import torch.nn as nn

class ActivationLayer(nn.Module):
    def __init__(self, activation_type='relu'):
        super(ActivationLayer, self).__init__()
        if activation_type == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation_type == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation_type == "sigmoid":
            self.activation = nn.Sigmoid(inplace=True)
        else:
            raise ValueError("Unsupported activation type")

    def forward(self, x):
        return self.activation(x)