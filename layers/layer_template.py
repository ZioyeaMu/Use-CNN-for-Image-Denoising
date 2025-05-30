import torch
import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        self.JuanJi = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # 激活层
        self.JiHuo = nn.ReLU()  # 激活层
        self.ChiHua = nn.MaxPool2d(kernel_size=2, stride=2)  # 池化层
        self.ShangCaiYang = nn.Upsample(scale_factor=2)  # 上采样层
        self.ShuChu = nn.Conv2d(32, 3, kernel_size=3, padding=1)  # 输出层

    def forward(self, x):
        x = self.JuanJi.forward(x)
        x = self.JiHuo.forward(x)
        x = self.ChiHua.forward(x)
        x = self.ShangCaiYang.forward(x)
        x = self.ShuChu.forward(x)

        return x


# 示例使用
if __name__ == "__main__":
    try:
        model = CNNModel()
        print(model)
        example_input = torch.randn(1, 3, 28, 28)  # 示例输入 (batch_size=1, channels=3, height=28, width=28)

        output = model.forward(example_input)  # 前向传播

        print(
            f"输入形状: \n大小：{example_input.shape[3]}x{example_input.shape[2]}\n维度：{example_input.shape[1]}\n图片数量：{example_input.shape[0]}")
        print(
            f"输出形状: \n大小：{output.shape[3]}x{output.shape[2]}\n维度：{output.shape[1]}\n图片数量：{output.shape[0]}")  # 应该是 (1, 3, 28, 28)
        if output.shape != example_input.shape:
            raise Exception("形状大小不匹配")
    except Exception as e:
        raise Exception(f'{e}\n出错了，快去找一些哪里有问题吧~')
