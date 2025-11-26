import torch
from torch import nn
from torchsummary import summary

"""
定义 LeNet-5 模型结构，用于 FashionMNIST 分类任务。
保持经典结构设置，便于与课本/论文中的结果对齐。
"""


class LeNet(nn.Module):
    """
    LeNet-5 经典卷积神经网络结构，针对单通道输入设计。
    该实现保留论文中的 Sigmoid 激活与平均池化，以便于复现基准。
    """

    def __init__(self):
        super(LeNet, self).__init__()
        # Sigmoid 在原论文中用于引入非线性；此处保持经典实现
        self.sig = nn.Sigmoid()
        self.flatten = nn.Flatten()
        # 两层卷积+平均池化提取特征
        self.conv1 = nn.Conv2d(1, 6, 5, 1, 2)   # 输入 1x28x28 -> 输出 6x28x28
        self.pool1 = nn.AvgPool2d(2, 2)         # 6x28x28 -> 6x14x14
        self.conv2 = nn.Conv2d(6, 16, 5)        # 6x14x14 -> 16x10x10
        self.pool2 = nn.AvgPool2d(2, 2)         # 16x10x10 -> 16x5x5
        # 三层全连接完成分类（16*5*5 = 400）
        self.fc3 = nn.Linear(400, 120)          # 特征压缩
        self.fc4 = nn.Linear(120, 84)           # 进一步提炼
        self.fc5 = nn.Linear(84, 10)            # 输出 10 类 logits

    def forward(self, x):
        # block1：卷积降噪 + 池化减半空间分辨率
        x = self.sig(self.conv1(x))
        x = self.pool1(x)
        # block2：更深层感受野提取语义特征
        x = self.sig(self.conv2(x))
        x = self.pool2(x)
        # 展平后进入多层感知机完成分类
        x = self.flatten(x)
        x = self.fc3(x)  # 第一层全连接，相当于 400 -> 120
        x = self.fc4(x)  # 第二层全连接，保留 84 维隐表示
        x = self.fc5(x)  # 最后一层输出 logits
        return x


if __name__ == '__main__':
    # 简单打印模型结构，确认参数量与输入尺寸匹配
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LeNet().to(device)
    # summary 将展示每层输出尺寸与参数数量，调试时非常有用
    print(summary(model, (1, 28, 28)))


