import torch
from torch import nn
from torchsummary import summary

class LeNet(nn.Module):
    """LeNet-5 网络结构，适用于 1 通道 28x28 输入"""
    def __init__(self):
        super(LeNet, self).__init__()
        self.sig = nn.Sigmoid()
        self.flatten = nn.Flatten()
        # 两层卷积+平均池化提取特征
        self.conv1 = nn.Conv2d(1, 6, 5, 1, 2)
        self.pool1 = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.AvgPool2d(2, 2)
        # 三层全连接完成分类
        self.fc3 = nn.Linear(400, 120)
        self.fc4 = nn.Linear(120, 84)
        self.fc5 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.sig(self.conv1(x))
        x = self.pool1(x)
        x = self.sig(self.conv2(x))
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        return x
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LeNet().to(device)
    print(summary(model, (1, 28, 28)))