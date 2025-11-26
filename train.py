"""
数据探索脚本：下载 FashionMNIST、应用与 LeNet 一致的预处理，并可视化一个批次。
"""

import os

# 解决 OpenMP 库冲突问题（Windows 上 PyTorch 常见问题）
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch.utils.data as Data
import matplotlib.pyplot as plt
import numpy as np

# 准备 FashionMNIST 训练集，并将图像 resize 到 LeNet 输入尺寸
train_data = FashionMNIST(
    root='./data',
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.Resize(size=224),  # LeNet 定义中使用 224x224 输入
        transforms.ToTensor()         # 将 PIL.Image 转为 [0,1] 张量
    ])
)

# 使用 DataLoader 进行批量加载
# num_workers=0 避免在 Windows 上的多进程开销
train_loader = Data.DataLoader(dataset=train_data, batch_size=64, shuffle=True, num_workers=0)

# 仅取一个批次数据用于可视化
for step, (b_x, b_y) in enumerate(train_loader):
    if step > 0:
        # 只需要一个批次即可检查预处理是否符合预期
        break

# squeeze(1) 去掉单通道维度以便 matplotlib 显示
batch_x = b_x.squeeze(1).numpy()
batch_y = b_y.numpy()
class_label = train_data.classes

# 将一个 batch 的图像排成网格，直观检查预处理是否正确
plt.figure(figsize=(12, 5))
for i in np.arange(len(batch_y)):
    plt.subplot(4, 16, i + 1)
    plt.imshow(batch_x[i, :, :], cmap=plt.cm.gray)
    plt.title(class_label[batch_y[i]], size=10)
    plt.axis('off')
    plt.subplots_adjust(wspace=0.05)
plt.show()



