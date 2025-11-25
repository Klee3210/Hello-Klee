import os
# 解决 OpenMP 库冲突问题（Windows 上 PyTorch 常见问题）
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch.utils.data as Data
import matplotlib.pyplot as plt
import numpy as np

# 准备 FashionMNIST 训练集，并将图像 resize 到 LeNet 输入尺寸
train_data = FashionMNIST(root='./data', train=True, download=True,
                          transform=transforms.Compose([transforms.Resize(size=224), transforms.ToTensor()]))
# 使用 DataLoader 进行批量加载
train_loader = Data.DataLoader(dataset=train_data, batch_size=64, shuffle=True, num_workers=0)

# 仅取一个批次数据用于可视化
for step, (b_x, b_y) in enumerate(train_loader):
    if step > 0:
        break
batch_x = b_x.squeeze(1).numpy()
batch_y = b_y.numpy()
class_label = train_data.classes
    
plt.figure(figsize=(12, 5))
for i in np.arange(len(batch_y)):
    plt.subplot(4, 16, i + 1)
    plt.imshow(batch_x[i, :, :], cmap=plt.cm.gray)
    plt.title(class_label[batch_y[i]], size=10)
    plt.axis('off')
    plt.subplots_adjust(wspace=0.05)
plt.show()
