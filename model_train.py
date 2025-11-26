import time
from torch import nn
from torchvision import transforms
from torchvision.datasets import FashionMNIST
import torch.utils.data as Data
import torch
import copy
from model import LeNet

"""
训练脚本：构建 FashionMNIST 数据流水线、定义训练/验证循环，并记录最优模型。
"""

def train_var_data():
    """
    加载和准备训练数据与验证数据
    
    Returns:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
    """
    # 加载FashionMNIST数据集，如果不存在则自动下载
    train_data = FashionMNIST(
        root='./data',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.Resize(size=224),  # 与 LeNet 输入分辨率保持一致
            transforms.ToTensor()         # 将像素缩放到 [0,1]
        ])
    )
    # 将训练数据按照8:2的比例分割为训练集和验证集
    train_data, val_data = Data.random_split(
        train_data,
        [0.8 * len(train_data), 0.2 * len(train_data)]
    )
    # 创建训练数据加载器，批次大小为64，打乱数据，使用8个工作线程
    train_loader = Data.DataLoader(train_data, batch_size=64, shuffle=True, num_workers=8)
    # 创建验证数据加载器
    # 注意：这里应该使用val_data而不是train_data
    val_loader = Data.DataLoader(val_data, batch_size=64, shuffle=True, num_workers=8)
    return train_loader, val_loader

def train_model(model,train_loader,val_loader,num_epochs):
    """
    训练模型

    Args:
        model: 要训练的模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        num_epochs: 训练轮数
    """
    # 检测并设置设备（GPU或CPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 使用Adam优化器，学习率为0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # 使用交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    # 将模型移动到指定设备（GPU或CPU）
    model = model.to(device)
    # 保存最佳模型权重的副本
    best_model_wts = copy.deepcopy(model.state_dict())
    # 初始化最佳准确率为0
    best_acc = 0.0
    # 记录每轮训练损失
    train_losses = []
    # 记录每轮验证损失
    val_losses = []
    # 记录训练开始时间
    train_accuracies = []  # 每个 epoch 的训练准确率
    val_accuracies = []    # 每个 epoch 的验证准确率
    since = time.time()    # 记录训练开始时间

    # 循环执行每个epoch的训练与验证流程
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # 每轮开始前重置累计指标
        train_loss = 0.0
        train_correct = 0
        val_loss = 0.0
        val_correct = 0
        train_num = 0
        val_num = 0
        # ------- 训练阶段 -------
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # 将一个batch的数据与标签转移到当前设备
            inputs, targets = inputs.to(device), targets.to(device)
            # 显式切换至训练模式，确保BN/Dropout等行为正确
            model.train()
            # 前向传播获得预测结果
            outputs = model(inputs)
            pre_labels = torch.argmax(outputs, dim=1)  # 记录预测标签以计算准确率
            # 计算本batch的损失
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()  # 反向传播
