# -*- coding: utf-8 -*-
"""
# @file name  : train_lenet.py
# @author     : 钟致远
# @date       : 2020-7-03
# @brief      : 猫狗分类
"""
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from matplotlib import pyplot as plt
from homework1.lenet import LeNet
from homework1.my_dataset import RMBDataset


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


set_seed()  # 设置随机种子
dogcat_label = {"cat": 0, "dog": 1}

# 参数设置
MAX_EPOCH = 100
BATCH_SIZE = 16
LR = 0.01
log_interval = 50
val_interval = 1

# ============================ step 1/5 数据 ============================

split_dir = os.path.join("data", "dogcat_split")
train_dir = os.path.join(split_dir, "train")
valid_dir = os.path.join(split_dir, "valid")

# # 求数据集的均值和标准差
# data_transform = transforms.transforms.Compose([
#     transforms.Resize((32, 32)),    # 缩放到32 * 32
#     transforms.RandomCrop(32, padding=4),    # 裁剪
#     transforms.ToTensor(),    # 转化为张量
# ])
#
# train_dataset = datasets.ImageFolder(root=train_dir, transform=data_transform)
# dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1)
# mean = torch.zeros(3)
# std = torch.zeros(3)
# print('==> Computing mean and std..')
# for inputs, targets in dataloader:
#     for i in range(3):
#         # print(inputs)
#         mean[i] += inputs[:, i, :, :].mean()
#         std[i] += inputs[:, i, :, :].std()
# mean.div_(len(train_dataset))
# std.div_(len(train_dataset))
# print("mean:{} and std: {}".format(mean,std))

# 后续调参时可直接使用求出的标准差和方差
mean = [0.4217, 0.3918, 0.3586]
std = [0.2562, 0.2450, 0.2364]

# Compose将一系列的transforms方法进行有序的组合
train_transform = transforms.Compose([
    transforms.Resize((32, 32)),    # 缩放到32 * 32
    transforms.RandomCrop(32, padding=4),    # 裁剪
    transforms.ToTensor(),    # 转化为张量
    transforms.Normalize(mean, std),    # 数据归一化
])

valid_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

# 构建MyDataset实例
train_data = RMBDataset(data_dir=train_dir, transform=train_transform)
valid_data = RMBDataset(data_dir=valid_dir, transform=valid_transform)

print(type(train_data))

# 构建DataLoder
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE)

# ============================ step 2/5 模型 ============================

net = LeNet(classes=2)
net.initialize_weights()

# ============================ step 3/5 损失函数 ============================
criterion = nn.CrossEntropyLoss()                                                   # 选择损失函数

# ============================ step 4/5 优化器 ============================
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)                        # 选择优化器
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)     # 设置学习率下降策略

# ============================ step 5/5 训练 ============================
train_curve = list()
valid_curve = list()

for epoch in range(MAX_EPOCH):

    loss_mean = 0.
    correct = 0.
    total = 0.

    net.train()
    # 数据获取
    for i, data in enumerate(train_loader):

        # forward
        inputs, labels = data
        outputs = net(inputs)

        # backward
        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        loss.backward()

        # update weights
        optimizer.step()

        # 统计分类情况
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).squeeze().sum().numpy()

        # 打印训练信息
        loss_mean += loss.item()
        train_curve.append(loss.item())
        if (i+1) % log_interval == 0:
            loss_mean = loss_mean / log_interval
            print("Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                epoch, MAX_EPOCH, i+1, len(train_loader), loss_mean, correct / total))
            loss_mean = 0.

    scheduler.step()  # 更新学习率

    # validate the model
    if (epoch+1) % val_interval == 0:

        correct_val = 0.
        total_val = 0.
        loss_val = 0.
        net.eval()
        with torch.no_grad():
            for j, data in enumerate(valid_loader):
                inputs, labels = data
                outputs = net(inputs)
                loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).squeeze().sum().numpy()

                loss_val += loss.item()

            loss_val_epoch = loss_val / len(valid_loader)
            valid_curve.append(loss_val_epoch)
            # valid_curve.append(loss.item())    # 记录整个epoch样本的loss，注意要取平均
            print("Valid:\t Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                epoch, MAX_EPOCH, j+1, len(valid_loader), loss_val_epoch, correct_val / total_val))


train_x = range(len(train_curve))
train_y = train_curve

train_iters = len(train_loader)
valid_x = np.arange(1, len(valid_curve)+1) * train_iters*val_interval # 由于valid中记录的是epochloss，需要对记录点进行转换到iterations
valid_y = valid_curve

plt.plot(train_x, train_y, label='Train')
plt.plot(valid_x, valid_y, label='Valid')

plt.legend(loc='upper right')
plt.ylabel('loss value')
plt.xlabel('Iteration')
plt.show()

# ============================ inference ============================

test_dir = os.path.join(split_dir, "test")
test_data = RMBDataset(data_dir=test_dir, transform=valid_transform)
valid_loader = DataLoader(dataset=test_data, batch_size=1)

total_val = 0
correct_val = 0

for i, data in enumerate(valid_loader):
    # forward
    inputs, labels = data
    outputs = net(inputs)
    _, predicted = torch.max(outputs.data, 1)
    total_val += labels.size(0)
    correct_val += (predicted == labels).squeeze().sum().numpy()

    # rs = 'cat' if predicted.numpy()[0] == 0 else 'dog'
    # print("模型分类结果{} 实际结果{}".format(rs,labels))

print("准确率为：{}".format(correct_val/total_val))