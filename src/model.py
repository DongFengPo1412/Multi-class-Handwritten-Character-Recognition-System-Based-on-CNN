import torch
import torch.nn as nn
import torch.nn.functional as F


class HandwrittenCNN(nn.Module):
    def __init__(self, num_classes=62):
        super(HandwrittenCNN, self).__init__()
        # 第一层卷积块
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # 第二层卷积块
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # 第三层卷积块（不再池化，保留 7x7 特征图）
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout_conv = nn.Dropout2d(0.15)
        self.dropout = nn.Dropout(0.5)

        # 全连接层 (输入大小: 128 * 7 * 7 = 6272)
        self.fc1 = nn.Linear(128 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))   # 28 -> 14
        x = self.dropout_conv(x)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))   # 14 -> 7
        x = self.dropout_conv(x)
        x = F.relu(self.bn3(self.conv3(x)))               # 7 -> 7 (无池化)
        x = x.view(-1, 128 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
