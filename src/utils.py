import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import numpy as np

# EMNIST ByClass 的映射表 (62类：0-9, A-Z, a-z)
label_map = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


class EMNISTRotateFlip:
    """EMNIST 原始数据是转置的，需进行 -90° 旋转并水平翻转以恢复人类视角"""

    def __call__(self, img):
        img = transforms.functional.rotate(img, -90)
        img = transforms.functional.hflip(img)
        return img


def get_dataloaders(batch_size=128):
    # 使用双线性插值 (BILINEAR) 提高缩放后的平滑度
    interp = transforms.InterpolationMode.BILINEAR

    # 1. 验证集与测试集预处理：保持数据纯净，评估模型真实泛化力
    val_transform = transforms.Compose([
        EMNISTRotateFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.1736,), (0.3317,))
    ])

    # 2. 训练集：全维度增强，专门针对 0.5mm 细笔迹与摄像头拍摄环境优化 [cite: 6, 12]
    train_transform = transforms.Compose([
        EMNISTRotateFlip(),
        # 空间变换：模拟作业纸倾斜、字符大小不一及书写倾斜
        transforms.RandomAffine(
            degrees=15, translate=(0.12, 0.12), scale=(0.8, 1.2),
            shear=12, interpolation=interp, fill=0
        ),
        # 透视变换：模拟摄像头非正对手写纸产生的几何畸变
        transforms.RandomPerspective(distortion_scale=0.2, p=0.4, fill=0),
        # 弹性变换：模拟纸张褶皱与笔画颤抖
        transforms.RandomApply([transforms.ElasticTransform(alpha=50.0)], p=0.2),
        # 光影增强：针对浅色作业纸在不同灯光下的阴影变化 [cite: 4]
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        # 模糊增强：模拟对焦不准，提升对 0.5mm 细笔画的提取能力
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),

        transforms.ToTensor(),
        # 随机擦除：强制模型学习字符的全局拓扑结构而非局部笔画
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0),
        transforms.Normalize((0.1736,), (0.3317,))
    ])

    # 加载原始 EMNIST 数据集 [cite: 8, 16]
    full_train_set = datasets.EMNIST('./data', split='byclass', train=True, download=True, transform=train_transform)
    test_set = datasets.EMNIST('./data', split='byclass', train=False, download=True, transform=val_transform)

    # 3. 算法优化：切分 10% 训练数据作为验证集，执行模型选择与性能监控
    train_size = int(0.9 * len(full_train_set))
    val_size = len(full_train_set) - train_size
    train_set, val_set = random_split(full_train_set, [train_size, val_size])

    # 4. 针对 RTX 4060 与 Windows 环境的数据加载加速
    # pin_memory 加速数据上传至 GPU，persistent_workers 保持子进程活跃减少开销
    loader_args = {
        'batch_size': batch_size,
        'num_workers': 4,
        'pin_memory': True,
        'persistent_workers': True if torch.cuda.is_available() else False
    }

    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)
    test_loader = DataLoader(test_set, batch_size=1000, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader