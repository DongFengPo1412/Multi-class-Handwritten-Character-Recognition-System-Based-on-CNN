import sys
import io
import torch
import torch.optim as optim
from src.model import HandwrittenCNN
from src.utils import get_dataloaders, label_map
import os
import time
import shutil




def train():
    # 1. 环境初始化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] 深度学习引擎启动 | 设备: {device} (RTX 4060 加速已就绪)")

    # 1.5 安全备份：如果已有已训练的模型权重，自动备份以防意外覆盖破坏
    if os.path.exists("checkpoints/emnist_model.pth"):
        try:
            shutil.copy("checkpoints/emnist_model.pth", "checkpoints/emnist_model_backup.pth")
            print("  [Backup] 成功备份现有模型权重到 checkpoints/emnist_model_backup.pth")
        except Exception as e:
            print(f"  [Backup Warning] 备份原有权重失败: {e}")

    # 2. 获取数据加载器 (对应优化的 utils.py，包含 10% 验证集)
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=128)

    # 2.5 导出数据增强样例 (对应 Requirement 1 预处理，用于 PPT 报告)
    try:
        import matplotlib
        matplotlib.use('Agg')  # 禁用 GUI 窗口，防止非桌面环境报错
        import matplotlib.pyplot as plt
        
        images, labels = next(iter(train_loader))
        plt.figure(figsize=(10, 8))
        plt.suptitle("EMNIST Preprocessing & Data Augmentation Samples (Req 1)", fontsize=14, fontweight='bold')
        for idx in range(min(16, len(images))):
            plt.subplot(4, 4, idx + 1)
            # 逆归一化还原图像原始色泽显示
            img_show = images[idx].squeeze().numpy() * 0.3317 + 0.1736
            plt.imshow(img_show, cmap='gray', vmin=0, vmax=1)
            lbl_char = label_map[labels[idx].item()]
            plt.title(f"Label: {lbl_char}", fontsize=10)
            plt.axis('off')
        plt.tight_layout()
        aug_path = os.path.join('checkpoints', 'data_augmentation_samples.png')
        plt.savefig(aug_path, dpi=300)
        plt.close()
        print(f"[Augmentations Saved] 已成功导出数据增强样例图：{aug_path} (可直接用于报告和PPT！)")
    except Exception as e:
        print(f"[-] 导出数据增强样例失败: {e}")

    # 3. 模型与损失函数
    model = HandwrittenCNN(num_classes=62).to(device)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    # 4. 优化器与智能调度 (加入权重衰减 L2 正则化)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    # 负反馈调度：如果 3 轮验证集 Loss 不降，则学习率减半
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, verbose=True)

    # 5. 混合精度加速
    use_cuda = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if use_cuda else None

    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')

    # 6. 训练监控变量与历史记录
    best_val_acc = 0.0
    epochs = 25  # 配合数据增强，适当增加轮数
    early_stop_patience = 7
    no_improve_epochs = 0
    
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": []
    }

    print(f"[Train Setup] 任务目标：识别 62 类手写字符 | 场景：0.5mm 签字笔/作业纸")
    print(f"开始执行 25 轮极限训练...")

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        # --- 训练阶段 ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_train = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            if use_cuda:
                with torch.amp.autocast('cuda'):
                    output = model(data)
                    loss = criterion(output, target)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

            train_loss += loss.item()
            pred = output.argmax(dim=1)
            train_correct += pred.eq(target).sum().item()
            total_train += target.size(0)

        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100.0 * train_correct / total_train

        # --- 验证阶段 ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        total_val = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                pred = output.argmax(dim=1)
                val_correct += pred.eq(target).sum().item()
                total_val += target.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100.0 * val_correct / total_val

        # 记录历史数据
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # 更新学习率调度器 (基于验证集 Loss)
        scheduler.step(avg_val_loss)

        epoch_time = time.time() - start_time
        print(f"Epoch [{epoch}/{epochs}] {epoch_time:.1f}s | "
              f"Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.2f}%")

        # --- 保存最优模型与早停逻辑 ---
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "checkpoints/emnist_model.pth")
            print(f"  [Model Update] 检测到模型提升，已更新权重 (Best Acc: {best_val_acc:.2f}%)")
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= early_stop_patience:
            print(f"[Early Stop] 触发早停机制：模型已连续 {early_stop_patience} 轮未提升，停止训练。")
            break

    # 6.5 绘制并保存训练曲线 (用于报告和 PPT 收敛展示)
    try:
        import matplotlib.pyplot as plt
        epochs_range = range(1, len(history["train_loss"]) + 1)
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, history["train_loss"], 'b-o', label='Train Loss')
        plt.plot(epochs_range, history["val_loss"], 'r-x', label='Val Loss')
        plt.title('Training & Validation Loss', fontsize=12, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, history["train_acc"], 'b-o', label='Train Acc')
        plt.plot(epochs_range, history["val_acc"], 'r-x', label='Val Acc')
        plt.title('Training & Validation Accuracy', fontsize=12, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        curve_path = os.path.join('checkpoints', 'training_curves.png')
        plt.savefig(curve_path, dpi=300)
        plt.close()
        print(f"[Curves Saved] 已成功生成训练收敛曲线图：{curve_path} (可直接用于报告和PPT！)")
    except Exception as e:
        print(f"[-] 生成训练收敛曲线图失败: {e}")

    # 7. 最终大考：在完全没见过的测试集上评估
    print("\n" + "=" * 30)
    print("[Test] 训练结束，开始最终测试集评估...")
    model.load_state_dict(torch.load("checkpoints/emnist_model.pth"))
    model.eval()
    test_correct = 0
    
    # 7.5 生成测试集混淆矩阵 (Confusion Matrix)
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        print("[Matrix] 正在收集测试集预测结果，用于生成混淆矩阵...")
        cm = np.zeros((62, 62), dtype=np.int32)
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1).cpu().numpy()
                t_np = target.cpu().numpy()
                for t_val, p_val in zip(t_np, pred):
                    cm[t_val, p_val] += 1
                test_correct += pred.eq(target).sum().item()
                
        # 绘制 62x62 混淆矩阵
        plt.figure(figsize=(14, 12))
        im = plt.imshow(cm, cmap='Blues', interpolation='nearest')
        plt.colorbar(im)
        tick_marks = np.arange(62)
        plt.xticks(tick_marks, list(label_map), rotation=45, fontsize=8)
        plt.yticks(tick_marks, list(label_map), fontsize=8)
        plt.title('EMNIST 62-Class Confusion Matrix (Req 3 Verification)', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()
        
        cm_path = os.path.join('checkpoints', 'confusion_matrix.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[Matrix Saved] 已成功生成混淆矩阵图：{cm_path} (可直接用于报告和PPT！)")
    except Exception as e:
        print(f"[-] 生成混淆矩阵失败: {e}")
        # Fallback evaluation
        test_correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                test_correct += pred.eq(target).sum().item()

    final_test_acc = 100.0 * test_correct / len(test_loader.dataset)
    print(f"[Success] 最终测试准确率: {final_test_acc:.2f}%")
    print(f"模型文件已就绪：checkpoints/emnist_model.pth")
    print("=" * 30)


if __name__ == "__main__":
    train()