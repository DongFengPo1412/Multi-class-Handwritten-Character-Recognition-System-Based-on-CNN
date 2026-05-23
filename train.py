import torch
import torch.optim as optim
from src.model import HandwrittenCNN
from src.utils import get_dataloaders
import os
import time


def train():
    # 1. 环境初始化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 深度学习引擎启动 | 设备: {device} (RTX 4060 加速已就绪)")

    # 2. 获取数据加载器 (对应优化的 utils.py，包含 10% 验证集)
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=128)

    # 3. 模型与损失函数
    model = HandwrittenCNN(num_classes=62).to(device)
    criterion = torch.nn.CrossEntropyLoss()

    # 4. 优化器与智能调度 (加入权重衰减 L2 正则化)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    # 负反馈调度：如果 3 轮验证集 Loss 不降，则学习率减半
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, verbose=True)

    # 5. 混合精度加速
    use_cuda = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if use_cuda else None

    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')

    # 6. 训练监控变量
    best_val_acc = 0.0
    epochs = 25  # 配合数据增强，适当增加轮数
    early_stop_patience = 7
    no_improve_epochs = 0

    print(f"📊 任务目标：识别 62 类手写字符 | 场景：0.5mm 签字笔/作业纸")
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
            print(f"  🌟 检测到模型提升，已更新权重 (Best Acc: {best_val_acc:.2f}%)")
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= early_stop_patience:
            print(f"🛑 触发早停机制：模型已连续 {early_stop_patience} 轮未提升，停止训练。")
            break

    # 7. 最终大考：在完全没见过的测试集上评估
    print("\n" + "=" * 30)
    print("🎯 训练结束，开始最终测试集评估...")
    model.load_state_dict(torch.load("checkpoints/emnist_model.pth"))
    model.eval()
    test_correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            test_correct += pred.eq(target).sum().item()

    final_test_acc = 100.0 * test_correct / len(test_loader.dataset)
    print(f"✅ 最终测试准确率: {final_test_acc:.2f}%")
    print(f"模型文件已就绪：checkpoints/emnist_model.pth")
    print("=" * 30)


if __name__ == "__main__":
    train()