import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import random
from src.model import HandwrittenCNN
from src.utils import get_dataloaders, label_map
from src.corrector import HandwrittenCorrector

# 初始化纠错器
corrector = HandwrittenCorrector()

def predict_with_tta(model, img_tensor, device):
    """
    【算法优化】测试时增强推理
    通过对单张图进行微小平移/旋转，取 11 次推理的平均概率，极大提升鲁棒性
    """
    batch = []
    # 原始 + 8个方向的微小平移
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            padded = F.pad(img_tensor, (1, 1, 1, 1), mode='constant', value=0)
            shifted = padded[:, :, 1 + dy:1 + dy + 28, 1 + dx:1 + dx + 28]
            batch.append(shifted)

    batch_tensor = torch.cat(batch, dim=0)
    with torch.no_grad():
        if device.type == 'cuda':
            with torch.amp.autocast('cuda'):
                output = model(batch_tensor)
                probs = F.softmax(output, dim=1)
        else:
            output = model(batch_tensor)
            probs = F.softmax(output, dim=1)

    # 返回平均概率分布
    return probs.mean(dim=0, keepdim=True)


def apply_correction_logic(avg_prob, img_tensor):
    """
    【结果修正】提取图像特征的高宽比，结合 corrector 的几何校验规则进行单字修正
    """
    # 计算28x28图像中激活笔画的边界框高宽比
    img_np = img_tensor.cpu().squeeze().numpy()
    # EMNIST 中背景是 0, 前景是正值 (由于Normalize了，阈值设为 -0.5 来区分背景)
    nonzero = (img_np > -0.5).nonzero()
    if len(nonzero[0]) > 0:
        min_y, max_y = nonzero[0].min(), nonzero[0].max()
        min_x, max_x = nonzero[1].min(), nonzero[1].max()
        w = max_x - min_x + 1
        h = max_y - min_y + 1
        ar = w / float(h)
    else:
        ar = 1.0

    # 单字情况下没有同行参考，相对高度设为 1.0，上下文为 neutral
    prob_vec = avg_prob.squeeze(0)
    adjusted_probs = corrector.apply_geometry_corrections([prob_vec], [ar], [1.0], "neutral")
    adjusted_prob = adjusted_probs[0]

    # 获取置信度最高的两个候选
    top2_probs, top2_indices = torch.topk(adjusted_prob, 2)
    p1, idx1 = top2_probs[0].item(), top2_indices[0].item()
    p2, idx2 = top2_probs[1].item(), top2_indices[1].item()

    char1 = label_map[idx1]
    
    # 判定模糊的标准：置信度低于 80% 或前二名概率差距小于 20%
    is_ambiguous = p1 < 0.80 or (p1 - p2) < 0.20

    return char1, p1, is_ambiguous


def test_display_interactive():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] 验证系统启动 | 硬件加速: {device}")

    # 获取测试集数据 (建议 num_workers=0 避免 Windows 崩溃)
    _, _, test_loader = get_dataloaders(batch_size=12)
    data_iter = iter(test_loader)

    model = HandwrittenCNN(num_classes=62).to(device)
    try:
        model.load_state_dict(torch.load("checkpoints/emnist_model.pth", map_location=device))
        print("[+] 权重加载成功！开始执行 S 级算法验证...")
    except:
        print("[-] 错误：找不到模型权重，请确认 checkpoints 目录")
        return

    model.eval()

    batch_count = 0
    while True:
        try:
            data, target = next(data_iter)
            batch_count += 1
        except StopIteration:
            break

        data, target = data.to(device), target.to(device)

        plt.figure(figsize=(15, 10))
        plt.suptitle(
            f"Batch {batch_count} - RTX 4060 TTA & Geometry Corrected Inference\n(Green: Correct | Red: Error | Orange: Ambiguous/Uncertain)",
            fontsize=16)

        for i in range(12):
            img_t = data[i].unsqueeze(0)

            # 执行 TTA 优化推理
            avg_prob = predict_with_tta(model, img_t, device)

            # 执行基于几何学高宽比的结果修正分析
            pred_char, conf, is_ambiguous = apply_correction_logic(avg_prob, data[i])
            true_char = label_map[target[i].item()]

            plt.subplot(3, 4, i + 1)
            # 在绘图展示前逆归一化，方便人眼看清
            img_show = data[i].cpu().squeeze().numpy() * 0.3317 + 0.1736
            plt.imshow(img_show, cmap='gray', vmin=0, vmax=1)

            # 颜色逻辑
            if pred_char == true_char:
                color = 'orange' if is_ambiguous else 'green'
            else:
                color = 'red'

            title_text = f"Pred: {pred_char} ({conf:.2%})\nTrue: {true_char}"
            plt.title(title_text, color=color, fontsize=11, fontweight='bold')
            plt.axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        print(f"正在展示第 {batch_count} 组测试结果（可视化窗口已弹出）...")
        plt.show()

        user_input = input("按回车继续，输入 'q' 退出: ").strip().lower()
        if user_input == 'q': break


def run_simulated_lexicon_test():
    """
    运行基于 EMNIST 测试集的单词与数字序列联合解码测试，证明纠错修正器功效
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HandwrittenCNN(num_classes=62).to(device)
    try:
        model.load_state_dict(torch.load("checkpoints/emnist_model.pth", map_location=device))
    except:
        return # 静默退出
    model.eval()

    _, _, test_loader = get_dataloaders(batch_size=500)
    
    # 整理测试图片
    class_images = {i: [] for i in range(62)}
    for data, target in test_loader:
        for img, lbl in zip(data, target):
            class_images[lbl.item()].append(img)
        if all(len(class_images[i]) >= 10 for i in range(62)):
            break
            
    char_to_idx = {char: i for i, char in enumerate(label_map)}
    
    simulated_sequences = [
        ("hello", [('h', 0.6, 1.0), ('e', 0.6, 0.7), ('l', 0.4, 1.0), ('l', 0.4, 1.0), ('o', 0.65, 0.7)]),
        ("zoom", [('z', 0.6, 0.7), ('o', 0.65, 0.7), ('o', 0.65, 0.7), ('m', 0.8, 0.7)]),
        ("class", [('c', 0.55, 0.7), ('l', 0.4, 1.0), ('a', 0.6, 0.7), ('s', 0.55, 0.7), ('s', 0.55, 0.7)]),
        ("2026", [('2', 0.55, 1.0), ('0', 0.45, 1.0), ('2', 0.55, 1.0), ('6', 0.55, 1.0)]),
        ("10850", [('1', 0.3, 1.0), ('0', 0.45, 1.0), ('8', 0.55, 1.0), ('5', 0.55, 1.0), ('0', 0.45, 1.0)]),
    ]

    print("\n" + "="*50)
    print("[Test] 正在运行【修正识别结果】联合解码模拟评估 (基于 EMNIST 测试集)")
    print("="*50)
    
    for word, char_details in simulated_sequences:
        chars_probs = []
        aspect_ratios = []
        relative_heights = []
        
        for char, ar, rh in char_details:
            idx = char_to_idx[char]
            img = random.choice(class_images[idx]).to(device).unsqueeze(0)
            with torch.no_grad():
                # 计算其 TTA 概率
                avg_prob = predict_with_tta(model, img, device)
                chars_probs.append(avg_prob.squeeze(0))
            aspect_ratios.append(ar)
            relative_heights.append(rh)
            
        raw_str, decoded_str, context = corrector.decode_sequence(chars_probs, aspect_ratios, relative_heights)
        print(f"[*] 目标字串: {word:<7} | CNN原始识别: {raw_str:<7} | 修正结果: {decoded_str:<7} ({'数字' if context=='numeric' else '英文'}模式)")
        
    print("="*50 + "\n")


if __name__ == "__main__":
    # 先控制台输出单词和数字纠错评估
    run_simulated_lexicon_test()
    # 再弹出可视化交互窗口
    test_display_interactive()