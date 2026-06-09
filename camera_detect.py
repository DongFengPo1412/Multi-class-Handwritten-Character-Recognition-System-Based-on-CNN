import cv2
import torch
import numpy as np
import torch.nn.functional as F
from src.model import HandwrittenCNN
from src.utils import label_map
from src.corrector import HandwrittenCorrector

# 初始化纠错器
corrector = HandwrittenCorrector()

def preprocess_for_emnist(char_img):
    """
    针对 0.5mm 签字笔优化的 EMNIST 风格预处理
    """
    # 1. 动态笔画加粗 (针对 0.5mm 签字笔的特征增强)
    density = np.sum(char_img > 0) / float(char_img.size)
    kernel = np.ones((2, 2), np.uint8)

    # 0.5mm 签字笔在图像中通常 density < 0.25，需要强化笔画
    if density < 0.25:
        char_img = cv2.dilate(char_img, kernel, iterations=1)
    elif density > 0.6:
        char_img = cv2.erode(char_img, kernel, iterations=1)

    h, w = char_img.shape
    # 2. 保持比例缩放，长边为 20
    if h > w:
        new_h, new_w = 20, max(int(20 * w / h), 1)
    else:
        new_h, new_w = max(int(20 * h / w), 1), 20

    char_resz = cv2.resize(char_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 3. 嵌入 28x28 画布
    temp = np.zeros((28, 28), dtype=np.uint8)
    pad_h = (28 - new_h) // 2
    pad_w = (28 - new_w) // 2
    temp[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = char_resz

    # 4. 基于质心的精确对齐 (EMNIST 核心标准)
    moments = cv2.moments(temp)
    if moments["m00"] != 0:
        cm_x = moments["m10"] / moments["m00"]
        cm_y = moments["m01"] / moments["m00"]
        shift_x = 14.0 - cm_x
        shift_y = 14.0 - cm_y
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        temp = cv2.warpAffine(temp, M, (28, 28))

    return temp


def predict_with_tta(model, img_tensor, device):
    """
    TTA 推理：通过 11 个变体的投票提升鲁棒性 (算法优化关键)
    """
    batch = []
    # 原始 + 8个方向的 1px 平移
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            padded = F.pad(img_tensor, (1, 1, 1, 1), mode='constant', value=0)
            shifted = padded[:, :, 1 + dy:1 + dy + 28, 1 + dx:1 + dx + 28]
            batch.append(shifted)

    # 2个小角度旋转变体
    for angle in [-5, 5]:
        rad = np.radians(angle)
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        theta = torch.tensor([[cos_a, -sin_a, 0], [sin_a, cos_a, 0]],
                             dtype=torch.float32, device=device).unsqueeze(0)
        grid = F.affine_grid(theta, img_tensor.size(), align_corners=False)
        rotated = F.grid_sample(img_tensor, grid, align_corners=False, padding_mode='zeros')
        batch.append(rotated)

    batch_tensor = torch.cat(batch, dim=0)
    with torch.no_grad():
        if device.type == 'cuda':
            with torch.amp.autocast('cuda'):  # 开启 RTX 4060 混合精度推理
                output = model(batch_tensor)
                probs = F.softmax(output, dim=1)
        else:
            output = model(batch_tensor)
            probs = F.softmax(output, dim=1)

    return probs.mean(dim=0, keepdim=True)


def scan_cameras(max_to_try=3):
    """
    扫描系统中可用的摄像头索引 (使用 DSHOW 避免 Windows 下的连接延迟)
    """
    available = []
    for i in range(max_to_try):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                available.append(i)
            cap.release()
    return available


def merge_bounding_boxes(raw_boxes, box_size):
    """
    连通域边界框融合算法 (解决 i/j 点体分离、笔画断开及嵌套框问题)
    """
    if not raw_boxes:
        return []
        
    # 先按照左边界 x 坐标排序
    raw_boxes.sort(key=lambda b: b[0])
    
    merged_boxes = []
    for box in raw_boxes:
        if not merged_boxes:
            merged_boxes.append(box)
            continue
            
        last_box = merged_boxes[-1]
        lx, ly, lw, lh = last_box
        x, y, w, h = box
        
        # 1. 垂直对齐合并 (例如 i/j 的点体融合)
        # 计算在 X 轴上的重叠宽度
        x_overlap = max(0, min(lx + lw, x + w) - max(lx, x))
        x_overlap_ratio = x_overlap / float(min(lw, w))
        
        # 计算垂直方向的间隔
        if ly < y:
            y_gap = y - (ly + lh)
        else:
            y_gap = ly - (y + h)
            
        # 如果 X 轴高度重叠，且垂直间隔很小 (小于较高框高度的 75%)
        is_vertical_aligned = (x_overlap_ratio > 0.25 or (x >= lx and x+w <= lx+lw) or (lx >= x and lx+lw <= x+w)) and (y_gap < max(lh, h) * 0.75)
        
        # 2. 水平极其邻近合并 (例如手写笔画断开)
        x_gap = x - (lx + lw)
        is_horizontal_near = (x_gap < 5) and (abs(ly - y) < max(lh, h) * 0.35)
        
        # 3. 包含/嵌套关系合并
        is_nested = (x >= lx - 2 and x + w <= lx + lw + 2 and y >= ly - 2 and y + h <= ly + lh + 2) or \
                    (lx >= x - 2 and lx + lw <= x + w + 2 and ly >= y - 2 and ly + lh <= y + h + 2)
                    
        if is_vertical_aligned or is_horizontal_near or is_nested:
            # 融合边界框
            new_x = min(lx, x)
            new_y = min(ly, y)
            new_w = max(lx + lw, x + w) - new_x
            new_h = max(ly + lh, y + h) - new_y
            merged_boxes[-1] = (new_x, new_y, new_w, new_h)
        else:
            merged_boxes.append(box)
            
    # 二次过滤：剔除合并后尺寸依然过小的杂质噪点 (宽<5 或 高<8 或 面积<100)
    valid_boxes = []
    for (x, y, w, h) in merged_boxes:
        if w >= 5 and h >= 8 and (w * h) >= 80:
            valid_boxes.append((x, y, w, h))
            
    return valid_boxes


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HandwrittenCNN(num_classes=62).to(device)
    try:
        model.load_state_dict(torch.load("checkpoints/emnist_model.pth", map_location=device))
        print("[+] 成功加载 CNN 模型权重！")
    except Exception as e:
        print(f"[-] 加载模型权重失败，请确认 checkpoints/emnist_model.pth 是否存在！错误信息: {e}")
        return

    model.eval()

    # 1. 扫描摄像头
    print("正在扫描系统可用摄像头...")
    available_cams = scan_cameras()
    if not available_cams:
        print("[Warning] 未检测到可用摄像头，默认尝试索引 0")
        available_cams = [0]
    else:
        print(f"[Camera] 发现可用摄像头索引: {available_cams}")

    cam_list_idx = 0
    current_cam = available_cams[cam_list_idx]
    
    # 使用 DSHOW 启动快速连接
    cap = cv2.VideoCapture(current_cam, cv2.CAP_DSHOW)
    box_size = 350
    final_result = ""
    raw_result = ""
    uncertain_infos = []

    print("\n" + "=" * 50)
    print("[Info] 实时字符识别系统已启动！")
    print("  * 【空格键】：对红框内手写字符进行顺序识别与纠错")
    print("  * 【c 键】  ：动态切换当前摄像头信号源")
    print("  * 【q 键】  ：安全退出系统")
    print("=" * 50 + "\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Warning] 无法读取摄像头帧，请检查摄像头占用或物理连接！")
            # 自动尝试重新打开
            cap.release()
            cv2.waitKey(1000)
            cap = cv2.VideoCapture(current_cam, cv2.CAP_DSHOW)
            continue

        h_f, w_f, _ = frame.shape
        x1, y1 = (w_f - box_size) // 2, (h_f - box_size) // 2
        x2, y2 = x1 + box_size, y1 + box_size

        # ROI 区域提取与预处理
        roi = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # ----------------------------------------------------
        # 【算法升级 1】：高斯差分消除不均匀阴影 (Illumination Map Subtraction)
        # ----------------------------------------------------
        bg = cv2.GaussianBlur(gray, (51, 51), 0)  # 估计光照背景
        gray_no_shadow = cv2.divide(gray, bg, scale=255)  # 差分消除阴影

        # CLAHE 局部对比度增强
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray_no_shadow)

        # 滤波去噪与自适应二值化
        blur = cv2.GaussianBlur(cv2.medianBlur(enhanced, 5), (3, 3), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 5)

        # ----------------------------------------------------
        # 在实时帧上绘制辅助框与提示信息
        # ----------------------------------------------------
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, f"Cam Index: {current_cam} (Press 'c' to switch)", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        cv2.putText(frame, f"Raw prediction: {raw_result}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        cv2.putText(frame, f"Corrected output: {final_result}", (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # 在屏幕下侧绘制多候选不确定结果
        y_offset = h_f - 30
        for i, info in enumerate(uncertain_infos[:3]):
            cv2.putText(frame, f"? {info}", (20, y_offset - i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

        # 实时展示窗口
        cv2.imshow('Handwritten OCR System (Main Frame)', frame)
        cv2.imshow('AI Vision (Binarization & Shadow Removal)', thresh)

        key = cv2.waitKey(1) & 0xFF
        
        # ----------------------------------------------------
        # 按键交互 1：执行识别与后处理修正
        # ----------------------------------------------------
        if key == ord(' '):
            final_result = ""
            raw_result = ""
            uncertain_infos = []
            
            # 提取所有外部轮廓
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 获取粗分割的原始边界框
            raw_boxes = []
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                area = cv2.contourArea(cnt)
                # 排除边缘框干扰，只保留内部可能轮廓
                if 40 < area < 30000 and 8 < x < box_size - 8 and 8 < y < box_size - 8:
                    raw_boxes.append((x, y, w, h))
            
            # ----------------------------------------------------
            # 【算法升级 2】：自适应连通域合并 (Box Merging)
            # ----------------------------------------------------
            valid_chars = merge_bounding_boxes(raw_boxes, box_size)
            print(f"\n[分割报告] 最终定位到 {len(valid_boxes)} 个独立字符")

            if len(valid_boxes) == 0:
                print("[-] 未在红框内检测到有效的手写字符！")
                continue

            # 获取字符的最大高度，用于后面的大小写高度比校正
            max_h = max(b[3] for b in valid_boxes)

            char_probs = []
            aspect_ratios = []
            relative_heights = []

            # 在主图上临时标出分割矩形
            debug_frame = roi.copy()

            # ----------------------------------------------------
            # 循环预测各个字符
            # ----------------------------------------------------
            for idx, (cx, cy, cw, ch) in enumerate(valid_chars):
                cv2.rectangle(debug_frame, (cx, cy), (cx + cw, cy + ch), (255, 255, 0), 2)
                
                # 裁剪并归一化
                char_crop = thresh[cy:cy + ch, cx:cx + cw]
                char_norm = preprocess_for_emnist(char_crop)

                img_t = torch.from_numpy(char_norm).float().to(device).view(1, 1, 28, 28) / 255.0
                img_t = (img_t - 0.1736) / 0.3317

                # 运行 TTA 变体多重采样推理
                avg_prob = predict_with_tta(model, img_t, device)
                prob_vec = avg_prob.squeeze(0)
                
                char_probs.append(prob_vec)
                aspect_ratios.append(cw / float(ch))
                relative_heights.append(ch / float(max_h))

                # 收集不确定字符的置信度信息 (Top-3)
                top3_vals, top3_indices = torch.topk(prob_vec, 3)
                val1, val2 = top3_vals[0].item(), top3_vals[1].item()
                
                # 判定不确定的标准：主候选概率低于 80% 或前两名差距小于 20%
                if val1 < 0.80 or (val1 - val2) < 0.20:
                    opts = ", ".join(f"'{label_map[top3_indices[i].item()]}' ({top3_vals[i].item():.1%})" for i in range(3))
                    uncertain_info = f"Char #{idx+1}: {opts}"
                    uncertain_infos.append(uncertain_info)
                    print(f"[Warning] 模糊字符警示 - {uncertain_info}")

            # 展示字符分割的调试图
            cv2.imshow('Debug: Segments', debug_frame)

            # ----------------------------------------------------
            # 【算法升级 3】：送入 Corrector 模块执行概率解码与几何校正
            # ----------------------------------------------------
            raw_str, decoded_str, context = corrector.decode_sequence(char_probs, aspect_ratios, relative_heights)
            
            raw_result = raw_str
            final_result = decoded_str
            
            print(f"CNN 原始输出 (ArgMax) : {raw_result}")
            print(f"推断上下文环境 (Context): {context}")
            print(f"[Result] 算法修正后最终结果   : {final_result}")

        # ----------------------------------------------------
        # 按键交互 2：动态切换当前摄像头信号源
        # ----------------------------------------------------
        elif key == ord('c'):
            cap.release()
            cam_list_idx = (cam_list_idx + 1) % len(available_cams)
            current_cam = available_cams[cam_list_idx]
            print(f"[Camera] 正在热切换摄像头设备到索引: {current_cam} ...")
            cap = cv2.VideoCapture(current_cam, cv2.CAP_DSHOW)
            final_result = ""
            raw_result = ""
            uncertain_infos = []

        # ----------------------------------------------------
        # 按键交互 3：退出
        # ----------------------------------------------------
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()