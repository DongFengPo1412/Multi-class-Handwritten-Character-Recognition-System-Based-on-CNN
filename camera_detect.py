import cv2
import torch
import numpy as np
import torch.nn.functional as F
from src.model import HandwrittenCNN
from src.utils import label_map


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
        with torch.amp.autocast('cuda'):  # 开启 RTX 4060 混合精度推理
            output = model(batch_tensor)
            probs = F.softmax(output, dim=1)

    return probs.mean(dim=0, keepdim=True)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HandwrittenCNN(num_classes=62).to(device)
    model.load_state_dict(torch.load("checkpoints/emnist_model.pth", map_location=device))
    model.eval()

    cap = cv2.VideoCapture(0)
    box_size = 350
    final_result = ""

    print("--- 实时识别系统已启动 (原始帧模式 | RTX 4060 加速) ---")
    print("操作提示: 画面非镜像。按下 [空格键] 进行多字符顺序识别")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 【修改点】移除了 cv2.flip(frame, 1)，现在输出的是摄像头采集的原始帧
        h_f, w_f, _ = frame.shape
        x1, y1 = (w_f - box_size) // 2, (h_f - box_size) // 2
        x2, y2 = x1 + box_size, y1 + box_size

        # 1. 预处理流水线
        roi = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # CLAHE 抗光影干扰
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        blur = cv2.GaussianBlur(cv2.medianBlur(gray, 5), (3, 3), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 5)

        # 绘制检测框和结果
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, f"Predict: {final_result}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

        cv2.imshow('Handwritten OCR System (Space to Run)', frame)
        cv2.imshow('Internal AI Vision', thresh)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            final_result = ""
            # 2. 轮廓分割与过滤
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            valid_chars = []
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                area = cv2.contourArea(cnt)
                solidity = area / float(w * h)
                # 过滤太小(噪点)或不符合字符长宽比的区域
                if 200 < area < 25000 and 0.2 < w / h < 5.0 and solidity > 0.2:
                    if 8 < x < box_size - 8 and 8 < y < box_size - 8:  # 避开边框干扰
                        valid_chars.append((x, y, w, h, cnt))

            # 3. 顺序排序：由于是原始帧，此处 sort 依然保证从左到右
            valid_chars.sort(key=lambda b: b[0])

            res_list = []
            for (cx, cy, cw, ch, cnt) in valid_chars:
                char_crop = thresh[cy:cy + ch, cx:cx + cw]
                char_norm = preprocess_for_emnist(char_crop)

                img_t = torch.from_numpy(char_norm).float().to(device).view(1, 1, 28, 28) / 255.0
                img_t = (img_t - 0.1736) / 0.3317

                # 4. 推理
                avg_prob = predict_with_tta(model, img_t, device)
                conf, pred = torch.max(avg_prob, 1)

                if conf.item() > 0.45:
                    res_list.append(label_map[pred.item()])

            final_result = "".join(res_list)
            print(f"识别结果 (从左至右): {final_result}")

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()