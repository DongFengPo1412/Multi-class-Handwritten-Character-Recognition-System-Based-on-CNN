import cv2
import torch
import numpy as np
import torch.nn.functional as F
from src.model import HandwrittenCNN
from src.utils import label_map
from src.corrector import HandwrittenCorrector
from src.baidu_ocr import BaiduOCRClient, BaiduOCRUnavailable
from src.local_ocr import segment_character_boxes
from concurrent.futures import ThreadPoolExecutor

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
    扫描系统中可用的摄像头索引
    """
    available = []
    for i in range(max_to_try):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                available.append(i)
            cap.release()
    return available


def create_cloud_ocr():
    """
    初始化百度手写 OCR。失败时不中断本地 CNN 演示。
    """
    try:
        client = BaiduOCRClient()
        client.ensure_ready()
        print("[Baidu OCR] 百度手写识别已启用")
        return client, True, "ready"
    except BaiduOCRUnavailable as e:
        print(f"[Baidu OCR] 未启用: {e}")
    except Exception as e:
        print(f"[Baidu OCR] 初始化失败: {e}")
    return None, False, "unavailable"


def box_center_y(box):
    return box[1] + box[3] / 2.0


def group_boxes_by_reading_lines(boxes):
    """
    多行文本按从上到下分组，每行内部从左到右排序。
    """
    if not boxes:
        return []

    heights = [h for _, _, _, h in boxes]
    median_h = float(np.median(heights)) if heights else 1.0
    line_threshold = max(18.0, median_h * 0.65)
    lines = []

    for box in sorted(boxes, key=box_center_y):
        cy = box_center_y(box)
        target_line = None
        best_distance = None
        for line in lines:
            distance = abs(cy - line["center"])
            if distance <= line_threshold and (best_distance is None or distance < best_distance):
                target_line = line
                best_distance = distance

        if target_line is None:
            lines.append({"center": cy, "boxes": [box]})
        else:
            target_line["boxes"].append(box)
            target_line["center"] = float(np.mean([box_center_y(b) for b in target_line["boxes"]]))

    return [
        sorted(line["boxes"], key=lambda b: b[0])
        for line in sorted(lines, key=lambda item: item["center"])
    ]


def sort_boxes_reading_order(boxes):
    """
    多行文本按从上到下、从左到右排序，避免两行字符按 X 坐标混排。
    """
    ordered = []
    for line in group_boxes_by_reading_lines(boxes):
        ordered.extend(line)
    return ordered


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
            
        combined_h = max(ly + lh, y + h) - min(ly, y)
        max_allowed_h = max(lh, h) * 1.55
        # 如果 X 轴重叠且垂直间隔很小才合并；限制合并后高度，避免跨行粘连。
        is_vertical_aligned = (
            x_overlap_ratio > 0.25
            or (x >= lx and x + w <= lx + lw)
            or (lx >= x and lx + lw <= x + w)
        ) and (y_gap < max(10, min(lh, h) * 1.4)) and (combined_h <= max_allowed_h)
        
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
            
    return sort_boxes_reading_order(valid_boxes)


UI = {
    "background": (246, 247, 249),
    "header": (36, 39, 44),
    "panel": (255, 255, 255),
    "border": (222, 225, 230),
    "text": (50, 53, 58),
    "muted": (132, 135, 142),
    "teal": (180, 150, 32),
    "green": (92, 160, 49),
    "orange": (42, 135, 230),
    "red": (70, 70, 220),
    "blue": (210, 118, 45),
}


def draw_label(image, text, origin, color=None, scale=0.48):
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color or UI["muted"],
        1,
        cv2.LINE_AA,
    )


def draw_value(image, text, origin, color=None, scale=0.82, max_width=370):
    value = text or "--"
    while value:
        width = cv2.getTextSize(value, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)[0][0]
        if width <= max_width:
            break
        value = value[:-1]
    if value != (text or "--"):
        value = value[:-3] + "..." if len(value) > 3 else value
    cv2.putText(
        image,
        value,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color or UI["text"],
        2,
        cv2.LINE_AA,
    )


def draw_status_badge(image, text, origin, color):
    x, y = origin
    label = text.upper()
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0]
    width = text_size[0] + 22
    cv2.rectangle(image, (x, y), (x + width, y + 26), color, -1)
    cv2.putText(
        image,
        label,
        (x + 11, y + 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def fit_image(image, width, height, background=(28, 30, 34)):
    canvas = np.full((height, width, 3), background, dtype=np.uint8)
    src_h, src_w = image.shape[:2]
    scale = min(width / src_w, height / src_h)
    new_w = max(1, int(src_w * scale))
    new_h = max(1, int(src_h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    x = (width - new_w) // 2
    y = (height - new_h) // 2
    canvas[y:y + new_h, x:x + new_w] = resized
    return canvas


def draw_capture_overlay(frame, roi_rect):
    x1, y1, x2, y2 = roi_rect
    overlay = frame.copy()
    shade = np.full_like(frame, 18)
    overlay = cv2.addWeighted(overlay, 0.45, shade, 0.55, 0)
    overlay[y1:y2, x1:x2] = frame[y1:y2, x1:x2]

    color = UI["teal"]
    length = max(18, min(x2 - x1, y2 - y1) // 9)
    thickness = 3
    for start, end in [
        ((x1, y1), (x1 + length, y1)), ((x1, y1), (x1, y1 + length)),
        ((x2, y1), (x2 - length, y1)), ((x2, y1), (x2, y1 + length)),
        ((x1, y2), (x1 + length, y2)), ((x1, y2), (x1, y2 - length)),
        ((x2, y2), (x2 - length, y2)), ((x2, y2), (x2, y2 - length)),
    ]:
        cv2.line(overlay, start, end, color, thickness, cv2.LINE_AA)

    cv2.rectangle(overlay, (x1, max(0, y1 - 27)), (x1 + 132, y1), color, -1)
    cv2.putText(
        overlay,
        "CAPTURE AREA",
        (x1 + 10, y1 - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return overlay


def render_dashboard(
    frame,
    roi_rect,
    thresh,
    current_cam,
    raw_result,
    final_result,
    cloud_result,
    cloud_status,
    cloud_enabled,
    cloud_confidence,
    uncertain_infos,
    debug_enabled,
):
    width, height = 1360, 760
    canvas = np.full((height, width, 3), UI["background"], dtype=np.uint8)

    cv2.rectangle(canvas, (0, 0), (width, 64), UI["header"], -1)
    cv2.putText(
        canvas,
        "HANDWRITTEN OCR STUDIO",
        (28, 39),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.82,
        (245, 247, 250),
        2,
        cv2.LINE_AA,
    )
    draw_status_badge(canvas, f"CAM {current_cam}", (1110, 19), UI["blue"])
    draw_status_badge(
        canvas,
        "BAIDU ON" if cloud_enabled else "BAIDU OFF",
        (1210, 19),
        UI["green"] if cloud_enabled else UI["muted"],
    )

    camera_x, camera_y, camera_w, camera_h = 24, 84, 852, 620
    cv2.rectangle(
        canvas,
        (camera_x - 1, camera_y - 1),
        (camera_x + camera_w + 1, camera_y + camera_h + 1),
        UI["border"],
        1,
    )
    camera_view = fit_image(draw_capture_overlay(frame, roi_rect), camera_w, camera_h)
    canvas[camera_y:camera_y + camera_h, camera_x:camera_x + camera_w] = camera_view

    panel_x, panel_y, panel_w, panel_h = 900, 84, 436, 620
    cv2.rectangle(canvas, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), UI["panel"], -1)
    cv2.rectangle(canvas, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), UI["border"], 1)

    draw_label(canvas, "LOCAL CNN / RAW", (924, 118))
    draw_value(canvas, raw_result, (924, 151), UI["orange"])
    cv2.line(canvas, (924, 171), (1312, 171), UI["border"], 1)

    draw_label(canvas, "CORRECTED RESULT", (924, 202))
    draw_value(canvas, final_result, (924, 242), UI["green"], scale=1.05)
    cv2.line(canvas, (924, 263), (1312, 263), UI["border"], 1)

    draw_label(canvas, "BAIDU HANDWRITING OCR", (924, 294))
    cloud_color = UI["green"] if cloud_status == "done" else UI["red"] if cloud_status == "failed" else UI["blue"]
    draw_value(canvas, cloud_result or cloud_status or "--", (924, 330), cloud_color, scale=0.88)
    if cloud_confidence is not None:
        draw_label(canvas, f"Average confidence  {cloud_confidence:.1%}", (924, 354), UI["muted"])
    cv2.line(canvas, (924, 373), (1312, 373), UI["border"], 1)

    draw_label(canvas, "AI VISION PREVIEW", (924, 404))
    preview = cv2.cvtColor(255 - thresh, cv2.COLOR_GRAY2BGR)
    preview = fit_image(preview, 388, 154, background=(248, 249, 250))
    canvas[420:574, 924:1312] = preview
    cv2.rectangle(canvas, (924, 420), (1312, 574), UI["border"], 1)

    if uncertain_infos:
        draw_status_badge(canvas, f"{len(uncertain_infos)} UNCERTAIN", (924, 590), UI["orange"])
        draw_label(canvas, uncertain_infos[0][:58], (924, 638), UI["text"], scale=0.43)
    else:
        draw_status_badge(canvas, "READY", (924, 590), UI["green"])
        draw_label(canvas, "Place handwriting inside the capture area.", (924, 638), UI["muted"], scale=0.43)

    footer = "SPACE  Recognize     B  Baidu OCR     C  Camera     D  Debug     Q  Quit"
    cv2.putText(
        canvas,
        footer,
        (28, 738),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        UI["text"],
        1,
        cv2.LINE_AA,
    )
    draw_status_badge(canvas, "DEBUG ON" if debug_enabled else "DEBUG OFF", (1215, 715), UI["blue"] if debug_enabled else UI["muted"])
    return canvas


def close_debug_windows():
    for name in ("Debug: Segments", "Debug: Baidu OCR Input", "AI Vision Debug"):
        try:
            cv2.destroyWindow(name)
        except cv2.error:
            pass


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
    
    # 启动默认摄像头后端
    cap = cv2.VideoCapture(current_cam)
    box_size = 350
    final_result = ""
    raw_result = ""
    cloud_result = ""
    cloud_status = ""
    cloud_confidence = None
    cloud_ocr, cloud_enabled, cloud_state = create_cloud_ocr()
    uncertain_infos = []
    debug_enabled = False
    latest_debug_frame = None
    executor = ThreadPoolExecutor(max_workers=1)
    cloud_future = None

    print("\n" + "=" * 50)
    print("[Info] 实时字符识别系统已启动！")
    print("  * 【空格键】：对红框内手写字符进行顺序识别与纠错")
    print("  * 【b 键】  ：开启/关闭百度手写 OCR")
    print("  * 【c 键】  ：动态切换当前摄像头信号源")
    print("  * 【d 键】  ：开启/关闭调试窗口")
    print("  * 【q 键】  ：安全退出系统")
    print("=" * 50 + "\n")

    window_name = "Handwritten OCR Studio"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1360, 760)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Warning] 无法读取摄像头帧，请检查摄像头占用或物理连接！")
            # 自动尝试重新打开
            cap.release()
            cv2.waitKey(1000)
            cap = cv2.VideoCapture(current_cam)
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
                                       cv2.THRESH_BINARY, 11, 5)
        
        # 自动对比度极性检测：统计图像边缘像素的分布
        # 取图像四周最外层的边缘像素
        h_t, w_t = thresh.shape
        border_pixels = np.concatenate([
            thresh[0, :],          # 上边界
            thresh[-1, :],         # 下边界
            thresh[:, 0],          # 左边界
            thresh[:, -1]          # 右边界
        ])
        # 如果边缘像素中白色(255)占了大多数，说明背景是亮色(如白纸)，我们需要反色让文字变白背景变黑
        if np.mean(border_pixels) > 127:
            thresh = 255 - thresh

        # 轮询百度云 OCR 识别线程结果
        if cloud_future is not None and cloud_future.done():
            try:
                res, err = cloud_future.result()
                if res is not None:
                    cloud_result = res.text or "(无文本)"
                    cloud_confidence = res.average_confidence
                    cloud_status = "done"
                    for line_index, line in enumerate(res.lines, 1):
                        confidence = f"{line.confidence:.1%}" if line.confidence is not None else "N/A"
                        print(f"[Baidu OCR] 第 {line_index} 行: {line.text} ({confidence})")
                    print(f"[Baidu OCR] 手写识别结果: {cloud_result}")
                else:
                    cloud_status = "failed"
                    print(f"[Baidu OCR] 云识别失败: {err}")
            except Exception as ex:
                cloud_status = "failed"
                print(f"[Baidu OCR] 云识别处理异常: {ex}")
            cloud_future = None

        dashboard = render_dashboard(
            frame=frame,
            roi_rect=(x1, y1, x2, y2),
            thresh=thresh,
            current_cam=current_cam,
            raw_result=raw_result,
            final_result=final_result,
            cloud_result=cloud_result,
            cloud_status=cloud_status or cloud_state,
            cloud_enabled=cloud_enabled,
            cloud_confidence=cloud_confidence,
            uncertain_infos=uncertain_infos,
            debug_enabled=debug_enabled,
        )
        cv2.imshow(window_name, dashboard)
        if debug_enabled:
            cv2.imshow("AI Vision Debug", thresh)
            if latest_debug_frame is not None:
                cv2.imshow("Debug: Segments", latest_debug_frame)

        key = cv2.waitKey(1) & 0xFF
        
        # ----------------------------------------------------
        # 按键交互 1：执行识别与后处理修正
        # ----------------------------------------------------
        if key == ord(' '):
            final_result = ""
            raw_result = ""
            cloud_result = ""
            cloud_status = ""
            cloud_confidence = None
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
            valid_chars = segment_character_boxes(thresh, raw_boxes, box_size)
            valid_lines = group_boxes_by_reading_lines(valid_chars)
            print(f"\n[分割报告] 最终定位到 {len(valid_chars)} 个独立字符 / {len(valid_lines)} 行（已按阅读顺序排序）")

            if len(valid_chars) == 0:
                print("[-] 未在红框内检测到有效的手写字符！")
                continue

            # 获取字符的最大高度，用于后面的大小写高度比校正
            max_h = max(b[3] for b in valid_chars)

            char_probs = []
            aspect_ratios = []
            relative_heights = []
            line_probs = []
            line_aspect_ratios = []
            line_relative_heights = []

            # 在主图上临时标出分割矩形
            debug_frame = roi.copy()

            # ----------------------------------------------------
            # 循环预测各个字符
            # ----------------------------------------------------
            global_idx = 0
            for line_idx, line_boxes in enumerate(valid_lines):
                current_line_probs = []
                current_line_aspect_ratios = []
                current_line_relative_heights = []

                for cx, cy, cw, ch in line_boxes:
                    global_idx += 1
                    cv2.rectangle(debug_frame, (cx, cy), (cx + cw, cy + ch), (255, 255, 0), 2)
                    cv2.putText(debug_frame, str(global_idx), (cx, max(cy - 5, 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

                    # 裁剪并归一化
                    char_crop = thresh[cy:cy + ch, cx:cx + cw]
                    char_norm = preprocess_for_emnist(char_crop)

                    img_t = torch.from_numpy(char_norm).float().to(device).view(1, 1, 28, 28) / 255.0
                    img_t = (img_t - 0.1736) / 0.3317

                    # 运行 TTA 变体多重采样推理
                    avg_prob = predict_with_tta(model, img_t, device)
                    prob_vec = avg_prob.squeeze(0)

                    ar = cw / float(ch)
                    rh = ch / float(max_h)
                    char_probs.append(prob_vec)
                    aspect_ratios.append(ar)
                    relative_heights.append(rh)
                    current_line_probs.append(prob_vec)
                    current_line_aspect_ratios.append(ar)
                    current_line_relative_heights.append(rh)

                    # 收集不确定字符的置信度信息 (Top-3)
                    top3_vals, top3_indices = torch.topk(prob_vec, 3)
                    val1, val2 = top3_vals[0].item(), top3_vals[1].item()

                    # 判定不确定的标准：主候选概率低于 80% 或前两名差距小于 20%
                    if val1 < 0.80 or (val1 - val2) < 0.20:
                        opts = ", ".join(f"'{label_map[top3_indices[i].item()]}' ({top3_vals[i].item():.1%})" for i in range(3))
                        uncertain_info = f"Line {line_idx+1} Char #{len(current_line_probs)}: {opts}"
                        uncertain_infos.append(uncertain_info)
                        print(f"[Warning] 模糊字符警示 - {uncertain_info}")

                line_probs.append(current_line_probs)
                line_aspect_ratios.append(current_line_aspect_ratios)
                line_relative_heights.append(current_line_relative_heights)

            latest_debug_frame = debug_frame
            if debug_enabled:
                cv2.imshow("Debug: Segments", debug_frame)

            # ----------------------------------------------------
            # 【算法升级 3】：送入 Corrector 模块执行概率解码与几何校正
            # ----------------------------------------------------
            raw_parts = []
            decoded_parts = []
            contexts = []
            for probs, ars, rhs in zip(line_probs, line_aspect_ratios, line_relative_heights):
                raw_part, decoded_part, line_context = corrector.decode_sequence(probs, ars, rhs)
                raw_parts.append(raw_part)
                decoded_parts.append(decoded_part)
                contexts.append(line_context)
            
            raw_result = " ".join(raw_parts)
            final_result = " ".join(decoded_parts)
            
            print(f"CNN 原始输出 (ArgMax) : {raw_result}")
            print(f"推断上下文环境 (Context): {' / '.join(contexts)}")
            print(f"[Result] 算法修正后最终结果   : {final_result}")

            if cloud_enabled and cloud_ocr is not None:
                print("[Baidu OCR] 正在上传原始红框整图...")
                cloud_status = "requesting"
                if debug_enabled:
                    cv2.imshow("Debug: Baidu OCR Input", roi)
                def run_cloud_task(image_roi):
                    try:
                        res = cloud_ocr.recognize_ndarray(image_roi)
                        return res, None
                    except Exception as ex:
                        return None, str(ex)
                cloud_future = executor.submit(run_cloud_task, roi.copy())
            elif cloud_ocr is None:
                cloud_status = "unavailable"
            else:
                cloud_status = "disabled"

        # ----------------------------------------------------
        # 按键交互 2：开启/关闭百度手写 OCR
        # ----------------------------------------------------
        elif key == ord('b'):
            if cloud_ocr is None:
                cloud_ocr, cloud_enabled, cloud_state = create_cloud_ocr()
            else:
                cloud_enabled = not cloud_enabled
                cloud_state = "enabled" if cloud_enabled else "disabled"
                print(f"[Baidu OCR] 百度手写识别已{'开启' if cloud_enabled else '关闭'}")
            cloud_result = ""
            cloud_status = cloud_state
            cloud_confidence = None

        # ----------------------------------------------------
        # 按键交互 3：动态切换当前摄像头信号源
        # ----------------------------------------------------
        elif key == ord('c'):
            cap.release()
            cam_list_idx = (cam_list_idx + 1) % len(available_cams)
            current_cam = available_cams[cam_list_idx]
            print(f"[Camera] 正在热切换摄像头设备到索引: {current_cam} ...")
            cap = cv2.VideoCapture(current_cam)
            final_result = ""
            raw_result = ""
            cloud_result = ""
            cloud_status = ""
            cloud_confidence = None
            uncertain_infos = []
            latest_debug_frame = None

        # ----------------------------------------------------
        # 按键交互 4：开启/关闭调试窗口
        # ----------------------------------------------------
        elif key == ord('d'):
            debug_enabled = not debug_enabled
            print(f"[Debug] 调试窗口已{'开启' if debug_enabled else '关闭'}")
            if not debug_enabled:
                close_debug_windows()

        # ----------------------------------------------------
        # 按键交互 5：退出
        # ----------------------------------------------------
        elif key == ord('q'):
            break

    cap.release()
    executor.shutdown(wait=False, cancel_futures=True)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
