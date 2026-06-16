import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

from src.model import HandwrittenCNN
from src.utils import label_map
from src.corrector import HandwrittenCorrector

def preprocess_for_emnist(char_img):
    """
    针对 0.5mm 签字笔优化的 EMNIST 风格预处理
    """
    density = np.sum(char_img > 0) / float(char_img.size)
    kernel = np.ones((2, 2), np.uint8)

    if density < 0.25:
        char_img = cv2.dilate(char_img, kernel, iterations=1)
    elif density > 0.6:
        char_img = cv2.erode(char_img, kernel, iterations=1)

    h, w = char_img.shape
    if h > w:
        new_h, new_w = 20, max(int(20 * w / h), 1)
    else:
        new_h, new_w = max(int(20 * h / w), 1), 20

    char_resz = cv2.resize(char_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    temp = np.zeros((28, 28), dtype=np.uint8)
    pad_h = (28 - new_h) // 2
    pad_w = (28 - new_w) // 2
    temp[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = char_resz

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
    TTA 推理：通过 11 个变体的投票提升鲁棒性
    """
    batch = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            padded = F.pad(img_tensor, (1, 1, 1, 1), mode='constant', value=0)
            shifted = padded[:, :, 1 + dy:1 + dy + 28, 1 + dx:1 + dx + 28]
            batch.append(shifted)

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
            with torch.amp.autocast('cuda'):
                output = model(batch_tensor)
                probs = F.softmax(output, dim=1)
        else:
            output = model(batch_tensor)
            probs = F.softmax(output, dim=1)

    return probs.mean(dim=0, keepdim=True)


def box_center_y(box):
    return box[1] + box[3] / 2.0


def group_boxes_by_reading_lines(boxes):
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
    ordered = []
    for line in group_boxes_by_reading_lines(boxes):
        ordered.extend(line)
    return ordered


def merge_close_runs(runs, max_gap):
    if not runs:
        return []

    merged = [runs[0]]
    for start, end in runs[1:]:
        last_start, last_end = merged[-1]
        if start - last_end - 1 <= max_gap:
            merged[-1] = (last_start, end)
        else:
            merged.append((start, end))
    return merged


def projection_runs(mask):
    runs = []
    start = None
    for idx, value in enumerate(mask):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            runs.append((start, idx - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def find_text_line_bands(binary_img):
    h, w = binary_img.shape
    if h == 0 or w == 0:
        return []

    row_projection = np.sum(binary_img > 0, axis=1).astype(np.float32)
    if row_projection.max() <= 0:
        return []

    kernel_size = max(3, min(15, int(h * 0.025) | 1))
    kernel = np.ones(kernel_size, dtype=np.float32) / float(kernel_size)
    smooth = np.convolve(row_projection, kernel, mode="same")
    active_threshold = max(2.0, min(w * 0.015, smooth.max() * 0.12))

    runs = projection_runs(smooth > active_threshold)
    runs = merge_close_runs(runs, max_gap=max(2, int(h * 0.01)))

    bands = []
    pad = max(2, int(h * 0.006))
    for start, end in runs:
        y1 = max(0, start - pad)
        y2 = min(h - 1, end + pad)
        band = binary_img[y1:y2 + 1, :]
        rows = np.where(np.sum(band > 0, axis=1) > 0)[0]
        cols = np.where(np.sum(band > 0, axis=0) > 0)[0]
        if len(rows) == 0 or len(cols) == 0:
            continue
        top = y1 + int(rows[0])
        bottom = y1 + int(rows[-1])
        if bottom - top + 1 >= 8 and len(cols) >= 4:
            bands.append((max(0, top - 1), min(h - 1, bottom + 1)))

    if not bands:
        return [(0, h - 1)]
    return bands


def contour_boxes_in_band(binary_img, y1, y2, box_size):
    band = binary_img[y1:y2 + 1, :]
    if band.size == 0:
        return []

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    closed = cv2.morphologyEx(band, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        x, local_y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        y = y1 + local_y
        margin_ok = 2 < x < box_size - 2 and 2 < y < box_size - 2
        if 15 < area < 30000 and margin_ok:
            boxes.append((x, y, w, h))
    return boxes


def split_wide_box(binary_img, box):
    x, y, w, h = box
    if w <= 0 or h <= 0:
        return []

    aspect = w / float(h)
    if aspect < 1.18 or w < 18:
        return [box]

    target_char_w = max(9.0, h * 0.56)
    expected_count = int(round(w / target_char_w))
    if expected_count < 2:
        return [box]
    expected_count = min(expected_count, 12)

    crop = binary_img[y:y + h, x:x + w]
    if crop.size == 0:
        return [box]

    projection = np.sum(crop > 0, axis=0).astype(np.float32)
    if projection.max() <= 0:
        return [box]

    kernel_size = max(3, min(9, int(w * 0.04) | 1))
    kernel = np.ones(kernel_size, dtype=np.float32) / float(kernel_size)
    smooth = np.convolve(projection, kernel, mode="same")

    low_threshold = max(1.0, smooth.max() * 0.16)
    low_cols = smooth <= low_threshold
    runs = []
    start = None
    for idx, is_low in enumerate(low_cols):
        if is_low and start is None:
            start = idx
        elif not is_low and start is not None:
            runs.append((start, idx - 1))
            start = None
    if start is not None:
        runs.append((start, w - 1))

    margin = max(4, int(w * 0.06))
    valley_centers = [
        (a + b) // 2
        for a, b in runs
        if b >= margin and a <= w - margin and (b - a + 1) <= max(10, int(w * 0.22))
    ]

    cuts = []
    min_gap = max(6, int(target_char_w * 0.45))
    for k in range(1, expected_count):
        ideal = int(round(k * w / expected_count))
        search_radius = max(7, int(target_char_w * 0.45))
        candidates = [c for c in valley_centers if abs(c - ideal) <= search_radius]
        if candidates:
            cut = min(candidates, key=lambda c: (smooth[c], abs(c - ideal)))
        else:
            left = max(margin, ideal - search_radius)
            right = min(w - margin, ideal + search_radius)
            if right <= left:
                cut = ideal
            else:
                local = smooth[left:right + 1]
                cut = left + int(np.argmin(local))

        if cuts and cut - cuts[-1] < min_gap:
            continue
        if cut < margin or w - cut < margin:
            continue
        cuts.append(cut)

    if not cuts:
        return [box]

    parts = []
    edges = [0] + cuts + [w]
    for left, right in zip(edges[:-1], edges[1:]):
        part = crop[:, left:right]
        cols = np.where(np.sum(part > 0, axis=0) > 0)[0]
        rows = np.where(np.sum(part > 0, axis=1) > 0)[0]
        if len(cols) == 0 or len(rows) == 0:
            continue
        px = x + left + int(cols[0])
        py = y + int(rows[0])
        pw = int(cols[-1] - cols[0] + 1)
        ph = int(rows[-1] - rows[0] + 1)
        if pw >= 4 and ph >= 8 and (pw * ph) >= 45:
            parts.append((px, py, pw, ph))

    if len(parts) < 2:
        return [box]
    return parts


def should_merge(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Calculate overlap and gaps
    x1_max, x2_max = x1 + w1, x2 + w2
    y1_max, y2_max = y1 + h1, y2 + h2
    
    x_overlap = max(0, min(x1_max, x2_max) - max(x1, x2))
    y_overlap = max(0, min(y1_max, y2_max) - max(y1, y2))
    
    # Gaps (0 if they overlap)
    x_gap = max(0, max(x1, x2) - min(x1_max, x2_max))
    y_gap = max(0, max(y1, y2) - min(y1_max, y2_max))
    
    min_w = min(w1, w2)
    max_w = max(w1, w2)
    min_h = min(h1, h2)
    max_h = max(h1, h2)
    
    x_overlap_ratio = x_overlap / float(min_w) if min_w > 0 else 0
    y_overlap_ratio = y_overlap / float(min_h) if min_h > 0 else 0
    
    # 1. Nested/Containing relationship (one box is inside another)
    # Include some tolerance (3 pixels)
    is_nested = (x1 >= x2 - 3 and x1_max <= x2_max + 3 and y1 >= y2 - 3 and y1_max <= y2_max + 3) or \
                (x2 >= x1 - 3 and x2_max <= x1_max + 3 and y2 >= y1 - 3 and y2_max <= y1_max + 3)
    if is_nested:
        return True
        
    # 2. Vertical alignment (e.g., dot of 'i' or 'j', or vertical parts of broken letters)
    # High horizontal overlap (X overlap ratio > 0.4)
    if x_overlap_ratio > 0.4 or (x1 >= x2 - 2 and x1_max <= x2_max + 2) or (x2 >= x1 - 2 and x2_max <= x1_max + 2):
        combined_h = max(y1_max, y2_max) - min(y1, y2)
        # Check if they are vertically close
        # Combined height should not exceed 2.2 * max_h to avoid merging different text lines
        if y_gap < max(15, min_h * 1.8) and combined_h <= max_h * 2.2:
            return True
            
    # 3. Horizontal splitting (broken stroke parts of the same letter side-by-side)
    # They should overlap vertically (same line)
    if y_overlap_ratio > 0.5:
        # If they actually overlap horizontally (x_overlap > 0), they must be part of the same letter
        if x_overlap > 0:
            return True
        # If they are side-by-side:
        # - Extremely close: gap <= 3 pixels
        if x_gap <= 3:
            return True
        # - Close, and one of them is very thin/small (likely a broken stroke fragment, not a full character)
        if x_gap <= 6 and (min_w <= 5 or (min_w * min_h) < 120):
            return True
            
    # 4. Diagonal close proximity for tiny fragments
    if x_gap <= 2 and y_gap <= 2 and (min_w * min_h < 100 or max_w * max_h < 150):
        return True
        
    return False


def merge_bounding_boxes(raw_boxes, box_size, binary_img=None):
    if not raw_boxes:
        return []
        
    # Iterative merge algorithm
    current_boxes = list(raw_boxes)
    while True:
        merged_any = False
        n = len(current_boxes)
        used = [False] * n
        next_boxes = []
        
        for i in range(n):
            if used[i]:
                continue
            box1 = current_boxes[i]
            for j in range(i + 1, n):
                if used[j]:
                    continue
                box2 = current_boxes[j]
                
                if should_merge(box1, box2):
                    # Merge them
                    x1, y1, w1, h1 = box1
                    x2, y2, w2, h2 = box2
                    new_x = min(x1, x2)
                    new_y = min(y1, y2)
                    new_w = max(x1 + w1, x2 + w2) - new_x
                    new_h = max(y1 + h1, y2 + h2) - new_y
                    box1 = (new_x, new_y, new_w, new_h)
                    used[j] = True
                    merged_any = True
            
            next_boxes.append(box1)
            used[i] = True
            
        current_boxes = next_boxes
        if not merged_any:
            break
            
    # Filter and sort
    valid_boxes = []
    for (x, y, w, h) in current_boxes:
        if w >= 5 and h >= 8 and (w * h) >= 80:
            valid_boxes.append((x, y, w, h))

    if binary_img is not None:
        split_boxes = []
        for line in group_boxes_by_reading_lines(valid_boxes):
            for box in line:
                split_boxes.extend(split_wide_box(binary_img, box))
        valid_boxes = split_boxes

    return sort_boxes_reading_order(valid_boxes)


def segment_character_boxes(binary_img, raw_boxes, box_size):
    line_boxes = []
    for y1, y2 in find_text_line_bands(binary_img):
        raw_line_boxes = contour_boxes_in_band(binary_img, y1, y2, box_size)
        if raw_line_boxes:
            line_boxes.extend(merge_bounding_boxes(raw_line_boxes, box_size, binary_img=binary_img))

    if line_boxes:
        return sort_boxes_reading_order(line_boxes)

    return merge_bounding_boxes(raw_boxes, box_size, binary_img=binary_img)



class LocalOCRResult:
    def __init__(self, raw_text: str, character_count: int, line_count: int, corrected_text: str, contexts: list[str], threshold: np.ndarray, uncertain: list[str], boxes: list = None):
        self.raw_text = raw_text
        self.character_count = character_count
        self.line_count = line_count
        self.corrected_text = corrected_text
        self.contexts = contexts
        self.threshold = threshold
        self.uncertain = uncertain
        self.boxes = boxes or []


class LocalOCRRecognizer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = HandwrittenCNN(num_classes=62).to(self.device)
        
        project_root = Path(__file__).resolve().parent.parent
        weights_path = project_root / "checkpoints" / "emnist_model.pth"
        if not weights_path.exists():
            weights_path = Path("checkpoints/emnist_model.pth")
            
        try:
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
            print(f"[+] LocalOCRRecognizer: Loaded CNN weights from {weights_path}.")
        except Exception as e:
            print(f"[-] LocalOCRRecognizer: Failed to load CNN weights: {e}")
            
        self.model.eval()
        self.corrector = HandwrittenCorrector()
        
        # Model Warmup to eliminate first-use inference latency
        try:
            with torch.no_grad():
                dummy = torch.zeros(1, 1, 28, 28).to(self.device)
                if self.device.type == 'cuda':
                    with torch.amp.autocast('cuda'):
                        self.model(dummy)
                else:
                    self.model(dummy)
            print("[+] LocalOCRRecognizer: Model warmup completed successfully.")
        except Exception as e:
            print(f"[-] LocalOCRRecognizer: Model warmup failed: {e}")

    def recognize(self, roi: np.ndarray) -> LocalOCRResult:
        box_size = roi.shape[0]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        bg = cv2.GaussianBlur(gray, (51, 51), 0)
        gray_no_shadow = cv2.divide(gray, bg, scale=255)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray_no_shadow)
        
        blur = cv2.GaussianBlur(cv2.medianBlur(enhanced, 3), (3, 3), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 4)
        
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
                                       
        # 使用形态学闭运算连接断裂的笔画以辅助寻找轮廓
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        raw_boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            if 40 < area < 30000 and 8 < x < box_size - 8 and 8 < y < box_size - 8:
                raw_boxes.append((x, y, w, h))
                
        valid_chars = segment_character_boxes(thresh, raw_boxes, box_size)
        valid_lines = group_boxes_by_reading_lines(valid_chars)
        
        character_count = len(valid_chars)
        line_count = len(valid_lines)
        
        if character_count == 0:
            return LocalOCRResult(
                raw_text="",
                character_count=0,
                line_count=0,
                corrected_text="",
                contexts=[],
                threshold=thresh,
                uncertain=[],
                boxes=[]
            )
            
        max_h = max(b[3] for b in valid_chars)
        
        line_probs = []
        line_aspect_ratios = []
        line_relative_heights = []
        uncertain_infos = []
        
        for line_idx, line_boxes in enumerate(valid_lines):
            current_line_probs = []
            current_line_aspect_ratios = []
            current_line_relative_heights = []
            
            for cx, cy, cw, ch in line_boxes:
                char_crop = thresh[cy:cy + ch, cx:cx + cw]
                char_norm = preprocess_for_emnist(char_crop)
                
                img_t = torch.from_numpy(char_norm).float().to(self.device).view(1, 1, 28, 28) / 255.0
                img_t = (img_t - 0.1736) / 0.3317
                
                avg_prob = predict_with_tta(self.model, img_t, self.device)
                prob_vec = avg_prob.squeeze(0)
                
                ar = cw / float(ch)
                rh = ch / float(max_h)
                
                current_line_probs.append(prob_vec)
                current_line_aspect_ratios.append(ar)
                current_line_relative_heights.append(rh)
                
                top3_vals, top3_indices = torch.topk(prob_vec, 3)
                val1, val2 = top3_vals[0].item(), top3_vals[1].item()
                if val1 < 0.80 or (val1 - val2) < 0.20:
                    opts = ", ".join(f"'{label_map[top3_indices[i].item()]}' ({top3_vals[i].item():.1%})" for i in range(3))
                    uncertain_info = f"Line {line_idx+1} Char #{len(current_line_probs)}: {opts}"
                    uncertain_infos.append(uncertain_info)
                    
            line_probs.append(current_line_probs)
            line_aspect_ratios.append(current_line_aspect_ratios)
            line_relative_heights.append(current_line_relative_heights)
            
        raw_parts = []
        decoded_parts = []
        contexts = []
        for probs, ars, rhs in zip(line_probs, line_aspect_ratios, line_relative_heights):
            raw_part, decoded_part, line_context = self.corrector.decode_sequence(probs, ars, rhs)
            raw_parts.append(raw_part)
            decoded_parts.append(decoded_part)
            contexts.append(line_context)
            
        raw_result = " ".join(raw_parts)
        final_result = " ".join(decoded_parts)
        
        return LocalOCRResult(
            raw_result,
            character_count,
            line_count,
            final_result,
            contexts,
            thresh,
            uncertain_infos,
            valid_chars
        )
