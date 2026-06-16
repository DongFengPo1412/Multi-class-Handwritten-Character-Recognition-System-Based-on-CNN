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


def merge_bounding_boxes(raw_boxes, box_size):
    if not raw_boxes:
        return []
        
    raw_boxes.sort(key=lambda b: b[0])
    
    merged_boxes = []
    for box in raw_boxes:
        if not merged_boxes:
            merged_boxes.append(box)
            continue
            
        last_box = merged_boxes[-1]
        lx, ly, lw, lh = last_box
        x, y, w, h = box
        
        x_overlap = max(0, min(lx + lw, x + w) - max(lx, x))
        x_overlap_ratio = x_overlap / float(min(lw, w))
        
        if ly < y:
            y_gap = y - (ly + lh)
        else:
            y_gap = ly - (y + h)
            
        combined_h = max(ly + lh, y + h) - min(ly, y)
        max_allowed_h = max(lh, h) * 1.55
        is_vertical_aligned = (
            x_overlap_ratio > 0.25
            or (x >= lx and x + w <= lx + lw)
            or (lx >= x and lx + lw <= x + w)
        ) and (y_gap < max(10, min(lh, h) * 1.4)) and (combined_h <= max_allowed_h)
        
        x_gap = x - (lx + lw)
        is_horizontal_near = (x_gap < 5) and (abs(ly - y) < max(lh, h) * 0.35)
        
        is_nested = (x >= lx - 2 and x + w <= lx + lw + 2 and y >= ly - 2 and y + h <= ly + lh + 2) or \
                    (lx >= x - 2 and lx + lw <= x + w + 2 and ly >= y - 2 and ly + lh <= y + h + 2)
                    
        if is_vertical_aligned or is_horizontal_near or is_nested:
            new_x = min(lx, x)
            new_y = min(ly, y)
            new_w = max(lx + lw, x + w) - new_x
            new_h = max(ly + lh, y + h) - new_y
            merged_boxes[-1] = (new_x, new_y, new_w, new_h)
        else:
            merged_boxes.append(box)
            
    valid_boxes = []
    for (x, y, w, h) in merged_boxes:
        if w >= 5 and h >= 8 and (w * h) >= 80:
            valid_boxes.append((x, y, w, h))
            
    return sort_boxes_reading_order(valid_boxes)


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
                                       
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        raw_boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            if 40 < area < 30000 and 8 < x < box_size - 8 and 8 < y < box_size - 8:
                raw_boxes.append((x, y, w, h))
                
        valid_chars = merge_bounding_boxes(raw_boxes, box_size)
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
