import os
import re
import base64
import requests
import cv2
from pathlib import Path

class BaiduOCRUnavailable(Exception):
    pass

class BaiduOCRLine:
    def __init__(self, text: str, confidence: float | None):
        self.text = text
        self.confidence = confidence

class BaiduOCRResponse:
    def __init__(self, text: str, average_confidence: float | None, lines: list[BaiduOCRLine]):
        self.text = text
        self.average_confidence = average_confidence
        self.lines = lines

class BaiduOCRClient:
    def __init__(self, key_file=None):
        self.api_key = os.getenv("BAIDU_OCR_API_KEY")
        self.secret_key = os.getenv("BAIDU_OCR_SECRET_KEY")
        self.access_token = None
        
        if not (self.api_key and self.secret_key):
            if key_file is None:
                project_root = Path(__file__).resolve().parent.parent
                key_file = project_root / "baiduocr" / "key.txt"
            else:
                key_file = Path(key_file)
                
            if key_file.is_file():
                try:
                    content = key_file.read_text(encoding="utf-8")
                    api_match = re.search(r"API\s*Key\s+(\S+)", content, re.IGNORECASE)
                    secret_match = re.search(r"Secret\s*Key\s+(\S+)", content, re.IGNORECASE)
                    if api_match and secret_match:
                        self.api_key = api_match.group(1)
                        self.secret_key = secret_match.group(1)
                except Exception as e:
                    raise BaiduOCRUnavailable(f"读取 key.txt 失败: {e}")
                    
        if not (self.api_key and self.secret_key):
            raise BaiduOCRUnavailable(
                "未找到百度 OCR 凭证。请设置 BAIDU_OCR_API_KEY 和 "
                "BAIDU_OCR_SECRET_KEY，或提供 baiduocr/key.txt。"
            )

    def ensure_ready(self):
        if self.access_token:
            return
        
        url = "https://aip.baidubce.com/oauth/2.0/token"
        params = {
            "grant_type": "client_credentials",
            "client_id": self.api_key,
            "client_secret": self.secret_key,
        }
        try:
            response = requests.post(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            token = data.get("access_token")
            if not token:
                msg = data.get("error_description") or data.get("error") or str(data)
                raise BaiduOCRUnavailable(f"获取 Access Token 失败: {msg}")
            self.access_token = token
        except Exception as e:
            if isinstance(e, BaiduOCRUnavailable):
                raise e
            raise BaiduOCRUnavailable(f"获取 Access Token 连接失败: {e}")

    def recognize_ndarray(self, roi_image):
        if not self.access_token:
            self.ensure_ready()
            
        ret, encoded_img = cv2.imencode('.jpg', roi_image)
        if not ret:
            raise RuntimeError("图像编码失败")
            
        image_base64 = base64.b64encode(encoded_img.tobytes()).decode("ascii")
        
        url = "https://aip.baidubce.com/rest/2.0/ocr/v1/handwriting"
        params = {"access_token": self.access_token}
        data = {
            "image": image_base64,
            "language_type": "CHN_ENG",
            "detect_direction": "true",
            "probability": "true",
        }
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        
        try:
            response = requests.post(url, params=params, data=data, headers=headers, timeout=30)
            response.raise_for_status()
            res_data = response.json()
        except Exception as e:
            raise RuntimeError(f"OCR 请求失败: {e}")
            
        if "error_code" in res_data:
            raise RuntimeError(f"OCR 调用失败 [{res_data['error_code']}]：{res_data.get('error_msg', '未知错误')}")
            
        words_result = res_data.get("words_result", [])
        lines = []
        text_parts = []
        for item in words_result:
            words = item.get("words", "")
            text_parts.append(words)
            prob_dict = item.get("probability", {})
            avg_prob = prob_dict.get("average", None)
            lines.append(BaiduOCRLine(words, avg_prob))
            
        combined_text = " ".join(text_parts)
        valid_probs = [line.confidence for line in lines if line.confidence is not None]
        avg_confidence = sum(valid_probs) / len(valid_probs) if valid_probs else None
        
        return BaiduOCRResponse(combined_text, avg_confidence, lines)
