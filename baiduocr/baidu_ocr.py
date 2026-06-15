import argparse
import base64
import json
import os
import re
import sys
from pathlib import Path

import requests


TOKEN_URL = "https://aip.baidubce.com/oauth/2.0/token"
HANDWRITING_URL = "https://aip.baidubce.com/rest/2.0/ocr/v1/handwriting"
TIMEOUT_SECONDS = 30


class BaiduOcrError(RuntimeError):
    pass


def load_credentials(key_file: Path) -> tuple[str, str]:
    api_key = os.getenv("BAIDU_OCR_API_KEY")
    secret_key = os.getenv("BAIDU_OCR_SECRET_KEY")
    if api_key and secret_key:
        return api_key, secret_key

    if not key_file.is_file():
        raise BaiduOcrError(
            "未找到百度 OCR 凭证。请设置 BAIDU_OCR_API_KEY 和 "
            "BAIDU_OCR_SECRET_KEY，或提供 key.txt。"
        )

    content = key_file.read_text(encoding="utf-8")
    api_match = re.search(r"API\s*Key\s+(\S+)", content, re.IGNORECASE)
    secret_match = re.search(r"Secret\s*Key\s+(\S+)", content, re.IGNORECASE)
    if not api_match or not secret_match:
        raise BaiduOcrError(f"无法从 {key_file} 解析 API Key 和 Secret Key。")
    return api_match.group(1), secret_match.group(1)


def get_access_token(api_key: str, secret_key: str) -> str:
    response = requests.post(
        TOKEN_URL,
        params={
            "grant_type": "client_credentials",
            "client_id": api_key,
            "client_secret": secret_key,
        },
        timeout=TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    data = response.json()
    token = data.get("access_token")
    if not token:
        message = data.get("error_description") or data.get("error") or str(data)
        raise BaiduOcrError(f"获取 Access Token 失败：{message}")
    return token


def recognize_handwriting(image_path: Path, access_token: str) -> dict:
    if not image_path.is_file():
        raise BaiduOcrError(f"图片不存在：{image_path}")

    image_base64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    response = requests.post(
        HANDWRITING_URL,
        params={"access_token": access_token},
        data={
            "image": image_base64,
            "language_type": "CHN_ENG",
            "detect_direction": "true",
            "probability": "true",
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    data = response.json()
    if "error_code" in data:
        raise BaiduOcrError(
            f"OCR 调用失败 [{data['error_code']}]：{data.get('error_msg', '未知错误')}"
        )
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="调用百度手写文字识别 API")
    parser.add_argument("image", type=Path, help="需要识别的本地图片")
    parser.add_argument(
        "--key-file",
        type=Path,
        default=Path(__file__).with_name("key.txt"),
        help="凭证文件路径，默认读取脚本同目录下的 key.txt",
    )
    parser.add_argument("--json-output", type=Path, help="保存完整接口响应 JSON")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        api_key, secret_key = load_credentials(args.key_file)
        token = get_access_token(api_key, secret_key)
        result = recognize_handwriting(args.image, token)
    except (BaiduOcrError, requests.RequestException, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1

    words = [item.get("words", "") for item in result.get("words_result", [])]
    print("\n".join(words))

    if args.json_output:
        args.json_output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\n完整结果已保存到：{args.json_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
