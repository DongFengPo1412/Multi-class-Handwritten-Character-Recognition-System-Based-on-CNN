import os

import requests

API_KEY = os.environ["BAIDU_OCR_API_KEY"]
SECRET_KEY = os.environ["BAIDU_OCR_SECRET_KEY"]

def main():
    url = "https://aip.baidubce.com/rest/2.0/ocr/v1/handwriting?access_token=" + get_access_token()

    payload='url=https%3A%2F%2Fbaidu-ai.bj.bcebos.com%2Focr%2Fhandwriting.jpeg&eng_granularity=letter&detect_direction=false&probability=false&detect_alteration=false'
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Accept': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload.encode("utf-8"))

    response.encoding = "utf-8"
    print(response.text)


def get_access_token():
    """
    使用 AK，SK 生成鉴权签名（Access Token）
    :return: access_token，或是None(如果错误)
    """
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {"grant_type": "client_credentials", "client_id": API_KEY, "client_secret": SECRET_KEY}
    return str(requests.post(url, params=params).json().get("access_token"))

if __name__ == '__main__':
    main()
