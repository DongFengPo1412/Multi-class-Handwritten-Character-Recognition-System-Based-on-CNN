import requests
import json
import os


def main():
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {
        "client_id": os.environ["BAIDU_OCR_API_KEY"],
        "client_secret": os.environ["BAIDU_OCR_SECRET_KEY"],
        "grant_type": "client_credentials",
    }

    payload = json.dumps("", ensure_ascii=False)
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }

    response = requests.request(
        "POST", url, params=params, headers=headers, data=payload.encode("utf-8")
    )

    response.encoding = "utf-8"
    print(response.text)


if __name__ == '__main__':
    main()
