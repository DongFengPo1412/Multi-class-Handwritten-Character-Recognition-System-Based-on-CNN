# 阿里云 OCR 接入说明

本文档用于把当前目录中的阿里云 OCR 调用方式迁移到其他 Python 项目中。

## 当前验证结果

当前已使用 `newassesskey.txt` 中的 AccessKey 成功调用阿里云 OCR 接口。

- 产品：阿里云 OCR API
- 接口：`RecognizeAllText`
- Endpoint：`ocr-api.cn-hangzhou.aliyuncs.com`
- SDK：`alibabacloud_ocr_api20210707`
- 测试状态：成功返回 HTTP `200`
- 测试识别结果：`OpenAPI 从这里开始，集成阿里云`

## 文件说明

当前目录关键文件：

- `DEMO.py`：阿里云官方生成的 OCR 调用 demo
- `newassesskey.txt`：可用的 AccessKey 文件
- `.venv`：本地 Python 虚拟环境，已安装所需 SDK

不要使用旧的 `assesskey.txt`，之前测试返回 `InvalidAccessKeyId`。

## AccessKey 文件格式

`newassesskey.txt` 当前不是纯两行 key，而是两行「字段名 + 空格 + 值」格式：

```text
accessKeyId <你的 AccessKey ID>
accessKeySecret <你的 AccessKey Secret>
```

读取时必须取每一行空格后面的值，不能把整行直接当作 key。

## 安装依赖

在其他 Python 项目中安装：

```bash
pip install alibabacloud_ocr_api20210707 alibabacloud_credentials alibabacloud_tea_openapi
```

如果项目使用虚拟环境：

```bash
python -m venv .venv
.venv\Scripts\python -m pip install alibabacloud_ocr_api20210707 alibabacloud_credentials alibabacloud_tea_openapi
```

## 推荐读取 Key 的方式

可以把下面函数复制到其他项目里：

```python
from pathlib import Path


def load_aliyun_keys(path: str = "newassesskey.txt") -> tuple[str, str]:
    lines = [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if len(lines) < 2:
        raise ValueError("AccessKey 文件至少需要两行：accessKeyId 和 accessKeySecret")

    values = []
    for line in lines[:2]:
        parts = line.split(None, 1)
        values.append(parts[1].strip() if len(parts) == 2 else parts[0].strip())

    return values[0], values[1]
```

## 最小可用 OCR 调用代码

下面代码会读取 `newassesskey.txt`，调用阿里云 OCR 识别图片 URL。

```python
import json
import os
from pathlib import Path

from alibabacloud_ocr_api20210707.client import Client as OcrClient
from alibabacloud_credentials.client import Client as CredentialClient
from alibabacloud_tea_openapi import models as open_api_models
from alibabacloud_ocr_api20210707 import models as ocr_models
from alibabacloud_tea_util import models as util_models


def load_aliyun_keys(path: str = "newassesskey.txt") -> tuple[str, str]:
    lines = [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(lines) < 2:
        raise ValueError("AccessKey 文件至少需要两行")

    values = []
    for line in lines[:2]:
        parts = line.split(None, 1)
        values.append(parts[1].strip() if len(parts) == 2 else parts[0].strip())

    return values[0], values[1]


def create_ocr_client() -> OcrClient:
    access_key_id, access_key_secret = load_aliyun_keys()

    os.environ["ALIBABA_CLOUD_ACCESS_KEY_ID"] = access_key_id
    os.environ["ALIBABA_CLOUD_ACCESS_KEY_SECRET"] = access_key_secret

    credential = CredentialClient()
    config = open_api_models.Config(credential=credential)
    config.endpoint = "ocr-api.cn-hangzhou.aliyuncs.com"
    return OcrClient(config)


def recognize_all_text(image_url: str) -> dict:
    client = create_ocr_client()
    request = ocr_models.RecognizeAllTextRequest(
        url=image_url,
        type="General",
    )
    runtime = util_models.RuntimeOptions()
    response = client.recognize_all_text_with_options(request, runtime)
    return response.to_map()


if __name__ == "__main__":
    result = recognize_all_text(
        "https://img.alicdn.com/imgextra/i1/O1CN01vCBpJz1hGYOgIkp4l_!!6000000004250-0-tps-564-294.jpg"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
```

## 返回结果位置

常用文本结果在：

```python
result["body"]["Data"]["Content"]
```

如果需要每个文字块：

```python
blocks = result["body"]["Data"]["SubImages"][0]["BlockInfo"]["BlockDetails"]
for block in blocks:
    print(block["BlockContent"], block["BlockConfidence"])
```

## 在其他项目中的接入建议

建议把 key 读取和 OCR 调用封装成单独模块，例如：

```text
your_project/
  aliyun_ocr.py
  newassesskey.txt
```

业务代码只调用：

```python
from aliyun_ocr import recognize_all_text

result = recognize_all_text(image_url)
text = result["body"]["Data"]["Content"]
```

## 常见错误

### InvalidAccessKeyId

含义：AccessKey ID 无效、被删除、被禁用，或读取格式错了。

本项目之前用旧 `assesskey.txt` 测试时就出现过这个错误。现在可用的是 `newassesskey.txt`。

### 读取 key 后仍然失败

优先检查：

- 是否取了空格后面的值，而不是整行
- `accessKeyId` 和 `accessKeySecret` 是否顺序反了
- 是否复制了额外空格或不可见字符
- 阿里云账号是否开通 OCR API 权限
- 当前 AccessKey 是否处于启用状态

### 如果使用 STS 临时凭证

如果 key 文件中以后变成三行，并包含 `SecurityToken`，则还需要额外设置：

```python
os.environ["ALIBABA_CLOUD_SECURITY_TOKEN"] = security_token
```

当前 `newassesskey.txt` 是长期 AK/SK，两行即可。

## 安全注意事项

- 不要把 `newassesskey.txt` 提交到 Git 仓库
- 不要在日志中打印 AccessKey Secret
- 生产环境更推荐使用环境变量、密钥管理服务或 RAM Role
- 如果 key 泄露，立刻在阿里云控制台禁用并重新创建

