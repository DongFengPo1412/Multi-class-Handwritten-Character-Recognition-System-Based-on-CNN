# 百度手写文字 OCR 调用示例

本项目通过百度智能云“手写文字识别”接口识别本地图片：

`POST https://aip.baidubce.com/rest/2.0/ocr/v1/handwriting`

## 已验证结果

2026-06-10 使用 `test.jpg` 真实调用成功：

```text
LIKE
YOU
```

接口返回 2 行文字，图片方向为正向。两行平均置信度分别约为
`0.99997` 和 `0.84656`。本次完整响应保存在 `result.json`。

## 环境要求

- Python 3.9+
- `requests`

安装依赖：

```powershell
python -m pip install -r requirements.txt
```

## 配置凭证

推荐使用环境变量：

```powershell
$env:BAIDU_OCR_API_KEY = "你的 API Key"
$env:BAIDU_OCR_SECRET_KEY = "你的 Secret Key"
```

也可以复制 `key.example.txt` 为 `key.txt`，然后填入凭证：

```text
API Key    你的 API Key
Secret Key 你的 Secret Key
```

脚本优先读取环境变量；未设置时，默认读取脚本目录下的 `key.txt`。

## 运行

只输出识别文字：

```powershell
python baidu_ocr.py test.jpg
```

同时保存百度接口的完整 JSON：

```powershell
python baidu_ocr.py test.jpg --json-output result.json
```

指定其他凭证文件：

```powershell
python baidu_ocr.py test.jpg --key-file C:\path\to\key.txt
```

## 调用流程

1. 使用 API Key 和 Secret Key 获取 Access Token。
2. 读取本地图片并进行 Base64 编码。
3. 以 `application/x-www-form-urlencoded` 格式调用手写文字识别接口。
4. 从响应的 `words_result` 中提取文字。

脚本默认启用图片方向检测和行置信度，并使用中英文混合识别。

## 常见错误

- `未找到百度 OCR 凭证`：设置环境变量或创建 `key.txt`。
- `获取 Access Token 失败`：检查 API Key、Secret Key 以及应用状态。
- `OCR 调用失败 [17]`：通常表示接口每日请求量已用完。
- `OCR 调用失败 [18]`：通常表示短时间请求过于频繁。
- 图片错误：确认格式为 JPG、JPEG、PNG 或 BMP，最长边不超过 4096px，
  Base64 和 URL 编码后的请求内容不超过 8 MB。

## 安全提醒

不要把 `key.txt` 或真实 AK/SK 提交到版本库。当前凭证曾明文保存在原始
demo 中，建议在百度智能云控制台轮换 Secret Key 后再用于正式环境。
