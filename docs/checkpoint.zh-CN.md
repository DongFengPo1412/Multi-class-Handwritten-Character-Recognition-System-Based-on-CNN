# 历史权重与复现

[简体中文](checkpoint.zh-CN.md) · [English](checkpoint.en.md) · [日本語](checkpoint.ja.md) · [Overview](../README.md)

本项目现在发布本地留存的 62 类历史检查点，供复现推理界面；不是本轮重新训练或重新评测的模型。原文件记录时间为 2026-06-16，不能据此重建当时的完整训练环境与数据划分。

检查点通过当前 `HandwrittenCNN(num_classes=62)` 严格加载，并用 1×1×28×28 零输入检查输出为有限的 1×62；这只验证结构和数值可用性，不是识别精度。原始训练数据、隐私图像和凭证不随此发布。

[Release](https://github.com/DongFengPo1412/Multi-class-Handwritten-Character-Recognition-System-Based-on-CNN/releases/tag/emnist62-checkpoint-20260616) · [Manifest](../checkpoints/checkpoint-manifest.json)

| Artifact | Value |
| --- | --- |
| Name | `emnist_model.pth` |
| Bytes | 13356700 |
| SHA-256 | `2cdf4b6300027ca1fb8202832fcfea37906b10b2094acf464f6c81b899113928` |
| Architecture | `src.model.HandwrittenCNN(num_classes=62)` |

## 安装与校验

```powershell
$task = Join-Path $env:USERPROFILE '.codex\cache\tasks\ocr-setup-<unique-id>'
python -B tools/download_checkpoint.py --cache-dir "$task\download"
python -B desktop_app.py
```

脚本只在大小与 SHA-256 相符后安装；已有不同权重会报错并保留，不自动覆盖。下载文件保留在指定缓存；模型仍被 Git 忽略，来自 Release 资产。

## 模型与评测边界

标签顺序为 `0–9/A–Z/a–z`。主桌面入口使用 11 视图 TTA；独立历史 predict.py 只有平移分支，应按入口区分。历史曲线不能作为当前修订版的新成绩，也没有跨书写者或真实相机准确率的新测量。
