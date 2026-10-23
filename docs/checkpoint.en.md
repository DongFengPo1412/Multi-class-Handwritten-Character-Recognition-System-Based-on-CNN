# Historical checkpoint and reproduction

[简体中文](checkpoint.zh-CN.md) · [English](checkpoint.en.md) · [日本語](checkpoint.ja.md) · [Overview](../README_EN.md)

A retained local 62-class checkpoint is published for reproducing inference and the workbench. It was not retrained or re-evaluated in this maintenance. The original file timestamp is 2026-06-16; that does not reconstruct its full training environment or exact split.

The checkpoint strictly loads into current HandwrittenCNN(num_classes=62). A zero input of 1x1x28x28 produces finite 1x62 output. This checks structure/numerical usability, not recognition accuracy. No raw training data, private images or credentials are released.

[Release](https://github.com/DongFengPo1412/Multi-class-Handwritten-Character-Recognition-System-Based-on-CNN/releases/tag/emnist62-checkpoint-20260616) · [Manifest](../checkpoints/checkpoint-manifest.json)

| Artifact | Value |
| --- | --- |
| Name | `emnist_model.pth` |
| Bytes | 13356700 |
| SHA-256 | `2cdf4b6300027ca1fb8202832fcfea37906b10b2094acf464f6c81b899113928` |
| Architecture | `src.model.HandwrittenCNN(num_classes=62)` |

## Install and verify

```powershell
$task = Join-Path $env:USERPROFILE ('.codex\cache\tasks\ocr-setup-' + [guid]::NewGuid().ToString('N'))
python -B tools/download_checkpoint.py --cache-dir "$task\download"
python -B desktop_app.py
```

The downloader installs only after checking byte size and SHA-256. A differing existing checkpoint is preserved and causes an error; it is never automatically overwritten. Downloads remain in the explicit cache. Weights stay Git-ignored and are distributed as a Release asset.

## Model and evaluation boundaries

Labels follow 0–9/A–Z/a–z. The primary desktop path uses 11-view TTA; the separate historical predict.py has translation variants only. Historical curves are not new results for the maintained code. No new writer-independent or real-camera accuracy was measured.
