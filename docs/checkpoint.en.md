# Checkpoint and reproduction requirements

[简体中文](checkpoint.zh-CN.md) · [English](checkpoint.en.md) · [日本語](checkpoint.ja.md) · [Overview](../README_EN.md)

A retained historical local checkpoint strictly loads into current `HandwrittenCNN(num_classes=62)`, but public binary distribution requires explicit permission. **No public download is currently available.** File timestamps or structure checks are not retraining or accuracy evaluation.

Supply compatible weights at `checkpoints/emnist_model.pth`, then run `python desktop_app.py`. Labels are 0–9/A–Z/a–z with 28x28 single-character input. The pending manifest makes the downloader refuse before network access; it does not create a random replacement. Existing weights are not overwritten.

If publication is authorized, a Release URL, byte size and SHA-256 will be provided and verified by downloading. Raw data, private images and credentials are excluded. The primary 11-view TTA and the separate historical translation-only predict.py are different paths; old figures are not fresh results.

[Manifest](../checkpoints/checkpoint-manifest.json) · [Downloader](../tools/download_checkpoint.py)
