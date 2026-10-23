# 检查点与复现条件

[简体中文](checkpoint.zh-CN.md) · [English](checkpoint.en.md) · [日本語](checkpoint.ja.md) · [Overview](../README.md)

本地留存的历史模型已检查能严格加载到当前 `HandwrittenCNN(num_classes=62)`，但公开分发二进制模型需要明确许可，**目前没有公开下载链接**。此前文件时间或结构检查不等于重新训练或准确率评测。

准备已有兼容权重到 `checkpoints/emnist_model.pth` 后运行 `python desktop_app.py`。标签顺序为 `0–9/A–Z/a–z`，输入为 28×28 单字图。当前清单明确标为 pending，下载工具会在访问网络前拒绝执行；它不生成或随机替代模型。已有权重不会自动覆盖。

未来若获准发布，将同时提供 Release 地址、字节数和 SHA-256，并进行实际下载校验；原始数据、私有图像与凭据不随发布。主入口11视图TTA与独立历史predict.py的平移分支需区分，不能复用旧图声称新成绩。

[Manifest](../checkpoints/checkpoint-manifest.json) · [Downloader](../tools/download_checkpoint.py)
