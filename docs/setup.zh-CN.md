# Handwritten Character Recognition and Correction — 部署与操作

**简体中文** · [English](setup.en.md) · [日本語](setup.ja.md) · [项目首页](../README.md)

<a id="structure"></a>

## 目录与必要资源

```text
.
├── desktop_app.py
├── camera_detect.py
├── train.py / predict.py / ablation_benchmark.py
├── src/   # CNN、分割、纠错、词表、百度客户端与检查
├── checkpoints/ablation_benchmark_results.json
├── baiduocr/   # 独立历史例程
├── requirements.txt
└── docs/       # 三语言指南与既有图片
```

`data/`、`checkpoints/*.pth` 与 `baiduocr/key.txt` 仍被忽略。可按[权重发布指南](checkpoint.zh-CN.md)从 Release 下载并校验历史检查点；清单纳入 Git，模型不纳入 Git。

<a id="environment"></a>

## 环境与安装

本轮离线检查使用Windows/Python3.10.20、Torch2.5.1+cu121、torchvision0.20.1+cu121、OpenCV4.11.0、NumPy1.26.4。它是检查环境，不是锁定依赖或CUDA性能评测。

```powershell
python -m pip install -r requirements.txt
```

需要Tkinter图形环境、可用摄像头；训练和预测还需EMNIST下载空间及网络。requirements不固定版本，新版PyTorch的AMP／调度器接口需核对，特别是`ReduceLROnPlateau(verbose=True)`兼容性。

<a id="operation"></a>

## 工作台与可选云参考

在仓库根目录启动`python desktop_app.py`，确认控制台显示CNN权重成功加载。摄像头扫描0–3，回退0，请核对实际设备。将内容放入中心方形ROI后按Space；可用Enter/Esc恢复，C换相机，B切换云参考，Q退出。

云端可通过`BAIDU_OCR_API_KEY`和`BAIDU_OCR_SECRET_KEY`环境变量，或忽略跟踪的`baiduocr/key.txt`配置；参照[key.example.txt](../baiduocr/key.example.txt)。有凭证时初始化会请求token；启用后ROI经JPEG/Base64发送百度服务。无凭证则本地路径仍可用。不要把密钥写入文档或提交。

历史入口`camera_detect.py`有不同界面与处理细节，不用它的截图或时序证明主入口所有行为。

<a id="training"></a>

## 训练与历史实验

```powershell
python train.py
python predict.py
python ablation_benchmark.py --samples 5000 --device cuda
```

这些是源码提供的入口，本轮没有执行。训练会备份并覆盖已有权重、生成曲线；predict下载数据并弹图；benchmark覆盖既有JSON。先保留自己的权重与结果，查看[评测问题](technical-notes.zh-CN.md#evaluation)，不能把再次运行旧方法称为解决了偏差。`revert_model.py`会用备份覆盖主权重，也不是默认运行步骤。

<a id="checks"></a>

## 离线检查与故障排查

```powershell
python src/test_segmentation.py
python src/test_new_patterns.py
```

前者3项分割断言，后者7组构造概率用例；不需要相机、训练权重或云凭证，不是识别准确率评估。

| 现象 | 核对 |
| :--- | :--- |
| 初始化失败 | 提供匹配HandwrittenCNN的权重；缺失/损坏会停止本地初始化，不会随机预测。 |
| 字符串被错误纠正 | 分割顺序、全局高度、触发格式、词典外专名。 |
| 暗底预览与结果不同 | 预览和推理使用不同预处理。 |
| 结果显示很慢 | 单工作线程中的CNN及云请求、80 ms轮询、实际硬件；无固定时延保证。 |
| 新环境训练报错 | Torch/torchvision版本、scheduler接口、数据下载和Windows多进程。 |

2026-10-23修复：缺失或不兼容权重立即报错，不使用随机模型继续识别；桌面入口给出初始化错误。7项新增可靠性测试、原3项分割断言和7组纠错用例通过，没有重新训练、相机评测或云请求。后续需要独立真实照片评测与数据版本记录。

```powershell
python -B -m unittest discover -s tests -v
```
