# Handwritten Character Recognition and Correction — 技术说明

**简体中文** · [English](technical-notes.en.md) · [日本語](technical-notes.ja.md) · [项目首页](../README.md)

<a id="source-map"></a>

## 源码与入口

| 路径 | 责任 |
| :--- | :--- |
| [desktop_app.py](../desktop_app.py) | 主工作台、相机、单工作线程、80 ms结果轮询、定格与框显示。 |
| [src/local_ocr.py](../src/local_ocr.py) | 预处理、分割、顺序、11变体推理及词级纠错。 |
| [src/model.py](../src/model.py)／[src/utils.py](../src/utils.py) | CNN、标签、EMNIST方向修正、增强和数据划分。 |
| [src/corrector.py](../src/corrector.py)／[词表](../src/words.txt) | 5459个去重合格词、概率调整、长度候选和格式掩码。 |
| [train.py](../train.py)／[predict.py](../predict.py) | 训练与独立演示评估；predict实际只做9种平移，不是主入口的11变体。 |
| [ablation_benchmark.py](../ablation_benchmark.py) | 单字与合成序列评测，条件和口径见下文。 |
| [src/baidu_ocr.py](../src/baidu_ocr.py) | 凭证读取、HTTP token、JPEG/Base64请求及云端结果解析。 |

`camera_detect.py`是另一套OpenCV界面，虽共用新版分割入口，仍保留独立预处理／后处理；不将它与Tkinter版本视为相同流水线。`baiduocr/`是早期独立例程，主程序使用`src/baidu_ocr.py`。

<a id="preprocessing"></a>

## 图像、字符与版式

`recognize`使用ROI边缘灰度均值127区分明暗背景。亮底背景核按最大边约8%取奇数；CLAHE为2.0/8×8，局部阈值窗口约1.5%且至少3，常数4。暗底不做除法，使用中值／高斯滤波和Otsu。实时预览仍用固定51核和不同阈值，不能认为预览就是推理输入。

分割先估计水平行带，再在带内做2×2闭／开运算、轮廓筛选和迭代合并；宽框按投影谷值切成多个子框，并非只等分两半。没有适用于所有连笔或竖排的保证。过滤参数依赖图像像素大小，ROI宽高处理还含方形假设。

字块按密度膨胀／腐蚀，长边缩到20后置入28×28画布，再用矩进行质心平移：

```math
\Delta x=14-\frac{M_{10}}{M_{00}},\qquad \Delta y=14-\frac{M_{01}}{M_{00}}
```

最近邻方向投票决定按行或列排序，间距超过`max(10,0.55×平均字符尺度)`分词。相对高度用全ROI高度基线，不是逐行最大高度；横排行结果最终以空格连接。

<a id="network"></a>

## CNN与训练配置

网络为`1→32→64→128`三个3×3卷积块，均有BatchNorm/SiLU；前两块2×2池化及0.15 Dropout2d。展平6272→512→62，全连接Dropout为0.5。卷积采用`kaiming_normal_(mode='fan_out', nonlinearity='relu')`初始化，实际激活是SiLU，不使用旧文档的输入通道数简化公式。

主推理把9种一像素平移和±5°旋转合为11个变体，平均Softmax：

```math
\bar p(c\mid x)=\frac1{11}\sum_{k=1}^{11}\mathrm{softmax}(f(T_kx))_c
```

EMNIST ByClass，方向修正为-90°旋转和水平翻转，归一化均值0.1736／标准差0.3317。增强含仿射、透视、弹性、亮度／对比度、模糊、随机擦除。训练90%／验证10%的`random_split`没有固定划分种子，并共享带随机增强的同一个数据集对象；独立测试集使用干净transform。

训练配置：batch128、Adam lr0.001/weight_decay1e-4、平滑交叉熵0.1、验证损失调度patience3/factor0.5，最多25轮、7轮精度不升早停，CUDA下AMP。源码配置和GPU打印字串不能证明实际训练设备、轮数或成绩。

<a id="decoding"></a>

## 概率纠错与交互边界

长宽比阈值0.52、相对高度0.78和上下文偏置重分配概率，再归一化。低置信度判据为Top-1低于0.80或前两名差低于0.20，输出Top-3，不等于校准过的正确率。

词典只搜索**相同长度**，对大小写概率求和后计算对数似然；没有词频先验或神经语言模型：

```math
W^*=\arg\max_{W\in D_N}\sum_{i=1}^{N}\ln\left(p_i(w_i^{lower})+p_i(w_i^{upper})+10^{-12}\right)
```

代码计算`pruned_candidates`，但实际打分循环仍遍历`candidates`，不能声称编辑距离剪枝已加速所有候选，更不能称O(1)解码或无损0.12 ms。

格式由长度与字符计数规则触发，不是完整正则校验；包含18位证件、11位号码、车牌形状、纯字母／数字。只限制输出类别，没有证件校验码、号码有效性或中文省份识别。原始预测触发错模式时会过度约束。

GUI的30 ms是额外回调间隔，80 ms为结果轮询；不保证30 FPS或22.6 ms。单后台任务依次执行本地与云端，启用云参考可延后共同显示。权重加载失败仅打印异常后继续使用初始化参数，使用前必须确认加载成功。

<a id="evaluation"></a>

## 历史结果及评测限制

[历史JSON](../checkpoints/ablation_benchmark_results.json)记载2026-09-14、CUDA、5000单字、50合成序列：

| 阶段 | 记录的单字% | 记录的序列% |
| :--- | ---: | ---: |
| M1 | 84.24 | 16 |
| M2 | 84.98 | 12 |
| M3 | 84.90 | 14 |
| M4 | 86.51 | 46 |
| M5 | 87.17 | 90 |

这些数字不作当前真实相机性能：M1额外随机平移，M2直接用原数据并未调用质心对齐；序列是按标签抽取测试字块，长宽比和高度由真值字符构造，M4还直接使用数字／字母真值语境。M1–M3单字和M4–M5单字来自不同集合，M5序列忽略大小写，不能跨行相减得模块独立收益。流程没有相机预处理与分割误差。

保留[训练曲线](images/training_curves.png)、[混淆矩阵](images/confusion_matrix.png)、[增强图](images/data_augmentation_samples.png)，仅作历史可视化，缺少配套原始逐轮记录、权重绑定和完整复现实验，83%不冒充当前独立测试准确率。

| 历史界面 | 历史界面 |
| :---: | :---: |
| ![数字](images/demo_digit_recognition.png) | ![混合](images/demo_mixed_alphanumeric.png) |
| ![多行](images/demo_multiline_letters.png) | ![号码](images/demo_phone_number_rule.png) |

<a id="implementation-status"></a>

## 验证状态与后续

本轮3项分割断言、7组纠错用例通过，未使用真实模型评测、相机或云端请求。参数、算法和GUI通过源码追踪；本机忽略跟踪的权重存在，不代表Git克隆携带模型。

18份追踪Python文件的静态解析中，17份通过；历史独立例程`baiduocr/demo2.py`第45行存在语法错误，不作为可直接执行的复现入口。主流水线及上述检查脚本解析通过。

优先建立独立验证transform／固定划分，重做测得几何的相机样本评测、统一阶段样本及大小写口径，再修正候选循环和权重加载失败处理。主源码保持不变。个人贡献与外部模型／数据需分别归属，库能力不写成项目创新指标。
