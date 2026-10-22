# Handwritten Character Recognition and Correction — 導入と操作

[简体中文](setup.zh-CN.md) · [English](setup.en.md) · **日本語** · [概要](../README_JA.md)

<a id="structure"></a>

## 構成と資源

```text
.
├── desktop_app.py
├── camera_detect.py
├── train.py / predict.py / ablation_benchmark.py
├── src/   # CNN、分割、補正、辞書、クラウド、検査
├── checkpoints/ablation_benchmark_results.json
├── baiduocr/   # 過去の独立例
├── requirements.txt
└── docs/       # 三言語ガイドと既存画像
```

`data/`、`checkpoints/*.pth`、`baiduocr/key.txt`はGit対象外です。対応する62クラス重みを別途用意し、読込失敗後の出力を学習済み結果とは扱いません。

<a id="environment"></a>

## 環境と導入

検査環境はWindows／Python3.10.20、Torch2.5.1+cu121、torchvision0.20.1+cu121、OpenCV4.11.0、NumPy1.26.4です。固定依存やCUDA性能評価ではありません。

```powershell
python -m pip install -r requirements.txt
```

Tkinter・画面・カメラが必要です。学習・予測にはEMNIST取得用の容量と通信が必要です。依存版は固定されておらず、新TorchのAMP・scheduler、特に`ReduceLROnPlateau(verbose=True)`の互換性を確認してください。

<a id="operation"></a>

## 画面操作と任意のクラウド

ルートで`python desktop_app.py`を実行し、CNN読込成功を確認します。カメラ0–3を走査し、見つからなければ0を使います。中央の正方形ROIに置き、Spaceで認識／再開、Enter/Escで再開、Cでカメラ、Bでクラウド、Qで終了です。

認証情報は`BAIDU_OCR_API_KEY`／`BAIDU_OCR_SECRET_KEY`、またはGit対象外の`baiduocr/key.txt`へ設定します。[key.example.txt](../baiduocr/key.example.txt)を参照してください。有効な情報があれば初期化時にトークンを取得し、有効化した認識ではROIをJPEG/Base64で百度へ送信します。認証情報なしでもローカル処理を使えます。秘密はコミットしません。

`camera_detect.py`は別の過去のUI・処理経路で、画面や時間を主入口の全挙動の根拠にはできません。

<a id="training"></a>

## 学習と過去の実験

```powershell
python train.py
python predict.py
python ablation_benchmark.py --samples 5000 --device cuda
```

提供された入口で、今回は未実行です。学習は重みをバックアップ・上書きし図を生成、predictはデータ取得と図表示、benchmarkはJSONを上書きします。先に重み・結果を保存し、[評価条件](technical-notes.ja.md#evaluation)を確認してください。旧手法を再実行しても偏りは解消しません。`revert_model.py`も主重みをバックアップで上書きするため、標準導入には含めません。

<a id="checks"></a>

## オフライン検査と切り分け

```powershell
python src/test_segmentation.py
python src/test_new_patterns.py
```

前者は分割3件、後者は構成した確率の7ケースです。カメラ・重み・クラウド情報は不要で、認識精度の試験ではありません。

| 現象 | 確認 |
| :--- | :--- |
| 初期化エラー | 対応するHandwrittenCNN重みを用意します。欠落・破損は停止し、無作為推論を行いません。 |
| 誤補正 | 分割順、全体高さ、書式判定、辞書外の名前。 |
| 暗背景の結果とプレビューが異なる | 前処理が別実装です。 |
| 表示まで遅い | 同一ワーカー内のCNN・クラウド、80 ms確認、実機性能。固定時間ではありません。 |
| 新環境で学習失敗 | Torch／torchvision、scheduler、ダウンロード、Windows多重プロセス。 |

2026-10-23修正：重み欠落・非互換は初期化エラーとして終了し、デスクトップに通知します。無作為モデルでの推論はしません。新規信頼性7件、既存分割3件・補正7件が通過しました。再学習、実撮影評価、クラウド呼出しは行っていません。実画像評価とデータ・重み版の記録は今後必要です。

```powershell
python -B -m unittest discover -s tests -v
```
