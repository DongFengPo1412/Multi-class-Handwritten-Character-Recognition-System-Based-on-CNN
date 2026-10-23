# チェックポイントと再現条件

[简体中文](checkpoint.zh-CN.md) · [English](checkpoint.en.md) · [日本語](checkpoint.ja.md) · [Overview](../README_JA.md)

保存済みの過去チェックポイントは現在の `HandwrittenCNN(num_classes=62)` に厳密読込できますが、バイナリの公開配布には明示許可が必要です。**現在、公開ダウンロードはありません。** ファイル時刻や構造検査は再学習・精度評価ではありません。

対応重みを `checkpoints/emnist_model.pth` に用意して `python desktop_app.py` を実行します。ラベルは0–9/A–Z/a–z、入力は28x28の単字です。pendingマニフェストではネットワーク前に取得を拒否し、ランダムな代替モデルは生成しません。既存重みを上書きしません。

公開許可後にRelease URL・バイト数・SHA-256を提示し、実ダウンロードで照合します。生データ・私有画像・資格情報は含めません。主入口11視点TTAと過去predict.pyの平行移動のみの経路は区別し、旧図を新成績としません。

[Manifest](../checkpoints/checkpoint-manifest.json) · [Downloader](../tools/download_checkpoint.py)
