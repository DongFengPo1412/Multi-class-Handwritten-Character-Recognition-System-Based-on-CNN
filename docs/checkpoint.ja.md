# 過去の重みと再現

[简体中文](checkpoint.zh-CN.md) · [English](checkpoint.en.md) · [日本語](checkpoint.ja.md) · [Overview](../README_JA.md)

保管されていた62クラスのチェックポイントを、推論・ワークベンチ再現用に公開します。今回再学習・再評価したモデルではありません。原ファイル時刻は2026-06-16ですが、当時の環境・分割を完全には再構成できません。

現在の HandwrittenCNN(num_classes=62) に厳密に読込でき、1x1x28x28のゼロ入力から有限の1x62出力を確認しました。構造・数値上の利用可能性の検査であり、認識精度ではありません。生の学習データ、私有画像、資格情報は公開しません。

[Release](https://github.com/DongFengPo1412/Multi-class-Handwritten-Character-Recognition-System-Based-on-CNN/releases/tag/emnist62-checkpoint-20260616) · [Manifest](../checkpoints/checkpoint-manifest.json)

| Artifact | Value |
| --- | --- |
| Name | `emnist_model.pth` |
| Bytes | 13356700 |
| SHA-256 | `2cdf4b6300027ca1fb8202832fcfea37906b10b2094acf464f6c81b899113928` |
| Architecture | `src.model.HandwrittenCNN(num_classes=62)` |

## 導入と検証

```powershell
$task = Join-Path $env:USERPROFILE '.codex\cache\tasks\ocr-setup-<unique-id>'
python -B tools/download_checkpoint.py --cache-dir "$task\download"
python -B desktop_app.py
```

バイト数とSHA-256一致後のみ配置します。既存の異なる重みは保持してエラーとし、自動上書きしません。取得物は指定キャッシュに残します。重みはGitの無視対象のままRelease資産として配布します。

## モデル・評価の範囲

ラベルは0–9/A–Z/a–z順です。主デスクトップ経路は11視点TTA、独立した過去のpredict.pyは平行移動分岐のみです。過去の曲線は保守版の新成績ではなく、筆者独立・実カメラ精度の新測定もありません。
