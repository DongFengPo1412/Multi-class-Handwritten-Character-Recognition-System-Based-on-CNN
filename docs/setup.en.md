# Handwritten Character Recognition and Correction — Setup and operation

[简体中文](setup.zh-CN.md) · **English** · [日本語](setup.ja.md) · [Overview](../README_EN.md)

<a id="structure"></a>

## Structure and assets

```text
.
├── desktop_app.py
├── camera_detect.py
├── train.py / predict.py / ablation_benchmark.py
├── src/   # CNN, segmentation, correction, lexicon, cloud client, checks
├── checkpoints/ablation_benchmark_results.json
├── baiduocr/   # earlier standalone examples
├── requirements.txt
└── docs/       # trilingual guides and existing images
```

`data/`, `checkpoints/*.pth` and `baiduocr/key.txt` remain ignored. Follow the [checkpoint guide](checkpoint.en.md) to download and verify the historical Release asset; its manifest is tracked, its weights are not.

<a id="environment"></a>

## Environment and installation

Offline checks used Windows/Python 3.10.20, Torch 2.5.1+cu121, torchvision 0.20.1+cu121, OpenCV 4.11.0, NumPy 1.26.4. This is a check environment, not a lockfile or CUDA benchmark.

```powershell
python -m pip install -r requirements.txt
```

Tkinter/display and a camera are needed for the workbench. Training/prediction require EMNIST download space/network. Requirements are unpinned; check AMP/scheduler compatibility, particularly `ReduceLROnPlateau(verbose=True)` on newer Torch.

<a id="operation"></a>

## Workbench and optional cloud reference

Run `python desktop_app.py` from the repository root and confirm successful CNN loading in the console. Camera indices 0–3 are scanned, falling back to 0. Place text in the central square ROI; Space recognizes/resumes, Enter/Esc resumes, C switches cameras, B toggles cloud, Q quits.

Configure cloud credentials through `BAIDU_OCR_API_KEY`/`BAIDU_OCR_SECRET_KEY` or ignored `baiduocr/key.txt`; see [key.example.txt](../baiduocr/key.example.txt). With credentials, initialization requests a token, and enabled recognition uploads JPEG/Base64 ROI data to Baidu. Local processing works without credentials. Never commit keys.

`camera_detect.py` is a separate historical interface/processing path; its screenshots or timing do not establish every behavior of the primary entry.

<a id="training"></a>

## Training and historical experiments

```powershell
python train.py
python predict.py
python ablation_benchmark.py --samples 5000 --device cuda
```

These supplied entries were not run in this audit. Training backs up/overwrites weights and generates plots; predict downloads data and opens plots; benchmark overwrites the JSON. Preserve your own weights/results and read [evaluation limitations](technical-notes.en.md#evaluation). Repeating the same method does not remove its bias. `revert_model.py` overwrites the primary checkpoint from backup and is not a default setup step.

<a id="checks"></a>

## Offline checks and troubleshooting

```powershell
python src/test_segmentation.py
python src/test_new_patterns.py
```

The former has three segmentation assertions, the latter seven constructed-probability cases. Neither requires camera, weights, or cloud credentials; neither estimates recognition accuracy.

| Symptom | Inspect |
| :--- | :--- |
| Initialization fails | Supply a compatible HandwrittenCNN checkpoint; missing/corrupt weights stop local initialization rather than producing random predictions. |
| Wrong text correction | Segmentation order, global height, format detection, out-of-lexicon names. |
| Dark preview differs from inference | Distinct preprocessing implementations. |
| Slow result display | CNN/cloud in one worker, 80 ms polling, actual hardware; no fixed latency. |
| Fresh training fails | Torch/torchvision versions, scheduler API, downloads, Windows multiprocessing. |

2026-10-23 fixes: missing or incompatible checkpoints abort initialization, with a desktop error dialog, instead of random-model inference. Seven new reliability tests plus the existing three segmentation assertions and seven correction cases passed. No retraining, camera evaluation, or cloud request was performed. Real-photo evaluation and dataset versioning remain future work.

```powershell
python -B -m unittest discover -s tests -v
```
