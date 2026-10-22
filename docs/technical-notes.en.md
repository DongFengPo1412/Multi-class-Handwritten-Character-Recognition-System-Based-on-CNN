# Handwritten Character Recognition and Correction — Technical notes

[简体中文](technical-notes.zh-CN.md) · **English** · [日本語](technical-notes.ja.md) · [Overview](../README_EN.md)

<a id="source-map"></a>

## Modules and entry points

| Path | Responsibility |
| :--- | :--- |
| [desktop_app.py](../desktop_app.py) | Main workbench, camera, one worker, 80 ms result polling, frozen-frame boxes. |
| [src/local_ocr.py](../src/local_ocr.py) | Preprocessing, segmentation, order, 11-view inference, word-level decoding. |
| [src/model.py](../src/model.py) / [src/utils.py](../src/utils.py) | CNN, labels, EMNIST orientation, augmentation, split. |
| [src/corrector.py](../src/corrector.py) / [lexicon](../src/words.txt) | 5459 unique accepted words, probability adjustment, length candidates, masks. |
| [train.py](../train.py) / [predict.py](../predict.py) | Training and separate evaluation demonstration; predict uses nine shifts, unlike the main 11-view path. |
| [ablation_benchmark.py](../ablation_benchmark.py) | Character/synthetic-sequence evaluation, subject to the limitations below. |
| [src/baidu_ocr.py](../src/baidu_ocr.py) | Credentials, HTTP token, JPEG/Base64 request, cloud response parsing. |

`camera_detect.py` is another OpenCV interface. It shares the updated segmentation entry but retains separate preprocessing/postprocessing; it is not identical to Tkinter. `baiduocr/` contains earlier standalone examples; the primary client is `src/baidu_ocr.py`.

<a id="preprocessing"></a>

## Images, characters, and layout

`recognize` uses mean border brightness 127 to select the background path. Light-background Gaussian kernels are about 8% of the maximum dimension, forced odd; CLAHE is 2.0/8×8, and adaptive windows are about 1.5%, at least 3, with constant 4. Dark backgrounds use median/Gaussian filtering and Otsu without division. Live preview still uses a fixed 51 kernel and different thresholds; it is not the exact inference input.

Segmentation estimates horizontal bands, then applies 2×2 closing/opening, contour filtering, and iterative merging within each band. Wide boxes can be split into multiple projection-based crops, not simply two equal halves. Cursive/vertical layouts are not universally supported; pixel thresholds and square-ROI assumptions remain.

Density controls dilation/erosion. Crops are resized to a longest side of 20, placed on 28×28, and translated by image moments:

```math
\Delta x=14-\frac{M_{10}}{M_{00}},\qquad \Delta y=14-\frac{M_{01}}{M_{00}}
```

Nearest-neighbor direction votes choose rows/columns. Gaps above `max(10,0.55×mean character size)` separate words. Relative height uses a global ROI baseline, not a per-line maximum; horizontal line results are ultimately joined with spaces.

<a id="network"></a>

## CNN and training configuration

Three 3×3 convolution blocks use channels `1→32→64→128`, BatchNorm/SiLU, and two 2×2 pools with Dropout2d 0.15. Flatten 6272→512→62 includes fully connected Dropout 0.5. Convolution initialization is `kaiming_normal_(mode='fan_out', nonlinearity='relu')`, although activation is SiLU; the former input-channel-only initialization equation was inaccurate.

Main inference averages Softmax probabilities across nine one-pixel shifts and ±5° rotations:

```math
\bar p(c\mid x)=\frac1{11}\sum_{k=1}^{11}\mathrm{softmax}(f(T_kx))_c
```

EMNIST ByClass is rotated -90° and horizontally flipped, normalized by mean 0.1736/std 0.3317. Augmentation includes affine/perspective/elastic transforms, brightness/contrast, blur, and erasing. The 90/10 `random_split` has no fixed split seed and shares one augmented dataset between training and validation. Only the independent test dataset uses the clean transform.

Training specifies batch 128, Adam lr 0.001/weight_decay1e-4, label smoothing 0.1, loss scheduler patience 3/factor 0.5, at most 25 epochs, accuracy-based early stopping after 7 stagnant epochs, and CUDA AMP. Configuration and GPU log strings do not prove the actual hardware, completed epochs, or results.

<a id="decoding"></a>

## Probability decoding and interaction

Aspect ratio 0.52, relative height 0.78, and context biases reweight and renormalize distributions. Uncertainty means Top-1 below 0.80 or a top-two gap below 0.20, exposing Top-3; this is not calibrated accuracy.

Lexicon search uses **equal-length** candidates and summed case probabilities. It has no explicit word-frequency prior or neural language model:

```math
W^*=\arg\max_{W\in D_N}\sum_{i=1}^{N}\ln\left(p_i(w_i^{lower})+p_i(w_i^{upper})+10^{-12}\right)
```

`pruned_candidates` is computed, but scoring still loops over `candidates`. It is inaccurate to claim all candidates receive effective edit-distance pruning, O(1) decoding, or lossless 0.12 ms processing.

Length/count rules select 18-character ID-like, 11-character phone-like, plate-like, alphabetic, or numeric masks. They do not validate checksums, real phone numbers, or Chinese province characters. A wrongly inferred format can overconstrain output.

GUI 30 ms is an additional callback delay, and results are polled every 80 ms; neither guarantees 30 FPS or 22.6 ms. One worker executes local then cloud processing, so the optional request delays joint display. Checkpoint-load failure only prints an error and leaves initialized parameters active; verify successful loading before use.

<a id="evaluation"></a>

## Historical results and limitations

The [archived JSON](../checkpoints/ablation_benchmark_results.json) records CUDA, 5000 glyphs, and 50 synthetic sequences on 2026-09-14:

| Stage | Recorded character % | Recorded sequence % |
| :--- | ---: | ---: |
| M1 | 84.24 | 16 |
| M2 | 84.98 | 12 |
| M3 | 84.90 | 14 |
| M4 | 86.51 | 46 |
| M5 | 87.17 | 90 |

These are not current camera metrics. M1 adds random shifts; M2 directly uses dataset inputs without calling centroid alignment. Sequences sample labeled test glyphs; aspect ratios/heights derive from true characters, and M4 receives true numeric/alphabetic context. M1–M3 and M4–M5 character scores use different populations; M5 sequence comparison ignores case. Cross-row differences do not isolate module contributions. Camera preprocessing and segmentation are absent.

[Training curves](images/training_curves.png), [confusion matrix](images/confusion_matrix.png), and [augmentation](images/data_augmentation_samples.png) remain historical illustrations. Raw epoch records, checkpoint binding, and full reproduction are missing, so 83% is not promoted to a current independently verified test accuracy.

| Historical UI | Historical UI |
| :---: | :---: |
| ![Digits](images/demo_digit_recognition.png) | ![Mixed](images/demo_mixed_alphanumeric.png) |
| ![Multiline](images/demo_multiline_letters.png) | ![Phone-like](images/demo_phone_number_rule.png) |

<a id="implementation-status"></a>

## Validation and further work

Three segmentation assertions and seven correction cases passed this audit without camera, cloud, or real-model evaluation. Source tracing establishes algorithms/configuration/GUI paths; ignored local weights do not mean models ship with Git clones.

Static parsing passed for 17 of 18 tracked Python files. Historical standalone `baiduocr/demo2.py` has a syntax error at line 45 and is not a ready-to-run reproduction entry. The main pipeline and the checks above parse successfully.

Priorities are an independent validation transform/fixed split, measured geometry on camera samples, identical stage populations and case rules, then candidate-loop and failed-loading corrections. Core source is unchanged. Separate individual contributions from external algorithms/data rather than presenting library capabilities as measured innovations.
