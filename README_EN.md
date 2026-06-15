# CNN-Based Multi-Class Handwritten Character Recognition & Correction System

[简体中文](README.md) | [English](README_EN.md) | [日本語](README_JA.md)

---

## 🌟 Project Introduction

This system is a high-performance handwritten character OCR recognition and correction system developed in accordance with the curriculum design requirements for junior-year "Intelligent Control". While **strictly keeping the underlying custom CNN network architecture (`HandwrittenCNN`) unchanged**, it incorporates advanced post-processing ideas from mature commercial OCR systems (such as QQ Text Extractor). The optimizations focus on **"Image Preprocessing"** and **"Result Correction"**:
* **Core Classifier**: Utilizes the pre-existing 3-layer `HandwrittenCNN` model with Test-Time Augmentation (TTA) multi-sampling.
* **Anti-Interference Preprocessing**: Implements a **Gaussian illumination subtraction algorithm for shadow removal**, an **adaptive bounding box clustering & merging algorithm**, and an **adaptive contrast polarity detection algorithm (seamlessly supporting blackboard chalk or colored papers)**.
* **Smart Result Correction**: Features a **Joint Probability Lexicon Decoder (MAP)** and **spatial geometric constraints (aspect-ratio/relative height)**.
* **Hardware & Interaction**: Supports **hot-swapping cameras via the `c` key** (easily toggling between built-in and external webcams) and outputs **Top-3 candidates** for low-confidence characters.

---

## 🛠️ Core Algorithms

### 1. Data Collection & Image Preprocessing (Requirement 1)
Under webcam capturing conditions, environment shadows (cast by phone or hand) often create large black blotches after simple binarization.
* **Background Illumination Subtraction**: Estimates the illumination map of the ROI using a large Gaussian filter ($51 \times 51$). By dividing the original gray image by the background illumination map (`cv2.divide`), we get a clean white background with dark strokes, completely eliminating shadows.
* **CLAHE & Adaptive Thresholding**: Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) and local Adaptive Thresholding on the shadow-free image to extract smooth, noise-free binarized strokes.
* **Adaptive Contrast Polarity Detection**: Standard binarization assumes a "dark characters on a light background" scenario. To support "white chalk on a dark blackboard" or custom colored paper/ink combinations, the system automatically checks the average intensity of the outermost border pixels of the thresholded image. If white pixels dominate (indicating a light background), it automatically inverts the image to ensure the character is passed to the CNN as a white glyph on a black background. Otherwise, it keeps it as-is. This enables 100% automated environment adaptation without manual toggle buttons.


### 2. Feature Selection & Bounding Box Segmentation (Requirement 2)
* **Adaptive Bounding Box Clustering & Merging (Box Merging)**:
  - Automatically groups nearby bounding boxes of contours.
  - **Vertical Merging**: Integrates separated components (e.g. the dot and body of lowercase letters `i` and `j`) when their horizontal overlap ratio is high and vertical gap is small.
  - **Horizontal Near Merging**: Merges strokes that are disconnected due to fast writing or thin pen strokes (e.g. the crossbar of letter `A` missing).
* **EMNIST Alignment**: Computes the center of mass (Image Moments) of the merged box and translates the character to the center $(14, 14)$ of a standardized $28 \times 28$ canvas, feeding it into the CNN.

### 3. Recognition Result Correction (Requirement 3)
Handwritten characters like `oO0`, `1Il`, `2Zz`, and `5Ss` are geometrically identical when normalized to $28 \times 28$. The system corrects them using:
* **Context Classification**: Automatically determines if the recognized sequence is a "word" or a "number/mixed string".
* **Joint Probability Lexicon Decoding (MAP)**: In an alphabetical context, instead of hard-decoding using ArgMax, the system scores candidate words in the lexicon (`words.txt`, 10,000 common words) by summing the log-probabilities of each character position case-insensitively:
  $$\text{Score}(W) = \sum_{i=1}^{N} \ln ( P(c_i.lower() | x_i) + P(c_i.upper() | x_i) )$$
  The candidate word with the highest log-likelihood is chosen (e.g., correcting raw CNN output `he11O` to `hello`).
* **Aspect Ratio constraint for `0` vs `O`**: Uses aspect ratio `w/h < 0.52` to weight candidate `0`, otherwise favoring letters `O/o`.
* **Relative Height ratio for Casing**: Computes height ratio $r = h_i / \max(H)$ within the line. Confused letters (like `Z`) are mapped to lowercase `z` if $r < 0.78$, otherwise remaining uppercase.

---

## 📂 Project Structure

```text
PythonProject3/
├── src/
│   ├── __init__.py
│   ├── model.py              # Custom CNN structure (HandwrittenCNN)
│   ├── utils.py              # Data loader and augmentation pipeline
│   ├── corrector.py          # Lexicon decoding and geometric checks
│   └── words.txt             # 10,000 common English words list
├── checkpoints/
│   └── emnist_model.pth      # Pre-trained weights (Unchanged, loaded directly)
├── data/                     # Automatically downloaded EMNIST dataset
├── train.py                  # Neural network training script
├── predict.py                # Batch evaluation script (with lexicon test & matplotlib GUI)
├── camera_detect.py          # Real-time camera OCR and correction script (Demo entry)
├── revert_model.py           # Model weight reversion utility (one-click restore of backup weights)
├── README.md                 # Simplified Chinese Documentation
├── README_EN.md              # English Documentation
└── README_JA.md              # Japanese Documentation
```

---

## 🚀 Running and Operation Instructions

### 1. Launch Real-time Camera OCR
```bash
python camera_detect.py
```
* **Dynamic Camera Switching (`c` Key)**:
  - The program scans and opens camera index `0` (built-in webcam) on startup.
  - Press the **`c` key** at any time to cycle through all detected active camera devices (built-in webcam, external USB webcam, virtual camera) without restarting the script.
* **Crop & Recognize (Space Key)**:
  - Place a white paper with hand-written words/digits inside the red focal box (0.5mm black gel pen recommended).
  - The `AI Vision` window displays the binarized stroke with shadows removed.
  - Press the **[Space Key]** to execute character segmentation and spelling correction. The terminal and the screen will immediately display the `Raw prediction` and `Corrected output`.
* **Uncertainty Candidate Reporting**:
  - If a character has low confidence (top-1 < 80% or difference between top-1 and top-2 is < 20%), a yellow warning `⚠️ 模糊字符警示` (Top-3 possibilities and percentages) is printed in the terminal and displayed on the screen.
* **Exit (`q` Key)**:
  - Press the **`q` key** to release the video capture and close all windows safely.

### 2. Launch Visual Test Evaluation
```bash
python predict.py
```
- Prints correction results of simulated test cases (`hello`, `zoom`, `class`, `2026`, `10850`) before/after correction in the terminal.
- Randomly loads batches of 12 images from the EMNIST test set and plots them in a Matplotlib GUI (Green for correct, Red for wrong, Orange for ambiguous/uncertain). Press Enter to cycle to the next batch.

---

## 📊 Advanced Model Training & Academic Asset Generation

The training pipeline in [train.py](file:///c:/Users/Liu/PycharmProjects/PythonProject3%20-%20%E5%89%AF%E6%9C%AC/train.py) has been heavily optimized with modern deep learning techniques. It automatically generates academic-grade visual assets inside the `checkpoints/` directory to be directly used in your curriculum reports and defense slides:

### 1. Architecture & Optimizer Improvements
*   **SiLU (Swish) Activation**: Swapped standard ReLU activations in the `HandwrittenCNN` (defined in [src/model.py](file:///c:/Users/Liu/PycharmProjects/PythonProject3%20-%20%E5%89%AF%E6%9C%AC/src/model.py)) with **SiLU (Swish)**, enhancing the model's non-linear capacity on thin gel pen strokes.
*   **He (Kaiming) Initialization**: Applies Normal He weight initialization to convolutional layers and normal distribution to dense layers, dramatically speeding up training convergence.
*   **Label Smoothing**: Employs `label_smoothing=0.1` in CrossEntropyLoss to reduce overfitting caused by minor labeling errors in the raw dataset.
*   **L2 Weight Decay**: Adds `weight_decay=1e-4` to the Adam optimizer to constrain weight magnitude and improve model generalization.
*   **ReduceLROnPlateau Scheduler**: Automatically halves the learning rate when validation loss plateaus for 3 consecutive epochs.

### 2. Auto-Generated Figures (Saved in `checkpoints/`)
*   **Data Augmentation Samples (`data_augmentation_samples.png`)**:
    *   **Course Requirement**: Preprocessing & Data Augmentation (Requirement 1).
    *   **Details**: Generates a 4x4 grid showcasing augmented EMNIST samples (including elastic deformation, rotation, lighting noise, and translation) used to boost model robustness.
*   **Training Curves (`training_curves.png`)**:
    *   **Course Requirement**: Model Training & Evaluation (Requirement 2/3).
    *   **Details**: Plots the training/validation loss and accuracy curves across 25 epochs to demonstrate model convergence.
*   **62-Class Confusion Matrix (`confusion_matrix.png`)**:
    *   **Course Requirement**: Error Analysis & Revision (Requirement 3).
    *   **Details**: A high-resolution heatmap visualizing classification errors. Unveils the high error rates among visually similar pairs (like `1`/`l`/`I`, `0`/`O`), proving the necessity of our corrector module ([src/corrector.py](file:///c:/Users/Liu/PycharmProjects/PythonProject3%20-%20%E5%89%AF%E6%9C%AC/src/corrector.py)).

---

## ⚡ Industrial-grade Performance & Robustness Optimizations

To deliver a commercial-grade, secure, and lag-free demonstration experience during defense, the following engineering optimizations are integrated:

### 1. Inference Warmup
*   **Problem**: In PyTorch, the initial forward pass compiles the computing graph and allocates GPU/CPU memory, causing a visible lag (up to 2–3 seconds) on the first recognized character.
*   **Solution**: During the initialization of `LocalOCRRecognizer` (defined in [src/local_ocr.py](file:///c:/Users/Liu/PycharmProjects/PythonProject3%20-%20%E5%89%AF%E6%9C%AC/src/local_ocr.py)), the system runs a dummy tensor through the network (Warmup). When the main application opens, the initial recognition is instant and lag-free.

### 2. Multithreaded Decoupling
*   **Problem**: If inference and Baidu cloud requests run on the GUI main thread, the desktop application window freezes (displaying OS "Not Responding" warnings) during recognition.
*   **Solution**: We decouple tasks using `ThreadPoolExecutor` in `desktop_app.py`. Camera frame updates run continuously on the main thread (guaranteeing a smooth 30 FPS video feed), while heavy model inference and API calls are assigned to a background worker thread, updating result cards asynchronously.

### 3. Model Weight Auto-Backup & One-Click Reversion
*   **Problem**: Retraining or tuning parameters in `train.py` can corrupt existing weights if the process is terminated abnormally or if the new model performs worse.
*   **Solution**:
    *   **Auto-Backup**: When training starts, the script duplicates the current `checkpoints/emnist_model.pth` to `emnist_model_backup.pth`.
    *   **Quick Restoration**: Running `python revert_model.py` in the root directory overwrites the active weights with the backup file in 1 second, securing your system under all conditions.
