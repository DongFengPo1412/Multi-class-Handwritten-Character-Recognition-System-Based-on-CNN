# CNN-Based Multi-Class Handwritten Character Recognition & Correction System

[简体中文](README.md) | [English](README_EN.md) | [日本語](README_JA.md)

---

## 🌟 Project Introduction

This system is a high-performance handwritten character OCR recognition and correction system developed in accordance with the curriculum design requirements for junior-year "Intelligent Control". While **strictly keeping the underlying custom CNN network architecture (`HandwrittenCNN`) unchanged**, it incorporates advanced post-processing ideas from mature commercial OCR systems (such as QQ Text Extractor). The optimizations focus on **"Image Preprocessing"** and **"Result Correction"**:
* **Core Classifier**: Utilizes the pre-existing 3-layer `HandwrittenCNN` model with Test-Time Augmentation (TTA) multi-sampling.
* **Anti-Interference Preprocessing**: Implements a **Gaussian illumination subtraction algorithm for shadow removal** and an **adaptive bounding box clustering & merging algorithm**.
* **Smart Result Correction**: Features a **Joint Probability Lexicon Decoder (MAP)** and **spatial geometric constraints (aspect-ratio/relative height)**.
* **Hardware & Interaction**: Supports **hot-swapping cameras via the `c` key** (easily toggling between built-in and external webcams) and outputs **Top-3 candidates** for low-confidence characters.

---

## 🛠️ Core Algorithms

### 1. Data Collection & Image Preprocessing (Requirement 1)
Under webcam capturing conditions, environment shadows (cast by phone or hand) often create large black blotches after simple binarization.
* **Background Illumination Subtraction**: Estimates the illumination map of the ROI using a large Gaussian filter ($51 \times 51$). By dividing the original gray image by the background illumination map (`cv2.divide`), we get a clean white background with dark strokes, completely eliminating shadows.
* **CLAHE & Adaptive Thresholding**: Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) and local Adaptive Thresholding on the shadow-free image to extract smooth, noise-free binarized strokes.

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
