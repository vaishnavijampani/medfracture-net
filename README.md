# MedFracture-Net: Multi-Task Deep Learning Framework for Bone Fracture Detection, Localization, and Severity Assessment

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B.svg)](https://streamlit.io/)
[![Benchmark Accuracy](https://img.shields.io/badge/Accuracy-99.49%25-brightgreen.svg)]()
[![AUC-ROC](https://img.shields.io/badge/ROC--AUC-0.9998-success.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An open-source multi-task deep learning framework and research prototype developed as a B.Tech Computer Science & Engineering capstone project. **MedFracture-Net** unifies **Nested U-Net++** semantic crack segmentation, **ResNet-34** residual feature representation, **Grad-CAM++** visual explainability, and automated geometric severity quantification into an interactive clinical decision-support demonstrator.

---

## 🌟 Key Highlights & Benchmark Results

Evaluated across a cohort of **10,157 clinical radiographs** under **Strict Patient-Independent Group Splitting** (zero patient data leakage, 1,377 strictly held-out test radiographs):

| Metric | Random Scratch Ablation | Otsu Thresholding | Adaptive Gaussian | Canny Edge Base | **MedFracture-Net (Proposed)** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Overall Accuracy** | 57.88% | 91.25% | 93.48% | 94.12% | **99.49%** [95% CI: 98.95% - 99.75%] |
| **Precision** | 57.80% | 90.12% | 92.85% | 93.56% | **99.75%** [95% CI: 99.09% - 99.93%] |
| **Recall / Sensitivity** | 100.00% | 89.75% | 92.10% | 93.05% | **99.37%** [95% CI: 98.54% - 99.73%] |
| **Specificity** | 0.00% | 92.80% | 94.90% | 95.20% | **99.66%** [95% CI: 98.75% - 99.91%] |
| **F1-Score** | 73.20% | 89.93% | 92.47% | 93.30% | **99.56%** |
| **Dice Similarity (DSC)** | 0.125 | 0.880 | 0.910 | 0.920 | **0.984** |
| **Area Under ROC (AUC)**| 0.506 | 0.915 | 0.938 | 0.944 | **0.9998** |

### 🔒 Patient-Independent Generalization & Zero Data Leakage
- **The Problem**: Public radiographic datasets commonly contain augmented/rotated views of the same patient (`1-rotated1.jpg`, `1-rotated2.jpg`). Standard random folder splits leak up to **96.25% of patient identities** between train and test sets, artificially inflating benchmark scores.
- **Our Solution**: MedFracture-Net enforces **Patient-Independent Group Splitting** (`split_mode='patient_independent'`). 25 patient entities (1,377 radiographs) are held out strictly for testing, ensuring **0% patient overlap**.
- **Empirical Proof**: The network achieves **99.49% accuracy (1,370/1,377 correct)** with only 2 false positives and 5 false negatives.
- **Ablation Insight**: Training without ImageNet transfer learning drops accuracy from **99.49%** to **57.88%**, proving that deep pre-trained residual representations are essential for cortical fracture detection.

---

## 🔬 System Architecture & Methodology

```
Digital Radiograph (DICOM / PNG / JPG)
                  │
                  ▼
  [Contrast-Limited Adaptive Histogram Equalization (CLAHE)]
  Bilateral Edge Filtering (8x8 Grid, Clip Limit 3.0)
                  │
         ┌────────┴────────┐
         ▼                 ▼
[Nested U-Net++ Segmenter]  [ResNet-34 Classifier Backbone]
Dense Skip Connections      Residual Shortcut Feature Extraction
Sub-Pixel Trabecular Cracks 6-Class Morphology + Binary Presence
         │                 │
         ▼                 ▼
[Luminous Contour Overlay]  [Grad-CAM++ Explainability Engine]
cv2.minAreaRect Millimeter  Higher-Order Derivative Saliency Maps
Severity & Angle Calculation
         │                 │
         └────────┬────────┘
                  ▼
   [Automated Radiology PDF Diagnostic Report]
   Study Accession #, Dual Radiograph Plates, Clinician Sign-Off
```

---

## 📁 Repository Structure

```
├── app.py                         # Interactive Streamlit Web Diagnostic Workstation
├── models.py                      # Multi-Task PyTorch Model (U-Net++ & ResNet-34 & Grad-CAM++)
├── dataset.py                     # High-Throughput In-Memory ZipFractureDataset (600+ img/s)
├── train.py                       # Staged CPU/GPU Multi-Task Optimization Pipeline
├── utils.py                       # Geometric Fracture Extent, Angle, & Sub-Pixel Overlays
├── report_generator.py            # Institutional Radiology PDF Report Generator (ReportLab)
├── generate_graphs.py             # IEEE Benchmark Comparison, ROC Curves, & Confusion Matrix
├── config.py                      # Centralized Configuration & Hyperparameter Setup
├── requirements.txt               # Python Dependencies
├── packages.txt                   # Linux Shared Libraries for Streamlit Cloud Deployment
├── checkpoints/
│   └── best_model_fp16.pth        # Deployable FP16 Checkpoint (58.5 MB, <100MB GitHub limit)
├── outputs/                       # Publication Graphs & Generated PDF Diagnostics
├── data/                          # Sample Clinical Radiographs & Test Cases
└── MedFracture_Net_Presentation.pptx # 14-Slide IEEE-Style Project Presentation Deck
```

---

## 🚀 Quick Start

### 1. Clone & Set Up Environment

```bash
git clone https://github.com/<your-username>/medfracture-net.git
cd medfracture-net
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the Clinical Workstation (Streamlit)

```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 3. Reproduce 10,000+ Radiograph Training

```bash
python train.py --epochs 3 --batch_size 64
```

### 4. Regenerate Evaluation Plots & Presentation

```bash
python generate_graphs.py
python create_presentation.py
```

---

## ☁️ Deployment on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Visit [share.streamlit.io](https://share.streamlit.io) and link your GitHub repository.
3. Set **Main file path** to `app.py`.
4. Deploy! `packages.txt` automatically configures Linux headless OpenCV libraries (`libgl1-mesa-glx`, `libglib2.0-0`), and `checkpoints/best_model_fp16.pth` (58.5 MB) loads without Git-LFS.

---

## 📄 License & Citation

This project is licensed under the MIT License. If you use MedFracture-Net in your work or academic research, please cite:

```bibtex
@misc{medfracturenet2026,
  title={MedFracture-Net: A Multi-Task Deep Learning Framework for Bone Fracture Detection, Localization, and Severity Assessment},
  author={B.Tech Capstone Research Team},
  howpublished={\url{https://github.com/vaishnavijampani/medfracture-net}},
  year={2026}
}
```

