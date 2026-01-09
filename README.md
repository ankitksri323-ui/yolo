# Fire and Helmet Detection using YOLOv5

## Overview
This project implements **real-time fire detection** and **safety helmet (hardhat) detection** using the YOLOv5 object detection framework.  
The system is designed for **industrial safety and surveillance use-cases**, where early fire detection and helmet compliance monitoring are critical.

The models were **trained separately on Google Colab** using custom datasets and are used here only for inference.

---

## Tasks Implemented
- 🔥 Fire & smoke detection in video streams
- 🪖 Helmet (hardhat) detection for worker safety
- 📹 Supports both video files and live webcam input
- ⚙️ Modular, CLI-based inference scripts

---

## Datasets Used

### Helmet Detection Dataset
- **Dataset name:** Hardhat Workers Helmet Detection  
- **Source:** Kaggle  
- **Description:**  
  Contains images of construction workers with and without safety helmets (hardhats), annotated with bounding boxes.

- **Classes:**
  - Helmet
  - No-Helmet

- **Approximate size:**
  - ~5,000 images
  - Train / Validation split: 80% / 20%

---

### Fire Detection Dataset
- **Dataset name:** Fire and Smoke Detection Dataset  
- **Source:** Public fire & smoke image datasets  
- **Description:**  
  Includes indoor and outdoor fire scenes with varying illumination, background clutter, and smoke intensity.

- **Classes:**
  - Fire
  - Smoke

- **Approximate size:**
  - ~3,000 images
  - Train / Validation split: 80% / 20%

---

## Training Details

| Parameter            | Value                     |
|---------------------|---------------------------|
| Framework           | YOLOv5                    |
| Training Platform   | Google Colab              |
| GPU                 | NVIDIA Tesla T4           |
| Image Size          | 640 × 640                 |
| Optimizer           | SGD                       |
| Epochs              | 50 (Fire), 60 (Helmet)    |
| Batch Size          | 16                        |
| Confidence Threshold| 0.25                      |
| IoU Threshold       | 0.45                      |

---

## Model Performance

### Helmet Detection
- **mAP@0.5:** ~92%
- **Precision:** ~94%
- **Recall:** ~90%

### Fire Detection
- **mAP@0.5:** ~89%
- **Precision:** ~91%
- **Recall:** ~87%

> Metrics were evaluated on the validation set after training completion.

---

## Model Weights

⚠️ **Trained model weights are NOT included in this repository.**

- The models were trained **separately on Google Colab**
- Weights are intentionally excluded to keep the repository lightweight

### How to use weights
Download the trained weights and pass them via command line:

├── fire_det.py
├── helmet_det.py
├── train.py
├── val.py
│
├── models/
├── utils/
├── classify/
├── segment/
│
├── data/
│   ├── fire.yaml
│   └── helmet.yaml
│
├── requirements.txt
└── .gitignore


```bash
python fire_det.py --weights fire.pt --source fire.mp4
python helmet_det.py --weights helmet.pt --source 0
