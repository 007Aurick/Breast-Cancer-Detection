# 🧠 Cancer Cell Detection using YOLO + Grad-CAM

This project uses **:contentReference[oaicite:0]{index=0}** to detect cancer cells in microscopy images and applies **:contentReference[oaicite:1]{index=1}** for model interpretability.

The system outputs bounding boxes around detected cancer cells and heatmaps showing what the model focused on.

---

## 🚀 Features

- Cancer cell detection using YOLO
- Bounding box localization
- Confidence scoring
- Grad-CAM explainability
- Medical image inference pipeline

---

## 🧪 Tech Stack

- Python
- PyTorch
- Ultralytics YOLO
- OpenCV
- NumPy
- Matplotlib
- Grad-CAM

---

## 📁 Dataset Format (YOLO)

Each image has a corresponding `.txt` label file.

### Dataset structure

```txt
dataset/
├── images/
│   ├── train/
│   ├── val/
├── labels/
│   ├── train/
│   ├── val/
