# Breast Cancer Cell Detection using YOLO + Grad-CAM

This project uses a method to detect cancer cells in microscopy images and applies it for model interpretability.

The system outputs bounding boxes around detected cancer cells and heatmaps showing which regions influenced the model's decision.

---

## Features

- Breast cancer cell detection using YOLO
- Bounding box localization with confidence scores
- Custom CNN classifier for image-level predictions
- Grad-CAM explainability overlays
- Real-time webcam inference pipeline

---

## Tech Stack

- Python
- PyTorch
- Ultralytics YOLO
- OpenCV
- NumPy
- Pandas
- Matplotlib
- pytorch-grad-cam

---

## Project Structure

```txt
BreastCancerDetection/
├── CancerCellDetection.py   # YOLO webcam inference + CNN / Grad-CAM setup
├── ImagePathways.csv        # Image paths and labels (0 = benign, 1 = malignant)
└── README.md
```

---

## Dataset Format (YOLO)

Each image has a corresponding `.txt` label file in YOLO format.

### Folder layout

```txt
dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

### Label format

Each `.txt` file contains one line per object:

```txt
<class_id> <x_center> <y_center> <width> <height>
```

All coordinates are normalized to `[0, 1]` relative to image width and height.

| Class ID | Label    |
|----------|----------|
| 0        | Benign   |
| 1        | Malignant |

### CSV format (classification)

For CNN training, images can also be referenced in `ImagePathways.csv`:

```csv
image_path,label
images/img001.png,0
images/img002.png,1
```

- `0` → Non-cancerous (benign)
- `1` → Cancerous (malignant)

---

## Installation

```bash
git clone https://github.com/AurickAnwar/Cancer-Cell-Detection.git
cd Cancer-Cell-Detection

pip install torch torchvision ultralytics opencv-python numpy pandas matplotlib pytorch-grad-cam
```

On first run, YOLO will download the `yolo11m.pt` weights automatically.

---

## Usage

Run the main script to start real-time detection from your webcam:

```bash
python CancerCellDetection.py
```

- A window shows YOLO detections with bounding boxes.
- Press `q` to quit.

Grad-CAM and CNN classification are wired up in the script for image-level explainability on static images.

---

## Pipeline

```txt
Image / Webcam Frame
        ↓
   YOLO Detection
        ↓
Bounding Boxes + Confidence
        ↓
   CNN Classification (optional)
        ↓
   Grad-CAM Heatmap
        ↓
Overlay on Original Image
```

---

## Grad-CAM

Grad-CAM highlights the regions in an image that contributed most to the model's prediction.

- **Red / yellow** → high-importance areas
- **Blue** → low-importance areas

This helps explain model decisions in medical imaging, where interpretability matters.

---

## Goals

- Detect breast cancer cells in microscopy images
- Localize suspicious regions with YOLO
- Explain predictions with Grad-CAM heatmaps
- Build a foundation for medical AI and computer vision

---

## Future Improvements

- Train YOLO on a breast cancer histopathology dataset
- Connect Grad-CAM to the CNN classifier for static image inference
- Add batch inference for folders of images
- Expand to multi-class grading (e.g. tumor stage)
- Improve accuracy with deeper architectures (EfficientNet, ResNet)

---

## Disclaimer

This project is for **educational and research purposes only**. It is not a medical device and should not be used for clinical diagnosis.
