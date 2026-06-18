from ultralytics import YOLO
import cv2
import os

IMG_DIRS = ["train/images", "valid/images", "test/images"]#list of image folders to process

yolo_model = YOLO("best.pt")
os.makedirs("crops/benign", exist_ok=True)
os.makedirs("crops/tumor", exist_ok=True)

for img_dir in IMG_DIRS:
    for img_file in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img_file)
        image = cv2.imread(img_path)
        if image is None:
            continue

        results = yolo_model(img_path, verbose=False)
        boxes = results[0].boxes.xyxy
        classes = results[0].boxes.cls

        for i, (box, cls) in enumerate(zip(boxes, classes)):
            x1, y1, x2, y2 = map(int, box)
            crop = image[y1:y2, x1:x2]
            stem = os.path.splitext(img_file)[0]
            if int(cls) == 0:  # class '1' = benign
                cv2.imwrite(f"crops/benign/{stem}_crop_{i}.jpg", crop)
            else:              # class 'tumor' = malignant
                cv2.imwrite(f"crops/tumor/{stem}_crop_{i}.jpg", crop)
