from ultralytics import YOLO
import cv2
import os

IMG_DIRS = ["train/images", "valid/images", "test/images"]#list of image folders to process

yolo_model = YOLO("best.pt")#load the YOLO model
os.makedirs("crops/benign", exist_ok=True)#created the benign and tumor folders if they don't exist
os.makedirs("crops/tumor", exist_ok=True)

for img_dir in IMG_DIRS:#iterate through the image folder
    for img_file in os.listdir(img_dir):#iterate through the images in the folder
        img_path = os.path.join(img_dir, img_file)#join the image folder and file name
        image = cv2.imread(img_path)#read the image
        if image is None:
            continue

        results = yolo_model(img_path, verbose=False)#run the YOLO model on the image
        boxes = results[0].boxes.xyxy#get the bounding boxes coordinates
        classes = results[0].boxes.cls#get the benign or tumor class

        for i, (box, cls) in enumerate(zip(boxes, classes)):
            x1, y1, x2, y2 = map(int, box)#convert the bounding box coordinates to integers
            crop = image[y1:y2, x1:x2]#cut out 
            stem = os.path.splitext(img_file)[0]
            if int(cls) == 0:  # class '1' = benign
                cv2.imwrite(f"crops/benign/{stem}_crop_{i}.jpg", crop)
            else:              # class 'tumor' = malignant
                cv2.imwrite(f"crops/tumor/{stem}_crop_{i}.jpg", crop)
