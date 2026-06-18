from ultralytics import YOLO
import cv2

model = YOLO("best.pt")

img_path = "valid/images/116_jpg.rf.40b2319a800ddb713b9435284f9f6ffc.jpg"
results = model(img_path)
annotated_frame = results[0].plot()
cv2.imshow("Cancer Cell Detection", annotated_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()