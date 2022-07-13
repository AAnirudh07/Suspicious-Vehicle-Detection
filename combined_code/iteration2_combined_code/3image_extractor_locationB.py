
import numpy as np
import cv2
import time
import torch
import json
default_model = torch.hub.load('ultralytics/yolov5', 'yolov5n')
cap = cv2.VideoCapture("vids/teastall(310-340).mp4")
default_model.conf = 0.5  # NMS confidence threshold
default_model.iou = 0.5  # NMS IoU threshold
default_model.agnostic = True  # NMS class-agnostic
default_model.multi_label = False  # NMS multiple labels per box
# (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
# ['bicycle', 'car', 'motorcycle', 'bus', 'truck']
default_model.classes = [1, 2, 3, 5, 7]
default_model.max_det = 100  # maximum number of detections per image
default_model.amp = False  # Automatic Mixed Precision (AMP) inference

# cv2.namedWindow("Output Frame", cv2.WINDOW_NORMAL)

# creating the videocapture object
# and reading from the input file
# Change it to 0 if reading from webcam
cap = cv2.VideoCapture("vids/teastall(310-340).mp4")
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

frame_num = 0
uniq = 0
# Reading the video file until finished
while(cap.isOpened()):

    # Capture frame-by-frame

    ret, frame = cap.read()

    # if video finished or no Video Input
    if not ret:
        break
    img = frame
    frame_num = frame_num + 1

    # Inference
    if(frame_num % 60 != 0):
        continue
    cv2.imwrite("hello.jpg", img)
    results = default_model("hello.jpg")
    json1 = json.loads(
        str(results.pandas().xyxy[0].to_json(orient="records")))

    for row in json1:
        if(img is not None):
            img_crop = img[int(row['ymin']):int(row['ymax']),
                           int(row['xmin']):int(row['xmax']), :]
            cv2.putText(img, str(row['name']), (int(row['xmin']), int(
                row['ymin'])-10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
            # img = cv2.rectangle(img, (int(row['xmin']), int(row['ymin'])), (int(
            #     row['xmax']), int(row['ymax'])), (255, 255, 255), 2)
            cv2.imwrite("tea_images/"+str(frame_num) +
                        "vehicle"+str(uniq)+".jpeg", img_crop)
            uniq = uniq+1
    print(frame_num, "/", length)


# When everything done, release the capture
cap.release()
# Destroy the all windows now
cv2.destroyAllWindows()
