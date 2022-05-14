import cv2 
import numpy as np

from object_detection import ObjectDetection
od = ObjectDetection()

cap = cv2.VideoCapture("D:/Projects/suspicious-vehicle-detection/data/videos/Ch8_20220112161012.mp4")
count = 0
center_points = []

while True:
    ret, frame = cap.read()
    count += 1
    if not ret:
        break

    (class_ids, scores, boxes) = od.detect(frame)

    for id,box in enumerate(boxes):
        if class_ids[id] in [1,2,3,4,5,7]:
            (x,y,w,h) = box
            cx = int((x+x+w)/2)
            cy = int((y+y+h)/2)
            center_points.append((cx,cy))
            print(f"FRAME:{count} COORDS:{x,y,w,h}")
            
            cv2.rectangle(frame, (x,y),(x+w,y+h), (0,255,0), 2)
            cv2.circle(frame, (cx,cy), 5, (0,255,0), -1)

    for points in center_points:
        cv2.circle(frame, (points[0],points[1]), 5, (0,255,0), -1)

    cv2.imshow("Frame",frame)
    key = cv2.waitKey(0)
    if key == 27:
        break
