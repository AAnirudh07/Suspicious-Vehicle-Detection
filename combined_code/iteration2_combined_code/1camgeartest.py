# import required libraries
from vidgear.gears import VideoGear
import cv2
import os
import skvideo.io
import scipy
import numpy as np
import os
import re
import cv2  # opencv library
import numpy as np
import torch
import cv2
import tensorflow as tf
import json
from imutils.video import WebcamVideoStream
# or yolov5m, yolov5l, yolov5x, custom
default_model = torch.hub.load('ultralytics/yolov5', 'yolov5n')
cap = cv2.VideoCapture("vids/toll5.mp4")
default_model.conf = 0.5  # NMS confidence threshold
default_model.iou = 0.5  # NMS IoU threshold
default_model.agnostic = True  # NMS class-agnostic
default_model.multi_label = False  # NMS multiple labels per box
# (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
# ['bicycle', 'car', 'motorcycle', 'bus', 'truck']
default_model.classes = [1, 2, 3, 5, 7]
default_model.max_det = 100  # maximum number of detections per image
default_model.amp = False  # Automatic Mixed Precision (AMP) inference
# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
# cv2.namedWindow("Output Frame", cv2.WINDOW_NORMAL)


# open any valid video stream(for e.g `myvideo.avi` file)
stream = VideoGear(source="vids/toll5.mp4").start()

# loop over
i = 0
uniq = 0
while True:

    # read frames from stream
    frame = stream.read()
    i += 1
    # check for frame if Nonetype
    if frame is None:
        break
    img = frame
    cv2.imwrite("hello.jpg", img)
    # Inference
    results = default_model("hello.jpg")
    json1 = json.loads(
        str(results.pandas().xyxy[0].to_json(orient="records")))

    for row in json1:
        if(img is not None):
            img_crop = img[int(row['ymin']):int(row['ymax']),
                           int(row['xmin']):int(row['xmax']), :]
            cv2.putText(img, str(row['name']), (int(row['xmin']), int(
                row['ymin'])-10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
            img = cv2.rectangle(img, (int(row['xmin']), int(row['ymin'])), (int(
                row['xmax']), int(row['ymax'])), (255, 255, 255), 2)
            cv2.imwrite("toll_images/"+"vehicle"+str(uniq)+".jpeg", img_crop)
            uniq = uniq+1
    print(i)
    # {do something with the frame here}

    # Show output window
    #cv2.imshow("Output Frame", img)

    # check for 'q' key if pressed
    # key = cv2.waitKey(1) & 0xFF
    # if key == ord("q"):
    #     break

# close output window
cv2.destroyAllWindows()

# safely close video stream
stream.stop()
