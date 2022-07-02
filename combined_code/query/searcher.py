from torchreid.utils import FeatureExtractor
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
import sys

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
cv2.namedWindow("WIN", cv2.WINDOW_NORMAL)
extractor = FeatureExtractor(
    model_name='osnet_x1_0',
    device='cuda'
)

query = "toll_images"+"/vehicle7.jpeg"
query_img = cv2.imread(query)
# choose codec according to format needed
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
outputvid = []
NUM_CLUSTERS = 5
vid = WebcamVideoStream(src="vids/merged_tea_stall.mp4").start()
i = 0
uniq = 0
while(True):
    i = i+1
    # Images
    print("iter= ", i)
    img = vid.read()  # or file, Path, PIL, OpenCV, numpy, list
    if(cv2.waitKey(33) == 27):
        break
    cv2.imwrite("hello.jpg", img)
    # Inference
    results = default_model("hello.jpg")
    json1 = json.loads(
        str(results.pandas().xyxy[0].to_json(orient="records")))

    for row in json1:
        if(img is not None):
            img_crop = img[int(row['ymin']):int(row['ymax']),
                           int(row['xmin']):int(row['xmax']), :]
            features = extractor([img_crop, query_img])
            ans = torch.nn.functional.cosine_similarity(
                features[0], features[1], dim=0)
            cv2.putText(img, str("{0:.2f}".format(ans))+str(row['name']), (int(row['xmin']), int(
                row['ymin'])-10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
            img = cv2.rectangle(img, (int(row['xmin']), int(row['ymin'])), (int(
                row['xmax']), int(row['ymax'])), (255, 255, 255), 2)
            if(ans.item() >= 0.7):
                cv2.imwrite("tea_images/"+"spotted"+str(uniq)+".jpeg", img)
                cv2.imwrite("tea_images/"+"spotted_object" +
                            str(uniq)+".jpeg", img_crop)
    # # # Results
    cv2.imwrite("tea_images/"+str(i)+".jpeg", img)
    cv2.imshow("WIN", img)
    # results.print()
cv2.destroyAllWindows()
vid.stop()
# STOP
