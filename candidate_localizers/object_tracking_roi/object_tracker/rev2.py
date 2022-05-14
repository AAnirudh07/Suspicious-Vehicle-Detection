import os
import re
import cv2
import numpy as np
from os.path import isfile, join
import matplotlib.pyplot as plt
from tracker import *

tracker = EuclideanDistTracker()

kernel = np.ones((13,13),np.uint8)
font = cv2.FONT_HERSHEY_SIMPLEX

video_path = "D:/Projects/vehicle classification/data/videos/Ch8_20220112161012.mp4"

cap = cv2.VideoCapture(video_path)
ret ,new_frame = cap.read()
old_frame = new_frame

while ret:
    
    grayA = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
    diff_image = cv2.absdiff(grayB, grayA)
    
    ret, thresh = cv2.threshold(diff_image, 100, 255, cv2.THRESH_BINARY)
    
    dilated = cv2.dilate(thresh,kernel,iterations = 1)
    
    contours, hierarchy = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    
    dmy = old_frame.copy()
    mask = np.zeros(dmy.shape[:2],dtype=np.uint8)

    '''
    Given the following:
    point P(x1,y1)
    line Ax+By+C=0 (in general form)

    We could check the value of y on the line which would correspond to x1.
    This will be y= (-Ax1-C)/B.

    The point would then be above the line if y1 is greater than y. Therefore,
    y1-y>0
    y1 - (-Ax1-C)/B>0
    (Ax1+By2+C)/B>0
    '''
    detections = []
    for cntr in contours:
        M = cv2.moments(cntr)
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            if cy - ((-9/16)*cx + 1080) > 0 and (cv2.contourArea(cntr) >= 5000 and cv2.contourArea(cntr) < 300000) :
                x,y,w,h  = cv2.boundingRect(cntr)
                detections.append([x,y,w,h])
                cv2.rectangle(mask,(x,y),(x+w,y+h),(255),-1)

    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cv2.putText(dmy, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.rectangle(dmy, (x, y), (x + w, y + h), (0, 255, 0), 3)

    cv2.line(dmy, (0, 1080),(1920,0),(100, 255, 255))
    cv2.imshow("", dmy)
    key = cv2.waitKey(5) 
    if key == 27: #esc key stops the process
        break
    
    old_frame = new_frame
    ret, new_frame = cap.read()

print("Finished")