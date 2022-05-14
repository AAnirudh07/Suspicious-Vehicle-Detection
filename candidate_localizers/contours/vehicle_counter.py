import os
import re
import cv2
import numpy as np
from os.path import isfile, join
import matplotlib.pyplot as plt

kernel = np.ones((9,9),np.uint8)
font = cv2.FONT_HERSHEY_SIMPLEX

# get file names of the frames
col_frames = os.listdir('data/frames/')
col_frames.sort(key=lambda f: int(re.sub('\D', '', f)))

col_images=[]

knt = 0

for i in col_frames:
    knt+=1
    img = cv2.imread('data/frames/'+i)
    col_images.append(img)
    if knt==300:
        break


for i in range(len(col_images)-1):

    # frame differencing
    grayA = cv2.cvtColor(col_images[i], cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(col_images[i+1], cv2.COLOR_BGR2GRAY)
    diff_image = cv2.absdiff(grayB, grayA)
    
    # image thresholding
    ret, thresh = cv2.threshold(diff_image, 70, 255, cv2.THRESH_BINARY)
    
    # image dilation
    dilated = cv2.dilate(thresh,kernel,iterations = 1)
    
    # find contours
    contours, hierarchy = cv2.findContours(dilated.copy(), cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    
    # shortlist contours appearing in the detection zone

    valid_cntrs = []
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
    for cntr in contours:
        M = cv2.moments(cntr)
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            if cy - ((-9/16)*cx + 1080) > 0 and cv2.contourArea(cntr) >= 12000:
                valid_cntrs.append(cntr)

    # add contours to original frames
    
    dmy = col_images[i].copy()
    cv2.drawContours(dmy, valid_cntrs, -1, (127,200,0), 2)
    
    cv2.putText(dmy, "vehicles detected: " + str(len(valid_cntrs)), (55, 15), font, 0.6, (0, 180, 0), 2)
    cv2.line(dmy, (0, 1080),(1920,0),(100, 255, 255))
    cv2.imshow("", dmy)
    key = cv2.waitKey(5) 
    if key == 27: #esc key stops the process
        break
