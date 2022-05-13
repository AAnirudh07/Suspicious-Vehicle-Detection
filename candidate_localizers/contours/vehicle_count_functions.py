import os
import re
import cv2
import numpy as np
from os.path import isfile, join
import matplotlib.pyplot as plt

# get file names of the frames
col_frames = os.listdir('data/frames/')
col_frames.sort(key=lambda f: int(re.sub('\D', '', f)))

col_images=[]

knt = 0

for i in col_frames:
    knt+=1
    img = cv2.imread('data/frames/'+i)
    col_images.append(img)
    if knt==100:
        break

#check if loaded
i = 1

for frame in [i, i+1]:
    plt.imshow(cv2.cvtColor(col_images[frame], cv2.COLOR_BGR2RGB))
    plt.title("frame: "+str(frame))
    plt.show()

# convert the frames to grayscale
grayA = cv2.cvtColor(col_images[i], cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(col_images[i+1], cv2.COLOR_BGR2GRAY)
plt.imshow(cv2.absdiff(grayB, grayA), cmap = 'gray')
plt.show()

diff_image = cv2.absdiff(grayB, grayA)

# perform image thresholding
ret, thresh = cv2.threshold(diff_image, 70, 255, cv2.THRESH_BINARY)
plt.imshow(thresh, cmap = 'gray')
plt.show()

# apply image dilation
kernel = np.ones((17,17),np.uint8)
dilated = cv2.dilate(thresh,kernel,iterations = 1)
plt.imshow(dilated, cmap = 'gray')
plt.show()

# plot vehicle detection zone
cv2.line(dilated, (0, 1080),(1920,0),(100, 0, 0),5)
plt.imshow(dilated)
plt.show()

contours, hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

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

for i,cntr in enumerate(contours):
    M = cv2.moments(cntr)
    if M['m00'] != 0:
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        if cy - ((-9/16)*cx + 1080) > 0 and cv2.contourArea(cntr) >= 25:
            valid_cntrs.append(cntr)

print(len(valid_cntrs))

dmy = col_images[0].copy()

cv2.drawContours(dmy, valid_cntrs, -1, (127,200,0), 2)
cv2.line(dmy, (0, 1080),(1920,0),(0, 0, 0))
plt.imshow(dmy)
plt.show()