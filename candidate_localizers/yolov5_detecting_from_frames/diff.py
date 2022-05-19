import pandas as pd
import cv2
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as pp
import numpy as np

import PIL
import os
from PIL import Image
import time
df1 = pd.read_csv("in.csv")
df2 = pd.read_csv("out.csv")
result=[]
start=time.time()
count=0
for ind1 in df1.index:
    original = cv2.imread("in/"+df1['file'][ind1])
    frame_num1 = df1['frame_num'][ind1]
    result.append(0)
    for ind2 in df2.index:
        try:
            original = cv2.imread("in/"+df2['file'][ind1])
            frame_num2 = df2['frame_num'][ind2]
            count+=1
            duplicate = cv2.imread("out/"+df2['file'][ind2])
            #difference = cv2.subtract(original, duplicate)
            #b, g, r = cv2.split(difference)
            dim = (64,64)
            grayA = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            grayB = cv2.cvtColor(duplicate, cv2.COLOR_BGR2GRAY)
            grayA = cv2.resize(grayA, (100, 50)) 
            grayB = cv2.resize(grayB, (100, 50)) 
            if ssim(grayA,grayB)>=65:
                result[-1]=abs(frame_num1-frame_num2)
                break
        except:
            print(df1['file'][ind1],"not found")
        print(frame_num1,frame_num2,result[-1])
        print(count,"df12:: ",ind1*len(df2['file'])+ind2,"/",len(df2['file'])*len(df1['file']))

df1['diff'] = result
df1.to_csv("diff.csv") 
print(start-time.time())