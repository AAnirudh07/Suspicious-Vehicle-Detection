import cv2
import time
import torch
import json
import sys
import pandas as pd


if(len(sys.argv)!=2):
    print("enter folder to enter pics and create pandas")
    exit()

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=False)
classes = ['car', 'motorcycle', 'vehicle', 'bus', 'truck', 'auto rickshaw', 'rickshaw', 'SUV', 'scooter', 'sedan', 'coupe', 'station wagon', 'hatchback', 'convertible', 'van']
start = time.time()
frame_num=0
values = []
fol = sys.argv[1]
cap = cv2.VideoCapture(fol+'.mp4')
while True:
    ret,frame = cap.read()
    if(not ret):
        break
    if(frame_num%100==0):
        cv2.imwrite("frame.jpg", frame)
        result = model("frame.jpg")
        res = json.loads(str(result.pandas().xyxy[0].to_json(orient="records")))
        for count,row in enumerate(res):
            value = []
            if(row['name'] in classes):
                img_crop = frame[int(row['ymin']):int(row['ymax']), int(row['xmin']):int(row['xmax']), :]
                name = str(frame_num)+"_"+str(count)+".jpg"
                cv2.imwrite(fol+"\\"+name,img_crop)
                value.append(name)
                value.append(row['name'])
                value.append('red')
                value.append(frame_num)
                values.append(value)
    frame_num+=1

df =  pd.DataFrame(values,columns=['file','type','color','frame_num'])
df.to_csv(fol+'.csv')
print((time.time() - start)/1000)