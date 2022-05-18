import torch
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=False)
import cv2
import json
import numpy as np
import skvideo
import json
import gc
from imutils.video import WebcamVideoStream

gc.collect()
ffmpeg_path = "D:/Apps/ffmpeg/bin"
skvideo.setFFmpegPath(ffmpeg_path)

import math

import skvideo
import skvideo.io
import os

cv2.namedWindow("frame", cv2.WINDOW_NORMAL) 
def create_mask(img, colors):
    """
    Creates a binary mask from HSV image using given colors.
    """
    HSV_RANGES = {
        # red is a major color
        'red': [
            {
                'lower': np.array([159, 50, 70]),
                'upper': np.array([179, 255, 255])
            },
            {
                'lower': np.array([0, 50, 70]),
                'upper': np.array([9, 255, 255])
            }
        ],
        # yellow is a minor color
        'yellow': [
            {
                'lower': np.array([15, 0, 0]),
                'upper': np.array([36, 255, 255])
            }
        ],
        # green is a major color
        'green': [
            {
                'lower': np.array([36, 0, 0]),
                'upper': np.array([86, 255, 255])
            }
        ],
        # cyan is a minor color
        'cyan': [
            {
                'lower': np.array([81, 39, 64]),
                'upper': np.array([100, 255, 255])
            }
        ],
        # blue is a major color
        'blue': [
            {
                'lower': np.array([100, 150, 0]),
                'upper': np.array([140, 255, 255])
            }
        ],
        'orange': [
            {
                'lower': np.array([18, 40, 90]),
                'upper': np.array([27, 255, 255])
            }
        ],
        # violet is a minor color
        'violet': [
            {
                'lower': np.array([141, 39, 64]),
                'upper': np.array([160, 255, 255])
            }
        ],
        # next are the monochrome ranges
        # black is all H & S values, but only the lower 25% of V
        'black': [
            {
                'lower': np.array([97, 39, 0]),
                'upper': np.array([179, 255, 30])
            }
        ],
        # gray is all H values, lower 15% of S, & between 26-89% of V
        'gray': [
            {
                'lower': np.array([0, 0, 64]),
                'upper': np.array([179, 38, 228])
            }
        ],
        # white is all H values, lower 15% of S, & upper 10% of V
        'white': [
            {
                'lower': np.array([0, 0, 229]),
                'upper': np.array([180, 38, 255])
            }
        ]
    }
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # noinspection PyUnresolvedReferences
    mask = np.zeros((hsv_img.shape[0], hsv_img.shape[1]), dtype=np.uint8)

    for color in colors:
        for color_range in HSV_RANGES[color]:
            # noinspection PyUnresolvedReferences
            mask += cv2.inRange(
                hsv_img,
                color_range['lower'],
                color_range['upper']
            )
    return cv2.countNonZero(mask)  # np.count_nonzero(mask==1)


def color_picker(img):
    masks = []
    colors = {"0": "red", '1': "blue", '2': "yellow",
              '3': "green", '4': "orange", '5': "black", '6': "white"}
    masks.append(('0', create_mask(img, ['red'])))
    masks.append(('1', create_mask(img, ['blue'])/2))
    masks.append(('2', create_mask(img, ['yellow'])/4))
    masks.append(('3', create_mask(img, ['green'])/5))
    masks.append(('4', create_mask(img, ['orange'])))
    masks.append(('5', create_mask(img, ['black'])/10))
    masks.append(('6', create_mask(img, ['white'])))
    masks.sort(key=lambda x: x[1], reverse=True)
    return colors[masks[0][0]]

width = 0
height = 0

def predict(vid_path):
    json_data=[]
    count = 0
    v_id=0
    center_points_prev_frame = []
    tracking_objects = {}
    center_points_prev_frame = []
    vid = WebcamVideoStream(vid_path).start()
    output = []
    video = skvideo.io.FFmpegWriter('output.mp4',{"-pix_fmt":"bgr24"},{"-q":"0", "-color_primaries": "3", "-color_trc": "1", "-colorspace": "1"})
    while(True):
        #print(idx)
        img = vid.read()       
        #cv2.line(img, pt1=(100,100), pt2=(400,400), color=(0,0,255), thickness=10)
        count+=1
        if img is None:
            break
        center_points_cur_frame = []
        cv2.imwrite("frame.jpg", img)
        result = model("frame.jpg")
        print(count)
        res = json.loads(
            str(result.pandas().xyxy[0].to_json(orient="records")))
        json_data.append([])
        for idx,pt2 in tracking_objects.items() :
            tracking_objects[idx][1]=False
        for row in res:
            if(row['name'] in ['car', 'vehicle', 'bus', 'truck', 'auto rickshaw', 'rickshaw', 'SUV', 'scooter', 'sedan', 'coupe', 'station wagon', 'hatchback', 'convertible', 'van'] and int(row['ymax'])>600 and int(row['xmin'])>400):
                
                cv2.rectangle(img, (int(row['xmin']),int(row['ymin'])),(int(row['xmax']),int(row['ymax'])),(255, 0, 0), 2)
                img_crop = img[int(row['ymin']):int(row['ymax']), int(row['xmin']):int(row['xmax']), :]
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                hsv_img = cv2.cvtColor(img_crop, cv2.COLOR_BGR2HSV)
                x, y, w, h = int(row['xmin']),int(row['ymin']),int(row['xmax'])-int(row['xmin']),int(row['ymax'])-int(row['ymin'])
                #print(type(x),type(y),type(w),type(h))
                # h, s, v = hsv_img[:, :, 0], hsv_img[:, :, 1], hsv_img[:, :, 2]
                # clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
                cx = (x + x + w) // 2
                cy = (y + y + h) // 2
                center_points_cur_frame.append((cx, cy))
                # h, s, v = h, s, clahe.apply(v)
                # hsv_img = np.dstack((h, s, v))
                # img_crop = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
                color = color_picker(np.array(img_crop.copy(), dtype=np.uint8))
                vehicleid = -1
                for idx,pt2 in tracking_objects.items() :
                            distance = math.hypot(pt2[0][0] - cx, pt2[0][1] - cy)
                            if distance < 350:
                                tracking_objects[idx] = [(cx,cy),True]
                                vehicleid=idx
                                break
                if(vehicleid==-1):
                    vehicleid = v_id
                    tracking_objects[v_id] = [(cx,cy),True]
                    v_id+=1

                cv2.putText(img,"Vehicle"+str(vehicleid)+"{"+color+str(row['name'])+"}", (int(row['xmin']), int(row['ymin'])-10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
                json_data[count-1].append({"vehicle"+str(vehicleid) :"{"+color+str(row['name'])+"}"})
        to_be_del=[]
        for idx,pt2 in tracking_objects.items() :
            if(pt2[1]==False):
                to_be_del.append(idx)
        for idx in to_be_del:
            del tracking_objects[idx]
                # center_points_prev_frame=center_points_cur_frame
        cv2.imshow("frame",img)
        key = cv2.waitKey(1)
        if key == 27:
            break
        video.writeFrame(img)
    output = np.array(output)
    fps = 25
    crf = 17
    json_string = json.dumps(json_data)
    # Writing to sample.json
    with open("output.json", "w") as outfile:
        outfile.write(json_string)
    # outputdata = output.astype(np.uint8)
    # skvideo.io.vwrite("output.mp4", outputdata,outputdict={"-pix_fmt":"rgb24"})
    print(os.path.join(os.getcwd(), "output.mp4"))

predict("input.mp4")