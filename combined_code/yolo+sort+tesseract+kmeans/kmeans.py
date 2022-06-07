import os
import cv2
import numpy as np
from collections import Counter
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import colorsys
FOLDER = "D:\Desktop\Test_vehicle\IN\\runs\\track\yolov5l_osnet_x1_03\\crops\\bus\\36.0"
CLT = KMeans(n_clusters=1)
KERNEL = np.ones((13,13),np.uint8)

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

def img_comparison(img1, img2):
    f, ax = plt.subplots(1, 2, figsize=(10,10))
    ax[0].imshow(img1)
    ax[1].imshow(img2)
    ax[0].axis('off') 
    ax[1].axis('off')
    f.tight_layout()
    plt.show()


def palette_perc(k_cluster):
    width = 300
    palette = np.zeros((50, width, 3), np.uint8)
    
    n_pixels = len(k_cluster.labels_)
    counter = Counter(k_cluster.labels_)
    perc = {}
    for i in counter:
        perc[i] = np.round(counter[i]/n_pixels, 2)
    perc = dict(sorted(perc.items()))
    
    step = 0
    
    for idx, centers in enumerate(k_cluster.cluster_centers_): 
        palette[:, step:int(step + perc[idx]*width+1), :] = centers
        step += int(perc[idx]*width+1)
        
    return palette

