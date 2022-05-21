import os
import cv2
import numpy as np
from collections import Counter
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

FOLDER = "D:/Projects/suspicious-vehicle-detection/candidate_classifiers/color/test_images"
CLT = KMeans(n_clusters=1)
KERNEL = np.ones((13,13),np.uint8)

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
    
    #for logging purposes
    print(perc)
    print(k_cluster.cluster_centers_)
    
    step = 0
    
    for idx, centers in enumerate(k_cluster.cluster_centers_): 
        palette[:, step:int(step + perc[idx]*width+1), :] = centers
        step += int(perc[idx]*width+1)
        
    return palette

for filename in os.listdir(FOLDER):
    img = cv2.imread(FOLDER+"/"+filename)
    color_hists = CLT.fit(img.reshape(-1, 3))
    img_comparison(img, palette_perc(color_hists))