# Paralelizing
import torch
from torchreid.utils import FeatureExtractor
import os
import cv2
extractor = FeatureExtractor(
    model_name='osnet_x1_0',
    device='cpu'
)
# features
image_list = []

# import required module
# assign directory
directory = "uniq_toll_images"
# iterate over files in
# that directory
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(f):
        image_list.append(f)

features = extractor(image_list)
# print(image_list)
# print(features.shape)  # output (5, 512)
# print(image_list[:3])
# print(features.shape)
for i in range(len(features)):
    flag = 0
    ans = 0
    for j in range(i+1, min(len(features), i+10)):
        ans = float(max(torch.nn.functional.cosine_similarity(
            features[i], features[j], dim=0), ans))
        if(ans > 0.71):
            flag = 1
            break
    if(flag == 0):
        img = cv2.imread(image_list[i])
        # print(write_directory+"\\"+image_list[i].split("\\")[1])
        cv2.imwrite(write_directory+"\\"+image_list[i].split("\\")[1], img)
