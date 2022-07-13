import torch
from torchreid.utils import FeatureExtractor
import os
import cv2
extractor = FeatureExtractor(
    model_name='osnet_x1_0',
    device='cuda'
)
# features
image_list = []

# import required module
# assign directory
directory = "tea_images"
write_directory = "uniq_tea_images"
# iterate over files in
# that directory
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(f):
        image_list.append(f)

# print(image_list)
# print(features.shape)  # output (5, 512)
# print(image_list[:3])
# print(features.shape)
for i in range(len(image_list)):
    flag = 0
    ans = 0
    temp_image_list = []
    for k in range(i, i+11):
        temp_image_list.append(image_list[k])
    features = extractor(temp_image_list)
    print(i, "/", len(image_list))
    for j in range(i+1, min(len(temp_image_list)-1, i+10)):
        ans = float(max(torch.nn.functional.cosine_similarity(
            features[0], features[j-i], dim=0), ans))
        if(ans > 0.71):
            flag = 1
            break
    if(flag == 0):
        img = cv2.imread(image_list[i])
        # print(write_directory+"\\"+image_list[i].split("\\")[1])
        cv2.imwrite(write_directory+"\\"+image_list[i].split("\\")[1], img)
