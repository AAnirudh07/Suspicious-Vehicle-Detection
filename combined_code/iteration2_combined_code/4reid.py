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
locationA = "uniq_toll_images"
locationB = "uniq_tea_images"
# iterate over files in
# that directory
for filename in os.listdir(locationA):
    f = os.path.join(locationA, filename)
    # checking if it is a file
    if os.path.isfile(f):
        image_list.append(f)

# print(image_list)
# print(features.shape)  # output (5, 512)
# print(image_list[:3])
# print(features.shape)
directory = "tracks"
print(image_list[:5])
for i in range(len(image_list)):
    path = os.path.join(directory, image_list[i].split("\\")[1][:-5])
    os.mkdir(path)
    cv2.imwrite(path+"\\"+"original.jpeg", cv2.imread(image_list[i]))
    origfeature = extractor([path+"\\"+"original.jpeg"])[0]
    print(i, "/", len(image_list))
    for index, filename in enumerate(os.listdir(locationB)):
        f = os.path.join(locationB, filename)
        feature = extractor([f])[0]
        ans = float(torch.nn.functional.cosine_similarity(
            origfeature, feature, dim=0))
        if(ans > 0.71):
            cv2.imwrite(path+"\\"+filename, cv2.imread(f))
