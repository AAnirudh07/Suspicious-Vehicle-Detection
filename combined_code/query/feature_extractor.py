import os
from torchreid.utils import FeatureExtractor
import torch
extractor = FeatureExtractor(
    model_name='osnet_x1_0',
    device='cuda'
)
# features 
image_list = []

# import required module
# assign directory
directory = 'only_image'

# iterate over files in
# that directory
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(f):
        image_list.append(f)

features = extractor(image_list)
print(image_list)
print(features.shape)  # output (5, 512)
for i in range(len(features)):
    for j in range(len(features)):
        ans = torch.nn.functional.cosine_similarity(
            features[i], features[j], dim=0)
        print(i, j, "::", ans, ans.item())
