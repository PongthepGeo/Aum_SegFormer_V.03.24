#-----------------------------------------------------------------------------------------#
import sys
sys.path.append("./Libs") 
import architectures as A
import utilities as U
#-----------------------------------------------------------------------------------------#
import os
import pandas as pd
import cv2
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
#-----------------------------------------------------------------------------------------#
from sklearn.metrics import accuracy_score
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerConfig
from transformers import SegformerImageProcessor
from torch import nn
#-----------------------------------------------------------------------------------------#
params = {
	'savefig.dpi': 300,  
	'figure.dpi' : 300,
	'axes.labelsize':10,  
	'axes.titlesize':10,
	'axes.titleweight': 'bold',
	'legend.fontsize': 8,
	'xtick.labelsize':8,
	'ytick.labelsize':8,
	'font.family': 'serif',
}
matplotlib.rcParams.update(params)
#-----------------------------------------------------------------------------------------#

DATASET_PATH = "AUM_crop_datasets"
img_test = "/images/test/img_sen2_2019_01_14_0_1024.png"
mask_test = "/mask/test/label_sen2_2019_01_14_0_1024.png"

learning_rate = 1e-2
weight_decay = 1e-4
# gamma = 0.95
gamma = 1e-2
step_size = 5
model_name = f"lr{learning_rate}_wd{weight_decay}_g{gamma}_ss{step_size}"
CHECKPOINT_PATH = "save_trained_model/" + model_name + ".ckpt"

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)

#-----------------------------------------------------------------------------------------#

class_labels = {
	0: "background",
	1: "Rice",
	2: "Forest",
	3: "Pond"
}
id2label = class_labels
label2id = {v: k for k, v in id2label.items()}
print(label2id)
print(f"number of classes: {len(id2label)}")
class_colors = {
	0: (255, 255, 255),          # white for background     
	1: (231, 113, 72),           # Rice                  
	2: (112, 230, 68),           # Forest                
	3: (89, 159, 243)            # Pond                  
}

#-----------------------------------------------------------------------------------------#

config = SegformerConfig(
    backbone_name="mit_b0",
    num_classes=len(id2label),
    id2label=id2label,
    label2id=label2id
)

model = SegformerForSemanticSegmentation(config=config)
state_dict = torch.load(CHECKPOINT_PATH)
model.load_state_dict(state_dict)
model = model.to(device)

#-----------------------------------------------------------------------------------------#

image = cv2.imread(DATASET_PATH + img_test)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
mask = Image.open(DATASET_PATH + mask_test).convert("L")

mask_array = np.array(mask)
reverted_mask_array = U.revert_pixels(mask_array, class_colors)
fig, axs = plt.subplots(1, 2, figsize=(20, 10))
axs[0].imshow(image)
axs[1].imshow(reverted_mask_array)
plt.show()

#-----------------------------------------------------------------------------------------#

image_processor_inference = SegformerImageProcessor(do_random_crop=False, do_pad=False)
pixel_values = image_processor_inference(image, return_tensors="pt").pixel_values.to(device)
print(pixel_values.shape)

model.eval()
outputs = model(pixel_values=pixel_values)
logits = outputs.logits.cpu()
print(logits.shape)

#-----------------------------------------------------------------------------------------#

size=(image.shape[0], image.shape[1]), 
upsampled_logits = nn.functional.interpolate(logits,
                size=(image.shape[0], image.shape[1]),  
                mode="bilinear",
                align_corners=False)
seg = upsampled_logits.argmax(dim=1)[0]

#-----------------------------------------------------------------------------------------#

color_seg = U.revert_pixels_pytorch(seg, class_colors)
img = np.array(image) * 0.5 + color_seg * 0.5
img = img.astype(np.uint8)
fig, axs = plt.subplots(1, 2, figsize=(20, 10))
axs[0].imshow(img)
axs[1].imshow(color_seg)
plt.show()

#-----------------------------------------------------------------------------------------#