#-----------------------------------------------------------------------------------------#
import sys
sys.path.append("./Libs") 
import architectures as A
import utilities as U
#-----------------------------------------------------------------------------------------#
import torch
import pandas as pd
import numpy as np
import albumentations as aug
import matplotlib.pyplot as plt
import torch.optim as optim
import os
#-----------------------------------------------------------------------------------------#
from torch.utils.data import DataLoader
from transformers import AdamW
from torch import nn
from sklearn.metrics import accuracy_score
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor, SegformerConfig
from torchvision.transforms import Resize
from tqdm import tqdm
#-----------------------------------------------------------------------------------------#

"""
Step 0: Predefined Parameters.
"""

image_size = 512
DATASET_PATH = "AUM_crop_datasets"
select_label = 60
# learning_rate = 1e-5
learning_rate = 1e-4
# learning_rate = 1e-2
weight_decay = 1e-4
# gamma = 0.95
gamma = 1e-2
step_size = 5
patience = 6
batch_size = 32
# batch_size = 8
# num_epochs = 200
num_epochs = 400
# num_epochs = 1200
model_name = f"lr{learning_rate}_wd{weight_decay}_g{gamma}_ss{step_size}"
CHECKPOINT_PATH = "save_trained_model/" + model_name + ".ckpt"

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)

save_file = "save_trained_model"
if not os.path.exists(save_file):
    os.makedirs(save_file)

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

feature_extractor = SegformerImageProcessor(align=False,
                                            reduce_zero_label=False)
train_dataset = A.ImageSegmentationDataset(root_dir=DATASET_PATH,
                                           feature_extractor=feature_extractor,
                                           transforms=aug.Compose([aug.Resize(
                                                                   height=image_size,
                                                                   width=image_size),
                                                                   aug.Flip(p=0.5)]))
valid_dataset = A.ImageSegmentationDataset(root_dir=DATASET_PATH,
                                           feature_extractor=feature_extractor,
                                           transforms=aug.Resize(height=image_size,
                                                                 width=image_size),
                                           train=False)
print(f"number of training: {len(train_dataset)}")
print(f"number of validation: {len(valid_dataset)}")
encoded_inputs = train_dataset[select_label]
U.view_converted_inputs(encoded_inputs, class_colors)

#-----------------------------------------------------------------------------------------#

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size)
batch = next(iter(train_dataloader))
for k, v in batch.items():
    print(k, v.shape)
print(batch["labels"].shape)

#-----------------------------------------------------------------------------------------#

config = SegformerConfig(
    backbone_name="mit_b0",
    num_classes=len(id2label),
    id2label=id2label,
    label2id=label2id,
    class_colors=class_colors
)

model = SegformerForSemanticSegmentation(config)
optimizer = torch.optim.AdamW(model.parameters(),
                              lr=learning_rate,
                              weight_decay=weight_decay)
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                    mode='min',
                                                    factor=gamma,
                                                    patience=patience,
                                                    min_lr=1e-6)
model.to(device)
print("Model Initialized!")

#-----------------------------------------------------------------------------------------#

best_valid_loss = float("inf")
train_loss_history = []
valid_loss_history = []

for epoch in range(1, num_epochs):  # loop over the dataset multiple times
    print("Epoch:", epoch)
    pbar = tqdm(train_dataloader)
    accuracies = []
    losses = []
    val_accuracies = []
    val_losses = []
    model.train()
    for idx, batch in enumerate(pbar):
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        optimizer.zero_grad()
        outputs = model(pixel_values=pixel_values, labels=labels)
        upsampled_logits = nn.functional.interpolate(outputs.logits,
                                                     size=labels.shape[-2:],
                                                     mode="bilinear",
                                                     align_corners=False)
        predicted = upsampled_logits.argmax(dim=1)
        mask = (labels != 255) 
        pred_labels = predicted[mask].detach().cpu().numpy()
        true_labels = labels[mask].detach().cpu().numpy()
        accuracy = accuracy_score(pred_labels, true_labels)
        loss = outputs.loss
        accuracies.append(accuracy)
        losses.append(loss.item())
        pbar.set_postfix({"Batch": idx,
                          "Pixel-wise accuracy": sum(accuracies)/len(accuracies),
                          "Loss": sum(losses)/len(losses)})
        loss.backward()
        optimizer.step()
    else:
        model.eval()
        with torch.no_grad():
            for idx, batch in enumerate(valid_dataloader):
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(pixel_values=pixel_values, labels=labels)
                upsampled_logits = nn.functional.interpolate(outputs.logits,
                                                             size=labels.shape[-2:],
                                                             mode="bilinear",
                                                             align_corners=False)
                predicted = upsampled_logits.argmax(dim=1)
                mask = (labels != 255) 
                pred_labels = predicted[mask].detach().cpu().numpy()
                true_labels = labels[mask].detach().cpu().numpy()
                accuracy = accuracy_score(pred_labels, true_labels)
                val_loss = outputs.loss
                val_accuracies.append(accuracy)
                val_losses.append(val_loss.item())
                if val_loss < best_valid_loss:
                    best_valid_loss = val_loss
                    torch.save(model.state_dict(), CHECKPOINT_PATH)
    train_loss = sum(losses) / len(losses)
    valid_loss = sum(val_losses) / len(val_losses)
    train_loss_history.append(train_loss)
    valid_loss_history.append(valid_loss)
    print(f"Train Pixel-wise accuracy: {sum(accuracies)/len(accuracies)}\
         Train Loss: {sum(losses)/len(losses)}\
         Val Pixel-wise accuracy: {sum(val_accuracies)/len(val_accuracies)}\
         Val Loss: {sum(val_losses)/len(val_losses)}")
    lr_scheduler.step(valid_loss)
U.loss_history_plot(train_loss_history, valid_loss_history, model_name)

#-----------------------------------------------------------------------------------------#