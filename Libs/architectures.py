import os
import cv2
#-----------------------------------------------------------------------------------------#
from torch.utils.data import Dataset, DataLoader
#-----------------------------------------------------------------------------------------#

class ImageSegmentationDataset(Dataset):
    def __init__(self, root_dir, feature_extractor, transforms=None, train=True):
        self.root_dir = root_dir
        self.feature_extractor = feature_extractor
        self.train = train
        self.transforms = transforms
        sub_path = "train" if self.train else "test"
        self.img_dir = os.path.join(self.root_dir, "images", sub_path)
        self.ann_dir = os.path.join(self.root_dir, "mask", sub_path)
        image_file_names = []
        for root, dirs, files in os.walk(self.img_dir):
            image_file_names.extend(files)
        self.images = sorted(image_file_names)
        annotation_file_names = []
        for root, dirs, files in os.walk(self.ann_dir):
            annotation_file_names.extend(files)
        self.annotations = sorted(annotation_file_names)
        assert len(self.images) == len(self.annotations), "There must be as many images as there are segmentation maps"
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        image = cv2.imread(os.path.join(self.img_dir, self.images[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        segmentation_map = cv2.imread(os.path.join(self.ann_dir, self.annotations[idx]))
        segmentation_map = cv2.cvtColor(segmentation_map, cv2.COLOR_BGR2GRAY)
        if self.transforms is not None:
            augmented = self.transforms(image=image, mask=segmentation_map)
            encoded_inputs = self.feature_extractor(augmented['image'], augmented['mask'], return_tensors="pt")
        else:
            encoded_inputs = self.feature_extractor(image, segmentation_map, return_tensors="pt")
        for k,v in encoded_inputs.items():
            encoded_inputs[k].squeeze_() 
        return encoded_inputs



