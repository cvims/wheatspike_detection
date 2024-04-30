"""
Wheat dataset loader for training.
"""
# =============================================================================
# Imports
# =============================================================================
import os
import re
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torchvision.ops import box_convert
from torchvision.utils import draw_bounding_boxes
from torchvision.utils import draw_segmentation_masks
from torch.utils.data import Dataset

sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)))
from utils.utils_dataset import extract_coco_annotations, nostdout, get_transform_albumentation, set_seed


# =============================================================================
class WheatDataset(Dataset):
    def __init__(self, root, transforms, plot):
        self.root = root
        self.transforms = transforms
        self.plot = plot

        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.annotations_path = os.path.join(self.root, "annotations.json")

    def __getitem__(self, idx):
        # loads picture
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        img = np.array(Image.open(img_path).convert("RGB"))

        # get bbox, area, segmentation, classes, and mask
        with nostdout():
            anno, class_name, mask = extract_coco_annotations(self.annotations_path, idx)
        boxes, classes, area, segmentations = [], [], [], []
        for dict in anno:
            segmentations.append(dict["segmentation"])
            boxes.append(dict["bbox"])
            area.append(dict["area"])
            classes.append(class_name)
        
        # convert and create target
        boxes = torch.as_tensor(boxes, dtype=torch.float32).numpy()
        mask = mask.to(dtype=torch.uint8).permute(1,2,0).numpy()
        labels = torch.ones((len(boxes),), dtype=torch.int64)
        img_id = int(re.findall(r'\d+', img_path)[-2])
        image_id = torch.tensor([img_id])
        anno_idx = torch.tensor([idx+1])
        area = torch.as_tensor(area, dtype=torch.float32)
        iscrowd = torch.zeros((len(boxes)), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["masks"] = mask
        target["labels"] = labels
        target["image_id"] = image_id
        target["anno_idx"] = anno_idx
        target["area"] = area
        target["iscrowd"] = iscrowd
        
        # transform img, bboxes and img
        if self.transforms is not None:
            transformed = self.transforms(
                                image=img, 
                                mask=target["masks"], 
                                bboxes=target["boxes"], 
                                class_labels=target["labels"])

            img = torch.from_numpy(transformed['image']).permute(2,0,1).to(torch.float32) / 255
            target["masks"] = torch.from_numpy(transformed['mask']).to(torch.uint8).permute(2,0,1)
            target["boxes"] = box_convert(torch.Tensor(transformed["bboxes"]), "xywh", "xyxy").to(torch.float32)

        # plot img, boxes and masks
        if self.plot:
            bboxes = draw_bounding_boxes((img*255).to(torch.uint8), target["boxes"], colors=(255,0,0))
            masks = draw_segmentation_masks(bboxes, (target["masks"].sum(dim=0)) > 0, colors=(0,0,255), alpha=0.5)
            fig, ax = plt.subplots(figsize=(14, 11))
            plt.imshow(masks.permute(1,2,0))
            plt.show()

        return img, target

    def __len__(self):
        return len(self.imgs)