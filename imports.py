import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.models import efficientnet_b0
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def create_model(num_classes, load):
    model = fasterrcnn_resnet50_fpn_v2(weights='DEFAULT')  # type:ignore
    in_features = model.roi_heads.box_predictor.cls_score.in_features  # type:ignore
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    if load:
        model.load_state_dict(torch.load('box_model.pt'))
    return model

def plate_model(load=False):
    model = efficientnet_b0(weights='DEFAULT')
    model.classifier[1] = nn.Linear(in_features=1280, out_features=234, bias=True)
    if load:
        model.load_state_dict(torch.load('plate_model.pt'))
    return model

class CCPDDataset(Dataset):
    def __init__(self, annotations_file, transform=None):
        self.file = pd.read_csv(annotations_file)
        self.img_dir = '../data/CCPD2019'
        self.transform = transform

    def __len__(self):
        return len(self.file)

    def __getitem__(self, idx):
        name = self.file.iloc[idx, 0]
        image = read_image(os.path.join(self.img_dir, name)).div(255)
        corners = torch.tensor(list(self.file.iloc[idx, 2:10]))
        top = min(corners[5], corners[7])
        bottom = max(corners[1], corners[3])
        left = min(corners[2], corners[4])
        right = max(corners[0], corners[6])
        boxes = torch.tensor([[left, top, right, bottom]])
        labels = torch.tensor(list(self.file.iloc[idx, 10:]))
        if self.transform:
            image = self.transform(image)
        return image, {'boxes': boxes, 'labels': labels}


class BoxDataset(Dataset):
    def __init__(self, annotations_file, transform=None):
        self.file = pd.read_csv(annotations_file)
        self.img_dir = '../data/CCPD2019'
        self.transform = transform

    def __len__(self):
        return len(self.file)

    def __getitem__(self, idx):
        name = self.file.iloc[idx, 0]
        image = read_image(os.path.join(self.img_dir, name)).div(255)
        corners = torch.tensor(list(self.file.iloc[idx, 2:10]))
        top = min(corners[5], corners[7])
        bottom = max(corners[1], corners[3])
        left = min(corners[2], corners[4])
        right = max(corners[0], corners[6])
        boxes = torch.tensor([[left, top, right, bottom]])
        labels = torch.tensor([1])
        if self.transform:
            image = self.transform(image)
        return image, {'boxes': boxes, 'labels': labels}, self.file.iloc[idx, :]


class PlateDataset(Dataset):
    def __init__(self, split, transform=None):
        self.img_dir = os.path.join('../data/plates', split)
        self.files = os.listdir(self.img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        image = read_image(os.path.join(self.img_dir, name)).div(255)
        labels = torch.tensor([int(x) for x in name.split('.')[0].split('_')])
        if self.transform:
            image = self.transform(image)
        return image, labels
