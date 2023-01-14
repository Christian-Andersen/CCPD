import os
import pandas as pd
import torch
import torchvision
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset


class CCPDDataset(Dataset):
    def __init__(self, annotations_file):
        self.file = pd.read_csv(annotations_file)
        self.img_dir = 'CCPD2019'

    def __len__(self):
        return len(self.file)

    def __getitem__(self, idx):
        name = self.file.iloc[idx, 0]
        name = os.path.join(self.img_dir, name)
        corners = torch.tensor(list(self.file.iloc[idx, 2:10]))
        top = min(corners[5], corners[7])
        bottom = max(corners[1], corners[3])
        left = min(corners[2], corners[4])
        right = max(corners[0], corners[6])
        boxes = torch.tensor([[left, top, right, bottom]])
        labels = torch.tensor(list(self.file.iloc[idx, 10:]))
        return name, {'boxes': boxes, 'labels': labels}


f = torchvision.transforms.PILToTensor()
f_inv = torchvision.transforms.ToPILImage()
# dataset1 = CCPDDataset('train.csv')
# dataset2 = CCPDDataset('val.csv')
# dataset = ConcatDataset(datasets=[dataset1, dataset2])
dataset = CCPDDataset('test.csv')
length = len(dataset)
for idx, datum in enumerate(dataset):
    name = datum[0]
    with Image.open(name) as img:
        save_location = os.path.join('plates', 'test', name.split('-')[-3]+'.'+str(idx)+'.jpg')
        box = datum[1]['boxes'][0].tolist()
        image = img.crop(box)
        image.save(save_location)
    if idx%100==0:
        print(str(100*idx/len(dataset))+'%')
