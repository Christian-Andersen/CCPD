import os
import torchvision
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from imports import create_model, plate_model
from torchvision.transforms.functional import crop
from torchvision.io import read_image

provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏",
             "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼",
             "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学",
             "O"]
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N',
             'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'O']
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P',
       'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4',
       '5', '6', '7', '8', '9', 'O']
BREAKS = [0, 
len(provinces),
len(provinces)+len(alphabets),
len(provinces)+len(alphabets)+1*len(ads),
len(provinces)+len(alphabets)+2*len(ads),
len(provinces)+len(alphabets)+3*len(ads),
len(provinces)+len(alphabets)+4*len(ads),
len(provinces)+len(alphabets)+5*len(ads)]

class CCPDDataset(Dataset):
    def __init__(self, annotations_file, transform=None):
        self.file = pd.read_csv(annotations_file)
        self.img_dir = 'CCPD2019'
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
        return image, {'boxes': boxes, 'labels': labels}, name

device = torch.device('cuda')
dataset = CCPDDataset('test.csv')
loader = DataLoader(dataset, pin_memory=True, shuffle=True)
model1 = create_model(2, True).to(device)
model1.eval()
model2 = plate_model(True).to(device)
model2.eval()
torch.set_grad_enabled(False)

def random_test(image, target):
    real_plate_ints = target['labels'].squeeze().tolist()
    output = model1(image.unsqueeze(dim=0).to(device))[0]
    if not output['labels'].numel():
        return False
    box = output['boxes'][0:1]
    top = round(box[0,1].item())
    left = round(box[0,0].item())
    height = round(box[0,3].item() - top)
    width = round(box[0,2].item() - left)
    plate_image = crop(image, top, left, height, width).unsqueeze(dim=0)
    logits = model2(torchvision.transforms.Resize((256, 256))(plate_image).to(device))[0]
    plate = ''
    real_plate = ''
    for i in range(7):
        if i==0:
            plate += provinces[logits[BREAKS[i]:BREAKS[i+1]].argmax()]
            real_plate += provinces[real_plate_ints[i]]
        if i==1:
            plate += alphabets[logits[BREAKS[i]:BREAKS[i+1]].argmax()]
            real_plate += alphabets[real_plate_ints[i]]
        else:
            plate += ads[logits[BREAKS[i]:BREAKS[i+1]].argmax()]
            real_plate += ads[real_plate_ints[i]]
    return plate==real_plate

correct = {'ccpd_db': 0, 'ccpd_blur': 0, 'ccpd_fn': 0, 'ccpd_rotate': 0, 'ccpd_tilt': 0, 'ccpd_challenge': 0}
total = {'ccpd_db': 0, 'ccpd_blur': 0, 'ccpd_fn': 0, 'ccpd_rotate': 0, 'ccpd_tilt': 0, 'ccpd_challenge': 0}
for batch_idx, (image, target, name) in enumerate(loader):
    folder = name[0].split('/')[0]
    if folder not in correct:
        correct[folder] = 0
        total[folder] = 0
    correct[folder] += random_test(image.squeeze(), target)
    total[folder] += 1
    if batch_idx%100==0:
        print("[{:.2%}]".format(batch_idx/len(loader)))
        for key in correct:
            if total[key]!=0:
                print("{}\t{:.2%}".format(key[5:11], correct[key]/total[key]))
        print("\t\tTotal AP: {:.2%}\n".format(sum(correct.values())/sum(total.values())))
