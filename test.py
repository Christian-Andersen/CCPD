import os
import pickle
import torchvision
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.io import read_image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from imports import create_model, CCPDDataset


def main():
    batch_size = 2
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    kwargs = {'batch_size': batch_size, 'shuffle': True, 'pin_memory': True, 'collate_fn': list}
    dataset = CCPDDataset('test.csv')
    loader = DataLoader(Subset(dataset, range(1000)), **kwargs)
    model = create_model(91)
    model.load_state_dict(torch.load('model_dict.pt'))
    model = model.to(device)
    model.train()
    loss = {'loss_classifier': [], 'loss_box_reg': [], 'loss_objectness': [], 'loss_rpn_box_reg': []}
    torch.set_grad_enabled(False)
    for batch_idx, batch in enumerate(loader):
        images = []
        targets = []
        for i in batch:
            images.append(i[0].to(device))
            targets.append({'boxes':i[1]['boxes'].to(device), 'labels':i[1]['labels'].to(device)})
        loss_dict = model(images, targets)
        for key in loss_dict:
            loss[key].append(loss_dict[key])
        if batch_idx%10==0:
            prog = batch_idx/len(loader)
            print("[{:.2%}]".format(prog), end='\r')
    for key, value in loss.items():
        print(key, sum(value)/len(value))

if __name__ == '__main__':
    main()
