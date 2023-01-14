import torchvision
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from imports import PlateDataset, plate_model

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


def main():
    batch_size = 20
    num_epochs = 100
    device = torch.device('cuda')
    print(device)
    kwargs = {'batch_size': batch_size, 'shuffle': True, 'pin_memory': True}
    dataset = PlateDataset('train', transform=torchvision.transforms.Resize((256, 256)))
    loader = DataLoader(dataset, **kwargs)
    model = plate_model(True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    model.train()
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            losses = torch.empty((7,))
            for i in range(7):
                losses[i] = F.cross_entropy(output[:,BREAKS[i]:BREAKS[i+1]], target[:,i])
            loss = losses.sum()
            loss.backward()
            optimizer.step()
            if batch_idx%100==0:
                torch.save(model.state_dict(), '../models/CCPD/plate_model.pt')
                print("Loss: {:.4f}\t[{:.2%}]".format(loss, batch_idx/len(loader)))
                torch.save(model.state_dict(), '../models/CCPD/plate_model2.pt')


if __name__ == '__main__':
    main()
