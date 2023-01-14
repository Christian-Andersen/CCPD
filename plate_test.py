import torchvision
import torch
import torch.nn as nn
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
    batch_size = 100
    device = torch.device('cpu')
    print(device)
    kwargs = {'batch_size': batch_size, 'shuffle': True, 'pin_memory': True}
    dataset = PlateDataset('test', transform=torchvision.transforms.Resize((256, 256)))
    loader = DataLoader(dataset, **kwargs)
    model = plate_model(True).to(device)
    torch.set_grad_enabled(False)
    model.eval()
    correct = [[] for _ in range(7)]
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        for i in range(7):
            pred = output[:,BREAKS[i]:BREAKS[i+1]].argmax(dim=1)
            real = target[:,i]
            correct[i] += pred.eq(real).tolist()
        total_correct = torch.tensor(correct).sum(dim=0)
        acc = round(100*total_correct.eq(7).sum().div(len(total_correct)).item(), 2)
        print(acc, [round(100*sum(x)/len(x), 2) for x in correct])


if __name__ == '__main__':
    main()
