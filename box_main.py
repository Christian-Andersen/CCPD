import torch
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader, ConcatDataset
from imports import create_model, BoxDataset


def main():
    batch_size = 2
    num_epochs = 100
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    kwargs = {'batch_size': batch_size, 'shuffle': True, 'pin_memory': True, 'collate_fn': list}
    transforms = T.RandomChoice([
        T.RandomHorizontalFlip(0.5),
        T.ColorJitter(0.5,0.5,0.5,0.5),
        T.GaussianBlur(5),
        T.RandomAffine(45),
    ])
    dataset = ConcatDataset([BoxDataset('train.csv', transforms), BoxDataset('val.csv', transforms)])
    loader = DataLoader(dataset, **kwargs)
    model = create_model(2, True)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    model.train()
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        for batch_idx, batch in enumerate(loader):
            images = []
            targets = []
            for i in batch:
                images.append(i[0].to(device))
                targets.append({'boxes':i[1]['boxes'].to(device), 'labels':i[1]['labels'].to(device)})
            optimizer.zero_grad()
            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())
            loss.backward() #type:ignore
            optimizer.step()
            if batch_idx%100==0:
                torch.save(model.state_dict(), 'box_model.pt')
                print("Loss: {:.4f}\t[{:.2%}]".format(loss, batch_idx/len(loader)))
                torch.save(model.state_dict(), 'box_model2.pt')


if __name__ == '__main__':
    main()
