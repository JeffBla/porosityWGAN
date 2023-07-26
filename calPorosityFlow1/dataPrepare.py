from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset
import torch.nn as nn
import torch

from targetGenerator import Generator

class ctPercentDataset(Dataset):
    def __init__(self, ct, percent, transform=None, target_transform=None):
        self.ct = ct
        self.percent = percent
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.ct.shape[0]

    def __getitem__(self, idx):
        image = self.ct[idx]
        label = self.percent[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

def dataSet(NUM_IMAGE= 1000, transform=None, target_transform=None, test_size=0.33, random_seed=None):
    ''' return train test dataset '''
    gen = Generator()
    gen.loadGeneraor()
    
    gen.set_extractPercent()

    fakes = torch.tensor([])
    percents = torch.tensor([])
    if gen.iscuda:
        fakes = fakes.cuda()
        percents = percents.cuda()

    for i in range(int(NUM_IMAGE/ gen.batchSize)):   
        fake_imgs, percent = gen.generateFakeImageAndPercent()     
        fakes = torch.cat((fakes, fake_imgs))
        percents = torch.cat((percents, percent))
    
    gen.unset_extractPercent()

    # return ct to percent data
    fakes = fakes.permute(0,2,3,1).view(-1, 1).detach().cpu()
    percents = percents.permute(0,2,3,1).reshape(-1, 3).detach().cpu()

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(fakes, percents,
                                                         test_size=test_size,
                                                         random_state=random_seed)

    return (ctPercentDataset(X_train, y_train, transform, target_transform),
            ctPercentDataset(X_test, y_test, transform, target_transform))


if __name__ == '__main__':
    dataSet()