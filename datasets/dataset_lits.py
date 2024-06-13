import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from torchvision.io import read_image
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample

class RandomRotationTransform:
    def __init__(self, degrees=(0, 180), size=(224,224)):
        self.degrees = degrees
        self.size = (224,224)
        
    def __call__(self, image, mask):
        image = transforms.functional.resize(image, self.size)
        mask = transforms.functional.resize(mask, self.size)
        angle = random.uniform(self.degrees[0], self.degrees[1])
        image = F.rotate(image, angle)
        mask = F.rotate(mask, angle)
        return image, mask

class LITSDataset(Dataset):
    def __init__(self,Xlist,Ylist,transform):
        self.xlist = Xlist
        self.ylist = Ylist
        self.transform = transform
    def __len__(self):
        return len(self.xlist)
    def __getitem__(self,idx):
        img = read_image(self.xlist[idx]).float()
        img=img/255.0
        mask = read_image(self.ylist[idx]).to(torch.int64)
        mask = mask/255
        if (self.transform):
            img,mask = self.transform(img,mask)
        sample = {'image': img, 'label': mask[0]}
        return sample

class LITSTestDataset(Dataset):
    def __init__(self,Xlist,Ylist,transform):
        self.xlist = Xlist
        self.ylist = Ylist
        self.transform = transform
    def __len__(self):
        return len(self.xlist)
    def __getitem__(self,idx):
        img = read_image(self.xlist[idx]).float()
        img=img/255.0
        mask = read_image(self.ylist[idx]).to(torch.int64)
        mask = mask/255
        if (self.transform):
            img,mask = self.transform(img,mask)
        sample = {'image': img, 'label': mask}
        return sample