import os

import numpy as np
import torch
from torch.utils.data.dataset import Subset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder

from datasets.cutpast_transformation import *
from PIL import Image
from glob import glob
import random

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from torch.utils.data import ConcatDataset



def get_exposure_set(image_size=224, count=5000, tiny_percent=0.2, category='carpet'):

    tiny_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.AutoAugment(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    train_transform_cutpasted = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((image_size, image_size)),
        CutPasteUnion(transform=transforms.Compose([transforms.ToTensor(), ])),
    ])

    tiny_count =  int(count*tiny_percent)
    cutpaste_count = int(count*(1-tiny_percent))

    train_ds_mvtech_cutpasted = []

    imagenet_exposure = ImageNetExposure(root='./tiny-imagenet-200', count=tiny_count, transform=tiny_transform)

    train_ds_mvtech_cutpasted.append(
        MVTecDataset_Cutpasted(root='./mvtec_anomaly_detection', train=True, category=category, transform=train_transform_cutpasted, count=cutpaste_count))

    exposureset = torch.utils.data.ConcatDataset([imagenet_exposure, train_ds_mvtech_cutpasted])
    return exposureset


class ImageNetExposure(Dataset):
    def __init__(self, root, count, transform=None):
        self.transform = transform
        image_files = glob(os.path.join(root, 'train', "*", "images", "*.JPEG"))
        if count==-1:
            final_length = len(image_files)
        else:
            random.shuffle(image_files)
            final_length = min(len(image_files), count)
        self.image_files = image_files[:final_length]
        self.image_files.sort(key=lambda y: y.lower())

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file)
        image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, 1

    def __len__(self):
        return len(self.image_files)

class MVTecDataset_Cutpasted(Dataset):
    def __init__(self, root, category, transform=None, train=True, count=-1):
        self.transform = transform
        self.image_files = []
        print("category MVTecDataset_Cutpasted:", category)
        if train:
            self.image_files = glob(os.path.join(root, category, "train", "good", "*.png"))
        else:
            image_files = glob(os.path.join(root, category, "test", "*", "*.png"))
            normal_image_files = glob(os.path.join(root, category, "test", "good", "*.png"))
            anomaly_image_files = list(set(image_files) - set(normal_image_files))
            self.image_files = image_files
        if count!=-1:
            if count<len(self.image_files):
                self.image_files = self.image_files[:count]
            else:
                t = len(self.image_files)
                for i in range(count-t):
                    self.image_files.append(random.choice(self.image_files[:t]))
        self.image_files.sort(key=lambda y: y.lower())
        self.train = train
    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file)
        image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, 1

    def __len__(self):
        return len(self.image_files)