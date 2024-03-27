from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import torch.utils.data as data
import matplotlib.image as mpimg
from torchvision import transforms
import random
import torch
from torch.utils.data import Dataset
import pandas as pd
from torchvision import transforms as T
from visualize.count_labels import count_unique_labels_of_dataset

class WBC_dataset(Dataset):
    def __init__(self, images_path="", csv_path="", resize=224, normal_class_label=1):
        self.path = images_path
        self.resize = resize
        self.normal_class_label = normal_class_label
        self.img_labels = pd.read_csv(csv_path)
        self.img_labels = self.img_labels[self.img_labels['class label'] != 5]
        self.transform = T.Compose([T.Resize(resize, Image.ANTIALIAS),
                                      T.CenterCrop(224),
                                      T.ToTensor(),
                                      T.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])])

    def __getitem__(self, idx):
        img_path = f"{self.path}/{str(self.img_labels.iloc[idx, 0]).zfill(3)}.bmp"
        # print(img_path)
        image = Image.open(img_path).convert('RGB')
        label = self.img_labels.iloc[idx, 1]

        image = self.transform(image)

        target = 1 if label == self.normal_class_label else 0
        return image, target

    def __len__(self):
        return len(self.img_labels)

    def transform(self, img):
        pass


def get_wbc1_train_and_test_dataset_for_anomaly_detection():
    df = pd.read_csv('./wbc/segmentation_WBC/Class Labels of Dataset 1.csv')
    df = df[df['class label'] != 5]
    train_data = df[df['class label'] == 1].sample(n=120, random_state=2)

    df = df.drop(train_data.index)

    test_data = pd.DataFrame()
    for label in [1, 2, 3, 4]:
        class_samples = df[df['class label'] == label]
        test_data = pd.concat([test_data, class_samples])

    train_data.to_csv('wbc1_train_dataset.csv', index=False)
    test_data.to_csv('wbc1_test_dataset.csv', index=False)

    train_dataset = WBC_dataset(csv_path='wbc1_train_dataset.csv', images_path='wbc/segmentation_WBC/Dataset 1')
    test_dataset = WBC_dataset(csv_path='wbc1_test_dataset.csv', images_path='wbc/segmentation_WBC/Dataset 1')

    count_unique_labels_of_dataset(train_dataset, "wbc1_train_dataset")
    count_unique_labels_of_dataset(test_dataset, "wbc1_test_dataset")

    return train_dataset, test_dataset


def get_just_wbc1_test_dataset_for_anomaly_detection():
    df = pd.read_csv('./wbc/segmentation_WBC/Class Labels of Dataset 1.csv')
    df = df[df['class label'] != 5]

    test_data = pd.DataFrame()
    for label in [1, 2, 3, 4]:
        class_samples = df[df['class label'] == label]
        test_data = pd.concat([test_data, class_samples])

    test_data.to_csv('wbc1_just_test_dataset.csv', index=False)

    test_dataset = WBC_dataset(csv_path='wbc1_just_test_dataset.csv', images_path='wbc/segmentation_WBC/Dataset 1')

    count_unique_labels_of_dataset(test_dataset, "wbc1_test_dataset")

    return test_dataset
