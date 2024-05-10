import torch
from dataset import get_data_transforms, get_strong_transforms
from torchvision.datasets import ImageFolder
import numpy as np
from glob import glob
import random
import os
from torch.utils.data import DataLoader
from models.resnet import resnet18, resnet34, resnet50, wide_resnet50_2, wide_resnet101_2
from models.de_resnet import de_wide_resnet50_2
from models.recontrast import ReContrast, ReContrast
from dataset import BrainTest, BrainTrain
import torch.backends.cudnn as cudnn
import argparse
from utils import evaluation_noseg, visualize_noseg
from utils import global_cosine, global_cosine_hm

from torch.nn import functional as F
from functools import partial
from PIL import Image
import warnings
import copy
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pickle
from torchvision import datasets, transforms

warnings.filterwarnings("ignore")


def get_gta_globs():
    from glob import glob
    nums = [f'0{i}' for i in range(1, 10)] + ['10']
    globs_id = []
    globs_ood = []
    for i in range(10):
        id_path = f'/kaggle/input/gta5-15-5-{nums[i]}/gta5_{i}/gta5_{i}/ID/*'
        ood_path = f'/kaggle/input/gta5-15-5-{nums[i]}/gta5_{i}/gta5_{i}/OOD/*'
        globs_id.append(glob(id_path))
        globs_ood.append(glob(ood_path))
        print(i, len(globs_id[-1]), len(globs_ood[-1]))

    glob_id = []
    glob_ood = []
    for i in range(len(globs_id)):
        glob_id += globs_id[i]
        glob_ood += globs_ood[i]

    random.seed(42)
    random.shuffle(glob_id)
    train_ratio = 0.7
    separator = int(train_ratio * len(glob_id))
    glob_train_id = glob_id[:separator]
    glob_test_id = glob_id[separator:]

    return glob_train_id, glob_test_id, glob_ood


class GTA(Dataset):
    def __init__(self, image_path, labels, transform=None, count=-1):
        self.transform = transform
        self.image_files = image_path
        self.labels = labels
        if count != -1:
            if count<len(self.image_files):
                self.image_files = self.image_files[:count]
                self.labels = self.labels[:count]
            else:
                t = len(self.image_files)
                for i in range(count-t):
                    self.image_files.append(random.choice(self.image_files[:t]))
                    self.labels.append(random.choice(self.labels[:t]))

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file)
        image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[index], 1

    def __len__(self):
        return len(self.image_files)



def show_images(images, labels, dataset_name):
    num_images = len(images)
    rows = int(np.ceil(num_images / 5))  # Use np.ceil to ensure enough rows

    fig, axes = plt.subplots(rows, 5, figsize=(15, rows * 3), squeeze=False)  # Ensure axes is always a 2D array

    for i, ax in enumerate(axes.flatten()):
        if i < num_images:
            # Check if image is a tensor, if so, convert to numpy
            if isinstance(images[i], torch.Tensor):
                image = images[i].numpy()
            else:
                image = images[i]
            # If image is in (C, H, W) format, transpose it to (H, W, C)
            if image.shape[0] in {1, 3}:  # Assuming grayscale (1 channel) or RGB (3 channels)
                image = image.transpose(1, 2, 0)
            if image.shape[2] == 1:  # If grayscale, convert to RGB for consistency
                image = np.repeat(image, 3, axis=2)
            ax.imshow(image)
            ax.set_title(f"Label: {labels[i].item()}")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(f'{dataset_name}_visualization.png')


def visualize_random_samples_from_clean_dataset(dataset, dataset_name):
    print(f"Start visualization of clean dataset: {dataset_name}")
    # Choose 20 random indices from the dataset
    if len(dataset) > 20:
        random_indices = random.sample(range(len(dataset)), 20)
    else:
        random_indices = [i for i in range(len(dataset))]

    # Retrieve corresponding samples
    random_samples = [dataset[i] for i in random_indices]

    # Separate images and labels
    images, labels, _ = zip(*random_samples)

    labels = torch.tensor(labels)

    # Show the 20 random samples
    show_images(images, labels, dataset_name)


def get_logger(name, save_path=None, level='INFO'):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))

    log_format = logging.Formatter('%(message)s')
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(log_format)
    logger.addHandler(streamHandler)

    if not save_path is None:
        os.makedirs(save_path, exist_ok=True)
        fileHandler = logging.FileHandler(os.path.join(save_path, 'log.txt'))
        fileHandler.setFormatter(log_format)
        logger.addHandler(fileHandler)

    return logger


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(_class_):
    print_fn(_class_)
    setup_seed(111)

    total_iters = 1000
    batch_size = 16
    image_size = 224
    crop_size = 224

    data_transform, gt_transform = get_data_transforms(image_size, crop_size)

    mean_train = [0.485, 0.456, 0.406]
    std_train = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_train,
                             std=std_train)
    ])
    glob_train_id, glob_test_id, glob_ood = get_gta_globs()

    train_data = GTA(image_path=glob_train_id, labels=[0] * len(glob_train_id), transform=transform)
    test_data1 = GTA(image_path=glob_test_id + glob_ood, labels=[0] * len(glob_test_id) + [1] * len(glob_ood),
                   transform=transform)
    shifted_normal_path = glob('/kaggle/input/cityscapes-5-10-threshold/cityscapes/ID/*')
    shifted_anomaly_path = glob('/kaggle/input/cityscapes-5-10-threshold/cityscapes/OOD/*')

    shifted_test_path = shifted_normal_path + shifted_anomaly_path
    shifted_test_label = [0] * len(shifted_normal_path) + [1] * len(shifted_anomaly_path)

    test_data2 = GTA(image_path=shifted_test_path, labels=shifted_test_label,
                   transform=transform)

    # visualize_random_samples_from_clean_dataset(train_data, 'train dataset aptos')
    # visualize_random_samples_from_clean_dataset(test_data1, f'test data aptos1')
    # visualize_random_samples_from_clean_dataset(test_data2, f'test data aptos2')

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    test_dataloader1 = torch.utils.data.DataLoader(test_data1, batch_size=1, shuffle=False, num_workers=1)
    test_dataloader2 = torch.utils.data.DataLoader(test_data2, batch_size=1, shuffle=False, num_workers=1)

    encoder, bn = wide_resnet50_2(pretrained=True)
    decoder = de_wide_resnet50_2(pretrained=False, output_conv=2)

    encoder = encoder.to(device)
    bn = bn.to(device)
    decoder = decoder.to(device)
    encoder_freeze = copy.deepcopy(encoder)

    model = ReContrast(encoder=encoder, encoder_freeze=encoder_freeze, bottleneck=bn, decoder=decoder)

    optimizer = torch.optim.AdamW(list(decoder.parameters()) + list(bn.parameters()),
                                  lr=2e-3, betas=(0.9, 0.999), weight_decay=1e-5)
    optimizer2 = torch.optim.AdamW(list(encoder.parameters()),
                                   lr=1e-5, betas=(0.9, 0.999), weight_decay=1e-5)
    print_fn('train image number:{}'.format(len(train_data)))
    print_fn('test1 image number:{}'.format(len(test_data1)))
    print_fn('test1 image number:{}'.format(len(test_data2)))

    auroc_sp_best = 0
    it = 0
    for epoch in range(total_iters // len(train_dataloader) + 1):
        model.train(encoder_bn_train=True)
        loss_list = []
        for img, label, _ in train_dataloader:
            img = img.to(device)
            en, de = model(img)

            alpha_final = 1
            alpha = min(-3 + (alpha_final - -3) * it / (total_iters * 0.1), alpha_final)
            loss = global_cosine_hm(en[:3], de[:3], alpha=alpha, factor=0.) / 2 + \
                   global_cosine_hm(en[3:], de[3:], alpha=alpha, factor=0.) / 2

            # loss = global_cosine(en[:3], de[:3]) / 2 + \
            #        global_cosine(en[3:], de[3:]) / 2

            optimizer.zero_grad()
            optimizer2.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer2.step()
            loss_list.append(loss.item())
            if (it + 1) % 1 == 0:
                auroc1, f1_1, acc1 = evaluation_noseg(model, test_dataloader1, device)
                print_fn('Test DataLoader 1 - AUROC:{:.4f}, F1:{:.4f}, ACC:{:.4f}'.format(auroc1, f1_1, acc1))

                # Evaluate on the second test dataloader
                auroc2, f1_2, acc2 = evaluation_noseg(model, test_dataloader2, device)
                print_fn('Test DataLoader 2 - AUROC:{:.4f}, F1:{:.4f}, ACC:{:.4f}'.format(auroc2, f1_2, acc2))

                # Set the model back to training mode
                model.train(encoder_bn_train=True)
                if auroc1 >= auroc_sp_best:
                    auroc_sp_best = auroc1
            it += 1
            if it == total_iters:
                break
        print_fn('iter [{}/{}], loss:{:.4f}'.format(it, total_iters, np.mean(loss_list)))

    # visualize_noseg(model, test_dataloader1, device, _class_=_class_)
    return


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--save_dir', type=str, default='./saved_results')
    parser.add_argument('--save_name', type=str,
                        default='recontrast_aptos_b32_it1k_lr2e31e5_wd1e5_hm1d01_s111')
    parser.add_argument('--gpu', default='0', type=str,
                        help='GPU id to use.')
    parser.add_argument('--test_id', default=1, type=int)
    args = parser.parse_args()

    logger = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name))
    print_fn = logger.info

    device = 'cuda:' + args.gpu if torch.cuda.is_available() else 'cpu'
    print_fn(device)

    item_list = ['WBC']
    for item in item_list:
        train(item)
