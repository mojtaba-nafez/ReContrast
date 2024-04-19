
import torch
from dataset import get_data_transforms, get_strong_transforms
from torchvision.datasets import ImageFolder
import numpy as np
import random
import os
from torch.utils.data import DataLoader
from models.resnet import resnet18, resnet34, resnet50, wide_resnet50_2, wide_resnet101_2
from models.de_resnet import de_wide_resnet50_2, de_resnet18
from models.recontrast import ReContrast, ReContrast
from dataset import BrainTest, BrainTrain
import torch.backends.cudnn as cudnn
import argparse
from utils import evaluation_noseg, visualize_noseg
from utils import global_cosine, global_cosine_hm

from torch.nn import functional as F
from functools import partial

import warnings
import copy
import logging
import numpy as np
import matplotlib.pyplot as plt


warnings.filterwarnings("ignore")

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


def train(_class_, unode1_checkpoint=None, unode2_checkpoint=None):
    print_fn(_class_)
    setup_seed(111)

    total_iters = 2000
    batch_size = 16
    image_size = 224
    crop_size = 224

    data_transform, gt_transform = get_data_transforms(image_size, crop_size)

    train_path = '../APTOS/'
    test_path = '../APTOS/'

    train_data = BrainTrain(transform=data_transform)
    test_data1 = BrainTest(transform=data_transform, test_id=1)
    test_data2 = BrainTest(transform=data_transform, test_id=2)


    visualize_random_samples_from_clean_dataset(train_data, 'train dataset brain')
    visualize_random_samples_from_clean_dataset(test_data1, f'test data brain1')
    visualize_random_samples_from_clean_dataset(test_data2, f'test data brain2')


    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4,
                                                   drop_last=False)
    test_dataloader1 = torch.utils.data.DataLoader(test_data1, batch_size=1, shuffle=False, num_workers=1)
    test_dataloader2 = torch.utils.data.DataLoader(test_data2, batch_size=1, shuffle=False, num_workers=1)

    encoder, bn = resnet18(pretrained=True)
    decoder = de_resnet18(pretrained=False, output_conv=2)

    encoder_freeze = copy.deepcopy(encoder)
    # encoder_freeze = encoder_freeze.to(device)

    if unode1_checkpoint is not None:  # encoder
        print('Applying U-node as encoder 1...')
        encoder, bn = resnet18(pretrained=True, progress=True, unode_path=unode1_checkpoint, fc=False)
        # decoder = de_resnet18(pretrained=False, progress=True, unode_path=unode1_checkpoint, output_conv=2)
        # encoder_freeze = copy.deepcopy(encoder)

    encoder = encoder.to(device)
    bn = bn.to(device)
    decoder = decoder.to(device)
    # encoder_freeze = copy.deepcopy(encoder)

    if unode2_checkpoint is not None:  # encoder_freeze
        print('Applying U-node as encoder 2...')
        encoder_freeze, _ = resnet18(pretrained=True, progress=True, unode_path=unode2_checkpoint, fc=False)

    encoder_freeze = encoder_freeze.to(device)

    model = ReContrast(encoder=encoder, encoder_freeze=encoder_freeze, bottleneck=bn, decoder=decoder)

    optimizer = torch.optim.AdamW(list(decoder.parameters()) + list(bn.parameters()),
                                  lr=2e-3, betas=(0.9, 0.999), weight_decay=1e-5)
    optimizer2 = torch.optim.AdamW(list(encoder.parameters()),
                                   lr=1e-5, betas=(0.9, 0.999), weight_decay=1e-5)
    print_fn('train image number:{}'.format(len(train_data)))
    print_fn('test1 image number:{}'.format(len(test_data1)))
    print_fn('test2 image number:{}'.format(len(test_data2)))

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
            if (it + 1) % 500 == 0:
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

    visualize_noseg(model, test_dataloader1, device, _class_=_class_)
    return auroc, auroc_sp_best


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--save_dir', type=str, default='./saved_results')
    parser.add_argument('--save_name', type=str,
                        default='recontrast_aptos_b32_it1k_lr2e31e5_wd1e5_hm1d01_s111')
    parser.add_argument('--gpu', default='0', type=str,
                        help='GPU id to use.')


    parser.add_argument('--encoder1_path', type=str, default='')
    parser.add_argument('--encoder2_path', type=str, default='')

    args = parser.parse_args()

    en1_path = args.encoder1_path if args.encoder1_path != '' else None
    en2_path = args.encoder2_path if args.encoder2_path != '' else None

    logger = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name))
    print_fn = logger.info

    device = 'cuda:' + args.gpu if torch.cuda.is_available() else 'cpu'
    print_fn(device)

    item_list = ['APTOS']
    for item in item_list:
        train(item, unode1_checkpoint=en1_path, unode2_checkpoint=en2_path)