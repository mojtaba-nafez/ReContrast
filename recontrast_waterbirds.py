import torch
from PIL import Image

from dataset import get_data_transforms, get_strong_transforms
from torchvision.datasets import ImageFolder
import numpy as np
import random
import os
from torch.utils.data import DataLoader
from models.resnet import resnet18, resnet34, resnet50, wide_resnet50_2, resnext50_32x4d
from models.de_resnet import de_wide_resnet50_2, de_resnet18, de_resnet34, de_resnet50, de_resnext50_32x4d
from models.recontrast import ReContrast, ReContrast
import torch.backends.cudnn as cudnn
import argparse
from utils import evaluation, visualize, global_cosine, modify_grad, visualize_noseg, global_cosine_hm, evaluation_noseg
from torch.nn import functional as F
from functools import partial
from ptflops import get_model_complexity_info
from torchvision import transforms
import matplotlib.pyplot as plt

import warnings
import copy
import logging
# from cutpaste_transformation import *

warnings.filterwarnings("ignore")


class Waterbird(torch.utils.data.Dataset):
    def __init__(self, root, df, transform, train=True, count_train_landbg=-1, count_train_waterbg=-1, mode='bg_all',
                 count=-1):
        self.transform = transform
        self.train = train
        self.df = df
        lb_on_l = df[(df['y'] == 0) & (df['place'] == 0)]
        lb_on_w = df[(df['y'] == 0) & (df['place'] == 1)]
        self.normal_paths = []
        self.labels = []

        normal_df = lb_on_l.iloc[:count_train_landbg]
        normal_df_np = normal_df['img_filename'].to_numpy()
        self.normal_paths.extend([os.path.join(root, x) for x in normal_df_np][:count_train_landbg])
        normal_df = lb_on_w.iloc[:count_train_waterbg]
        normal_df_np = normal_df['img_filename'].to_numpy()
        self.normal_paths.extend([os.path.join(root, x) for x in normal_df_np][:count_train_waterbg])

        if train:
            self.image_paths = self.normal_paths
        else:
            self.image_paths = []
            if mode == 'bg_all':
                dff = df
            elif mode == 'bg_water':
                dff = df[(df['place'] == 1)]
            elif mode == 'bg_land':
                dff = df[(df['place'] == 0)]
            else:
                print('Wrong mode!')
                raise ValueError('Wrong bg mode!')
            all_paths = dff[['img_filename', 'y']].to_numpy()
            for i in range(len(all_paths)):
                full_path = os.path.join(root, all_paths[i][0])
                if full_path not in self.normal_paths:
                    self.image_paths.append(full_path)
                    self.labels.append(all_paths[i][1])

        # if count != -1:
        #     random.shuffle(self.image_paths)
        #     if count < len(self.image_paths):
        #         self.image_paths = self.image_paths[:count]
        #         if not train:
        #             self.labels = self.labels[:count]
        #     else:
        #         t = len(self.image_paths)
        #         for i in range(count - t):
        #             self.image_paths.append(random.choice(self.image_paths[:t]))
        #             if not train:
        #                 self.labels.append(random.choice(self.labels[:t]))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        if self.train:
            return img, 0
        else:
            return img, self.labels[idx]


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


def show_images(images, labels, dataset_name):
    num_images = len(images)
    rows = int(np.ceil(num_images / 5))  # Use np.ceil to ensure enough rows
    fig, axes = plt.subplots(rows, 5, figsize=(15, rows * 3), squeeze=False)  # Ensure axes is always a 2D array
    for i, ax in enumerate(axes.flatten()):
        if i < num_images:
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
    if len(dataset) > 20:
        random_indices = random.sample(range(len(dataset)), 20)
    else:
        random_indices = [i for i in range(len(dataset))]
    random_samples = [dataset[i] for i in random_indices]
    try:
        images, _, labels, _ = zip(*random_samples)
    except:
        images, labels = zip(*random_samples)
    labels = torch.tensor(labels)
    show_images(images, labels, dataset_name)


def train(_class_, model='wide_res50', batch_size=16, total_iters=2000, evaluation_epochs=250, image_size=224):
    print_fn(_class_)
    setup_seed(111)

    total_iters = total_iters
    image_size = image_size
    crop_size = image_size

    data_transform, gt_transform = get_data_transforms(image_size, crop_size)  ## CHECK THIS IF NO GOOD RESULTS

    train_data_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.CenterCrop(crop_size),
    ])


    import pandas as pd
    df = pd.read_csv('/kaggle/input/waterbird/waterbird/metadata.csv')
    train_data = Waterbird(root='/kaggle/input/waterbird/waterbird', df=df,
                            transform=data_transform, train=True, count_train_landbg=3500, count_train_waterbg=100)
    test_data_landbg = Waterbird(root='/kaggle/input/waterbird/waterbird', df=df, transform=data_transform,
                                 train=False, count_train_landbg=3500, count_train_waterbg=100, mode='bg_land')
    test_data_waterbg = Waterbird(root='/kaggle/input/waterbird/waterbird', df=df, transform=data_transform,
                                 train=False, count_train_landbg=3500, count_train_waterbg=100, mode='bg_water')


    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader_landbg = torch.utils.data.DataLoader(test_data_landbg, batch_size=1, shuffle=False, num_workers=1)
    test_dataloader_waterbg = torch.utils.data.DataLoader(test_data_waterbg, batch_size=1, shuffle=False, num_workers=1)

    visualize_random_samples_from_clean_dataset(train_data, 'train dataset waterbirds')
    visualize_random_samples_from_clean_dataset(test_data_landbg, f'test data waterbirds landbg')
    visualize_random_samples_from_clean_dataset(test_data_waterbg, f'test data waterbirds waterbg')

    if model == 'wide_res50':
        encoder, bn = wide_resnet50_2(pretrained=True)
        decoder = de_wide_resnet50_2(pretrained=False, output_conv=2)
    elif model == 'res18':
        encoder, bn = resnet18(pretrained=True)
        decoder = de_resnet18(pretrained=False, output_conv=2)
    else:
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
    print_fn('test landbg image number:{}'.format(len(test_dataloader_landbg)))
    print_fn('test waterbg image number:{}'.format(len(test_dataloader_waterbg)))

    macs, params = get_model_complexity_info(model, (3, crop_size, crop_size),
                                             as_strings=True, print_per_layer_stat=False)
    print_fn('Computation:{}'.format(macs))
    print_fn('Parameters:{}'.format(params))

    auroc_px_best, auroc_sp_best, aupro_px_best = 0, 0, 0

    auroc_px_list = {"main": 0, "shifted": 0}
    auroc_px_list_best = {"main": 0, "shifted": 0}

    auroc_sp_list = {"main": 0, "shifted": 0}
    auroc_sp_list_best = {"main": 0, "shifted": 0}

    auroc_aupro_px_list = {"main": 0, "shifted": 0}
    auroc_aupro_px_list_best = {"main": 0, "shifted": 0}

    auroc_sp_best = 0
    it = 0
    for epoch in range(total_iters // len(train_dataloader) + 1):
        model.train(encoder_bn_train=True)
        loss_list = []
        for img, label in train_dataloader:
            img = img.to(device)
            en, de = model(img)

            if it == 0:
                print(img.shape, label.shape)

            alpha_final = 1
            alpha = min(-3 + (alpha_final - -3) * it / (total_iters * 0.1), alpha_final)
            loss = global_cosine_hm(en[:3], de[:3], alpha=alpha, factor=0.) / 2 + \
                   global_cosine_hm(en[3:], de[3:], alpha=alpha, factor=0.) / 2

            # loss = global_cosine(en[:3], de[:3]) / 2 + \
            #        global_cosine(en[3:], de[3:]) / 2

            optimizer.zero_grad()
            optimizer2.zero_grad()

            try:
                loss.backward()
            except:
                print("loss has not grad!")
                continue

            optimizer.step()
            optimizer2.step()
            loss_list.append(loss.item())
            if (it + 1) % evaluation_epochs == 0:
                print_fn('iter [{}/{}], loss:{:.4f}'.format(it, total_iters, np.mean(loss_list)))
                auroc, f1, acc = evaluation_noseg(model, test_dataloader_landbg, device, reduction='mean')
                model.train(encoder_bn_train=True)
                print_fn('AUROC:{:.4f}, F1:{:.4f}, ACC:{:.4f}'.format(auroc, f1, acc))
                if auroc >= auroc_sp_best:
                    auroc_sp_best = auroc

                shrink_factor = "main"
                # auroc, f1, acc = evaluation_noseg(model, test_dataloader1, device)
                auroc_px_list[str(shrink_factor)], auroc_sp_list[str(shrink_factor)], auroc_aupro_px_list[
                    str(shrink_factor)] = evaluation_noseg(model, test_dataloader_landbg, device)
                print_fn('Shrink Factor:{}, Sample Auroc:{:.3f}, F1:{:.3f}, acc:{:.3}'.format(shrink_factor,
                                                                                              auroc_px_list[
                                                                                                  str(shrink_factor)],
                                                                                              auroc_sp_list[
                                                                                                  str(shrink_factor)],
                                                                                              auroc_aupro_px_list[
                                                                                                  str(shrink_factor)]))
                if auroc_sp_list[str(shrink_factor)] >= auroc_sp_list_best[str(shrink_factor)]:
                    auroc_px_list_best[str(shrink_factor)], auroc_sp_list_best[str(shrink_factor)], \
                    auroc_aupro_px_list_best[str(shrink_factor)] = auroc_px_list[str(shrink_factor)], auroc_sp_list[
                        str(shrink_factor)], auroc_aupro_px_list[str(shrink_factor)]

                shrink_factor = "shifted"
                auroc_px_list[str(shrink_factor)], auroc_sp_list[str(shrink_factor)], auroc_aupro_px_list[
                    str(shrink_factor)] = evaluation_noseg(model, test_dataloader_waterbg, device)
                print_fn('Shrink Factor:{}, Sample Auroc:{:.3f}, F1:{:.3f}, acc:{:.3}'.format(shrink_factor,
                                                                                              auroc_px_list[
                                                                                                  str(shrink_factor)],
                                                                                              auroc_sp_list[
                                                                                                  str(shrink_factor)],
                                                                                              auroc_aupro_px_list[
                                                                                                  str(shrink_factor)]))
                if auroc_sp_list[str(shrink_factor)] >= auroc_sp_list_best[str(shrink_factor)]:
                    auroc_px_list_best[str(shrink_factor)], auroc_sp_list_best[str(shrink_factor)], \
                    auroc_aupro_px_list_best[str(shrink_factor)] = auroc_px_list[str(shrink_factor)], auroc_sp_list[
                        str(shrink_factor)], auroc_aupro_px_list[str(shrink_factor)]

                model.train(encoder_bn_train=True)
            it += 1
            if it == total_iters:
                break

        print_fn('iter [{}/{}], loss:{:.4f}'.format(it, total_iters, np.mean(loss_list)))

    # visualize_noseg(model, test_dataloader, device, _class_=_class_, save_name=args.save_name)
    return auroc, auroc_sp_best


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--save_dir', type=str, default='./saved_results')
    parser.add_argument('--save_name', type=str,
                        default='recontrast_isic_256224_b32_it2k_lr2e31e5_wd1e5_hm1d01_s111')
    parser.add_argument('--gpu', default='0', type=str,
                        help='GPU id to use.')
    parser.add_argument('--total_iters', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--evaluation_epochs', type=int, default=250)
    parser.add_argument('--model', type=str, default='wide_res50')
    parser.add_argument('--image_size', type=int, default=224)
    args = parser.parse_args()

    logger = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name))
    print_fn = logger.info

    device = 'cuda:' + args.gpu if torch.cuda.is_available() else 'cpu'
    print_fn(device)

    item_list = ['Waterbirds']
    for item in item_list:
        train(item, model=args.model, batch_size=args.batch_size, total_iters=args.total_iters,
              evaluation_epochs=args.evaluation_epochs, image_size=args.image_size)
