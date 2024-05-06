
import torch
from dataset import get_data_transforms, get_strong_transforms
from torchvision.datasets import ImageFolder
import numpy as np
import random
import os
from torch.utils.data import DataLoader
from models.resnet import resnet18, resnet34, resnet50, wide_resnet50_2, resnext50_32x4d
from models.de_resnet import de_wide_resnet50_2, de_resnet18, de_resnet34, de_resnet50, de_resnext50_32x4d
from models.recontrast import ReContrast, ReContrast
from dataset import MVTecDataset
import torch.backends.cudnn as cudnn
import argparse
from utils import evaluation, visualize, global_cosine, global_cosine_hm
from torch.nn import functional as F
from functools import partial
from ptflops import get_model_complexity_info

from visualize.visualize_dataset import visualize_random_samples_from_clean_dataset
from torchvision import transforms
import matplotlib.pyplot as plt

from datasets.wbc1 import get_wbc1_train_and_test_dataset_for_anomaly_detection, get_just_wbc1_test_dataset_for_anomaly_detection
from datasets.wbc2 import get_wbc2_train_and_test_dataset_for_anomaly_detection, get_just_wbc2_test_dataset_for_anomaly_detection

import warnings
import copy
import logging

warnings.filterwarnings("ignore")


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

def train(_class_, total_iters=2000):
    print_fn(_class_)
    setup_seed(111)

    total_iters = total_iters
    batch_size = 16
    image_size = 256
    crop_size = 256

    data_transform, gt_transform = get_data_transforms(image_size, crop_size)

    # train_path = '/kaggle/input/mvtec-ad/' + _class_ + '/train'
    # test_path = '/kaggle/input/mvtec-ad/' + _class_
    #
    # train_data = ImageFolder(root=train_path, transform=data_transform)
    # test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")

    train_data, test_data = get_wbc1_train_and_test_dataset_for_anomaly_detection()
    just_test_data = get_just_wbc2_test_dataset_for_anomaly_detection()

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4,
                                                   drop_last=False)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=1)
    just_test_dataloader = torch.utils.data.DataLoader(just_test_data, batch_size=batch_size, shuffle=False, num_workers=1)

    visualize_random_samples_from_clean_dataset(train_data, f"train_data")
    visualize_random_samples_from_clean_dataset(test_data, f"test_data")

    encoder, bn = wide_resnet50_2(pretrained=True)
    decoder = de_wide_resnet50_2(pretrained=False, output_conv=2)

    encoder = encoder.to(device)
    bn = bn.to(device)
    decoder = decoder.to(device)
    encoder_freeze = copy.deepcopy(encoder)

    model = ReContrast(encoder=encoder, encoder_freeze=encoder_freeze, bottleneck=bn, decoder=decoder)
    # for m in encoder.modules():
    #     if isinstance(m, torch.nn.BatchNorm2d):
    #         m.eps = 1e-8

    optimizer = torch.optim.AdamW(list(decoder.parameters()) + list(bn.parameters()),
                                  lr=2e-3, betas=(0.9, 0.999), weight_decay=1e-5)
    optimizer2 = torch.optim.AdamW(list(encoder.parameters()),
                                   lr=1e-5, betas=(0.9, 0.999), weight_decay=1e-5)
    print_fn('train image number:{}'.format(len(train_data)))
    print_fn('test image number:{}'.format(len(test_data)))
    macs, params = get_model_complexity_info(model, (3, crop_size, crop_size),
                                             as_strings=True, print_per_layer_stat=False)
    print_fn('Computation:{}'.format(macs))
    print_fn('Parameters:{}'.format(params))

    auroc_px_best, auroc_sp_best, aupro_px_best = 0, 0, 0
    it = 0
    for epoch in range(int(np.ceil(total_iters / len(train_dataloader)))):
        # encoder batchnorm in eval for these classes.
        model.train(encoder_bn_train=_class_ not in ['toothbrush', 'leather', 'grid', 'tile', 'wood', 'screw'])

        loss_list = []
        for img, label in train_dataloader:
            img = img.to(device)
            en, de = model(img)

            alpha_final = 1
            alpha = min(-3 + (alpha_final - -3) * it / (total_iters * 0.1), alpha_final)
            loss = global_cosine_hm(en[:3], de[:3], alpha=alpha, factor=0.) / 2 + \
                   global_cosine_hm(en[3:], de[3:], alpha=alpha, factor=0.) / 2

            # loss = global_cosine(en[:3], de[:3], stop_grad=False) / 2 + \
            #        global_cosine(en[3:], de[3:], stop_grad=False) / 2

            optimizer.zero_grad()
            optimizer2.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            optimizer2.step()
            loss_list.append(loss.item())
            if (it + 1) % 250 == 0:
                auroc_px, auroc_sp, aupro_px = evaluation(model, test_dataloader, device)
                just_test_auroc_px, just_test_auroc_sp, just_test_aupro_px = evaluation(model, just_test_dataloader, device)
                model.train(encoder_bn_train=_class_ not in ['toothbrush', 'leather', 'grid', 'tile', 'wood', 'screw'])

                print_fn('Pixel Auroc:{:.3f}, Sample Auroc:{:.3f}, Pixel Aupro:{:.3}'.format(auroc_px, auroc_sp, aupro_px))
                print_fn('Just test pixel Auroc:{:.3f}, Just test sample Auroc:{:.3f}, Just test pixel Aupro:{:.3}'.format(just_test_auroc_px, just_test_auroc_sp, just_test_aupro_px))
                if auroc_sp >= auroc_sp_best:
                    auroc_px_best, auroc_sp_best, aupro_px_best = auroc_px, auroc_sp, aupro_px
            it += 1
            if it == total_iters:
                break
        print_fn('iter [{}/{}], loss:{:.4f}'.format(it, total_iters, np.mean(loss_list)))

    # visualize(model, test_dataloader, device, _class_=_class_, save_name=args.save_name)
    return auroc_px, auroc_sp, aupro_px, auroc_px_best, auroc_sp_best, aupro_px_best


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--save_dir', type=str, default='./saved_results')
    parser.add_argument('--save_name', type=str,
                        default='recontrast_wbc_it2k_lr2e31e5_wd1e5_hm1d01_s111')
    parser.add_argument('--gpu', default='0', type=str,
                        help='GPU id to use.')
    parser.add_argument('--total_iters', type=int, default=2000)
    parser.add_argument('--train_wbc_dataset', type=str, default='wbc1', choices=['wbc1', 'wbc2'])
    args = parser.parse_args()

    item_list = ['carpet', 'bottle', 'hazelnut', 'leather', 'cable', 'capsule', 'grid', 'pill',
                 'transistor', 'metal_nut', 'screw', 'toothbrush', 'zipper', 'tile', 'wood']
    # item_list = ['toothbrush']
    logger = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name))
    print_fn = logger.info

    device = 'cuda:' + args.gpu if torch.cuda.is_available() else 'cpu'
    print_fn(device)

    result_list = []
    result_list_best = []
    for i, item in enumerate(item_list):
        auroc_px, auroc_sp, aupro_px, auroc_px_best, auroc_sp_best, aupro_px_best = train(item, total_iters=args.total_iters)
        result_list.append([item, auroc_px, auroc_sp, aupro_px])
        result_list_best.append([item, auroc_px_best, auroc_sp_best, aupro_px_best])

    mean_auroc_px = np.mean([result[1] for result in result_list])
    mean_auroc_sp = np.mean([result[2] for result in result_list])
    mean_aupro_px = np.mean([result[3] for result in result_list])
    print_fn(result_list)
    print_fn('mPixel Auroc:{:.4f}, mSample Auroc:{:.4f}, mPixel Aupro:{:.4}'.format(mean_auroc_px, mean_auroc_sp,
                                                                                    mean_aupro_px))

    best_auroc_px = np.mean([result[1] for result in result_list_best])
    best_auroc_sp = np.mean([result[2] for result in result_list_best])
    best_aupro_px = np.mean([result[3] for result in result_list_best])
    print_fn(result_list_best)
    print_fn('bPixel Auroc:{:.4f}, bSample Auroc:{:.4f}, bPixel Aupro:{:.4}'.format(best_auroc_px, best_auroc_sp,
                                                                                    best_aupro_px))
