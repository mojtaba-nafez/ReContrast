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
from torchvision import transforms
import matplotlib.pyplot as plt

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


def show_images(images, labels, dataset_name):
    num_images = len(images)
    rows = int(num_images / 5) + 1

    fig, axes = plt.subplots(rows, 5, figsize=(15, rows * 3))

    for i, ax in enumerate(axes.flatten()):
        if i < num_images:
            ax.imshow(images[i].permute(1, 2, 0))  # permute to (H, W, C) for displaying RGB images
            ax.set_title(f"Label: {labels[i]}")
        ax.axis("off")

    plt.savefig(f'{dataset_name}_visualization.png')


def visualize_random_samples_from_clean_dataset(dataset, dataset_name, train_data=True):
    print(f"Start visualization of clean dataset: {dataset_name}")
    # Choose 20 random indices from the dataset
    if len(dataset) > 20:
        random_indices = random.sample(range(len(dataset)), 20)
    else:
        random_indices = [i for i in range(len(dataset))]

    # Retrieve corresponding samples
    random_samples = [dataset[i] for i in random_indices]

    # Separate images and labels
    if train_data:
        images, labels = zip(*random_samples)
    else:
        images, _, labels, _ = zip(*random_samples)

    # print(f"len(labels): {len(labels)}")
    # print(f"type(labels): {type(labels)}")
    # print(f"type(images): {type(images)}")
    # print(f"type(labels[0]): {type(labels[0])}")
    # print(f"labels[0]: {labels[0]}")
    # print(f"labels.size(): {labels.size()}")

    # Convert PIL images to PyTorch tensors
    # transform = transforms.ToTensor()
    # images = [transform(image) for image in images]

    # Convert labels to PyTorch tensor
    print(f"len(labels): {len(labels)}")
    print(f"type(labels): {type(labels)}")
    print(f"type(labels[0]): {type(labels[0])}")
    print(f"labels[0]: {labels[0]}")
    labels = torch.tensor(labels)

    # Show the 20 random samples
    show_images(images, labels, dataset_name)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def _get_features(P, model, loader, interp=False, imagenet=False, simclr_aug=None,
                  sample_num=1, layers=('simclr', 'shift')):

    # layers = ['simclr', 'shift']
    if not isinstance(layers, (list, tuple)):
        layers = [layers]

    # check if arguments are valid
    assert simclr_aug is not None

    if imagenet is True:  # assume batch_size = 1 for ImageNet
        sample_num = 1

    # compute features in full dataset
    model.eval()
    feats_all = {layer: [] for layer in layers}  # initialize: empty list
    for i, (x, _) in enumerate(loader):
        # interp: False
        if interp:
            x_interp = (x + last) / 2 if i > 0 else x  # omit the first batch, assume batch sizes are equal
            last = x  # save the last batch
            x = x_interp  # use interp as current batch

        if imagenet is True:
            x = torch.cat(x[0], dim=0)  # augmented list of x

        x = x.to(device)  # gpu tensor

        # compute features in one batch
        feats_batch = {layer: [] for layer in layers}  # initialize: empty list
        # sample_num=10
        for seed in range(sample_num):
            set_random_seed(seed)

            if P.K_shift > 1:
                # train time call:
                #   x   = torch.Size([128, 3, 32, 32])
                #   x_t = torch.Size([512, 3, 32, 32])
                # test time call:
                #   x   = torch.Size([100, 3, 32, 32])
                #   x_t = torch.Size([400, 3, 32, 32])
                x_t = torch.cat([P.shift_trans(hflip(x), k) for k in range(P.K_shift)])
            else:
                x_t = x # No shifting: SimCLR
            x_t = simclr_aug(x_t)

            # compute augmented features
            with torch.no_grad():
                # layers = ['simclr', 'shift']
                kwargs = {layer: True for layer in layers}  # only forward selected layers
                _, output_aux = model(x_t, **kwargs)

            # add features in one batch
            for layer in layers:
                # train time call
                #   output_aux["shift"] torch.Size([512, 4])
                #   output_aux["simclr"] torch.Size([512, 128])
                # test time call
                #   output_aux["shift"] torch.Size([400, 4])
                #   output_aux["simclr"] torch.Size([400, 128])
                feats = output_aux[layer].cpu()
                # imagenet = False
                if imagenet is False:
                    # feats.chunk(P.K_shift):
                    #   Train:   4 * torch.Size([128, 128])   ||   4 * torch.Size([128, 4])
                    #   Test:    4 * torch.Size([100, 128])   ||   4 * torch.Size([100, 4])

                    # feats_batch[layer] = array of len=4
                    # train:    40 * torch.Size([128, 128])  ||  40 * torch.Size([128, 4])
                    # test:     40 * torch.Size([100, 128])  ||  40 * torch.Size([100, 4])
                    feats_batch[layer] += feats.chunk(P.K_shift)
                else:
                    feats_batch[layer] += [feats]  # (B, d) cpu tensor

        # concatenate features in one batch
        for key, val in feats_batch.items():
            if imagenet:
                feats_batch[key] = torch.stack(val, dim=0)  # (B, T, d)
            else:
                feats_batch[key] = torch.stack(val, dim=1)  # (B, T, d)
        # feats_batch
        # feats_batch["simclr"] = torch.Size([128, 40, 128]) || torch.Size([100, 40, 128])
        # feats_batch["shift"]  = torch.Size([128, 40, 4])   || torch.Size([100, 40, 4])

        # add features in full dataset
        for layer in layers:
            feats_all[layer] += [feats_batch[layer]]
    # feats_all["shift or simclr"] is an array len=40 --> element: [128, 40, 128]
    #   train time call:   simclr=[40, 128, 40, 128]  || shift=[40, 128, 40, 4]
    #   test time call:    simclr=[10, 100, 40, 128]  || shift=[10, 100, 40, 4]

    # concatenate features in full dataset
    for key, val in feats_all.items():
        feats_all[key] = torch.cat(val, dim=0)  # (N, T, d)
    # train time call feats_all[key]:
    #        torch.Size([5000, 40, 128])
    #        torch.Size([5000, 40, 4])
    # train time call feats_all[key]:
    #        torch.Size([1000, 40, 128])
    #        torch.Size([1000, 40, 4])

    # reshape order
    if imagenet is False:
        # Convert [1,2,3,4, 1,2,3,4] -> [1,1, 2,2, 3,3, 4,4]
        for key, val in feats_all.items():
            N, T, d = val.size()  # T = K * T'
            val = val.view(N, -1, P.K_shift, d)  # (N, T', K, d)
            val = val.transpose(2, 1)  # (N, 4, T', d)
            val = val.reshape(N, T, d)  # (N, T, d)
            feats_all[key] = val
    # train time call feats_all[key]:
    #        torch.Size([5000, 40, 128])
    #        torch.Size([5000, 40, 4])
    # train time call feats_all[key]:
    #        torch.Size([1000, 40, 128])
    #        torch.Size([1000, 40, 4])
    return feats_all


def get_features(P, data_name, model, loader, interp=False, prefix='',
                 simclr_aug=None, sample_num=1, layers=('simclr', 'shift')):

    if not isinstance(layers, (list, tuple)):
        layers = [layers]

    # load pre-computed features if exists
    feats_dict = dict()
    # for layer in layers:
    #     path = prefix + f'_{data_name}_{layer}.pth'
    #     if os.path.exists(path):
    #         feats_dict[layer] = torch.load(path)

    # pre-compute features and save to the path
    # left= ['simclr', 'shift']
    left = [layer for layer in layers if layer not in feats_dict.keys()]
    if len(left) > 0:
        _feats_dict = _get_features(P, model, loader, interp, P.dataset == 'imagenet',
                                    simclr_aug, sample_num, layers=left)

        for layer, feats in _feats_dict.items():
            path = prefix + f'_{data_name}_{layer}.pth'
            torch.save(_feats_dict[layer], path)
            feats_dict[layer] = feats  # update value

    return feats_dict


def train(_class_, shrink_factor=None, total_iters=2000, eval_only=False):
    print_fn(_class_)
    setup_seed(111)

    total_iters = total_iters
    batch_size = 16
    image_size = 256
    crop_size = 256

    data_transform, gt_transform = get_data_transforms(image_size, crop_size)

    train_path = '/kaggle/input/mvtec-ad/' + _class_ + '/train'
    test_path = '/kaggle/input/mvtec-ad/' + _class_

    train_data = ImageFolder(root=train_path, transform=data_transform)
    test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test",
                             shrink_factor=shrink_factor)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4,
                                                   drop_last=False)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=1)

    visualize_random_samples_from_clean_dataset(train_data, f"train_data_{_class_}", train_data=True)
    visualize_random_samples_from_clean_dataset(test_data, f"test_data_{_class_}", train_data=False)

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

    auroc_px_list = {"0.8": 0, "0.85": 0, "0.9": 0, "0.95": 0, "0.98": 0, "1.0": 0}
    auroc_px_list_best = {"0.8": 0, "0.85": 0, "0.9": 0, "0.95": 0, "0.98": 0, "1.0": 0}

    auroc_sp_list = {"0.8": 0, "0.85": 0, "0.9": 0, "0.95": 0, "0.98": 0, "1.0": 0}
    auroc_sp_list_best = {"0.8": 0, "0.85": 0, "0.9": 0, "0.95": 0, "0.98": 0, "1.0": 0}

    auroc_aupro_px_list = {"0.8": 0, "0.85": 0, "0.9": 0, "0.95": 0, "0.98": 0, "1.0": 0}
    auroc_aupro_px_list_best = {"0.8": 0, "0.85": 0, "0.9": 0, "0.95": 0, "0.98": 0, "1.0": 0}

    kwargs = {
        'simclr_aug': simclr_aug,
        'sample_num': P.ood_samples,
        'layers': P.ood_layer,
    }

    print('Pre-compute global statistics...')
    feats_train = get_features(P, f'{P.dataset}_train', model, train_dataloader, **kwargs)  # (M, T, d)

    if eval_only:
        model.load_state_dict(torch.load('model.pth'))
        auroc_px, auroc_sp, aupro_px = evaluation(model, test_dataloader, device)
        print_fn('Pixel Auroc:{:.3f}, Sample Auroc:{:.3f}, Pixel Aupro:{:.3}'.format(auroc_px, auroc_sp, aupro_px))
        return auroc_px, auroc_sp, aupro_px

    epochs = int(np.ceil(total_iters / len(train_dataloader)))
    for epoch in range(epochs):
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
            if epoch == epochs - 1:
                pad_size = [1.0, 0.98, 0.95, 0.9, 0.85, 0.8]

                for shrink_factor in pad_size:
                    test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform,
                                             phase="test", shrink_factor=shrink_factor)
                    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=1)

                    auroc_px_list[str(shrink_factor)], auroc_sp_list[str(shrink_factor)], auroc_aupro_px_list[
                        str(shrink_factor)] = evaluation(model, test_dataloader, device)
                    print_fn('Shrink Factor:{:.3f}, Pixel Auroc:{:.3f}, Sample Auroc:{:.3f}, Pixel Aupro:{:.3}'.format(
                        shrink_factor, auroc_px_list[str(shrink_factor)], auroc_sp_list[str(shrink_factor)],
                        auroc_aupro_px_list[str(shrink_factor)]))

                    if auroc_sp_list[str(shrink_factor)] >= auroc_sp_list_best[str(shrink_factor)]:
                        auroc_px_list_best[str(shrink_factor)], auroc_sp_list_best[str(shrink_factor)], \
                        auroc_aupro_px_list_best[str(shrink_factor)] = auroc_px_list[str(shrink_factor)], auroc_sp_list[
                            str(shrink_factor)], auroc_aupro_px_list[str(shrink_factor)]

                model.train(encoder_bn_train=_class_ not in ['toothbrush', 'leather', 'grid', 'tile', 'wood', 'screw'])
            it += 1
            if it == total_iters:
                break
        print_fn('iter [{}/{}], loss:{:.4f}'.format(it, total_iters, np.mean(loss_list)))

    torch.save(model.state_dict(), 'model.pth')
    # visualize(model, test_dataloader, device, _class_=_class_, save_name=args.save_name)
    return auroc_px_list, auroc_sp_list, auroc_aupro_px_list, auroc_px_list_best, auroc_sp_list_best, auroc_aupro_px_list_best


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--save_dir', type=str, default='./saved_results')
    parser.add_argument('--save_name', type=str,
                        default='recontrast_mvtec_it2k_lr2e31e5_wd1e5_hm1d01_s111')
    parser.add_argument('--gpu', default='0', type=str,
                        help='GPU id to use.')
    parser.add_argument('--shrink_factor', type=float, default=1)
    parser.add_argument('--total_iters', type=int, default=2000)
    parser.add_argument('--cls', type=int, default=11)
    parser.add_argument('--eval_only', action='store_true')
    args = parser.parse_args()

    item_list = ['carpet', 'bottle', 'hazelnut', 'leather', 'cable', 'capsule', 'grid', 'pill',
                 'transistor', 'metal_nut', 'screw', 'toothbrush', 'zipper', 'tile', 'wood']
    # item_list = ['toothbrush']
    logger = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name))
    print_fn = logger.info

    device = 'cuda:' + args.gpu if torch.cuda.is_available() else 'cpu'
    print_fn(device)

    result_list = {"0.8": [], "0.85": [], "0.9": [], "0.95": [], "0.98": [], "1.0": []}
    result_list_best = {"0.8": [], "0.85": [], "0.9": [], "0.95": [], "0.98": [], "1.0": []}
    pad_size = [1.0, 0.98, 0.95, 0.9, 0.85, 0.8]
    print(args.cls, item_list[args.cls])
    if args.eval_only:
        train(item_list[args.cls], eval_only=True)
        exit()
    for i, item in enumerate(item_list):
        print(f"+++++++++++++++++++++++++++++++++++++++{item}+++++++++++++++++++++++++++++++++++++++")
        auroc_px, auroc_sp, aupro_px, auroc_px_best, auroc_sp_best, aupro_px_best = train(item,
                                                                                          shrink_factor=args.shrink_factor,
                                                                                          total_iters=args.total_iters)
        for pad in pad_size:
            result_list[str(pad)].append([item, auroc_px[str(pad)], auroc_sp[str(pad)], aupro_px[str(pad)]])
            result_list_best[str(pad)].append(
                [item, auroc_px_best[str(pad)], auroc_sp_best[str(pad)], aupro_px_best[str(pad)]])

    for pad in pad_size:
        print(f'-------- shrink factor = {pad} --------')
        mean_auroc_px = np.mean([result[1] for result in result_list[str(pad)]])
        mean_auroc_sp = np.mean([result[2] for result in result_list[str(pad)]])
        mean_aupro_px = np.mean([result[3] for result in result_list[str(pad)]])
        print_fn(result_list[str(pad)])
        print_fn('mPixel Auroc:{:.4f}, mSample Auroc:{:.4f}, mPixel Aupro:{:.4}'.format(mean_auroc_px, mean_auroc_sp,
                                                                                        mean_aupro_px))

        best_auroc_px = np.mean([result[1] for result in result_list_best[str(pad)]])
        best_auroc_sp = np.mean([result[2] for result in result_list_best[str(pad)]])
        best_aupro_px = np.mean([result[3] for result in result_list_best[str(pad)]])
        print_fn(result_list_best[str(pad)])
        print_fn('bPixel Auroc:{:.4f}, bSample Auroc:{:.4f}, bPixel Aupro:{:.4}'.format(best_auroc_px, best_auroc_sp,
                                                                                        best_aupro_px))