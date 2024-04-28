import torch
from dataset import get_data_transforms, get_strong_transforms, RSNATEST, RSNATRAIN
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
from utils import evaluation, visualize, global_cosine, global_cosine_hm, NT_xent, contrastive_loss, \
    evaluation_noseg_brain
from torch.nn import functional as F
from functools import partial
from ptflops import get_model_complexity_info
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn as nn

import warnings
import copy
import logging
import glob
import pandas as pd
from PIL import Image
from cutpaste_transformation import *

warnings.filterwarnings("ignore")


class BrainTest(torch.utils.data.Dataset):
    def __init__(self, transform, test_id=1):

        self.transform = transform
        self.test_id = test_id

        test_normal_path = glob.glob('./Br35H/dataset/test/normal/*')
        test_anomaly_path = glob.glob('./Br35H/dataset/test/anomaly/*')

        self.test_path = test_normal_path + test_anomaly_path
        self.test_label = [0] * len(test_normal_path) + [1] * len(test_anomaly_path)

        if self.test_id == 2:
            test_normal_path = glob.glob('./brats/dataset/test/normal/*')
            test_anomaly_path = glob.glob('./brats/dataset/test/anomaly/*')

            self.test_path = test_normal_path + test_anomaly_path
            self.test_label = [0] * len(test_normal_path) + [1] * len(test_anomaly_path)

    def __len__(self):
        return len(self.test_path)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.test_path[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        has_anomaly = 0 if self.test_label[idx] == 0 else 1

        # this is fake:)
        gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        gt[:, :, 1:3] = 1
        # return img, , has_anomaly, img_path
        return img, gt, has_anomaly, img_path


class BrainTrain(torch.utils.data.Dataset):
    def __init__(self, transform):
        self.transform = transform
        self.image_paths = glob.glob('./Br35H/dataset/train/normal/*')
        brats_mod = glob.glob('./brats/dataset/train/normal/*')
        random.seed(1)
        random_brats_images = random.sample(brats_mod, 150)
        self.image_paths.extend(random_brats_images)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        return img, 0



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


class BinaryClassifier(nn.Module):
    def __init__(self, in_channels=1024):
        super(BinaryClassifier, self).__init__()
        # input shape: [Batch size, 256, 16, 16]
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        # output shape: [Batch size, 256, 1, 1]
        self.flatten = nn.Flatten()
        # output shape: [Batch size, 256]
        self.fc = nn.Linear(in_channels, 2)

    def forward(self, x):
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class BinaryClassifier2(nn.Module):
    def __init__(self, in_channels=2048):
        super(BinaryClassifier2, self).__init__()
        # input shape: [Batch size, 256, 16, 16]
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        # output shape: [Batch size, 256, 1, 1]
        self.flatten = nn.Flatten()
        # output shape: [Batch size, 256]
        self.fc = nn.Linear(in_channels, 2)

    def forward(self, x):
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


def train(_class_, shrink_factor=None, total_iters=2000, evaluation_epochs=250, training_using_pad=False, max_ratio=0,
          augmented_view=False, batch_size=16, model='wide_res50', different_view=False, head_end=False,
          image_size=256, unode_path=None, trainable_encoder_path=None, decoder_path=None, cls_path=None):
    print_fn(_class_)
    setup_seed(111)

    total_iters = total_iters
    image_size = image_size
    crop_size = image_size

    if augmented_view:
        train_data_transforms = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ColorJitter(0.8, 0.8, 0.8, 0.2),  # Color jitter
            transforms.RandomGrayscale(p=0.2),  # Random grayscale
            transforms.ToTensor(),
            transforms.CenterCrop(crop_size),
        ])
    else:
        train_data_transforms = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.CenterCrop(crop_size),
        ])
    data_transform, gt_transform = get_data_transforms(image_size, image_size)

    train_data = RSNATRAIN(transform=data_transform)
    test_data1 = RSNATEST(transform=data_transform, test_id=1)
    test_data2 = RSNATEST(transform=data_transform, test_id=2)
    test_data3 = RSNATEST(transform=data_transform, test_id=3)


    # visualize_random_samples_from_clean_dataset(train_data, 'train dataset RSNA')
    # visualize_random_samples_from_clean_dataset(test_data1, f'test data RSNA')
    # visualize_random_samples_from_clean_dataset(test_data3, f'test data CXRP')


    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4,
                                                   drop_last=False)
    train_dataloader2 = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4,
                                                   drop_last=False)
    test_dataloader1 = torch.utils.data.DataLoader(test_data1, batch_size=1, shuffle=False, num_workers=1)
    test_dataloader2 = torch.utils.data.DataLoader(test_data2, batch_size=1, shuffle=False, num_workers=1)
    test_dataloader3 = torch.utils.data.DataLoader(test_data3, batch_size=1, shuffle=False, num_workers=1)

    print('len Trainset(main)', len(train_data))
    print('len Testset(main)', len(test_data1))
    print('len Testset(shifted CXRP)', len(test_data3))
    
    in_channels = 1024
    if model == 'wide_res50':
        encoder, bn = wide_resnet50_2(pretrained=True, head_end=head_end)
        decoder = de_wide_resnet50_2(pretrained=False, output_conv=2)
    elif model == 'res18':
        encoder, bn = resnet18(pretrained=True, head_end=head_end)
        decoder = de_resnet18(pretrained=False, output_conv=2)
        in_channels = 256
    else:
        encoder, bn = wide_resnet50_2(pretrained=True, head_end=head_end)
        decoder = de_wide_resnet50_2(pretrained=False, output_conv=2)
    if not head_end:
        cls = BinaryClassifier(in_channels)
    else:
        cls = BinaryClassifier2(2 * in_channels)
    
    if cls_path is not None:
        cls_dic = torch.load(cls_path)
        cls.load_state_dict(cls_dic)
        print("cls loaded!")

    if unode_path is None:
        encoder_freeze = copy.deepcopy(encoder)
    else:
        if model != 'res18':
            print('Only res18 implemented!')
            exit(1)
        encoder_freeze, _ = resnet18(pretrained=True, unode_path=unode_path, head_end=head_end, is_unode_model=True)

    if trainable_encoder_path is not None:
        if model != 'res18':
            print('Only res18 implemented!')
            exit(1)
        encoder, _ = resnet18(pretrained=True, unode_path=trainable_encoder_path, head_end=head_end)

    if decoder_path is not None:
        if model != 'res18':
            print('Only res18 implemented!')
            exit(1)
        decoder = de_resnet18(pretrained=True, output_conv=2, decoder_path=decoder_path)    

    encoder = encoder.to(device)
    bn = bn.to(device)
    decoder = decoder.to(device)
    encoder_freeze = encoder_freeze.to(device)
    cls = cls.to(device)
    
    encoder_freeze.eval()
    model = ReContrast(encoder=encoder, encoder_freeze=encoder_freeze, bottleneck=bn, decoder=decoder,
                       image_size=image_size, crop_size=crop_size, device=device, head_end=head_end)
    # for m in encoder.modules():
    #     if isinstance(m, torch.nn.BatchNorm2d):
    #         m.eps = 1e-8

    criterion = nn.CrossEntropyLoss()
    optimizer_cls = torch.optim.AdamW(list(cls.parameters()), lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-5)
    optimizer = torch.optim.AdamW(list(decoder.parameters()) + list(bn.parameters()),
                                  lr=2e-3, betas=(0.9, 0.999), weight_decay=1e-5)
    optimizer2 = torch.optim.AdamW(list(encoder.parameters()),
                                   lr=1e-5, betas=(0.9, 0.999), weight_decay=1e-5)
    print_fn('train image number:{}'.format(len(train_data)))
    print_fn('test image number:{}'.format(len(test_data1)))
    macs, params = get_model_complexity_info(model, (3, crop_size, crop_size),
                                             as_strings=True, print_per_layer_stat=False)
    print_fn('Computation:{}'.format(macs))
    print_fn('Parameters:{}'.format(params))

    auroc_px_best, auroc_sp_best, aupro_px_best = 0, 0, 0
    it = 0

    auroc_px_list = {"main": 0, "shifted": 0}
    auroc_px_list_best = {"main": 0, "shifted": 0}

    auroc_sp_list = {"main": 0, "shifted": 0}
    auroc_sp_list_best = {"main": 0, "shifted": 0}

    auroc_aupro_px_list = {"main": 0, "shifted": 0}
    auroc_aupro_px_list_best = {"main": 0, "shifted": 0}

    auroc_cls_auc_list = {"main": 0, "shifted": 0}
    auroc_cls_auc_list_best = {"main": 0, "shifted": 0}

    auroc_mixed_auc_list = {"main": 0, "shifted": 0}
    auroc_mixed_auc_list_best = {"main": 0, "shifted": 0}

    auroc_mix_auc_list = {"main": 0, "shifted": 0}
    auroc_mix_auc_list_best = {"main": 0, "shifted": 0}

    anomaly_transforms = transforms.Compose([
        transforms.ToPILImage(),
        CutPasteUnion(transform=transforms.Compose([transforms.ToTensor(), ])),
    ])
    print('len(train_dataloader):', len(train_dataloader))
    for epoch in range(int(np.ceil(total_iters / len(train_dataloader)))):
        # encoder batchnorm in eval for these classes.
        model.train(encoder_bn_train=True)

        loss_list = []
        for img, label in train_dataloader:
            # img : [16, 3, 256, 256]
            # img = torch.cat([img, img.clone()])

            img = img.to(device)
            anomaly_data = np.ones(len(img))
            # en : [[16,256,64,64], [16,512,32,32], [16,1024,16,16], [16,256,64,64], [16,512,32,32], [16,1024,16,16]]
            # de : [[16,256,64,64], [16,512,32,32], [16,1024,16,16], [16,256,64,64], [16,512,32,32], [16,1024,16,16]]
            en, de = model(img)
            alpha_final = 1
            alpha = min(-3 + (alpha_final - -3) * it / (total_iters * 0.1), alpha_final)

            loss = global_cosine_hm(en[:3], de[:3], anomaly_data=anomaly_data, alpha=alpha, factor=0.) / 2 + \
                    global_cosine_hm(en[3:], de[3:], anomaly_data=anomaly_data, alpha=alpha, factor=0.) / 2
            
            '''
            loss2 = contrastive_loss(en[:3], de[:3], anomaly_data=anomaly_data, layer_num=0) + contrastive_loss(en[:3], de[:3], anomaly_data=anomaly_data, layer_num=1) + contrastive_loss(en[:3], de[:3], anomaly_data=anomaly_data, layer_num=2)
            loss3 = contrastive_loss(en[3:], de[3:], anomaly_data=anomaly_data, layer_num=0) +  contrastive_loss(en[3:], de[3:], anomaly_data=anomaly_data, layer_num=1) +  contrastive_loss(en[3:], de[3:], anomaly_data=anomaly_data, layer_num=2)
            loss = loss1 + (loss2/6) + (loss3/6)
            '''
            # loss = global_cosine(en[:3], de[:3], stop_grad=False) / 2 + \
            #        global_cosine(en[3:], de[3:], stop_grad=False) / 2
            # optimizer_cls.zero_grad()
            optimizer.zero_grad()
            optimizer2.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            # optimizer_cls.step()
            optimizer.step()
            optimizer2.step()
            loss_list.append(loss.item())

            if (it + 1) % evaluation_epochs == 0:
                cls.eval()
                model.train(mode=False)
                shrink_factor = "main"
                # auroc, f1, acc = evaluation_noseg(model, test_dataloader1, device)
                auroc_px_list[str(shrink_factor)], auroc_sp_list[str(shrink_factor)], auroc_aupro_px_list[
                    str(shrink_factor)], auroc_cls_auc_list[str(shrink_factor)], auroc_mix_auc_list[
                    str(shrink_factor)] = evaluation_noseg_brain(model,
                                                                 test_dataloader1,
                                                                 device,
                                                                 cls=cls,
                                                                 head_end=head_end,
                                                                 train_loader=train_dataloader2,
                                                                 anomaly_transforms=anomaly_transforms)
                print_fn(
                    'Shrink Factor:{}, Recon_Diff Auroc:{:.3f}, F1:{:.3f}, Acc:{:.3}, Unode Auroc:{:.3f}, Recon_Diff + UNODE:{:.3f}'.format(
                        shrink_factor,
                        auroc_px_list[
                            str(shrink_factor)],
                        auroc_sp_list[
                            str(shrink_factor)],
                        auroc_aupro_px_list[
                            str(shrink_factor)],
                        auroc_cls_auc_list[str(shrink_factor)],
                        auroc_mix_auc_list[shrink_factor]))

                if auroc_sp_list[str(shrink_factor)] >= auroc_sp_list_best[str(shrink_factor)]:
                    auroc_px_list_best[str(shrink_factor)], auroc_sp_list_best[str(shrink_factor)], \
                    auroc_aupro_px_list_best[str(shrink_factor)], auroc_cls_auc_list_best[str(shrink_factor)] = \
                        auroc_px_list[str(shrink_factor)], auroc_sp_list[
                            str(shrink_factor)], auroc_aupro_px_list[str(shrink_factor)], auroc_cls_auc_list[
                            str(shrink_factor)]

                shrink_factor = "shifted"
                auroc_px_list[str(shrink_factor)], auroc_sp_list[str(shrink_factor)], auroc_aupro_px_list[
                    str(shrink_factor)], auroc_cls_auc_list[str(shrink_factor)], auroc_mix_auc_list[
                    str(shrink_factor)] = evaluation_noseg_brain(model,
                                                                 test_dataloader3,
                                                                 device,
                                                                 cls=cls,
                                                                 head_end=head_end,
                                                                 train_loader=train_dataloader2,
                                                                 anomaly_transforms=anomaly_transforms)
                print_fn(
                    'Shrink Factor:{}, Recon_Diff Auroc:{:.3f}, F1:{:.3f}, Acc:{:.3}, Unode Auroc:{:.3f}, Recon_Diff + UNODE:{:.3f}'.format(
                        shrink_factor,
                        auroc_px_list[
                            str(shrink_factor)],
                        auroc_sp_list[
                            str(shrink_factor)],
                        auroc_aupro_px_list[
                            str(shrink_factor)],
                        auroc_cls_auc_list[str(shrink_factor)],
                        auroc_mix_auc_list[shrink_factor]))

                if auroc_sp_list[str(shrink_factor)] >= auroc_sp_list_best[str(shrink_factor)]:
                    auroc_px_list_best[str(shrink_factor)], auroc_sp_list_best[str(shrink_factor)], \
                    auroc_aupro_px_list_best[str(shrink_factor)], auroc_cls_auc_list_best[str(shrink_factor)] = \
                        auroc_px_list[str(shrink_factor)], auroc_sp_list[
                            str(shrink_factor)], auroc_aupro_px_list[str(shrink_factor)], auroc_cls_auc_list[
                            str(shrink_factor)]

                model.train(encoder_bn_train=True)
                cls.train()

            it += 1
            if it == total_iters:
                break
        print_fn('iter [{}/{}], loss:{:.4f}'.format(it, total_iters, np.mean(loss_list)))
    
    model.save_models()
    torch.save(cls.state_dict(), './cls.pt')
    # visualize(model, test_dataloader, device, _class_=_class_, save_name=args.save_name)
    return auroc_px_list, auroc_sp_list, auroc_aupro_px_list, auroc_cls_auc_list, auroc_mixed_auc_list, \
           auroc_px_list_best, auroc_sp_list_best, auroc_aupro_px_list_best, auroc_cls_auc_list_best, auroc_mixed_auc_list_best


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--save_dir', type=str, default='./saved_results')
    parser.add_argument('--save_name', type=str,
                        default='recontrast_mvtec_it2k_lr2e31e5_wd1e5_hm1d01_s111')
    parser.add_argument('--gpu', default='0', type=str,
                        help='GPU id to use.')
    parser.add_argument('--shrink_factor', type=float, default=None)
    parser.add_argument('--total_iters', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--evaluation_epochs', type=int, default=250)
    parser.add_argument('--training_shrink_factor', action='store_true')
    parser.add_argument('--training_using_pad', action='store_true')
    parser.add_argument('--max_ratio', type=float, default=0)
    parser.add_argument('--augmented_view', action='store_true')
    parser.add_argument('--model', type=str, default='wide_res50')
    parser.add_argument('--item_list', type=str, default='0,15')
    parser.add_argument('--different_view', action='store_true')
    parser.add_argument('--head_end', action='store_true',
                        help='put the cls head at the end of the encoder (instead of the 3rd layer)')
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--unode_path', type=str, default=None)
    parser.add_argument('--trainable_encoder_path', type=str, default=None)
    parser.add_argument('--decoder_path', type=str, default=None)
    parser.add_argument('--cls_path', type=str, default=None)

    args = parser.parse_args()
    image_size = args.image_size

    if args.training_shrink_factor:
        args.training_using_pad = True

    logger = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name))
    print_fn = logger.info

    head_end = args.head_end

    device = 'cuda:' + args.gpu if torch.cuda.is_available() else 'cpu'
    print_fn(device)

    result_list = {"main": [], "shifted": []}
    result_list_best = {"main": [], "shifted": []}
    pad_size = ["main", "shifted"]
    item = 'brain'
    print(f"+++++++++++++++++++++++++++++++++++++++{item}+++++++++++++++++++++++++++++++++++++++")
    auroc_px, auroc_sp, aupro_px, auroc_sp_cls, auroc_mixed, auroc_px_best, auroc_sp_best, aupro_px_best, auroc_sp_cls_best, auroc_mixed_best = train(
        item,
        shrink_factor=args.shrink_factor,
        total_iters=args.total_iters,
        evaluation_epochs=args.evaluation_epochs,
        training_using_pad=args.training_using_pad,
        max_ratio=args.max_ratio,
        augmented_view=args.augmented_view,
        batch_size=args.batch_size,
        model=args.model,
        different_view=args.different_view,
        head_end=head_end,
        image_size=image_size,
        unode_path=args.unode_path,
        trainable_encoder_path=args.trainable_encoder_path,
        decoder_path=args.decoder_path,
        cls_path=args.cls_path)
    # for pad in pad_size:
    #     result_list[str(pad)].append(
    #         [item, auroc_px[str(pad)], auroc_sp[str(pad)], aupro_px[str(pad)], auroc_sp_cls[str(pad)]])
    #     result_list_best[str(pad)].append(
    #         [item, auroc_px_best[str(pad)], auroc_sp_best[str(pad)], aupro_px_best[str(pad)],
    #          auroc_sp_cls_best[str(pad)]])
    '''
    for pad in pad_size:
        print(f'-------- shrink factor = {pad} --------')
        mean_auroc_px = np.mean([result[1] for result in result_list[str(pad)]])
        mean_auroc_sp = np.mean([result[2] for result in result_list[str(pad)]])
        mean_aupro_px = np.mean([result[3] for result in result_list[str(pad)]])
        mean_auc_sp_cls = np.mean([result[4] for result in result_list[str(pad)]])
        print_fn(result_list[str(pad)])
        print_fn('mPixel Auroc:{:.4f}, mSample Map Auroc:{:.4f}, mPixel Aupro:{:.4}, mSample AUC cls:{:.4}'.format(
            mean_auroc_px, mean_auroc_sp,
            mean_aupro_px, mean_auc_sp_cls))

        best_auroc_px = np.mean([result[1] for result in result_list_best[str(pad)]])
        best_auroc_sp = np.mean([result[2] for result in result_list_best[str(pad)]])
        best_aupro_px = np.mean([result[3] for result in result_list_best[str(pad)]])
        best_auc_sp_cls = np.mean([result[4] for result in result_list_best[str(pad)]])
        print_fn(result_list_best[str(pad)])
        print_fn('bPixel Auroc:{:.4f}, bSample Map Auroc:{:.4f}, bPixel Aupro:{:.4}, bSample Auroc cls:{:.4}'.format(
            best_auroc_px, best_auroc_sp,
            best_aupro_px, best_auc_sp_cls))
    '''
