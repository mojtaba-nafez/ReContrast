import torch
from torchvision.datasets.folder import default_loader

from dataset import get_data_transforms, get_strong_transforms
from torchvision.datasets import ImageFolder
import numpy as np
import torch.nn as nn
import random
import os
from torch.utils.data import DataLoader, Dataset
from models.resnet import resnet18, resnet34, resnet50, wide_resnet50_2, resnext50_32x4d
from models.de_resnet import de_wide_resnet50_2, de_resnet18, de_resnet34, de_resnet50, de_resnext50_32x4d
from models.recontrast import ReContrast, ReContrast
import torch.backends.cudnn as cudnn
import argparse
from utils import evaluation, visualize, global_cosine, modify_grad, NT_xent, contrastive_loss, evaluation_noseg_brain
from torch.nn import functional as F
from functools import partial
from ptflops import get_model_complexity_info
from torchvision import transforms
import matplotlib.pyplot as plt
from glob import glob
from PIL import Image
import warnings
import copy
import logging
from cutpaste_transformation import *
import pandas as pd

import copy
# from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
# from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
# from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision

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


def global_cosine_hm_1(a, b, anomaly_data, alpha=1., factor=0.):
    # a(enc), b(dec): [[16,256,64,64], [16,512,32,32], [16,1024,16,16]]
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    weight = [1, 1, 1]
    for item in range(len(a)):
        a_ = a[item].detach()
        b_ = b[item]

        a_ = a_[anomaly_data == 1]
        b_ = b_[anomaly_data == 1]
        if (len(a_) < 1) or (len(b_) < 1):
            print("a_, b_", a_, b_)
            print("a, b", a, b)
            print("anomaly_data:", anomaly_data)
            continue

        with torch.no_grad():
            point_dist = 1 - cos_loss(a_, b_).unsqueeze(1)

        # mean_dist, std_dist: just are a number
        mean_dist = point_dist.mean()
        std_dist = point_dist.reshape(-1).std()

        # cos_loss(a_.view(a_.shape[0], -1),b_.view(b_.shape[0], -1)): torch.Size([8])
        # loss += (torch.mean((1 - cos_loss(a_.view(a_.shape[0], -1),b_.view(b_.shape[0], -1)))*anomaly_data)) * weight[item]
        loss += torch.mean(1 - cos_loss(a_.view(a_.shape[0], -1), b_.view(b_.shape[0], -1))) * weight[item]
        thresh = mean_dist + alpha * std_dist
        partial_func = partial(modify_grad, inds=point_dist < thresh, factor=factor)
        b_.register_hook(partial_func)

    return loss


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


class RandomRotationTransform:
    def __init__(self, transform=None):
        # List of angles
        self.angles = [90, 180, 270]
        self.transform = transform

    def __call__(self, x):
        # Select a random angle
        angle = random.choice(self.angles)
        if self.transform:
            x = self.transform(x)
        return transforms.functional.rotate(x, angle)


def three_digits(a: int):
    x = str(a)
    if len(x) == 1:
        return f'00{a}'
    if len(x) == 2:
        return f'0{a}'
    return x


class WBCDataset(torch.utils.data.Dataset):  ####  FOR MAIN / SHIFTED
    def __init__(self, root1, root2,
                 labels1: pd.DataFrame, labels2: pd.DataFrame, transform=None, train=True, test_id=1, ratio=0.7, count=-1):
        self.transform = transform
        self.root1 = root1
        self.root2 = root2
        self.labels1 = labels1
        self.labels2 = labels2
        self.train = train
        self.test_id = test_id
        self.targets = []
        labels1 = labels1[labels1['class label'] != 5]
        labels2 = labels2[labels2['class'] != 5]

        normal_df = labels1[labels1['class label'] == 1]
        self.normal_paths = [os.path.join(root1, f'{three_digits(x)}.bmp') for x in list(normal_df['image ID'])]
        random.seed(42)
        random.shuffle(self.normal_paths)
        self.separator = int(ratio * len(self.normal_paths))
        self.train_paths = self.normal_paths[:self.separator]

        if self.train:
            self.image_paths = self.train_paths
            self.targets = [0] * len(self.image_paths)
        else:
            if self.test_id == 1:
                all_images = glob(os.path.join(root1, '*.bmp'))
                self.image_paths = [x for x in all_images if x not in self.train_paths]
                self.image_paths = [x for x in self.image_paths if
                                    int(os.path.basename(x).split('.')[0]) in labels1['image ID'].values]
                ids = [os.path.basename(x).split('.')[0] for x in self.image_paths]
                ids_labels = list(labels1[labels1['image ID'] == int(x)]['class label'] for x in ids)
                self.targets = [0 if x.item() == 1 else 1 for x in ids_labels]
            else:
                self.image_paths = glob(os.path.join(root2, '*.bmp'))
                self.image_paths = [x for x in self.image_paths if int(os.path.basename(x).split('.')[0])
                                    in labels2['image ID'].values]
                self.targets = [
                    0 if labels2[labels2['image ID'] == int(os.path.basename(x).split('.')[0])]['class'].item() == 1
                    else 1 for x in self.image_paths]

        if count != -1:
            if count < len(self.image_paths):
                self.image_paths = self.image_paths[:count]
                self.targets = self.targets[:count]
            else:
                t = len(self.image_paths)
                for i in range(count - t):
                    self.image_paths.append(random.choice(self.image_paths[:t]))
                    self.targets.append(random.choice(self.targets[:t]))
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.train:
            return img, 0
        gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        gt[:, :, 1:3] = 1
        return img, gt, self.targets[idx], img_path

def train(_class_, shrink_factor=None, total_iters=2000, evaluation_epochs=250, training_using_pad=False, max_ratio=0,
          augmented_view=False, batch_size=16, model='wide_res50', different_view=False, head_end=False,
          image_size=256, unode_path=None, trainable_encoder_path=None, decoder_path=None, cls_path=None, grad=False,
          pretrain_unode_weghts=False):
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
    data_transform, gt_transform = get_data_transforms(image_size, crop_size)

    root1 = '/kaggle/working/segmentation_WBC/Dataset 1'
    root2 = '/kaggle/working/segmentation_WBC/Dataset 2'
    import pandas as pd
    df1 = pd.read_csv('/kaggle/working/segmentation_WBC/Class Labels of Dataset 1.csv')
    df2 = pd.read_csv('/kaggle/working/segmentation_WBC/Class Labels of Dataset 2.csv')
    train_data = WBCDataset(root1=root1, root2=root2,
                               labels1=df1, labels2=df2, transform=data_transform, train=True)
    test_data1 = WBCDataset(root1=root1, root2=root2,
                              labels1=df1, labels2=df2, transform=data_transform, train=False, test_id=1)
    test_data2 = WBCDataset(root1=root1, root2=root2,
                              labels1=df1, labels2=df2, transform=data_transform, train=False, test_id=2)

    visualize_random_samples_from_clean_dataset(train_data, 'train dataset')
    visualize_random_samples_from_clean_dataset(test_data1, f'test data main')
    visualize_random_samples_from_clean_dataset(test_data2, f'test data shifted')

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader1 = torch.utils.data.DataLoader(test_data1, batch_size=1, shuffle=False, num_workers=1)
    test_dataloader2 = torch.utils.data.DataLoader(test_data2, batch_size=1, shuffle=False, num_workers=1)

    # visualize_random_samples_from_clean_dataset(train_data, f"train_data_{_class_}", train_data=True)
    # visualize_random_samples_from_clean_dataset(test_data, f"test_data_{_class_}", train_data=False)

    in_channels = 1024
    if model == 'wide_res50':
        encoder, bn = wide_resnet50_2(pretrained=True, head_end=head_end, pretrain_unode_weghts=pretrain_unode_weghts)
        decoder = de_wide_resnet50_2(pretrained=False, output_conv=2)
    elif model == 'res18':
        encoder, bn = resnet18(pretrained=True, head_end=head_end, pretrain_unode_weghts=pretrain_unode_weghts)
        decoder = de_resnet18(pretrained=False, output_conv=2)
        in_channels = 256
    else:
        encoder, bn = wide_resnet50_2(pretrained=True, head_end=head_end, pretrain_unode_weghts=pretrain_unode_weghts)
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
        if model not in ['res18', 'wide_res50']:
            print('Only res18 and wide_res50 implemented!')
            exit(1)
        if model == 'res18':
            encoder_freeze, _ = resnet18(pretrained=True, unode_path=unode_path, head_end=head_end, is_unode_model=True,
                                         pretrain_unode_weghts=pretrain_unode_weghts)
        if model == 'wide_res50':
            encoder_freeze, _ = wide_resnet50_2(pretrained=True, unode_path=unode_path, head_end=head_end,
                                                is_unode_model=True, pretrain_unode_weghts=pretrain_unode_weghts)

    if trainable_encoder_path is not None:
        if model not in ['res18', 'wide_res50']:
            print('Only res18 and wide_res50 implemented!')
            exit(1)
        if model == 'res18':
            encoder, _ = resnet18(pretrained=True, unode_path=trainable_encoder_path, head_end=head_end,
                                  pretrain_unode_weghts=pretrain_unode_weghts)
        if model == 'wide_res50':
            encoder, _ = wide_resnet50_2(pretrained=True, unode_path=trainable_encoder_path, head_end=head_end,
                                         pretrain_unode_weghts=pretrain_unode_weghts)

    if decoder_path is not None:
        if model not in ['res18', 'wide_res50']:
            print('Only res18 and wide_res50 implemented!')
            exit(1)
        if model == 'res18':
            decoder = de_resnet18(pretrained=True, output_conv=2, decoder_path=decoder_path)
        if model == 'wide_res50':
            decoder = de_wide_resnet50_2(pretrained=True, output_conv=2, decoder_path=decoder_path)

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
    print_fn('test main image number:{}'.format(len(test_data1)))
    print_fn('test shifted image number:{}'.format(len(test_data2)))
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

    auroc_mix_auc_list = {"main": 0, "shifted": 0}
    auroc_mix_auc_list_best = {"main": 0, "shifted": 0}

    anomaly_transforms = transforms.Compose([
        transforms.ToPILImage(),
        CutPasteUnion(transform=transforms.Compose([transforms.ToTensor(), ])),
    ])

    if total_iters == 0:
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
                                                         train_loader=train_dataloader,
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
                                                         test_dataloader2,
                                                         device,
                                                         cls=cls,
                                                         head_end=head_end,
                                                         train_loader=train_dataloader,
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

    for epoch in range(int(np.ceil(total_iters / len(train_dataloader)))):
        # encoder batchnorm in eval for these classes.
        model.train(encoder_bn_train=True)

        loss_list = []
        for img, label in train_dataloader:
            # img : [16, 3, 256, 256]
            # img = torch.cat([img, img.clone()])

            img = img.to(device)
            anomaly_data = np.ones(len(img))
            anomaly_data[int(len(anomaly_data) / 2):] = -1
            for i in range(len(anomaly_data)):
                if anomaly_data[i] == -1:
                    img[i] = anomaly_transforms(img[i])
            anomaly_data = torch.tensor(anomaly_data).to(device)
            # we also need one where instead on -1s we have 1s
            anomaly_one = [1 if x == -1 else 0 for x in anomaly_data]
            anomaly_one = torch.tensor(anomaly_one).to(device)
            # en : [[16,256,64,64], [16,512,32,32], [16,1024,16,16], [16,256,64,64], [16,512,32,32], [16,1024,16,16]]
            # de : [[16,256,64,64], [16,512,32,32], [16,1024,16,16], [16,256,64,64], [16,512,32,32], [16,1024,16,16]]
            if not head_end:
                en, de = model(img, head_end=head_end)
                cls_output = cls(en[5])
            else:
                en, de, en3 = model(img, head_end=head_end)
                cls_output = cls(en3)

            cls_loss = criterion(cls_output, anomaly_one.to(torch.int64))
            alpha_final = 1
            alpha = min(-3 + (alpha_final - -3) * it / (total_iters * 0.1), alpha_final)
            loss = None
            if sum(anomaly_data[anomaly_data == 1]) > 0:
                loss1 = global_cosine_hm_1(en[:3], de[:3], anomaly_data=anomaly_data, alpha=alpha, factor=0.) / 2 + \
                        global_cosine_hm_1(en[3:], de[3:], anomaly_data=anomaly_data, alpha=alpha, factor=0.) / 2
                loss2 = (contrastive_loss(en[:3], de[:3], anomaly_data=anomaly_data, layer_num=2) / 2) + \
                        (contrastive_loss(en[3:], de[3:], anomaly_data=anomaly_data, layer_num=2) / 2)
                loss = loss1 + loss2 + cls_loss
            '''
            loss2 = contrastive_loss(en[:3], de[:3], anomaly_data=anomaly_data, layer_num=0) + contrastive_loss(en[:3], de[:3], anomaly_data=anomaly_data, layer_num=1) + contrastive_loss(en[:3], de[:3], anomaly_data=anomaly_data, layer_num=2)
            loss3 = contrastive_loss(en[3:], de[3:], anomaly_data=anomaly_data, layer_num=0) +  contrastive_loss(en[3:], de[3:], anomaly_data=anomaly_data, layer_num=1) +  contrastive_loss(en[3:], de[3:], anomaly_data=anomaly_data, layer_num=2)
            loss = loss1 + (loss2/6) + (loss3/6)
            '''
            # loss = global_cosine(en[:3], de[:3], stop_grad=False) / 2 + \
            #        global_cosine(en[3:], de[3:], stop_grad=False) / 2

            optimizer_cls.zero_grad()
            optimizer.zero_grad()
            optimizer2.zero_grad()
            try:
                loss.backward()
            except:
                print("loss has not grad!")
                continue
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer_cls.step()
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
                                                                 train_loader=train_dataloader,
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
                                                                 test_dataloader2,
                                                                 device,
                                                                 cls=cls,
                                                                 head_end=head_end,
                                                                 train_loader=train_dataloader,
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
    return auroc_px_list, auroc_sp_list, auroc_aupro_px_list, auroc_cls_auc_list, \
           auroc_px_list_best, auroc_sp_list_best, auroc_aupro_px_list_best, auroc_cls_auc_list_best


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
    parser.add_argument('--different_view', action='store_true')
    parser.add_argument('--model', type=str, default='wide_res50')
    parser.add_argument('--item_list', type=str, default='0,15')
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--unode_path', type=str, default=None)
    parser.add_argument('--head_end', action='store_true',
                        help='put the cls head at the end of the encoder (instead of the 3rd layer)')
    parser.add_argument('--trainable_encoder_path', type=str, default=None)
    parser.add_argument('--decoder_path', type=str, default=None)
    parser.add_argument('--cls_path', type=str, default=None)
    parser.add_argument('--grad', action='store_true')
    parser.add_argument('--pretrain_unode_weghts', action='store_true')

    args = parser.parse_args()

    unode_path = args.unode_path
    image_size = args.image_size

    if args.training_shrink_factor:
        args.training_using_pad = True

    # item_list = ['toothbrush']

    logger = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name))
    print_fn = logger.info

    head_end = args.head_end

    device = 'cuda:' + args.gpu if torch.cuda.is_available() else 'cpu'
    print_fn(device)

    result_list = {"main": [], "shifted": []}
    result_list_best = {"main": [], "shifted": []}
    pad_size = ["main", "shifted"]
    item = 'wbc'
    print(f"+++++++++++++++++++++++++++++++++++++++{item}+++++++++++++++++++++++++++++++++++++++")
    auroc_px, auroc_sp, aupro_px, auroc_sp_cls, auroc_px_best, auroc_sp_best, aupro_px_best, auroc_sp_cls_best = train(
        item,
        shrink_factor=args.shrink_factor,
        total_iters=args.total_iters,
        evaluation_epochs=args.evaluation_epochs,
        training_using_pad=args.training_using_pad,
        max_ratio=args.max_ratio,
        augmented_view=args.augmented_view,
        batch_size=args.batch_size,
        model=args.model,
        head_end=head_end,
        image_size=image_size,
        unode_path=unode_path,
        trainable_encoder_path=args.trainable_encoder_path,
        decoder_path=args.decoder_path,
        cls_path=args.cls_path,
        grad=args.grad,
        pretrain_unode_weghts=args.pretrain_unode_weghts)
