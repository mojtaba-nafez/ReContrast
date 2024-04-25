import torch
import torch.nn as nn
from cutpaste_transformation import CutPasteUnion
from dataset import get_data_transforms, get_strong_transforms
from torchvision.datasets import ImageFolder
import numpy as np
import random
import os
from torch.utils.data import DataLoader
from models.resnet import resnet18, resnet34, resnet50, wide_resnet50_2, resnext50_32x4d
from models.de_resnet import de_wide_resnet50_2, de_resnet18, de_resnet34, de_resnet50, de_resnext50_32x4d
from models.recontrast import ReContrast, ReContrast
from dataset import MVTecDataset, Train_MVTecDataset
import torch.backends.cudnn as cudnn
import argparse
from utils import evaluation, visualize, global_cosine, global_cosine_hm, NT_xent, contrastive_loss
from torch.nn import functional as F
from functools import partial
from ptflops import get_model_complexity_info
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn as nn

import warnings
import copy
import logging
from cutpaste_transformation import *

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



class NewModel(nn.Module):

    def __init__(self, encoder, bn, decoder):
        super(NewModel, self).__init__()
        self.encoder = encoder
        self.bn = bn
        self.decoder = decoder
        self.classifier = nn.Linear(256, 2).to('cuda')

        self.encoder.eval()
        self.bn.eval()
        self.decoder.train()

    def forward(self, img):
        with torch.no_grad():
            en = self.encoder(img)
            en2 = [torch.cat([a, b], dim=0) for a, b in zip(en, en)]

            bottle = self.bn(en2)

        de = self.decoder(bottle)

        de = [a.chunk(dim=0, chunks=2) for a in de]

        ## USING de[2][0]
        logits = de[2][0].mean(dim=(-2, -1), keepdim=True).squeeze()
        out = self.classifier(logits)

        return out



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
          augmented_view=False, batch_size=16, model='wide_res50', lr_cls=1e-3, head_end=False, update_decoder=False,
          unode1_checkpoint=None, unode2_checkpoint=None):
    anomaly_transforms = transforms.Compose([
        transforms.ToPILImage(),
        CutPasteUnion(transform=transforms.Compose([transforms.ToTensor(), ]))])
    print_fn(_class_)
    setup_seed(111)

    total_iters = total_iters
    image_size = 256
    crop_size = 256

    if augmented_view:
        train_data_transforms = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),  # Random horizontal flip
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

    test_path = '/kaggle/input/mvtec-ad/' + _class_
    if training_using_pad:
        train_path = '/kaggle/input/mvtec-ad/'
        train_data = Train_MVTecDataset(root=train_path, category=_class_, transform=train_data_transforms)
    else:
        train_path = '/kaggle/input/mvtec-ad/' + _class_ + '/train'
        train_data = ImageFolder(root=train_path, transform=train_data_transforms)

    train_data = ImageFolder(root=train_path, transform=data_transform)

    test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test",
                             shrink_factor=shrink_factor)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4,
                                                   drop_last=False)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=1)

    visualize_random_samples_from_clean_dataset(train_data, f"train_data_{_class_}", train_data=True)
    visualize_random_samples_from_clean_dataset(test_data, f"test_data_{_class_}", train_data=False)

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

    #     if update_decoder:
    #         print('updating decoder...')
    #         anomaly_transforms = transforms.Compose([
    #             transforms.ToPILImage(),
    #             CutPasteUnion(transform=transforms.Compose([transforms.ToTensor(), ])),
    #         ])

    #         new_model = NewModel(encoder, bn, decoder)
    #         criteron = nn.CrossEntropyLoss()
    #         optimizer = torch.optim.AdamW(list(new_model.parameters()),
    #                                       lr=2e-3, betas=(0.9, 0.999), weight_decay=1e-5)
    #         encoder.eval()
    #         bn.eval()
    #         decoder.train()
    #         for epoch in range(21):
    #             loss_list = []
    #             for img, label in train_dataloader:
    #                 img = img.to(device)

    #                 anomaly_data = np.ones(len(img)) * 0
    #                 numbers = list(range(len(img)))
    #                 random.shuffle(numbers)
    #                 anomaly_data[numbers[:int(len(numbers) / 2)]] = 1

    #                 for i in range(len(anomaly_data)):
    #                     if anomaly_data[i] == 1:
    #                         img[i] = anomaly_transforms(img[i])
    #                 anomaly_data = torch.tensor(anomaly_data).to(device)

    #                 logits = new_model(img)
    #                 anomaly_data = anomaly_data.to(torch.long)
    #                 loss = criteron(logits, anomaly_data)
    #                 optimizer.zero_grad()
    #                 loss.backward()
    #                 optimizer.step()
    #                 loss_list.append(loss.item())
    #             print('loss:', np.mean(loss_list))

    #             if epoch % 10 == 0:
    #                 decoder.eval()
    #                 correct = 0
    #                 total = 0
    #                 for img, _, label, _ in test_dataloader:
    #                     img = img.to(device)
    #                     label = label.to(device)
    #                     with torch.no_grad():
    #                         output = new_model(img)
    #                         total += len(img)
    #                         if len(img) > 1:
    #                             _, pred = torch.max(output, dim=1)
    #                         else:
    #                             pred = 0 if output[0] > output[1] else 1
    #                         correct += (pred == label).sum().item()

    #                 accuracy = 100 * correct / total
    #                 print(f'Accuracy on test data: {accuracy:.2f}%')

    #                 correct = 0
    #                 total = 0
    #                 for img, label in train_dataloader:
    #                     img = img.to(device)
    #                     label = label.to(device)
    #                     with torch.no_grad():
    #                         output = new_model(img)
    #                         total += len(img)
    #                         _, pred = torch.max(output, dim=1)
    #                         correct += (pred == label).sum().item()

    #                 accuracy = 100 * correct / total
    #                 print(f'Accuracy on train data: {accuracy:.2f}%')

    #                 decoder.train()


    #         torch.save(decoder.state_dict(), 'decoder_trained.pth')
    #         print('saved decoder...')

    #     model = ReContrast(encoder=encoder, encoder_freeze=encoder_freeze, bottleneck=bn, decoder=decoder)

    in_channels = 1024
    if model == 'wide_res50':
        encoder, bn = wide_resnet50_2(pretrained=True)
        decoder = de_wide_resnet50_2(pretrained=False, output_conv=2)
    elif model == 'res18':
        encoder, bn = resnet18(pretrained=True)
        decoder = de_resnet18(pretrained=False, output_conv=2)
        in_channels = 256
    else:
        encoder, bn = wide_resnet50_2(pretrained=True)
        decoder = de_wide_resnet50_2(pretrained=False, output_conv=2)
    if not head_end:
        cls = BinaryClassifier(in_channels=in_channels)
    else:
        cls = BinaryClassifier2(in_channels=2 * in_channels)
    encoder = encoder.to(device)
    bn = bn.to(device)
    decoder = decoder.to(device)
    encoder_freeze = copy.deepcopy(encoder)
    cls = cls.to(device)
    model = ReContrast(encoder=encoder, encoder_freeze=encoder_freeze, bottleneck=bn, decoder=decoder,
                       image_size=image_size, crop_size=crop_size, device=device, head_end=head_end)


    criterion = nn.CrossEntropyLoss()
    optimizer_cls = torch.optim.AdamW(list(cls.parameters()), lr=lr_cls, betas=(0.9, 0.999), weight_decay=1e-5)

    optimizer = torch.optim.AdamW(list(decoder.parameters()) + list(bn.parameters()),
                                  lr=2e-3, betas=(0.9, 0.999), weight_decay=1e-5)
    optimizer2 = torch.optim.AdamW(list(encoder.parameters()),
                                   lr=1e-5, betas=(0.9, 0.999), weight_decay=1e-5)
    print_fn('train image number:{}'.format(len(train_data)))
    print_fn('test image number:{}'.format(len(test_data)))
    macs, params = get_model_complexity_info(model, (3, crop_size, crop_size),
                                             as_strings=True, print_per_layer_stat=False)
    print_fn('Computation:{}'.format(macs))  # NONE vs 10.11 GMac
    print_fn('Parameters:{}'.format(params))  # NONe vs 21.65 M

    auroc_px_best, auroc_sp_best, aupro_px_best = 0, 0, 0
    it = 0
    print('len train dat aloader: ', len(train_dataloader))


    auroc_px_list = {"0.8": 0, "0.85": 0, "0.9": 0, "0.95": 0, "0.98": 0, "1.0": 0}
    auroc_px_list_best = {"0.8": 0, "0.85": 0, "0.9": 0, "0.95": 0, "0.98": 0, "1.0": 0}

    auroc_sp_list = {"0.8": 0, "0.85": 0, "0.9": 0, "0.95": 0, "0.98": 0, "1.0": 0}
    auroc_sp_list_best = {"0.8": 0, "0.85": 0, "0.9": 0, "0.95": 0, "0.98": 0, "1.0": 0}

    auroc_aupro_px_list = {"0.8": 0, "0.85": 0, "0.9": 0, "0.95": 0, "0.98": 0, "1.0": 0}
    auroc_aupro_px_list_best = {"0.8": 0, "0.85": 0, "0.9": 0, "0.95": 0, "0.98": 0, "1.0": 0}


    auroc_cls_auc_list = {"0.8": 0, "0.85": 0, "0.9": 0, "0.95": 0, "0.98": 0, "1.0": 0}
    auroc_cls_auc_list_best = {"0.8": 0, "0.85": 0, "0.9": 0, "0.95": 0, "0.98": 0, "1.0": 0}

    anomaly_transforms = transforms.Compose([
        transforms.ToPILImage(),
        CutPasteUnion(transform=transforms.Compose([transforms.ToTensor(), ])),
    ])
    for epoch in range(int(np.ceil(total_iters / len(train_dataloader)))):
        # encoder batchnorm in eval for these classes.
        model.train(encoder_bn_train=_class_ not in ['toothbrush', 'leather', 'grid', 'tile', 'wood', 'screw'])

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


            loss1 = global_cosine_hm(en[:3], de[:3], anomaly_data=anomaly_data, alpha=alpha, factor=0.) / 2 + \
                    global_cosine_hm(en[3:], de[3:], anomaly_data=anomaly_data, alpha=alpha, factor=0.) / 2
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
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer_cls.step()
            optimizer.step()
            optimizer2.step()
            loss_list.append(loss.item())

            if (it + 1) % evaluation_epochs == 0:
                pad_size = [1.0, 0.98, 0.95, 0.9, 0.85, 0.8]

                for shrink_factor in pad_size:
                    test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform,
                                             phase="test", shrink_factor=shrink_factor)

                    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=1)

                    auroc_px_list[str(shrink_factor)], auroc_sp_list[str(shrink_factor)], auroc_aupro_px_list[
                        str(shrink_factor)], auroc_cls_auc_list[str(shrink_factor)] = evaluation(model, test_dataloader,
                                                                                                 device,
                                                                                                 max_ratio=max_ratio,
                                                                                                 cls=cls,
                                                                                                 head_end=head_end)
                    print_fn(
                        'Shrink Factor:{:.3f}, Pixel Auroc:{:.3f}, Sample Map Auroc:{:.3f}, Pixel Aupro:{:.3}, Sample CLS AUROC:{:.3}'.format(
                            shrink_factor, auroc_px_list[str(shrink_factor)], auroc_sp_list[str(shrink_factor)],
                            auroc_aupro_px_list[str(shrink_factor)], auroc_cls_auc_list[str(shrink_factor)]))

                    if auroc_sp_list[str(shrink_factor)] >= auroc_sp_list_best[str(shrink_factor)]:
                        auroc_px_list_best[str(shrink_factor)], auroc_sp_list_best[str(shrink_factor)], \
                        auroc_aupro_px_list_best[str(shrink_factor)], auroc_cls_auc_list_best[str(shrink_factor)] = \
                            auroc_px_list[str(shrink_factor)], auroc_sp_list[
                                str(shrink_factor)], auroc_aupro_px_list[str(shrink_factor)], auroc_cls_auc_list[
                                str(shrink_factor)]

                model.train(encoder_bn_train=_class_ not in ['toothbrush', 'leather', 'grid', 'tile', 'wood', 'screw'])
                cls.train()
            it += 1
            if it == total_iters:
                break
        print_fn('iter [{}/{}], loss:{:.4f}'.format(it, total_iters, np.mean(loss_list)))

    # visualize(model, test_dataloader, device, _class_=_class_, save_name=args.save_name)
    model.save_models()
    return auroc_px_list, auroc_sp_list, auroc_aupro_px_list, auroc_cls_auc_list, auroc_px_list_best, auroc_sp_list_best, auroc_aupro_px_list_best, auroc_cls_auc_list_best


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
    parser.add_argument('--lr_cls', type=float, default=0.001)
    parser.add_argument('--encoder1_path', type=str, default='')
    parser.add_argument('--encoder2_path', type=str, default='')
    parser.add_argument('--update_decoder', type=str, default='0')
      
    parser.add_argument('--head_end', action='store_true',
                        help='put the cls head at the end of the encoder (instead of the 3rd layer)')

    args = parser.parse_args()

    if args.training_shrink_factor:
        args.training_using_pad = True

    item_list = ['screw', 'cable', 'transistor', 'carpet', 'bottle', 'hazelnut', 'leather', 'capsule', 'grid', 'pill',
                 'metal_nut', 'toothbrush', 'zipper', 'tile', 'wood']

    items = args.item_list.split(',')
    st = int(items[0])
    ed = int(items[1])
    item_list = item_list[st: ed]
    print(item_list)
    # item_list = ['toothbrush']
    head_end = args.head_end

    logger = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name))
    print_fn = logger.info

    device = 'cuda:' + args.gpu if torch.cuda.is_available() else 'cpu'
    print_fn(device)


    result_list = {"0.8": [], "0.85": [], "0.9": [], "0.95": [], "0.98": [], "1.0": []}
    result_list_best = {"0.8": [], "0.85": [], "0.9": [], "0.95": [], "0.98": [], "1.0": []}
    pad_size = [1.0, 0.98, 0.95, 0.9, 0.85, 0.8]

    for i, item in enumerate(item_list):
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
            lr_cls=args.lr_cls,
            head_end=head_end)
        for pad in pad_size:
            result_list[str(pad)].append(
                [item, auroc_px[str(pad)], auroc_sp[str(pad)], aupro_px[str(pad)], auroc_sp_cls[str(pad)]])
            result_list_best[str(pad)].append(
                [item, auroc_px_best[str(pad)], auroc_sp_best[str(pad)], aupro_px_best[str(pad)],
                 auroc_sp_cls_best[str(pad)]])

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
