
import torch
from dataset import get_data_transforms, get_strong_transforms
from torchvision.datasets import ImageFolder
import numpy as np
import random
import os
from torch.utils.data import DataLoader
from models.resnet import resnet18, resnet34, resnet50, wide_resnet50_2, wide_resnet101_2
from models.de_resnet import de_wide_resnet50_2
from models.recontrast import ReContrast, ReContrast
from dataset import ISICTrain, ISICTest
import torch.backends.cudnn as cudnn
import argparse
from utils import evaluation_noseg, visualize_noseg
from utils import global_cosine, global_cosine_hm
from cutpaste_transformation import *
import torch.nn as nn

from torch.nn import functional as F
from functools import partial

import warnings
import copy
import logging
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")
class BinaryClassifier2(nn.Module):

    def __init__(self):
        super(BinaryClassifier2, self).__init__()
        self.fc = nn.Linear(1000, 2)

    def forward(self, x):
        return self.fc(x)
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



def train(_class_, count=-1):
    print_fn(_class_)
    setup_seed(111)

    total_iters = 2000
    batch_size = 8
    image_size = 256
    crop_size = 256

    data_transform, gt_transform = get_data_transforms(image_size, crop_size)

    train_path = '../ISIC2018/'
    test_path = '../ISIC2018/'

    train_data = ISICTrain(transform=data_transform, count=count)
    test_data1 = ISICTest(transform=data_transform, test_id=1)
    test_data2 = ISICTest(transform=data_transform, test_id=2)

    visualize_random_samples_from_clean_dataset(train_data, 'train dataset isic')
    visualize_random_samples_from_clean_dataset(test_data1, f'test data isic1')
    visualize_random_samples_from_clean_dataset(test_data2, f'test data isic2')

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4,
                                                   drop_last=False)
    test_dataloader1 = torch.utils.data.DataLoader(test_data1, batch_size=1, shuffle=False, num_workers=1)
    test_dataloader2 = torch.utils.data.DataLoader(test_data2, batch_size=1, shuffle=False, num_workers=1)

    encoder, bn = wide_resnet50_2(pretrained=True)
    decoder = de_wide_resnet50_2(pretrained=False, output_conv=2)

    encoder = encoder.to(device)
    bn = bn.to(device)
    decoder = decoder.to(device)
    encoder_freeze = copy.deepcopy(encoder)

    model = ReContrast(encoder=encoder, encoder_freeze=encoder_freeze, bottleneck=bn, decoder=decoder)
    to_binary = BinaryClassifier2()
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(list(decoder.parameters()) + list(bn.parameters()),
                                  lr=2e-3, betas=(0.9, 0.999), weight_decay=1e-5)
    optimizer2 = torch.optim.AdamW(list(encoder.parameters()),
                                   lr=1e-5, betas=(0.9, 0.999), weight_decay=1e-5)
    print_fn('train image number:{}'.format(len(train_data)))
    print_fn('test image number:{}'.format(len(test_data1)))
    print_fn('test image number:{}'.format(len(test_data2)))

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

    model.to(device)
    to_binary.to(device)

    anomaly_transforms = transforms.Compose([
        transforms.ToPILImage(),
        CutPasteUnion(transform=transforms.Compose([transforms.ToTensor(), ])),
    ])

    for epoch in range(total_iters // len(train_dataloader) + 1):
        model.train(encoder_bn_train=True)
        loss_list = []
        for img, label, _ in train_dataloader:

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

            en1, de1, out = model(img, head=True)
            out = to_binary(out)
            head_loss = criterion(out, anomaly_one.to(torch.int64))

            en, de = model(img[:len(img) // 2])

            alpha_final = 1
            alpha = min(-3 + (alpha_final - -3) * epoch / (total_iters * 0.1), alpha_final)
            model_loss = global_cosine_hm(en[:3], de[:3], alpha=alpha, factor=0.) / 2 + \
                         global_cosine_hm(en[3:], de[3:], alpha=alpha, factor=0.) / 2

            loss = model_loss + head_loss

            optimizer.zero_grad()
            optimizer2.zero_grad()
            loss.backward()

            optimizer.step()
            optimizer2.step()
            loss_list.append(loss.item())



            if (it + 1) % 50 == 0:
                data_type = "main"
                auroc_px_list[str(data_type)], auroc_sp_list[str(data_type)], auroc_aupro_px_list[
                    str(data_type)], auroc_cls_auc_list[str(data_type)] = evaluation_noseg(
                    model, test_dataloader1, device, cls=to_binary)
                #  auroc, f1, acc
                # auroc_px_list[str(data_type)], auroc_sp_list[str(data_type)], auroc_aupro_px_list[str(data_type)] = evaluation_brain(model, test_dataloader1, device, max_ratio=max_ratio)
                print_fn('Shrink Factor:{}, Sample Auroc:{:.3f}, F1:{:.3f}, Acc:{:.3}, CLS Auroc:{:.3f}'.format(
                    data_type,
                    auroc_px_list[
                        str(data_type)],
                    auroc_sp_list[
                        str(data_type)],
                    auroc_aupro_px_list[
                        str(data_type)],
                    auroc_cls_auc_list[str(data_type)]))
                if auroc_sp_list[str(data_type)] >= auroc_sp_list_best[str(data_type)]:
                    auroc_px_list_best[str(data_type)], auroc_sp_list_best[str(data_type)], \
                        auroc_aupro_px_list_best[str(data_type)], auroc_cls_auc_list_best[str(data_type)] = \
                        auroc_px_list[str(data_type)], auroc_sp_list[
                            str(data_type)], auroc_aupro_px_list[str(data_type)], auroc_cls_auc_list[
                            str(data_type)]

                data_type = "shifted"
                auroc_px_list[str(data_type)], auroc_sp_list[str(data_type)], auroc_aupro_px_list[
                    str(data_type)], auroc_cls_auc_list[str(data_type)] = evaluation_noseg(model,
                                                                                           test_dataloader2,
                                                                                           device,
                                                                                           cls=to_binary)
                # auroc_px_list[str(data_type)], auroc_sp_list[str(data_type)], auroc_aupro_px_list[str(data_type)] = evaluation_brain(model, test_dataloader2, device, max_ratio=max_ratio)
                print_fn('Shrink Factor:{}, Sample Auroc:{:.3f}, F1:{:.3f}, Acc:{:.3}, CLS Auroc:{:.3f}'.format(
                    data_type,
                    auroc_px_list[
                        str(data_type)],
                    auroc_sp_list[
                        str(data_type)],
                    auroc_aupro_px_list[
                        str(data_type)],
                    auroc_cls_auc_list[str(data_type)]))
                if auroc_sp_list[str(data_type)] >= auroc_sp_list_best[str(data_type)]:
                    auroc_px_list_best[str(data_type)], auroc_sp_list_best[str(data_type)], \
                        auroc_aupro_px_list_best[str(data_type)], auroc_cls_auc_list_best[str(data_type)] = \
                        auroc_px_list[str(data_type)], auroc_sp_list[
                            str(data_type)], auroc_aupro_px_list[str(data_type)], auroc_cls_auc_list[
                            str(data_type)]
            it += 1
            if it == total_iters:
                break

    visualize_noseg(model, test_dataloader1, device, _class_=_class_, save_name=args.save_name)
    return auroc_px_list, auroc_sp_list, auroc_aupro_px_list, auroc_cls_auc_list, \
        auroc_px_list_best, auroc_sp_list_best, auroc_aupro_px_list_best, auroc_cls_auc_list_best


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--save_dir', type=str, default='./saved_results')
    parser.add_argument('--save_name', type=str,
                        default='recontrast_isic_256224_b32_it2k_lr2e31e5_wd1e5_hm1d01_s111')
    parser.add_argument('--gpu', default='0', type=str,
                        help='GPU id to use.')
    parser.add_argument('--data_count', type=int, default=5000)

    args = parser.parse_args()

    logger = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name))
    print_fn = logger.info

    device = 'cuda:' + args.gpu if torch.cuda.is_available() else 'cpu'
    print_fn(device)

    result_list = {"main": [], "shifted": []}
    result_list_best = {"main": [], "shifted": []}
    data_types = ["main", "shifted"]
    item = 'brain'

    print(f"+++++++++++++++++++++++++++++++++++++++{item}+++++++++++++++++++++++++++++++++++++++")
    auroc_px, auroc_sp, aupro_px, auroc_sp_cls, auroc_px_best, auroc_sp_best, aupro_px_best, auroc_sp_cls_best = train(
        item,
    )
    for type in data_types:
        result_list[str(type)].append(
            [item, auroc_px[str(type)], auroc_sp[str(type)], aupro_px[str(type)], auroc_sp_cls[str(type)]])
        result_list_best[str(type)].append(
            [item, auroc_px_best[str(type)], auroc_sp_best[str(type)], aupro_px_best[str(type)],
             auroc_sp_cls_best[str(type)]])

    for type in data_types:
        print(f'-------- shrink factor = {type} --------')
        mean_auroc_px = np.mean([result[1] for result in result_list[str(type)]])
        mean_auroc_sp = np.mean([result[2] for result in result_list[str(type)]])
        mean_aupro_px = np.mean([result[3] for result in result_list[str(type)]])
        mean_auc_sp_cls = np.mean([result[4] for result in result_list[str(type)]])

        print_fn(result_list[str(type)])
        print_fn('Sample Auroc:{:.4f}, F1:{:.4f}, Acc:{:.4}'.format(mean_auroc_px, mean_auroc_sp,
                                                                    mean_aupro_px, mean_auc_sp_cls))

        best_auroc_px = np.mean([result[1] for result in result_list_best[str(type)]])
        best_auroc_sp = np.mean([result[2] for result in result_list_best[str(type)]])
        best_aupro_px = np.mean([result[3] for result in result_list_best[str(type)]])
        best_auc_sp_cls = np.mean([result[4] for result in result_list_best[str(type)]])

        print_fn(result_list_best[str(type)])
        print_fn('Sample Auroc:{:.4f}, F1:{:.4f}, Acc:{:.4}'.format(best_auroc_px, best_auroc_sp,
                                                                    best_aupro_px, best_auc_sp_cls))