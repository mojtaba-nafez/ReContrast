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
from dataset import MVTecDataset
import torch.backends.cudnn as cudnn
import argparse
from utils import evaluation, visualize, global_cosine, global_cosine_hm
from torch.nn import functional as F
from functools import partial
from ptflops import get_model_complexity_info
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.data import ConcatDataset
from exposure_dataset import get_exposure_set


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

def populate_dataset_to_fixed_count(dataset, target_length):

    if not dataset:
        raise ValueError("Dataset cannot be empty.")

    original_length = len(dataset)
    if original_length >= target_length:
        return dataset

    # Calculate the number of times the dataset needs to be repeated
    repeat_count = (target_length // original_length) + 1

    # Extend the dataset by repeating it
    extended_dataset = dataset * repeat_count

    # If the extended dataset is longer than the target, slice it to the target length
    return extended_dataset[:target_length]

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


def train(_class_, shrink_factor=None, total_iters=2000,
          unode1_checkpoint=None, unode2_checkpoint=None, data_count=10000):
    anomaly_transforms = transforms.Compose([
        transforms.ToPILImage(),
        CutPasteUnion(transform=transforms.Compose([transforms.ToTensor(), ])),
    ])

    print_fn(_class_)
    setup_seed(111)

    total_iters = total_iters
    batch_size = 16
    image_size = 256
    crop_size = 256

    data_transform, gt_transform = get_data_transforms(image_size, crop_size)

    train_path = '/kaggle/input/mvtec-ad/' + _class_
    test_path = '/kaggle/input/mvtec-ad/' + _class_

    train_data = MVTecDataset(root=train_path, transform=data_transform, gt_transform=gt_transform, phase='train', count=data_count//2)
    test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test",
                             shrink_factor=shrink_factor)

    exposure_dataset = get_exposure_set(image_size=image_size, category=_class_, count=data_count // 2)

    combined_dataset = ConcatDataset([exposure_dataset, train_data])

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4,
                                                   drop_last=False)


    encoder_train_dataloader = torch.utils.data.DataLoader(combined_dataset, batch_size=batch_size, shuffle=True, num_workers=4,
                                                   drop_last=False)


    print('len train set: ', len(train_data))
    print('len exposure set: ', len(exposure_dataset))
    print('len combined set: ', len(combined_dataset))



    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=1)

    visualize_random_samples_from_clean_dataset(train_data, f"train_data_{_class_}", train_data=True)
    visualize_random_samples_from_clean_dataset(test_data, f"test_data_{_class_}", train_data=False)
    visualize_random_samples_from_clean_dataset(exposure_dataset, f"exposure_data_{_class_}", train_data=True)

    encoder, bn = resnet18(pretrained=True)
    decoder = de_resnet18(pretrained=False, output_conv=2)



    encoder_freeze = copy.deepcopy(encoder)
    # encoder_freeze = encoder_freeze.to(device)

    if unode1_checkpoint is not None:  # encoder
        print('Applying U-node as encoder 1...')
        encoder, bn = resnet18(pretrained=True, progress=True, unode_path=unode1_checkpoint, fc=False)



        # last_layer = encoder.fc
        # print("Type of last layer:", type(last_layer))
        # print("Output features of last layer:", last_layer.out_features)
        print(encoder)

    encoder = encoder.to(device)
    bn = bn.to(device)
    decoder = decoder.to(device)
    # encoder_freeze = copy.deepcopy(encoder)

    if unode2_checkpoint is not None:  # encoder_freeze
        print('Applying U-node as encoder 2...')
        encoder_freeze, _ = resnet18(pretrained=True, progress=True, unode_path=unode2_checkpoint, fc=False)

    encoder_freeze = encoder_freeze.to(device)

    model = ReContrast(encoder=encoder, encoder_freeze=encoder_freeze, bottleneck=bn, decoder=decoder)
    # for m in encoder.modules():
    #     if isinstance(m, torch.nn.BatchNorm2d):
    #         m.eps = 1e-8

    optimizer = torch.optim.AdamW(list(decoder.parameters()) + list(bn.parameters()),
                                  lr=2e-3, betas=(0.9, 0.999), weight_decay=1e-5)
    optimizer2 = torch.optim.AdamW(list(encoder.parameters()),
                                   lr=1e-5, betas=(0.9, 0.999), weight_decay=1e-5)

    criterion = nn.BCELoss()

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

    # IMPORTANT: total_iters should be >= 250 so that return values get computed


    for epoch in range(int(np.ceil(total_iters / len(train_dataloader)))):
        # encoder batchnorm in eval for these classes.
        print(f"Epoch {epoch + 1}/{int(np.ceil(total_iters / len(train_dataloader)))}")
        loss_list = []

        if epoch % 2 == 1:  # Even epochs
            # Train only the encoder's head, rest of the encoder is frozen
            for param in model.encoder.parameters():
                param.requires_grad = True  # Freeze the encoder except its head
            model.encoder.fc.requires_grad = True  # Unfreeze the head

            for img, label in encoder_train_dataloader:  # Different dataset for encoder training
                img = img.to(device)
                label = label.to(device)  # Assuming label is for binary classification
                output = model.encoder(img)
                # print("Output :", output)
                # print("Label :", label)
                # print('label type: ', type(label))
                print(type(output))
                loss = criterion(output, label.float())
                optimizer2.zero_grad()
                loss.backward()
                optimizer2.step()
        else:  # Odd epochs
            # Train the entire ReContrast model, encoder's head is frozen
            model.train(encoder_bn_train=_class_ not in ['toothbrush', 'leather', 'grid', 'tile', 'wood', 'screw'])
            for param in model.parameters():
                param.requires_grad = True  # Unfreeze all
            model.encoder.fc.requires_grad = False  # Freeze the head

            for img, label in train_dataloader:
                img = img.to(device)
                en, de = model(img)
                alpha_final = 1
                alpha = min(-3 + (alpha_final - -3) * epoch / (total_iters * 0.1), alpha_final)
                loss = global_cosine_hm(en[:3], de[:3], alpha=alpha, factor=0.) / 2 + \
                       global_cosine_hm(en[3:], de[3:], alpha=alpha, factor=0.) / 2

                optimizer.zero_grad()
                optimizer2.zero_grad()
                loss.backward()

                optimizer.step()
                optimizer2.step()
                loss_list.append(loss.item())


            if (it + 1) % (total_iters / 2) == 0:
                pad_size = [0.8, 0.85, 0.9, 0.95, 0.98, 1.0]

                for shrink_factor in pad_size:
                    test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform,
                                             phase="test", shrink_factor=shrink_factor)
                    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False,
                                                                  num_workers=1)

                    auroc_px_list[str(shrink_factor)], auroc_sp_list[str(shrink_factor)], auroc_aupro_px_list[
                        str(shrink_factor)] = evaluation(model, test_dataloader, device)
                    print_fn(
                        'Shrink Factor:{:.3f}, Pixel Auroc:{:.3f}, Sample Auroc:{:.3f}, Pixel Aupro:{:.3}'.format(
                            shrink_factor, auroc_px_list[str(shrink_factor)], auroc_sp_list[str(shrink_factor)],
                            auroc_aupro_px_list[str(shrink_factor)]))

                    if auroc_sp_list[str(shrink_factor)] >= auroc_sp_list_best[str(shrink_factor)]:
                        auroc_px_list_best[str(shrink_factor)], auroc_sp_list_best[str(shrink_factor)], \
                        auroc_aupro_px_list_best[str(shrink_factor)] = auroc_px_list[str(shrink_factor)], \
                                                                       auroc_sp_list[
                                                                           str(shrink_factor)], auroc_aupro_px_list[
                                                                           str(shrink_factor)]

            it += 1
            if it == total_iters:
                break
        print_fn('iter [{}/{}], loss:{:.4f}'.format(it, total_iters, np.mean(loss_list)))

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
    parser.add_argument('--shrink_factor', type=float, default=None)
    parser.add_argument('--total_iters', type=int, default=2000)
    parser.add_argument('--evaluation_epochs', type=int, default=250)
    parser.add_argument('--data_count', type=int, default=1000)

    # ADDING U NODE
    parser.add_argument('--encoder1_path', type=str, default='')
    parser.add_argument('--encoder2_path', type=str, default='')
    parser.add_argument('--classes', type=str, default='0,1,2,3,4,5,6,7,8,9,10,11,12,13,14', help='classes of mvtec')

    args = parser.parse_args()

    classes = args.classes.split(',')
    print('classes: ', classes)

    item_list = ['carpet', 'bottle', 'hazelnut', 'leather', 'cable', 'capsule', 'grid', 'pill',
                 'transistor', 'metal_nut', 'screw', 'toothbrush', 'zipper', 'tile', 'wood']
    print(item_list)
    # item_list = ['toothbrush']
    logger = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name))
    print_fn = logger.info

    device = 'cuda:' + args.gpu if torch.cuda.is_available() else 'cpu'
    print_fn(device)

    result_list = []
    result_list_best = []


    # decoder_path = args.use_new_decoder if args.use_new_decoder != '' else None
    #
    # print('decoder path:', decoder_path)

    result_list = {"0.8": [], "0.85": [], "0.9": [], "0.95": [], "0.98": [], "1.0": []}
    result_list_best = {"0.8": [], "0.85": [], "0.9": [], "0.95": [], "0.98": [], "1.0": []}
    pad_size = [0.8, 0.85, 0.9, 0.95, 0.98, 1.0]

    en1_path = args.encoder1_path if args.encoder1_path != '' else None
    en2_path = args.encoder2_path if args.encoder2_path != '' else None

    print('en1_path: ', en1_path)
    print('en2_path: ', en2_path)

    # num_classes = int(args.num_classes)

    for i in range(len(classes)):
        item = item_list[int(classes[i])]
        print(f"+++++++++++++++++++++++++++++++++++++++{item}+++++++++++++++++++++++++++++++++++++++")
        auroc_px, auroc_sp, aupro_px, auroc_px_best, auroc_sp_best, aupro_px_best = train(item,
                                                                                          shrink_factor=args.shrink_factor,
                                                                                          total_iters=args.total_iters,
                                                                                          unode1_checkpoint=en1_path,
                                                                                          unode2_checkpoint=en2_path,
                                                                                          data_count=args.data_count
                                                                                          )
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
