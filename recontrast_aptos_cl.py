
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
from dataset import AptosTest, AptosTrain
import torch.backends.cudnn as cudnn
import argparse
from utils import evaluation, visualize, global_cosine, global_cosine_hm, NT_xent, contrastive_loss, evaluation_noseg_brain
from torch.nn import functional as F
from functools import partial
from ptflops import get_model_complexity_info
from torchvision import transforms
import matplotlib.pyplot as plt

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

def train(_class_, shrink_factor=None, total_iters=2000, evaluation_epochs=250, training_using_pad=False, max_ratio=0, augmented_view=False, batch_size=16, model='wide_res50'):
    print_fn(_class_)
    setup_seed(111)

    total_iters = total_iters
    image_size = 256
    crop_size = 256
    
    if augmented_view:
        train_data_transforms = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ColorJitter(0.8, 0.8, 0.8, 0.2),  # Color jitter
            transforms.RandomGrayscale(p=0.2),    # Random grayscale
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

    train_path = '../APTOS/'
    test_path = '../APTOS/'

    train_data = AptosTrain(transform=train_data_transforms)
    test_data1 = AptosTest(transform=data_transform, test_id=1)
    test_data2 = AptosTest(transform=data_transform, test_id=2)


    visualize_random_samples_from_clean_dataset(train_data, 'train dataset aptos')
    visualize_random_samples_from_clean_dataset(test_data1, f'test data aptos1')
    visualize_random_samples_from_clean_dataset(test_data2, f'test data aptos2')



    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4,
                                                   drop_last=False)
    test_dataloader1 = torch.utils.data.DataLoader(test_data1, batch_size=1, shuffle=False, num_workers=1)
    test_dataloader2 = torch.utils.data.DataLoader(test_data2, batch_size=1, shuffle=False, num_workers=1)


    # visualize_random_samples_from_clean_dataset(train_data, f"train_data_{_class_}", train_data=True)
    # visualize_random_samples_from_clean_dataset(test_data, f"test_data_{_class_}", train_data=False)

    if model=='wide_res50':
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
    model = ReContrast(encoder=encoder, encoder_freeze=encoder_freeze, bottleneck=bn, decoder=decoder, image_size=image_size, crop_size=crop_size, device=device)
    # for m in encoder.modules():
    #     if isinstance(m, torch.nn.BatchNorm2d):
    #         m.eps = 1e-8

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

    auroc_px_list = {"main":0, "shifted":0}
    auroc_px_list_best = {"main":0, "shifted":0}

    auroc_sp_list = {"main":0, "shifted":0}
    auroc_sp_list_best = {"main":0, "shifted":0}

    auroc_aupro_px_list = {"main":0, "shifted":0}
    auroc_aupro_px_list_best = {"main":0, "shifted":0}
    
    anomaly_transforms = transforms.Compose([
        transforms.ToPILImage(),
        CutPasteUnion(transform = transforms.Compose([transforms.ToTensor(),])),
    ])
    for epoch in range(int(np.ceil(total_iters / len(train_dataloader)))):
        # encoder batchnorm in eval for these classes.
        model.train(encoder_bn_train=True)

        loss_list = []
        for img, label in train_dataloader:
            # img : [16, 3, 256, 256]
            # img = torch.cat([img, img.clone()])

            img = img.to(device)
            anomaly_data = np.ones(len(img))
            anomaly_data[int(len(anomaly_data)/2):] = -1
            for i in range(len(anomaly_data)):
                if anomaly_data[i] == -1:
                    img[i] = anomaly_transforms(img[i])
            anomaly_data = torch.tensor(anomaly_data).to(device)
            # en : [[16,256,64,64], [16,512,32,32], [16,1024,16,16], [16,256,64,64], [16,512,32,32], [16,1024,16,16]]
            # de : [[16,256,64,64], [16,512,32,32], [16,1024,16,16], [16,256,64,64], [16,512,32,32], [16,1024,16,16]]
            en, de = model(img)
            alpha_final = 1
            alpha = min(-3 + (alpha_final - -3) * it / (total_iters * 0.1), alpha_final)
            
            loss1 = global_cosine_hm(en[:3], de[:3], anomaly_data=anomaly_data, alpha=alpha, factor=0.) / 2 + \
                   global_cosine_hm(en[3:], de[3:], anomaly_data=anomaly_data, alpha=alpha, factor=0.) / 2
            loss2 = (contrastive_loss(en[:3], de[:3], anomaly_data=anomaly_data, layer_num=2) / 2) + \
                        (contrastive_loss(en[3:], de[3:], anomaly_data=anomaly_data, layer_num=2) / 2)
            loss = loss1  + loss2
            '''
            loss2 = contrastive_loss(en[:3], de[:3], anomaly_data=anomaly_data, layer_num=0) + contrastive_loss(en[:3], de[:3], anomaly_data=anomaly_data, layer_num=1) + contrastive_loss(en[:3], de[:3], anomaly_data=anomaly_data, layer_num=2)
            loss3 = contrastive_loss(en[3:], de[3:], anomaly_data=anomaly_data, layer_num=0) +  contrastive_loss(en[3:], de[3:], anomaly_data=anomaly_data, layer_num=1) +  contrastive_loss(en[3:], de[3:], anomaly_data=anomaly_data, layer_num=2)
            loss = loss1 + (loss2/6) + (loss3/6)
            '''
            # loss = global_cosine(en[:3], de[:3], stop_grad=False) / 2 + \
            #        global_cosine(en[3:], de[3:], stop_grad=False) / 2

            optimizer.zero_grad()
            optimizer2.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            optimizer2.step()
            loss_list.append(loss.item())
            if (it + 1) % evaluation_epochs == 0:

                shrink_factor = "main" 
                # auroc, f1, acc = evaluation_noseg(model, test_dataloader1, device)
                auroc_px_list[str(shrink_factor)], auroc_sp_list[str(shrink_factor)], auroc_aupro_px_list[str(shrink_factor)] = evaluation_noseg_brain(model, test_dataloader1, device)
                print_fn('Shrink Factor:{}, Sample Auroc:{:.3f}, F1:{:.3f}, acc:{:.3}'.format(shrink_factor, auroc_px_list[str(shrink_factor)], auroc_sp_list[str(shrink_factor)], auroc_aupro_px_list[str(shrink_factor)]))
                if auroc_sp_list[str(shrink_factor)] >= auroc_sp_list_best[str(shrink_factor)]:
                    auroc_px_list_best[str(shrink_factor)], auroc_sp_list_best[str(shrink_factor)], auroc_aupro_px_list_best[str(shrink_factor)] = auroc_px_list[str(shrink_factor)], auroc_sp_list[str(shrink_factor)], auroc_aupro_px_list[str(shrink_factor)]
                
                shrink_factor = "shifted"
                auroc_px_list[str(shrink_factor)], auroc_sp_list[str(shrink_factor)], auroc_aupro_px_list[str(shrink_factor)] = evaluation_noseg_brain(model, test_dataloader2, device)
                print_fn('Shrink Factor:{}, Sample Auroc:{:.3f}, F1:{:.3f}, acc:{:.3}'.format(shrink_factor, auroc_px_list[str(shrink_factor)], auroc_sp_list[str(shrink_factor)], auroc_aupro_px_list[str(shrink_factor)]))
                if auroc_sp_list[str(shrink_factor)] >= auroc_sp_list_best[str(shrink_factor)]:
                    auroc_px_list_best[str(shrink_factor)], auroc_sp_list_best[str(shrink_factor)], auroc_aupro_px_list_best[str(shrink_factor)] = auroc_px_list[str(shrink_factor)], auroc_sp_list[str(shrink_factor)], auroc_aupro_px_list[str(shrink_factor)]
                
                model.train(encoder_bn_train=True)
                
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
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--evaluation_epochs', type=int, default=250)
    parser.add_argument('--training_shrink_factor', action='store_true')
    parser.add_argument('--training_using_pad', action='store_true')
    parser.add_argument('--max_ratio', type=float, default=0)
    parser.add_argument('--augmented_view', action='store_true')
    parser.add_argument('--model', type=str, default='wide_res50')
    parser.add_argument('--item_list', type=int, default=0)

    args = parser.parse_args()

    if args.training_shrink_factor:
        args.training_using_pad = True
    
    
    
    logger = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name))
    print_fn = logger.info

    device = 'cuda:' + args.gpu if torch.cuda.is_available() else 'cpu'
    print_fn(device)

    result_list = {"main":[], "shifted":[]}
    result_list_best = {"main":[], "shifted":[]}
    pad_size = ["main", "shifted"]
    item = 'brain'
    print(f"+++++++++++++++++++++++++++++++++++++++{item}+++++++++++++++++++++++++++++++++++++++")
    auroc_px, auroc_sp, aupro_px, auroc_px_best, auroc_sp_best, aupro_px_best = train(item, shrink_factor=args.shrink_factor, total_iters=args.total_iters, evaluation_epochs=args.evaluation_epochs, training_using_pad=args.training_using_pad, max_ratio=args.max_ratio, augmented_view=args.augmented_view, batch_size=args.batch_size, model=args.model)
    for pad in pad_size:
        result_list[str(pad)].append([item, auroc_px[str(pad)], auroc_sp[str(pad)], aupro_px[str(pad)]])
        result_list_best[str(pad)].append([item, auroc_px_best[str(pad)], auroc_sp_best[str(pad)], aupro_px_best[str(pad)]])

    for pad in pad_size:
        print(f'-------- shrink factor = {pad} --------')
        mean_auroc_px = np.mean([result[1] for result in result_list[str(pad)]])
        mean_auroc_sp = np.mean([result[2] for result in result_list[str(pad)]])
        mean_aupro_px = np.mean([result[3] for result in result_list[str(pad)]])
        print_fn(result_list[str(pad)])
        print_fn('Sample Auroc:{:.4f}, F1:{:.4f}, Acc:{:.4}'.format(mean_auroc_px, mean_auroc_sp,
                                                                                        mean_aupro_px))

        best_auroc_px = np.mean([result[1] for result in result_list_best[str(pad)]])
        best_auroc_sp = np.mean([result[2] for result in result_list_best[str(pad)]])
        best_aupro_px = np.mean([result[3] for result in result_list_best[str(pad)]])
        print_fn(result_list_best[str(pad)])
        print_fn('Sample Auroc:{:.4f}, F1:{:.4f}, Acc:{:.4}'.format(best_auroc_px, best_auroc_sp,
                                                                                        best_aupro_px))


