from torchvision import transforms
from PIL import Image
import os
import torch
import glob
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST, ImageFolder
import numpy as np
import torch.multiprocessing
from torch.utils.data import Dataset
from PIL import ImageFilter, Image, ImageOps
from torchvision.datasets.folder import default_loader
import os
import random

torch.multiprocessing.set_sharing_strategy('file_system')


def get_data_transforms(size, isize, mean_train=None, std_train=None):
    mean_train = [0.485, 0.456, 0.406] if mean_train is None else mean_train
    std_train = [0.229, 0.224, 0.225] if std_train is None else std_train
    data_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.CenterCrop(isize),])
        # transforms.Normalize(mean=mean_train,
        #                      std=std_train)])
    gt_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.CenterCrop(isize),
        transforms.ToTensor()])
    return data_transforms, gt_transforms


def get_strong_transforms(size, isize, mean_train=None, std_train=None):
    mean_train = [0.485, 0.456, 0.406] if mean_train is None else mean_train
    std_train = [0.229, 0.224, 0.225] if std_train is None else std_train
    data_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        # transforms.RandomRotation(15, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.CenterCrop(isize),
        transforms.Normalize(mean=mean_train,
                             std=std_train)])
    return data_transforms

class IMAGENET30_TEST_DATASET(Dataset):
    def __init__(self, root_dir="/kaggle/input/imagenet30-dataset/one_class_test/one_class_test/", transform=None):
        """
        Args:
            root_dir (string): Directory with all the classes.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.img_path_list = []
        self.targets = []

        # Map each class to an index
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(sorted(os.listdir(root_dir)))}
        # print(f"self.class_to_idx in ImageNet30_Test_Dataset:\n{self.class_to_idx}")

        # Walk through the directory and collect information about the images and their labels
        for i, class_name in enumerate(os.listdir(root_dir)):
            class_path = os.path.join(root_dir, class_name)
            for instance_folder in os.listdir(class_path):
                instance_path = os.path.join(class_path, instance_folder)
                if instance_path != "/kaggle/input/imagenet30-dataset/one_class_test/one_class_test/airliner/._1.JPEG":
                    for img_name in os.listdir(instance_path):
                        if img_name.endswith('.JPEG'):
                            img_path = os.path.join(instance_path, img_name)
                            # image = Image.open(img_path).convert('RGB')
                            self.img_path_list.append(img_path)
                            self.targets.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, idx):
        img_path = self.img_path_list[idx]
        image = default_loader(img_path)
        label = self.targets[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

def center_paste(large_img, small_img):
    # Calculate the center position
    large_width, large_height = large_img.size
    small_width, small_height = small_img.size

    # Calculate the top-left position
    left = (large_width - small_width) // 2
    top = (large_height - small_height) // 2

    # Create a copy of the large image to keep the original unchanged
    result_img = large_img.copy()

    # Paste the small image onto the large one at the calculated position
    result_img.paste(small_img, (left, top))

    return result_img


class MVTecDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, phase, shrink_factor=None):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'ground_truth')
        self.transform = transform
        self.gt_transform = gt_transform
        # load dataset
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()  # self.labels => good : 0, anomaly : 1
        self.imagenet30_testset = IMAGENET30_TEST_DATASET()
        self.shrink_factor = shrink_factor
        print(f"self.shrink_factor: {self.shrink_factor}")

    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.JPG")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.JPG")
                gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*.png")
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        if self.shrink_factor:
            pad_img, _ = self.imagenet30_testset[int(random.random() * len(self.imagenet30_testset))]
            pad_img = pad_img.resize(img.size)

            img = img.resize((int(img.size[0] * self.shrink_factor), int(img.size[1] * self.shrink_factor)))

            img = center_paste(pad_img, img)

        img = self.transform(img)
        if gt == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, img_path


class MVTecSegDataset(torch.utils.data.Dataset):
    def __init__(self, root, mask_path, transform, gt_transform, phase):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'ground_truth')
            self.mask_path = os.path.join(mask_path, 'test')

        self.transform = transform
        self.gt_transform = gt_transform
        # load dataset
        self.img_paths, self.gt_paths, self.mask_paths, self.labels, self.types = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        mask_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                mask_paths = glob.glob(os.path.join(self.mask_path, defect_type) + "/*.png")

                img_paths.sort()
                mask_paths.sort()

                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                mask_tot_paths.extend(mask_paths)

                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*.png")
                mask_paths = glob.glob(os.path.join(self.mask_path, defect_type) + "/*.png")

                img_paths.sort()
                gt_paths.sort()
                mask_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                mask_tot_paths.extend(mask_paths)

                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, mask_tot_paths, tot_labels, tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        if gt == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        mask = Image.open(self.mask_paths[idx])
        mask = self.gt_transform(mask)

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, mask, img_path




def center_paste_2(large_img, small_img, shrink_factor):
    width , height = small_img.size
    large_img = large_img.resize((width, height))

    new_width = int(width * shrink_factor)
    new_height = int(height * shrink_factor)

    small_img = small_img.resize((new_width, new_height))
    small_width, small_height = small_img.size

    left = (width - small_width) // 2
    top = (height - small_height) // 2

    result_img = large_img.copy()
    result_img.paste(small_img, (left, top))
    return result_img


class IMAGENET30_TEST_DATASET_2(Dataset):
    def __init__(self, root_dir="/kaggle/input/imagenet30-dataset/one_class_test/one_class_test/", transform=None):
        """
        Args:
            root_dir (string): Directory with all the classes.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.img_path_list = []
        self.targets = []

        # Map each class to an index
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(sorted(os.listdir(root_dir)))}
        # print(f"self.class_to_idx in ImageNet30_Test_Dataset:\n{self.class_to_idx}")

        # Walk through the directory and collect information about the images and their labels
        for i, class_name in enumerate(os.listdir(root_dir)):
            class_path = os.path.join(root_dir, class_name)
            for instance_folder in os.listdir(class_path):
                instance_path = os.path.join(class_path, instance_folder)
                if instance_path != "/kaggle/input/imagenet30-dataset/one_class_test/one_class_test/airliner/._1.JPEG":
                    for img_name in os.listdir(instance_path):
                        if img_name.endswith('.JPEG'):
                            img_path = os.path.join(instance_path, img_name)
                            # image = Image.open(img_path).convert('RGB')
                            self.img_path_list.append(img_path)
                            self.targets.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, idx):
        img_path = self.img_path_list[idx]
        label = self.targets[idx]
        img = Image.open(img_path)
        return img

class Train_MVTecDataset(Dataset):
    def __init__(self, root, category, transform=None, shrink_factor=1, train=True, count=-1):
        self.transform = transform
        self.image_files = []
        self.shrink_factor = shrink_factor
        print("category MVTecDataset:", category)
        if train:
            good_images = glob.glob(os.path.join(root, category, "train", "good", "*.png"))
            if count != -1:
                if count < len(good_images):
                    good_images = good_images[:count]
                else:
                    t = len(good_images)
                    for i in range(count - t):
                        good_images.append(random.choice(good_images[:t]))
            good_images.sort(key=lambda y: y.lower())
            augmented_images = []
            for img in good_images:
                augmented_images.append((img, True))  # Mark this image for augmentation
            good_images = [(img, False) for img in good_images]  # Original images not for augmentation
            self.image_files = good_images + augmented_images
        else:
            image_files = glob(os.path.join(root, category, "test", "*", "*.png"))
            normal_image_files = glob(os.path.join(root, category, "test", "good", "*.png"))
            anomaly_image_files = list(set(image_files) - set(normal_image_files))
            self.image_files = image_files
            self.image_files = [(img, False) for img in image_files]
        self.train = train
        self.imagenet_30 = IMAGENET30_TEST_DATASET_2()


    def __getitem__(self, index):
        image_file , augment= self.image_files[index]
        image = Image.open(image_file)
        image = image.convert('RGB')
        
        # if self.train and augment:
        if augment:
            random_index = int(random.random() * len(self.imagenet_30))
            imagenet30_img = self.imagenet_30[random_index]
            imagenet30_img = imagenet30_img.convert('RGB')
            factors = [0.98, 0.95, 0.93, 0.91, 0.88, 0.82, 0.90, 0.97, 0.85, 0.80]
            image  = center_paste_2(imagenet30_img, image, random.choice(factors))
        '''
        if not self.train:
            imagenet_30 = IMAGENET30_TEST_DATASET()
            random_index = int(random.random() * len(imagenet_30))
            imagenet30_img = imagenet_30[random_index]
            imagenet30_img = imagenet30_img.convert('RGB')
            image = center_paste(imagenet30_img, image, self.shrink_factor)
        '''
        if self.transform is not None:
            image = self.transform(image)
        if os.path.dirname(image_file).endswith("good"):
            target = 0
        else:
            target = 1
        return image, target

    def __len__(self):
        return len(self.image_files)

class LOCODataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, phase):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'ground_truth')
        self.transform = transform
        self.gt_transform = gt_transform
        # load dataset
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*/000.png")
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        size = (img.size[1], img.size[0])
        img = self.transform(img)
        type = self.types[idx]
        if gt == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, img_path, type, size


class MedicalDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, phase):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
        self.transform = transform
        self.phase = phase
        # load dataset
        self.img_paths, self.labels = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):

        img_tot_paths = []
        tot_labels = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'NORMAL':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*")
                img_tot_paths.extend(img_paths)
                tot_labels.extend([0] * len(img_paths))
            else:
                if self.phase == 'train':
                    continue
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*")
                img_tot_paths.extend(img_paths)
                tot_labels.extend([1] * len(img_paths))

        return img_tot_paths, tot_labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, label = self.img_paths[idx], self.labels[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        return img, label, img_path
