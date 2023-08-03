import copy
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms


def get_dataloaders_from_folders(train_folder, test_folder=None, val_folder=None, image_size=256, batch_size=16,
                                 train_transform=None, eval_transform=None, fix_imbalanced=False, ratio=0.8):
    if train_transform is None:
        train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(size=(image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(degrees=10)
            ]
        )
    if eval_transform is None:
        eval_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(size=(image_size, image_size))
            ]
        )
    train_dataset = ImageFolder(root=train_folder, transform=train_transform)

    val_dataset = None
    if val_folder is not None:
        val_dataset = ImageFolder(root=val_folder, transform=eval_transform)

    if val_dataset is None:
        train_dataset, val_dataset = split_train_val(train_dataset, ratio, eval_transform)

    test_dataset = None
    if test_folder is not None:
        test_dataset = ImageFolder(root=test_folder, transform=eval_transform)

    train_dataloader, val_dataloader, test_dataloader, classes, train_info = get_dataloaders_from_dataset(train_dataset,
                                                                                                          test_dataset,
                                                                                                          val_dataset,
                                                                                                          batch_size=batch_size,
                                                                                                          fix_imbalanced=fix_imbalanced)

    return train_dataloader, val_dataloader, test_dataloader, classes, train_info


def split_train_val(train_dataset, ratio=0.8, transform=None):
    train_dataset, val_dataset = __split_dataset(train_dataset, ratio)
    val_dataset = copy.deepcopy(val_dataset)
    if transform is not None:
        val_dataset.transform = transform
    return train_dataset, val_dataset


def get_dataloaders_from_dataset(train_dataset, test_dataset, val_dataset=None, batch_size=16,
                                 fix_imbalanced=False):
    train_sampler = None
    if fix_imbalanced:
        train_sampler = get_sampler_for_imbalanced(train_dataset)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, sampler=train_sampler,
                                  shuffle=train_sampler is None)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    classes, train_info = get_dataset_info(train_dataset)
    return train_dataloader, val_dataloader, test_dataloader, classes, train_info


def get_dataset_info(train_dataset):
    classes = train_dataset.classes
    train_info = np.unique(train_dataset.targets, return_counts=True)
    return classes, train_info


def get_sampler_for_imbalanced(train_dataset):
    c = Counter(j for i, j, k in train_dataset)
    weights = [1 / c.get(j) for i, j, k in train_dataset]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights))
    return sampler


def __split_dataset(dataset, ratio):
    train_size = int(ratio * len(dataset))
    val_size = len(dataset) - train_size
    dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    return dataset, val_dataset


def k_fold(k, dataset, num_val_samples):
    for i in range(k):
        valid_idx = np.arange(len(dataset))[i * num_val_samples:(i + 1) * num_val_samples]
        train_idx = np.concatenate(
            [np.arange(len(dataset))[:i * num_val_samples], np.arange(len(dataset))[(i + 1) * num_val_samples:]],
            axis=0)
        train_dataset = Subset(dataset, train_idx)
        valid_dataset = Subset(dataset, valid_idx)

        yield train_dataset, valid_dataset

