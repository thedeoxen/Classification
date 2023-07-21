import copy

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms


def main():
    pass


if __name__ == '__main__':
    main()


def get_dataloaders(train_folder, test_folder, image_size, batch_size):
    train_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(size=(image_size, image_size)),
            # transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=10)
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(size=(image_size, image_size))
        ]
    )
    train_dataset = ImageFolder(root=train_folder, transform=train_transform)

    train_info = np.unique(train_dataset.targets, return_counts=True)

    test_dataset = ImageFolder(root=test_folder, transform=eval_transform)
    classes = train_dataset.classes
    ratio = 0.8
    train_dataset, val_dataset = split_dataset(train_dataset, ratio)
    # val_dataset.dataset.transform = eval_transform

    val_dataset = copy.deepcopy(val_dataset)
    val_dataset.dataset.transform = eval_transform

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader, classes, train_info


def split_dataset(dataset, ratio):
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