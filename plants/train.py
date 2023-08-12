import os

import albumentations as A
import cv2
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from LRFinder import plot_lr_finder, LRFinder
from Trainer import Trainer
from classification_models import get_mobilenetv3_model, get_resnet_model
from tools import logging_tools as log
from tools.data_tool import split_train_val
from tools.torch_tool import get_device, set_seed

batch_size = 2
lr = 1e-3
epochs = 20
image_size = 224
validation_step = 1
early_stop = -1

freeze_pretrained = False

train_images_folder = "dataset/images"
train_csv = "dataset/train.csv"


class PlantsDataset(Dataset):
    def __init__(self, folder_path, csv_file, transform):
        self.data = pd.read_csv(csv_file)
        self.folder_path = folder_path
        self.classes = self.get_unique_labels()
        self.targets = self.data['labels'].tolist()
        self.transform = transform
        self.class_distribution = self.get_class_distribution()

    def __len__(self):
        return len(self.data)

    def get_class_distribution(self):
        class_distribution = {label: 0 for label in self.classes}
        all_labels = self.data['labels'].str.split().tolist()
        for labels in all_labels:
            for label in labels:
                if label in self.classes:
                    class_distribution[label] += 1
        class_labels = list(class_distribution.keys())
        class_counts = list(class_distribution.values())

        return class_labels, class_counts

    def get_unique_labels(self):
        all_labels = self.data['labels'].str.split().tolist()
        unique_labels = set([label for sublist in all_labels for label in sublist])
        return sorted(unique_labels)

    def __getitem__(self, idx):
        file_name = self.data['image'].iloc[idx]
        label = self.data['labels'].iloc[idx]
        image_path = os.path.join(self.folder_path, file_name)
        # image = Image.open(image_path).convert('RGB')
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image=image)["image"]

        label_encoding = torch.zeros(len(self.classes), dtype=torch.float32)
        labels = label.split()
        for label in labels:
            if label in self.classes:
                label_idx = self.classes.index(label)
                label_encoding[label_idx] = 1.0

        return image, label_encoding


def main():
    # Init
    device = get_device(True)
    set_seed()

    writer = SummaryWriter()

    train_transform = A.Compose(
        [
            A.Resize(
                height=image_size,
                width=image_size,
            ),
            A.ShiftScaleRotate(always_apply=False,
                               shift_limit=0.0625,
                               scale_limit=0.1,
                               rotate_limit=45, ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(always_apply=False, p=0.5),
            A.GridDistortion(always_apply=False, p=0.5, num_steps=5, distort_limit=(-0.15, 0.15)),
            A.GaussNoise(always_apply=False, p=0.5, var_limit=(10.0, 50.0)),
            A.RandomResizedCrop(image_size, image_size, ratio=(1, 1), scale=(0.8, 1.0)),
            A.GaussianBlur(always_apply=False, p=0.5, blur_limit=(3, 3)),

            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
            ToTensorV2(),

        ]
    )

    eval_transform = A.Compose(
        [
            A.Resize(
                height=image_size,
                width=image_size,
            ),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
            ToTensorV2(),
        ]
    )

    train_dataset = PlantsDataset(train_images_folder, train_csv, train_transform)
    train_dataset, val_dataset = split_train_val(train_dataset, transform=eval_transform)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    # Prepare datasets
    classes, train_info = train_dataset.dataset.classes, train_dataset.dataset.class_distribution

    log.log_train_labels_distribution_image(train_info[0], train_info[1], writer)
    log.log_images_examples(classes, train_dataloader, writer)

    # Define model

    # model, model_name = get_efficientnet_model(device, freeze_pretrained=freeze_pretrained, classes=len(classes))
    # model, model_name = get_vit_model(device, freeze_pretrained=freeze_pretrained, classes=len(classes))
    # model, model_name = get_resnet_model(device, freeze_pretrained=freeze_pretrained, classes=len(classes))
    model, model_name = get_mobilenetv3_model(device, freeze_pretrained=freeze_pretrained, classes=len(classes))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                    max_lr=lr,
                                                    steps_per_epoch=len(train_dataloader),
                                                    epochs=epochs)

    # find_lr(criterion, device, model, optimizer, train_dataloader, writer)

    # Train Model
    trainer = Trainer(model, optimizer, criterion, scheduler, writer, early_stop=early_stop, multilabel=True)
    val_loss, val_acc, train_loss, train_acc = trainer.train(train_dataloader, val_dataloader, epochs, validation_step)

    # Log training results and hyperarams
    hyper_params_dict = {
        "lr": lr,
        "batch_size": batch_size,
        "image_size": image_size,
        "freeze_pretrained": freeze_pretrained,
        "model": model_name
    }
    metrics_dict = {
        "~loss/val": val_loss
    }
    log.log_hyperparams(hyper_params_dict, metrics_dict, writer)

    # Evaluate and log results
    images, labels, probs = get_predictions(model, val_dataloader, device)

    log.log_confusion_matrix(classes, labels, probs, writer, multilabel=True)

    incorrect_examples = get_incorrect_examples(images, labels, probs)
    log.log_incorrect_examples(classes, incorrect_examples, writer)

    # Save model
    torch.save(model.state_dict(), 'model.pt')


def find_lr(criterion, device, model, optimizer, train_dataloader, writer):
    END_LR = 10
    NUM_ITER = 300
    lr_finder = LRFinder(model, optimizer, criterion, device)
    lrs, losses = lr_finder.range_test(train_dataloader, END_LR, NUM_ITER)
    plot_lr_finder(lrs, losses, skip_start=0, skip_end=0)
    writer.add_figure('lr_finder', plt.gcf(), 0)


def get_incorrect_examples(images, labels, probs):
    y_pred = torch.where(probs > 0.5, 1, 0)
    corrects = torch.eq(labels, y_pred)
    incorrect_examples = []
    for image, label, prob, correct in zip(images, labels, probs, corrects):
        if any(correct == False):
            incorrect_examples.append((image, label, prob))
    incorrect_examples.sort(reverse=True, key=lambda x: torch.max(x[2], dim=0).values)
    return incorrect_examples


def get_predictions(model, iterator, device):
    model.eval()

    images = []
    labels = []
    probs = []

    with torch.no_grad():
        for (x, y) in iterator:
            x = x.to(device)

            y_pred = model(x)

            images.append(x.cpu())
            labels.append(y.cpu())
            probs.append(y_pred.cpu())

    images = torch.cat(images, dim=0)
    labels = torch.cat(labels, dim=0)
    probs = torch.cat(probs, dim=0)

    return images, labels, probs


if __name__ == '__main__':
    main()
