import copy

import torch
import torch.nn.functional as F
import torchinfo
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import ImageFolder
from torchvision.models import efficientnet_b0
from torchvision.transforms import transforms
from tqdm import tqdm

from image_helper import plot_images, plot_confusion_matrix, plot_most_incorrect
from torch_tool import get_device, get_optimizer_lr, set_seed

batch_size = 16

lr = 0.0001
epochs = 3
image_size = 256


def main():
    device = get_device()
    set_seed()

    train_folder = "dataset/train"
    test_folder = "dataset/test"
    train_dataloader, val_dataloader, test_dataloader, classes = get_dataloaders(train_folder, test_folder)

    model = get_model(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                    max_lr=lr * 1e1,
                                                    steps_per_epoch=len(train_dataloader),
                                                    epochs=epochs)

    writer = SummaryWriter()

    for images, labels in train_dataloader:
        break
    plot_images(images, labels, classes)
    writer.add_figure('train_images', plt.gcf(), 0)

    for epoch in tqdm(range(epochs)):
        model.train()
        train_loss = 0
        train_acc = 0
        for i, train_data in enumerate(train_dataloader):
            x, y = train_data

            y_pred = model(x.to(device))

            loss = criterion(y_pred, y.to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss
            acc_iter = accuracy_score(y, torch.argmax(y_pred.cpu(), dim=1))
            train_acc += acc_iter

        train_loss = train_loss / (i + 1)
        train_acc = train_acc / (i + 1)

        if epoch % 5 == 0:
            print(f"Epoch:{epoch}")
            print(f"Training Loss:{train_loss:.5f}\tTraining Acc:{train_acc * 100:.5f}%")
            print("------------------------------------------------------------------")

            val_loss = 0
            val_acc = 0
            model.eval()
            for i, val_data in enumerate(val_dataloader):
                x, y = val_data
                y_pred = model(x.to(device))

                loss = criterion(torch.softmax(y_pred, dim=1), y.to(device))
                val_loss += loss
                prediction_indexes = torch.softmax(y_pred.cpu(), dim=1)
                acc_iter = accuracy_score(y, torch.argmax(prediction_indexes, dim=1))
                val_acc += acc_iter

            val_loss = val_loss / (i + 1)
            val_acc = val_acc / (i + 1)

            print(f"Epoch:{epoch}")
            print(f"Val Loss:{val_loss:.5f}\tVal Acc:{val_acc * 100:.5f}%")
            print("------------------------------------------------------------------")

            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("Acc/val", val_acc, epoch)
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Acc/train", train_acc, epoch)
            writer.add_scalar("LR", get_optimizer_lr(optimizer), epoch)

    writer.add_hparams(
        {
            "lr": lr,
            "batch_size": batch_size,
            "image_size": image_size
        },
        {
            "loss/val": val_loss,
            "acc/val": val_acc,
            "loss/train": train_loss,
            "acc/train": train_acc
        }

    )

    images, labels, probs = get_predictions(model, val_dataloader, device)
    pred_labels = torch.argmax(probs, 1)

    plot_confusion_matrix(labels, pred_labels, classes)
    writer.add_figure('val_confusion_matrix', plt.gcf(), 0)

    incorrect_examples = get_incorrect_examples(images, labels, probs, pred_labels)
    plot_most_incorrect(incorrect_examples, classes, 10)
    writer.add_figure('val_most_incorrect', plt.gcf(), 0)


def get_incorrect_examples(images, labels, probs, pred_labels):
    corrects = torch.eq(labels, pred_labels)
    incorrect_examples = []
    for image, label, prob, correct in zip(images, labels, probs, corrects):
        if not correct:
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

            y_prob = F.softmax(y_pred, dim=-1)

            images.append(x.cpu())
            labels.append(y.cpu())
            probs.append(y_prob.cpu())

    images = torch.cat(images, dim=0)
    labels = torch.cat(labels, dim=0)
    probs = torch.cat(probs, dim=0)

    return images, labels, probs


def get_model(device):
    model = efficientnet_b0(weights="IMAGENET1K_V1")

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=1280, out_features=3, bias=True),
        nn.Softmax(dim=-1)
    )
    model.to(device)
    torchinfo.summary(model)
    return model


def get_dataloaders(train_folder, test_folder):
    train_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(size=(image_size, image_size)),
            # transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            # transforms.RandomRotation(degrees=30)
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(size=(image_size, image_size))
        ]
    )
    train_dataset = ImageFolder(root=train_folder, transform=train_transform)
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

    return train_dataloader, val_dataloader, test_dataloader, classes


def split_dataset(dataset, ratio):
    train_size = int(ratio * len(dataset))
    val_size = len(dataset) - train_size
    dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    return dataset, val_dataset


if __name__ == '__main__':
    main()
