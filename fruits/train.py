import torch
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchinfo import torchinfo
from torchvision.models import resnet50
from tqdm import tqdm

from LRFinder import plot_lr_finder, LRFinder
from fruits.FruitImageDataset import FruitImageDataset
from tools import logging_tools as log
from tools.data_tool import get_dataloaders_from_dataset
from tools.logging_tools import log_metrics
from tools.torch_tool import get_device, set_seed, get_optimizer_lr

batch_size = 16
lr = 1e-3
epochs = 100
image_size = 256
validation_step = 1
early_stop = 10

freeze_pretrained = True

train_folder = "dataset/Train"
test_folder = "dataset/Test"

torch.autograd.set_detect_anomaly(True)


class Model(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = resnet50(weights="IMAGENET1K_V1")
        params = list(self.model.parameters())
        for index, param in enumerate(params):
            param.requires_grad = False
        self.model.fc = nn.Sequential(nn.Linear(self.model.fc.in_features, 100))
        self.model.out_1 = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(100, 9),
            nn.Softmax(dim=-1)
        )

        self.model.out_2 = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(100, 2),
            nn.Softmax(dim=-1)
        )
        self.model.to(get_device())
        torchinfo.summary(self.model)

    def forward(self, x):
        x = self.model(x)
        y1, y2 = self.model.out_1(x), self.model.out_2(x)
        return y1, y2


def main():
    # Init
    device = get_device(True)
    set_seed()

    writer = SummaryWriter()

    train_dataset = FruitImageDataset(train_folder)
    test_dataset = FruitImageDataset(test_folder)

    train_dataloader, val_dataloader, test_dataloader, classes, train_info = get_dataloaders_from_dataset(test_dataset,
                                                                                                          train_dataset,
                                                                                                          ratio=0.8,
                                                                                                          batch_size=batch_size,
                                                                                                          fix_imbalanced=True)

    log.log_train_labels_distribution_image(classes, train_info, writer)
    log.log_images_examples(classes, train_dataloader, writer)

    # Define model
    model = Model()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                    max_lr=lr,
                                                    steps_per_epoch=len(train_dataloader),
                                                    epochs=epochs)

    # find_lr(criterion, device, model, optimizer, train_dataloader, writer)

    # Train Model
    best_val_acc = 0
    for epoch in tqdm(range(epochs), desc="Epoch"):
        model.train()
        train_loss = 0
        train_acc = 0
        iterations = len(train_dataloader)
        for i, train_data in tqdm(enumerate(train_dataloader),
                                  desc=f"Epoch: {epoch} / {epochs} - Iteration",
                                  total=iterations,
                                  miniters=int(iterations / 200)):
            x, y1, y2 = train_data

            y_pred1, y_pred2 = model(x.to(device))

            loss = criterion(y_pred1, y1.to(device)) + criterion(y_pred2, y2.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss
            acc_iter = accuracy_score(y1, torch.argmax(y_pred1.cpu(), dim=1))
            train_acc += acc_iter

        train_loss = train_loss / (i + 1)
        train_acc = train_acc / (i + 1)

        if epoch % validation_step == 0:
            print(f"Epoch:{epoch}")
            print(f"Training Loss:{train_loss:.5f}\tTraining Acc:{train_acc * 100:.5f}%")
            print("------------------------------------------------------------------")

            val_loss = 0
            val_acc = 0
            model.eval()
            for i, val_data in enumerate(val_dataloader):
                x, y1, y2 = val_data
                y_pred1, y_pred2 = model(x.to(device))

                loss = criterion(y_pred1, y1.to(device)) + criterion(y_pred2, y2.to(device))
                val_loss += loss
                prediction_indexes = y_pred1.cpu()
                acc_iter = accuracy_score(y1, torch.argmax(prediction_indexes, dim=1))
                val_acc += acc_iter

            val_loss = val_loss / (i + 1)
            val_acc = val_acc / (i + 1)

            if best_val_acc < val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), 'fruits.pt')

            print(f"Epoch:{epoch}")
            print(f"Val Loss:{val_loss:.5f}\tVal Acc:{val_acc * 100:.5f}%")
            print("------------------------------------------------------------------")

            optimizer_lr = get_optimizer_lr(optimizer)
            log_metrics(writer, epoch, optimizer_lr, train_acc, train_loss, val_acc, val_loss)

    return val_loss, val_acc, train_loss, train_acc

    # Log training results and hyperarams
    hyper_params_dict = {
        "lr": lr,
        "batch_size": batch_size,
        "image_size": image_size,
        "freeze_pretrained": freeze_pretrained
    }
    metrics_dict = {
        "loss/val": val_loss,
        "acc/val": val_acc,
        "loss/train": train_loss,
        "acc/train": train_acc
    }
    log.log_hyperparams(hyper_params_dict, metrics_dict, writer)

    # Evaluate and log results
    images, labels, probs = get_predictions(model, val_dataloader, device)
    pred_labels = torch.argmax(probs, 1)

    log.lof_confusion_matrix(classes, labels, pred_labels, writer)

    incorrect_examples = get_incorrect_examples(images, labels, probs, pred_labels)
    log.log_incorrect_examples(classes, incorrect_examples, writer)

    # Save model
    torch.save(model.state_dict(), 'eggs-model.pt')


def find_lr(criterion, device, model, optimizer, train_dataloader, writer):
    END_LR = 10
    NUM_ITER = 300
    lr_finder = LRFinder(model, optimizer, criterion, device)
    lrs, losses = lr_finder.range_test(train_dataloader, END_LR, NUM_ITER)
    plot_lr_finder(lrs, losses, skip_start=0, skip_end=0)
    writer.add_figure('lr_finder', plt.gcf(), 0)


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

            images.append(x.cpu())
            labels.append(y.cpu())
            probs.append(y_pred.cpu())

    images = torch.cat(images, dim=0)
    labels = torch.cat(labels, dim=0)
    probs = torch.cat(probs, dim=0)

    return images, labels, probs


if __name__ == '__main__':
    main()
