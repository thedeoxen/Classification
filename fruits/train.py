import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchinfo import torchinfo
from torchvision.models import resnet50
from tqdm import tqdm

from fruits.FruitImageDataset import FruitImageDataset
from tools import logging_tools as log
from tools.data_tool import get_dataloaders_from_dataset
from tools.torch_tool import get_device, set_seed, get_optimizer_lr

batch_size = 16
lr = 1e-3
epochs = 1
image_size = 256
validation_step = 1
early_stop = 5

freeze_pretrained = True

train_folder = "dataset/Train"
test_folder = "dataset/Test"


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

    train_dataset = FruitImageDataset(train_folder, image_size=image_size)
    test_dataset = FruitImageDataset(test_folder, image_size=image_size, train=False)

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

    # Train Model
    best_val_acc = 0
    for epoch in tqdm(range(epochs), desc="Epoch"):
        model.train()
        train_loss = 0
        train_acc1 = 0
        train_acc2 = 0
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
            acc_iter_fruits = accuracy_score(y1, torch.argmax(y_pred1.cpu(), dim=1))
            acc_iter2 = accuracy_score(y2, torch.argmax(y_pred2.cpu(), dim=1))
            train_acc1 += acc_iter_fruits
            train_acc2 += acc_iter2

        train_loss = train_loss / (i + 1)
        train_acc1 = train_acc1 / (i + 1)
        train_acc2 = train_acc2 / (i + 1)

        if epoch % validation_step == 0:
            print(f"Epoch:{epoch}")
            print(
                f"Training Loss:{train_loss:.5f}\tTraining Acc1:{train_acc1 * 100:.5f}% tTraining Acc2:{train_acc2 * 100:.5f}%")
            print("------------------------------------------------------------------")

            val_loss = 0

            val_acc_fruits = 0
            val_precision_fruits = 0
            val_recall_fruits = 0
            val_f1_fruits = 0

            val_acc_fresh = 0
            val_precision_fresh = 0
            val_recall_fresh = 0
            val_f1_fresh = 0

            model.eval()
            for i, val_data in enumerate(val_dataloader):
                x, y1, y2 = val_data
                y_pred1, y_pred2 = model(x.to(device))
                loss = criterion(y_pred1, y1.to(device)) + criterion(y_pred2, y2.to(device))
                val_loss += loss
                prediction_indexes1 = y_pred1.cpu()
                prediction_indexes2 = y_pred2.cpu()

                index_fruits = torch.argmax(prediction_indexes1, dim=1)
                index_fresh = torch.argmax(prediction_indexes2, dim=1)

                acc_fruits, f1_fruits, precision_fruits, recall_fruits = get_classification_metrics(index_fruits, y1)
                acc_fresh, f1_fresh, precision_fresh, recall_fresh = get_classification_metrics(index_fresh, y2)

                val_acc_fruits += acc_fruits
                val_precision_fruits += precision_fruits
                val_recall_fruits += recall_fruits
                val_f1_fruits += f1_fruits

                val_acc_fresh += acc_fresh
                val_precision_fresh += precision_fresh
                val_recall_fresh += recall_fresh
                val_f1_fresh += f1_fresh

            size = (i + 1)

            val_acc_fruits = val_acc_fruits / size
            val_precision_fruits = val_precision_fruits / size
            val_recall_fruits = val_recall_fruits / size
            val_f1_fruits = val_f1_fruits / size

            val_acc_fresh = val_acc_fresh / size
            val_precision_fresh = val_precision_fresh / size
            val_recall_fresh = val_recall_fresh / size
            val_f1_fresh = val_f1_fresh / size

            if best_val_acc < val_acc_fruits:
                best_val_acc = val_acc_fruits
                torch.save(model.state_dict(), 'fruits.pt')

            print(f"Epoch:{epoch}")
            print(f"Val Loss:{val_loss:.5f}\tVal Acc:{val_acc_fruits * 100:.5f}%")
            print("------------------------------------------------------------------")

            optimizer_lr = get_optimizer_lr(optimizer)

            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("Loss/train", train_loss, epoch)

            writer.add_scalar("acc_fruits/val", val_acc_fruits, epoch)
            writer.add_scalar("precision_fruits/val", val_precision_fruits, epoch)
            writer.add_scalar("recall_fruits/val", val_recall_fruits, epoch)
            writer.add_scalar("f1_fruits/val", val_f1_fruits, epoch)
            writer.add_scalar("acc_fresh/val", val_acc_fresh, epoch)
            writer.add_scalar("precision_fresh/val", val_precision_fresh, epoch)
            writer.add_scalar("recall_fresh/val", val_recall_fresh, epoch)
            writer.add_scalar("f1_fresh/val", val_f1_fresh, epoch)

            writer.add_scalar("LR", optimizer_lr, epoch)

    # Log training results and hyperarams
    hyper_params_dict = {
        "lr": lr,
        "batch_size": batch_size,
        "image_size": image_size,
        "freeze_pretrained": freeze_pretrained
    }
    metrics_dict = {
        "loss/val": val_loss,
        "acc_fruits/val": val_acc_fruits,
        "acc_fruits/val": val_acc_fruits,
        "loss/train": train_loss,
        "acc1/train": train_acc1,
        "acc2/train": train_acc2
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


def get_classification_metrics(index_fruits, y1):
    acc_iter1 = accuracy_score(y1, index_fruits)
    precision = precision_score(y1, index_fruits)
    recall = recall_score(y1, index_fruits)
    f1 = f1_score(y1, index_fruits)
    return acc_iter1, f1, precision, recall


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
