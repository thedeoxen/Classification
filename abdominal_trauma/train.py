import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import torchinfo
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from abdominal_trauma.dataset import RSNADataset
from abdominal_trauma.model import Model

batch_size = 2
lr = 1e-4
epochs = 10000
image_size = 224
validation_step = 1
early_stop = -1

freeze_pretrained = False


def get_device(log=False):
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    if log:
        print("PYTORCH DEVICE: ", device)
    return device


def set_seed(seed: int = 0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def to_numpy(tensor):
    return tensor.detach().cpu().numpy()

if __name__ == '__main__':

    device = get_device(True)
    set_seed()

    train_csv = pd.read_csv("dataset/train.csv")
    classes = list(train_csv.columns[1:])


    def count_files_in_directories(root_dir):
        file_counts = []

        for root, dirs, files in os.walk(root_dir):
            if not dirs:  # Only consider directories without subdirectories
                file_counts.append(len(files))

        if file_counts:
            min_files = min(file_counts)
            max_files = max(file_counts)
            avg_files = sum(file_counts) / len(file_counts)

            print(f"Minimum files: {min_files}")
            print(f"Maximum files: {max_files}")
            print(f"Average files: {avg_files:.2f}")

            plt.hist(file_counts, bins=10, edgecolor='black')
            plt.title('Histogram of File Counts')
            plt.xlabel('Number of Files')
            plt.ylabel('Frequency')
            plt.show()


    train_csv.iloc[0].values[1:]
    patient_ids = train_csv['patient_id']
    labels = train_csv.drop(columns=['patient_id'])

    train_dataset = RSNADataset(train_csv)

    t = train_dataset[[0]]


    def collate_fn(data):
        return tuple(data)


    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                                  # num_workers=2,
                                  # collate_fn=collate_fn
                                  )

    t1, t2 = next(iter(train_dataloader))

    model = Model()
    model = model.to(device)
    # torchinfo.summary(model, input_size=(1, 128, 128, 128))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
    #                                                 max_lr=lr,
    #                                                 steps_per_epoch=len(train_dataloader),
    #                                                 epochs=epochs)

    for epoch in tqdm(range(epochs), desc="Epoch"):
        model.train()
        train_loss = 0
        train_acc_fruits = 0
        train_acc_fresh = 0
        iterations = len(train_dataloader)
        for i, train_data in tqdm(enumerate(train_dataloader),
                                  desc=f"Epoch: {epoch} / {epochs} - Iteration",
                                  total=iterations,
                                  miniters=int(iterations / 200)):

            x,y = train_data
            x,y = x.to(device), y.to(device)
            x = x.unsqueeze(dim=1)
            y_pred1 = model(x)
            loss = criterion(y_pred1, y.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()
            if epoch % 100 == 0:
                print(loss.item())

    #             train_loss += loss
    #             acc_iter_fruits = accuracy_score(y1, torch.argmax(y_pred1.cpu(), dim=1))
    #             acc_iter2 = accuracy_score(y2, torch.argmax(y_pred2.cpu(), dim=1))
    #             train_acc_fruits += acc_iter_fruits
    #             train_acc_fresh += acc_iter2

    #     train_loss = train_loss / (i + 1)
    #     train_acc_fruits = train_acc_fruits / (i + 1)
    #     train_acc_fresh = train_acc_fresh / (i + 1)

    #     if epoch % validate_each_step == 0:
    #         print(f"Epoch:{epoch}")
    #         print(
    #             f"Training Loss:{train_loss:.5f}\tTraining Acc1:{train_acc_fruits * 100:.5f}% tTraining Acc2:{train_acc_fresh * 100:.5f}%")
    #         print("------------------------------------------------------------------")

    #         val_loss, \
    #             val_acc_fresh, val_acc_fruits, \
    #             val_f1_fresh, val_f1_fruits, \
    #             val_precision_fresh, val_precision_fruits, \
    #             val_recall_fresh, val_recall_fruits = validation_step(val_dataloader, model, criterion)

    #         if best_val_acc < val_acc_fruits:
    #             best_val_acc = val_acc_fruits
    #             torch.save(model.state_dict(), 'fruits.pt')

    #         print(f"Epoch:{epoch}")
    #         print(f"Val Loss:{val_loss:.5f}\tVal Acc:{val_acc_fruits * 100:.5f}%")
    #         print("------------------------------------------------------------------")

    #         optimizer_lr = get_optimizer_lr(optimizer)

    #         writer.add_scalar("lr", optimizer_lr, epoch)

    #         writer.add_scalar("loss/train", train_loss, epoch)
    #         writer.add_scalars("acc/train", {"fruits": train_acc_fruits, "fresh": train_acc_fresh}, epoch)

    #         writer.add_scalar("loss/val", val_loss, epoch)
    #         writer.add_scalars("acc/val", {"fruits": val_acc_fruits, "fresh": val_acc_fresh}, epoch)
    #         writer.add_scalars("precision/val", {"fruits": val_precision_fruits, "fresh": val_precision_fresh}, epoch)
    #         writer.add_scalars("recall/val", {"fruits": val_recall_fruits, "fresh": val_recall_fresh}, epoch)
    #         writer.add_scalars("f1/val", {"fruits": val_f1_fruits, "fresh": val_f1_fresh}, epoch)
