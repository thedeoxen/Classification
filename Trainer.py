import numpy as np
import torch
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from tools.logging_tools import log_metrics
from tools.torch_tool import get_device, EarlyStopper, get_optimizer_lr, to_numpy


class Trainer:
    def __init__(self, model, optimizer, criterion, scheduler, writer, early_stop=-1, multilabel=False):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = get_device()
        self.writer = writer
        self.stopper = None
        if early_stop > 0:
            self.stopper = EarlyStopper(3)
        self.multilabel = multilabel

    def train(self, train_dataloader, val_dataloader, epochs, validation_step):
        best_val_acc = 0
        for epoch in tqdm(range(epochs), desc="Epoch"):
            self.model.train()
            train_loss = 0
            train_acc = 0
            iterations = len(train_dataloader)
            for i, train_data in tqdm(enumerate(train_dataloader),
                                      desc=f"Epoch: {epoch} / {epochs} - Iteration",
                                      total=iterations,
                                      miniters=int(iterations / 200)):
                x, y = train_data

                y_pred = self.model(x.to(self.device))

                loss = self.criterion(y_pred, y.to(self.device))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                train_loss += loss

                if self.multilabel:
                    acc_iter = self.get_multilabel_accuracy(y, y_pred)
                else:
                    prediction = y_pred.cpu().detach()
                    acc_iter = accuracy_score(y, torch.argmax(prediction, dim=1))
                train_acc += acc_iter

            train_loss = train_loss / (i + 1)
            train_acc = train_acc / (i + 1)

            if epoch % validation_step == 0:
                print(f"Epoch:{epoch}")
                print(f"Training Loss:{train_loss:.5f}\tTraining Acc:{train_acc * 100:.5f}%")
                print("------------------------------------------------------------------")

                val_loss = 0
                val_acc = 0
                self.model.eval()

                iterations = len(val_dataloader)
                for i, val_data in tqdm(enumerate(val_dataloader),
                                      desc=f"Validation Iteration",
                                      total=iterations,
                                      miniters=int(iterations / 200)):
                    x, y = val_data
                    y_pred = self.model(x.to(self.device))

                    loss = self.criterion(y_pred, y.to(self.device))
                    val_loss += loss

                    if self.multilabel:
                        acc_iter = self.get_multilabel_accuracy(y, y_pred)
                    else:
                        prediction_indexes = y_pred.cpu()
                        acc_iter = accuracy_score(y, torch.argmax(prediction_indexes, dim=1))

                    val_acc += acc_iter

                val_loss = val_loss / (i + 1)
                val_acc = val_acc / (i + 1)

                if best_val_acc < val_acc:
                    best_val_acc = val_acc
                    torch.save(self.model.state_dict(), 'model.pt')

                if self.stopper and self.stopper.early_stop(val_loss):
                    print("early_stop")
                    break

                print(f"Epoch:{epoch}")
                print(f"Val Loss:{val_loss:.5f}\tVal Acc:{val_acc * 100:.5f}%")
                print("------------------------------------------------------------------")

                optimizer_lr = get_optimizer_lr(self.optimizer)
                log_metrics(self.writer, epoch, optimizer_lr, train_acc, train_loss, val_acc, val_loss)

        return val_loss, val_acc, train_loss, train_acc

    def get_multilabel_accuracy(self, y, y_pred):
        prediction = y_pred.cpu().detach()
        y_pred = np.where(prediction > 0.5, 1, 0)
        y1 = y_pred.round().astype(np.float)
        y2 = to_numpy(y).round().astype(np.float)
        acc_iter = accuracy_score(y1, y2, normalize=True)
        return acc_iter
