import torch
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from tools.logging_tools import log_metrics
from tools.torch_tool import get_device, EarlyStopper, get_optimizer_lr


class Trainer:
    def __init__(self, model, optimizer, criterion, scheduler, writer, early_stop=-1):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = get_device()
        self.writer = writer
        if early_stop > 0:
            self.stopper = EarlyStopper(3)

    def train(self, train_dataloader, val_dataloader, epochs, validation_step):
        for epoch in tqdm(range(epochs)):
            self.model.train()
            train_loss = 0
            train_acc = 0
            for i, train_data in enumerate(train_dataloader):
                x, y = train_data

                y_pred = self.model(x.to(self.device))

                loss = self.criterion(y_pred, y.to(self.device))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                train_loss += loss
                acc_iter = accuracy_score(y, torch.argmax(y_pred.cpu(), dim=1))
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
                for i, val_data in enumerate(val_dataloader):
                    x, y = val_data
                    y_pred = self.model(x.to(self.device))

                    loss = self.criterion(y_pred, y.to(self.device))
                    val_loss += loss
                    prediction_indexes = y_pred.cpu()
                    acc_iter = accuracy_score(y, torch.argmax(prediction_indexes, dim=1))
                    val_acc += acc_iter

                val_loss = val_loss / (i + 1)
                val_acc = val_acc / (i + 1)

                if self.stopper and self.stopper.early_stop(val_loss):
                    break;
                    print("early_stop")

                print(f"Epoch:{epoch}")
                print(f"Val Loss:{val_loss:.5f}\tVal Acc:{val_acc * 100:.5f}%")
                print("------------------------------------------------------------------")

                optimizer_lr = get_optimizer_lr(self.optimizer)
                log_metrics(self.writer, epoch, optimizer_lr, train_acc, train_loss, val_acc, val_loss)

        return val_loss, val_acc, train_loss, train_acc
