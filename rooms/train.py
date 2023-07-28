import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms

from LRFinder import plot_lr_finder, LRFinder
from Trainer import Trainer
from classification_models import get_resnet_model, get_mobilenet_model, get_mobilenetv3_model
from tools import logging_tools as log
from tools.data_tool import get_dataloaders_from_folders
from tools.torch_tool import get_device, set_seed

batch_size = 16
lr = 1e-3
epochs = 100
image_size = 256
validation_step = 5
early_stop = -1

freeze_pretrained = True

train_folder = "dataset/train"
val_folder = "dataset/val"
test_folder = "dataset/test"


def main():
    # Init
    device = get_device(True)
    set_seed()

    writer = SummaryWriter()

    train_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(size=(image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=10)
        ]
    )

    # Prepare datasets
    train_dataloader, \
        val_dataloader, \
        test_dataloader, \
        classes, \
        train_info = get_dataloaders_from_folders(train_folder,
                                                  val_folder=val_folder,
                                                  image_size=image_size,
                                                  batch_size=batch_size,
                                                  train_transform=train_transform)

    log.log_train_labels_distribution_image(classes, train_info, writer)
    log.log_images_examples(classes, train_dataloader, writer)

    # Define model
    model, model_name = get_mobilenetv3_model(device, freeze_pretrained=freeze_pretrained, classes=len(classes))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                    max_lr=lr,
                                                    steps_per_epoch=len(train_dataloader),
                                                    epochs=epochs)

    # find_lr(criterion, device, model, optimizer, train_dataloader, writer)

    # Train Model
    trainer = Trainer(model, optimizer, criterion, scheduler, writer, early_stop=early_stop)
    val_loss, val_acc, train_loss, train_acc = trainer.train(train_dataloader, val_dataloader, epochs, validation_step)

    # Log training results and hyperarams
    hyper_params_dict = {
        "lr": lr,
        "batch_size": batch_size,
        "image_size": image_size,
        "freeze_pretrained": freeze_pretrained,
        "model":model_name
    }
    metrics_dict = {
        "~loss/val": val_loss
    }
    log.log_hyperparams(hyper_params_dict, metrics_dict, writer)

    # Evaluate and log results
    images, labels, probs = get_predictions(model, val_dataloader, device)
    pred_labels = torch.argmax(probs, 1)

    log.lof_confusion_matrix(classes, labels, pred_labels, writer)

    incorrect_examples = get_incorrect_examples(images, labels, probs, pred_labels)
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
