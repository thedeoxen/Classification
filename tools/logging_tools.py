from matplotlib import pyplot as plt

from tools.image_tools import plot_images, plot_confusion_matrix, plot_most_incorrect, plot_multilabel_confusion_matrix


def log_train_labels_distribution_image(classes, class_count, writer):
    plt.bar(classes, class_count)
    plt.xticks(rotation=90)
    writer.add_figure('train_labels_distribution', plt.gcf(), 0)


def log_images_examples(classes, train_dataloader, writer):
    for sample in train_dataloader:
        break
    plot_images(sample[0], sample[1], classes)
    writer.add_figure('train_images', plt.gcf(), 0)


def log_confusion_matrix(classes, labels, pred_labels, writer, multilabel=False):
    if multilabel:
        plot_multilabel_confusion_matrix(labels, pred_labels, classes)
    else:
        plot_confusion_matrix(labels, pred_labels, classes)
    writer.add_figure('val_confusion_matrix', plt.gcf(), 0)


def log_incorrect_examples(classes, incorrect_examples, writer):
    plot_most_incorrect(incorrect_examples, classes, 10)
    writer.add_figure('val_most_incorrect', plt.gcf(), 0)


def log_metrics(writer, epoch, optimizer_lr, train_acc, train_loss, val_acc, val_loss):
    writer.add_scalar("Loss/val", val_loss, epoch)
    writer.add_scalar("Acc/val", val_acc, epoch)
    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Acc/train", train_acc, epoch)
    writer.add_scalar("LR", optimizer_lr, epoch)


def log_hyperparams(hyper_params_dict, metrics_dict, writer):
    writer.add_hparams(
        hyper_params_dict,
        metrics_dict
    )
