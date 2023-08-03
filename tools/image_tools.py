import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, multilabel_confusion_matrix

images_mean = [0.485, 0.456, 0.406]
images_std = [0.229, 0.224, 0.225]

def plot_images(images, labels, classes):
    n_images = len(images)

    rows = int(np.sqrt(n_images))
    cols = int(np.sqrt(n_images))

    fig = plt.figure(figsize=(10, 10))

    for i in range(rows * cols):
        ax = fig.add_subplot(rows, cols, i + 1)
        image = images[i]
        ax.imshow(image.permute(1, 2, 0).cpu().numpy())
        ax.set_title(classes[labels[i]])
        ax.axis('off')


def plot_most_incorrect(incorrect, classes, n_images):
    n_images = min(n_images, len(incorrect))
    rows = int(np.sqrt(n_images))
    cols = int(np.sqrt(n_images))

    fig = plt.figure(figsize=(10, 10))

    for i in range(rows * cols):
        ax = fig.add_subplot(rows, cols, i + 1)

        image, true_label, probs = incorrect[i]
        image = image.permute(1, 2, 0)
        image = denormalize_image(image)

        if true_label.shape[0]>1:
            true_label = torch.argmax(true_label).item()
            true_prob = probs[true_label]
        else:
            true_prob = probs[true_label]
        incorrect_prob, incorrect_label = torch.max(probs, dim=0)
        true_class = classes[true_label]
        incorrect_class = classes[incorrect_label]

        ax.imshow(image.cpu().numpy())
        ax.set_title(f'true label: {true_class} ({true_prob:.3f})\n'
                     f'pred label: {incorrect_class} ({incorrect_prob:.3f})')
        ax.axis('off')

    fig.subplots_adjust(hspace=0.4)


def plot_confusion_matrix(labels, pred_labels, classes):
    fig = plt.figure(figsize=(10, 10));
    ax = fig.add_subplot(1, 1, 1);
    cm = confusion_matrix(labels, pred_labels);
    cm = ConfusionMatrixDisplay(cm, display_labels=classes);
    cm.plot(values_format='d', cmap='Blues', ax=ax)
    fig.delaxes(fig.axes[1])  # delete colorbar
    plt.xticks(rotation=90)
    plt.xlabel('Predicted Label', fontsize=50)
    plt.ylabel('True Label', fontsize=50)


def plot_multilabel_confusion_matrix(labels, pred_labels, classes):
    y_pred = np.where(pred_labels > 0.5, 1, 0)
    n = math.ceil(math.sqrt(len(classes)))
    f, axes = plt.subplots(n, n, figsize=(25, 15))
    axes = axes.ravel()
    for i in range(len(classes)):
        disp = ConfusionMatrixDisplay(confusion_matrix(labels[:, i],
                                                       y_pred[:, i]),
                                      display_labels=[0, i])
        disp.plot(ax=axes[i], values_format='.4g')
        disp.ax_.set_title(classes[i])
        if i < 10:
            disp.ax_.set_xlabel('')
        if i % 5 != 0:
            disp.ax_.set_ylabel('')
        disp.im_.colorbar.remove()

    plt.subplots_adjust(wspace=0.10, hspace=0.1)
    f.colorbar(disp.im_, ax=axes)


def denormalize_image(normalized_image_tensor):
    """
    Denormalizes the image tensor back to values in the range [0, 1].

    Args:
        normalized_image_tensor (torch.Tensor): The normalized image tensor (C x H x W).
        mean (list or tuple): List or tuple of mean values used for normalization.
        std (list or tuple): List or tuple of standard deviation values used for normalization.

    Returns:
        torch.Tensor: The denormalized image tensor with values in the range [0, 1].
    """
    # Step 1: Convert the image tensor to a NumPy array
    normalized_image_array = normalized_image_tensor.numpy()

    # Step 2: Reverse the normalization transformation
    original_image_array = (normalized_image_array * images_std) + images_mean

    # Step 3: Convert the array back to a tensor
    original_image_tensor = torch.tensor(original_image_array)

    return original_image_tensor