import os
import random

import numpy as np
import torch


def get_device():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
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



def find_lr(net, criterion, optimizer, trn_loader, init_value=1e-8, final_value=100., beta=0.98):
    epochs = 10
    num = (len(trn_loader) - 1) * epochs
    mult = (final_value / init_value) ** (1 / num)
    lr = init_value
    optimizer.param_groups[0]['lr'] = lr
    avg_loss = 0.
    best_loss = 0.
    batch_num = 0
    losses = []
    log_lrs = []

    for e in range(epochs):
        for data in trn_loader:
            batch_num += 1
            # As before, get the loss for this mini-batch of inputs/outputs
            inputs, labels = data
            device = get_device()
            # inputs, labels = Variable(inputs.to(device)), Variable(labels.to(device))
            optimizer.zero_grad()
            y_pred = net(inputs.to(device))
            loss = criterion(torch.softmax(y_pred, dim=1), labels.to(device))

            # Compute the smoothed loss
            avg_loss = beta * avg_loss + (1 - beta) * loss.data
            smoothed_loss = avg_loss / (1 - beta ** batch_num)
            # Stop if the loss is exploding
            if batch_num > 1 and smoothed_loss > 4 * best_loss:
                return log_lrs, losses
            # Record the best loss
            if smoothed_loss < best_loss or batch_num == 1:
                best_loss = smoothed_loss
            # Store the values
            losses.append(to_numpy(loss.data))
            lr_tensor = torch.FloatTensor([lr])
            loss_np = to_numpy(lr_tensor)
            log_lrs.append(np.log10(loss_np))
            # Do the SGD step
            loss.backward()
            optimizer.step()
            # Update the lr for the next step
            lr *= mult
            optimizer.param_groups[0]['lr'] = lr
    return log_lrs, losses

    # Example run. Select highest LR before graph looks erratic
    # model = BasicEmbModel(len(occupations), len(locations), len(industries), len(job_tags), len(user_ids), 25).to(device)
    # occu_warp = partial(warp_loss, num_labels=torch.FloatTensor([len(adids)]).to(device), device=device, limit_grad=False)
    # logs,losses = find_lr(model, occu_warp, torch.optim.RMSprop(model.parameters(), lr=0.05), train_loader, init_value=0.001, final_value=1000)
    # m = -5
    # plt.plot(logs[10:m],losses[10:m])


def to_numpy(tensor):
    return tensor.detach().cpu().numpy()


def get_optimizer_lr(optimizer):
    return optimizer.param_groups[0]['lr']
