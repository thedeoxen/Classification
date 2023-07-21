import torchinfo
from torch import nn
from torchvision.models import resnet50, efficientnet_b0


def get_resnet_model(device, freeze_pretrained=True, classes=3):
    model = resnet50(weights="IMAGENET1K_V1")
    if freeze_pretrained:
        __freeze_params(model)
    model.fc = nn.Linear(model.fc.in_features, classes)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(model.fc.in_features, classes),
        nn.Softmax(dim=-1)
    )

    model.to(device)
    torchinfo.summary(model)
    return model


def get_efficientnet_model(device, freeze_pretrained=True, classes=3):
    model = efficientnet_b0(weights="IMAGENET1K_V1")
    if freeze_pretrained:
        __freeze_params(model)

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=1280, out_features=classes, bias=True),
        nn.Softmax(dim=-1)
    )
    model.to(device)
    torchinfo.summary(model)
    return model


def __freeze_params(model):
    params = list(model.parameters())
    for index, param in enumerate(params):
        param.requires_grad = False
