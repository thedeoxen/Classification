import torchinfo
from torch import nn
from torchvision.models import resnet50, efficientnet_b0, mobilenet_v2, mobilenet_v3_small


def get_resnet_model(device, freeze_pretrained=True, classes=3):
    model = resnet50(weights="IMAGENET1K_V1")
    if freeze_pretrained:
        __freeze_params(model)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(model.fc.in_features, classes),
        nn.Softmax(dim=-1)
    )

    model.to(device)
    torchinfo.summary(model)
    return model, "resnet50"

def get_mobilenet_model(device, freeze_pretrained=True, classes=3):
    model = mobilenet_v2(weights="IMAGENET1K_V2")
    if freeze_pretrained:
        __freeze_params(model)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(model.last_channel, classes),
    )

    model.to(device)
    torchinfo.summary(model)
    return model, "mobilenet_v2"

def get_mobilenetv3_model(device, freeze_pretrained=True, classes=3):
    model = mobilenet_v3_small(weights="IMAGENET1K_V1")
    if freeze_pretrained:
        __freeze_params(model)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(model.classifier[0].in_features, classes),
    )

    model.to(device)
    torchinfo.summary(model)
    return model, "mobilenet_v3_small"


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
    return model, "efficientnet_b0"


def __freeze_params(model):
    params = list(model.parameters())
    for index, param in enumerate(params):
        param.requires_grad = False
