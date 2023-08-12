# import torch
# from torch import nn
#
#
# class Model(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv = nn.Sequential(self.conv_layer(in_chan=64, out_chan=128),
#                                   self.conv_layer(in_chan=128, out_chan=128),
#                                   self.conv_layer(in_chan=128, out_chan=256))
#
#         self.fc = nn.Sequential(nn.Linear(9216, 512),
#                                 nn.Dropout(p=0.15),
#                                 nn.Linear(512, 14))
#
#     def conv_layer(self, in_chan, out_chan):
#         conv_layer = nn.Sequential(
#             nn.Conv2d(in_chan, out_chan, kernel_size=(3, 3), padding=0),
#             nn.LeakyReLU(),
#             nn.MaxPool2d((2, 2)),
#             # nn.BatchNorm2d(out_chan)
#         )
#
#         return conv_layer
#
#     def forward(self, x):
#         out = self.conv(x)
#         out = out.view(out.size(0), -1)
#         out = self.fc(out)
#         # out = torch.sigmoid(out)
#         return out


import torch
import torch.nn as nn


class Conv3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_size, dropout_rate):
        super(Conv3DBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool3d(pool_size)
        self.batchnorm = nn.BatchNorm3d(out_channels)
        self.dropout = nn.Dropout3d(dropout_rate)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.batchnorm(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = Conv3DBlock(1, 32, kernel_size=3, pool_size=2, dropout_rate=0.0)
        self.conv2 = Conv3DBlock(32, 32, kernel_size=3, pool_size=2, dropout_rate=0.0)
        self.conv3 = Conv3DBlock(32, 64, kernel_size=3, pool_size=2, dropout_rate=0.01)
        self.conv4 = Conv3DBlock(64, 128, kernel_size=3, pool_size=2, dropout_rate=0.02)
        self.conv5 = Conv3DBlock(128, 256, kernel_size=3, pool_size=2, dropout_rate=0.03)
        self.conv6 = Conv3DBlock(256, 512, kernel_size=3, pool_size=2, dropout_rate=0.04)
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Linear(512, 1024)
        self.dropout = nn.Dropout(0.08)
        self.fc2 = nn.Linear(1024, 14)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


# Create an instance of the model
model = Model()