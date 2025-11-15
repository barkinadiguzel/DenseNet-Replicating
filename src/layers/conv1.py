import torch.nn as nn

def conv1_layer(in_channels=3, out_channels=64):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
