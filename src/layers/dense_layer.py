import torch
import torch.nn as nn
import torch.nn.functional as F

from config import growth_rate, bottleneck

class DenseLayer(nn.Module):
    def __init__(self, in_channels):
        super(DenseLayer, self).__init__()
        self.bottleneck = bottleneck

        if self.bottleneck:
            inter_channels = 4 * growth_rate
            self.bn1 = nn.BatchNorm2d(in_channels)
            self.relu1 = nn.ReLU(inplace=True)
            self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, bias=False)

            self.bn2 = nn.BatchNorm2d(inter_channels)
            self.relu2 = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(inter_channels, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.bn = nn.BatchNorm2d(in_channels)
            self.relu = nn.ReLU(inplace=True)
            self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        if self.bottleneck:
            out = self.conv1(self.relu1(self.bn1(x)))
            out = self.conv2(self.relu2(self.bn2(out)))
        else:
            out = self.conv(self.relu(self.bn(x)))
        out = torch.cat([x, out], 1)  
        return out
