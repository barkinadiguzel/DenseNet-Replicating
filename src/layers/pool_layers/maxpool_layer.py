import torch.nn as nn

def maxpool_layer(kernel_size=3, stride=2, padding=1):
    return nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
