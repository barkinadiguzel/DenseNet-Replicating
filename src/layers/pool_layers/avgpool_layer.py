import torch.nn as nn

def avgpool_layer():
    return nn.AdaptiveAvgPool2d((1, 1))
