import torch.nn as nn
from config import num_classes

def fc_layer(input_features):
    return nn.Linear(input_features, num_classes)
