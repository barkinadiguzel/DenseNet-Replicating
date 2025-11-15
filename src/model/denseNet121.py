import torch
import torch.nn as nn
from layers.conv1 import conv1
from layers.dense_layer import dense_layer
from layers.transition_layer import transition_layer
from layers.pool_layers import avgpool_layer
from layers.flatten_layer import flatten_layer
from layers.fc_layer import fc_layer
from config import growth_rate, input_channels, num_classes, block_layers

class DenseNet121(nn.Module):
    def __init__(self):
        super(DenseNet121, self).__init__()

        # Initial conv + maxpool
        self.conv1 = conv1(input_channels)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Dense Blocks
        num_features = 64
        self.dense_blocks = nn.ModuleList()
        self.trans_layers = nn.ModuleList()

        for i, num_layers in enumerate(block_layers):
            block_layers_list = []
            for _ in range(num_layers):
                block_layers_list.append(dense_layer(num_features, growth_rate))
                num_features += growth_rate
            self.dense_blocks.append(nn.Sequential(*block_layers_list))

            # Transition layer except after last block
            if i != len(block_layers) - 1:
                self.trans_layers.append(transition_layer(num_features))
                num_features = int(num_features * 0.5)  # compression

        # Final layers
        self.avgpool = avgpool_layer()
        self.flatten = flatten_layer()
        self.fc = fc_layer(num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        for i, dense_block in enumerate(self.dense_blocks):
            x = dense_block(x)
            if i < len(self.trans_layers):
                x = self.trans_layers[i](x)

        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
