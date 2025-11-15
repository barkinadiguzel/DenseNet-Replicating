# DenseNet variant
variant = "DenseNet-121"  # options: DenseNet-121, 169, 201, 264

# Input image
input_channels = 3         # RGB
input_size = (224, 224)    # ImageNet standard

# DenseNet growth parameters
growth_rate = 32
block_layers = [6, 12, 24, 16]  # DenseNet-121
bottleneck = True
compression = 1.0

# Classifier
num_classes = 1000         # ImageNet
