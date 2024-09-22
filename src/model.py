import torch
import torch.nn as nn
import torch.nn.functional as F

# define the CNN architecture


# Dense Layer which consists of BN -> ReLU -> Conv
class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(DenseLayer, self).__init__()
        
        # Apply Batch Norm, ReLU, and 3x3 Conv (bottleneck for growth)
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        )

    def forward(self, x):
        # Compute the output of the dense layer and concatenate input (x) with output
        out = self.layer(x)
        return torch.cat([x, out], dim=1)  # Concatenate along the channel dimension

# Dense Block which consists of multiple Dense Layers
class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate):
        super(DenseBlock, self).__init__()
        
        # Create a sequence of dense layers
        layers = []
        for i in range(num_layers):
            layers.append(DenseLayer(in_channels + i * growth_rate, growth_rate))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

# Transition Layer: Reduces feature map size using 1x1 conv and downsampling
class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        
        # Apply 1x1 Conv followed by Average Pooling
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.layer(x)

# DenseNet model
class MyModel(nn.Module):
    def __init__(self, num_classes=1000, growth_rate=32, num_layers_per_block=[6, 12, 24, 16],dropout=0.3):
        super(MyModel, self).__init__()
        
        # Initial convolutional layer
        self.init_conv = nn.Sequential(
            nn.Conv2d(3, 2 * growth_rate, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(2 * growth_rate),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        num_channels = 2 * growth_rate  # Starting number of channels after the initial convolution

        # Create Dense Blocks followed by Transition Layers
        self.features = nn.Sequential()
        for i, num_layers in enumerate(num_layers_per_block):
            # Create a Dense Block
            dense_block = DenseBlock(num_layers, num_channels, growth_rate)
            self.features.add_module(f'dense_block_{i+1}', dense_block)
            num_channels += num_layers * growth_rate  # Update the number of input channels for next block

            # If not the last block, add a Transition Layer to downsample
            if i != len(num_layers_per_block) - 1:
                out_channels = num_channels // 2  # Reduce the number of feature maps
                transition_layer = TransitionLayer(num_channels, out_channels)
                self.features.add_module(f'transition_layer_{i+1}', transition_layer)
                num_channels = out_channels  # Update the number of channels after transition layer

        # Final BatchNorm layer
        self.final_bn = nn.BatchNorm2d(num_channels)
        
        # Global average pooling and classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Adaptive average pool to 1x1
            nn.Flatten(),
            nn.Linear(num_channels, num_classes)
        )

    def forward(self, x):
        x = self.init_conv(x)         # Initial convolution
        x = self.features(x)          # Pass through Dense Blocks and Transition Layers
        x = self.final_bn(x)          # Final BatchNorm
        x = self.classifier(x)        # Global pooling and fully connected layer
        return x

######################################################################################
#                                     TESTS
######################################################################################



import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = next(dataiter)
    print(images.shape) 
    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
