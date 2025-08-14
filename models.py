import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# Updated ResNet-18 implementation

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes: int = 10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # Keep adaptive pooling for flexibility (supports larger inputs like 224×224)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * block.expansion, num_classes)
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, stride=s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out

def resnet18():
    """Return a ResNet-18 model (updated version)."""
    return ResNet(BasicBlock, [2, 2, 2, 2])

class StandardResNet(nn.Module):
    """Standard ResNet model for single-head classification from torchvision."""

    def __init__(self, dataset_name="CIFAR10", pretrained=False):
        super(StandardResNet, self).__init__()
        self.resnet = models.resnet18(pretrained=pretrained)

        if dataset_name == "MNIST":
            # Adapt for 1-channel grayscale images
            self.resnet.conv1 = nn.Conv2d(
                1, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            num_classes = 10
        elif dataset_name == "CIFAR10":
            # Use CIFAR-appropriate conv1 layer (smaller kernel, no stride)
            self.resnet.conv1 = nn.Conv2d(
                3, 64, kernel_size=3, stride=1, padding=1, bias=False
            )
            num_classes = 10
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.resnet(x)

# Multi-task model definition (uses the above ResNet as encoder)

class MTL_Two_Heads_ResNet(nn.Module):
    def __init__(self, dataset_name='MNIST', num_clients=10, pretrained=False, head_size: str = 'big'):
        super(MTL_Two_Heads_ResNet, self).__init__()

        # Use custom ResNet18 (CIFAR-style) as the encoder
        self.resnet = resnet18()

        # Adjust the first conv layer for MNIST (grayscale input)
        if dataset_name == 'MNIST':
            self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)

        # Depending on head_size, optionally project features to a smaller dimension
        _feature_dim_map = {
            'big': 512,
            'medium': 256,
            'small': 128,
        }
        self.feature_dim = _feature_dim_map[head_size]

        # Optional projection from 512 → desired feature_dim for medium/small
        self.feature_proj = None
        if self.feature_dim < 512:
            self.feature_proj = nn.Linear(512, self.feature_dim)

        # Remove the final classification layer of ResNet but KEEP adaptive pool
        self.encoder = nn.Sequential(*list(self.resnet.children())[:-1])  # excludes final linear layer
        self.flatten = nn.Flatten()

        # Validate head_size choice early
        if head_size not in {'big', 'medium', 'small'}:
            raise ValueError(f"head_size must be one of 'big', 'medium', or 'small'. Got {head_size}.")

        # Build task–specific heads with configurable size
        def _build_head(output_dim: int):
            """Utility to build a classification head given the desired hidden sizes."""
            # Hidden layer sizes depend on the (possibly reduced) input feature_dim
            hidden_layers_map = {
                'big':   [256],   # 512 → 256 → out
                'medium': [128],  # 256 → 128 → out
                'small':  [64],   # 128 → 64  → out
            }

            hidden_sizes = hidden_layers_map[head_size]
            input_dim = self.feature_dim
            layers = []
            for h in hidden_sizes:
                layers.append(nn.Linear(input_dim, h))
                layers.append(nn.ReLU())
                input_dim = h
            layers.append(nn.Linear(input_dim, output_dim))
            return nn.Sequential(*layers)

        # Digit classification head (10 classes)
        self.digit_head = _build_head(10)

        # Subset identification head (num_clients classes)
        self.subset_head = _build_head(num_clients)

        # Runtime controls for optionally suppressing one subset logit (by index)
        # When enabled, the specified class logit is set to a very negative value
        # during forward passes, effectively preventing that class from being predicted.
        self.kill_output_neuron: bool = False
        self.killed_subset_id: int | None = None

    def forward(self, x, return_features=True):
        # Shared encoding
        features = self.encoder(x)
        flattened = self.flatten(features)

        # Optional feature projection for medium/small models
        if self.feature_proj is not None:
            flattened = self.feature_proj(flattened)

        # Task-specific outputs
        digit_logits = self.digit_head(flattened)
        subset_logits = self.subset_head(flattened)

        # Optionally suppress one subset class logit (e.g., the forgotten client's ID)
        if self.kill_output_neuron and (self.killed_subset_id is not None):
            if 0 <= int(self.killed_subset_id) < subset_logits.size(1):
                subset_logits = subset_logits.clone()
                subset_logits[:, int(self.killed_subset_id)] = -1e9

        if return_features:
            return digit_logits, subset_logits, flattened
        else:
            return digit_logits, subset_logits 