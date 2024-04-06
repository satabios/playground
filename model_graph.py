import timm
import torch
import torch.nn as nn
from collections import OrderedDict
import re
import numpy as np
from torchsummary import summary
from collections import defaultdict, OrderedDict
import ipdb

from torchviz import make_dot



class VGG(nn.Module):
    ARCH = [64, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']

    def __init__(self) -> None:
        super().__init__()

        layers = []
        counts = defaultdict(int)

        def add(name: str, layer: nn.Module) -> None:
            layers.append((f"{name}{counts[name]}", layer))
            counts[name] += 1

        in_channels = 3
        for x in self.ARCH:
            if x != 'M':
                # conv-bn-relu
                add("conv", nn.Conv2d(in_channels, x, 3, padding=1, bias=False))
                add("bn", nn.BatchNorm2d(x))
                add("relu", nn.ReLU(True))
                in_channels = x
            else:
                # maxpool
                add("pool", nn.MaxPool2d(2))

        self.backbone = nn.Sequential(OrderedDict(layers))
        self.classifier = nn.Linear(512, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # backbone: [N, 3, 32, 32] => [N, 512, 2, 2]
        x = self.backbone(x)

        # avgpool: [N, 512, 2, 2] => [N, 512]
        x = x.mean([2, 3])

        # classifier: [N, 512] => [N, 10]
        x = self.classifier(x)
        return x


def custom_summary(model, input_size):
    def forward_hook(module, input, output):
        module_name = str(module.__class__).split('.')[-1].split("'")[0]
        input_shape = str(input[0].shape)
        output_shape = str(output.shape)
        num_params = sum(p.numel() for p in module.parameters())

        print(f"{module_name.ljust(20)} | "
              f"Input shape: {input_shape.ljust(30)} | "
              f"Output shape: {output_shape.ljust(30)} | "
              f"Parameters: {num_params}")

    hooks = []
    for layer in model.children():
        hook = layer.register_forward_hook(forward_hook)
        hooks.append(hook)

    try:
        model(torch.randn(input_size))
    finally:
        for hook in hooks:
            hook.remove()

model = VGG()

custom_summary(model, (1,3,32,32))