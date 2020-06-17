import torch
import torch.nn as nn
import torch.functional as F

class Bottleneck_densenet(nn.Module):
    def __init__(self, nChannels, growthrate):
        super(Bottleneck_densenet, self).__init__()
        interChannels = growthrate*4
        self.bottleneck = nn.Sequential(
            nn.BatchNorm2d(nChannels),
            nn.ReLU(inplace=True),
            nn.Conv2d(nChannels, interChannels, 1, bias=False),
            nn.BatchNorm2d(interChannels),
            nn.ReLU(inplace=True),
            nn.Conv2d(interChannels, growthrate, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        out = self.bottleneck(x)
        out = torch.cat((x, out), 1)
        return out

class Densenet(nn.Module):
    def __init__(self, nChannels, growthRate, nDenseBlock):
        super(Densenet, self).__init__()
        layers = []
        for i in range(int(nDenseBlock)):
            layers.append(Bottleneck_densenet(nChannels, growthRate))
            nChannels += growthRate
        self.denseblock = nn.Sequential(*layers)

    def forward(self, x):
        return self.denseblock(x)

