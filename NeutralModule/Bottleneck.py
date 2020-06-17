import torch.nn as nn

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_dim, out_dim, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 1, bias=False),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim, in_dim, 3, stride, 1, bias=False),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim, out_dim*self.expansion, 1, bias=False),
            nn.BatchNorm2d(out_dim*self.expansion)
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample;
        # self.downsample = nn.Sequential(
        #     nn.Conv2d(in_dim, out_dim, 1, 1),
        #     nn.BatchNorm2d(out_dim)
        # )

    def forward(self, x):
        indentity = x
        out = self.bottleneck(x)
        if self.downsample is not None:
            indentity = self.downsample(x)
        out += indentity
        out = self.relu(out)
        return out