import torch.nn as nn
import torch.nn.functional as F
from study_pytorch.NeutralModule.Bottleneck import Bottleneck

class FPN(nn.Module):
    def __init__(self, layers):
        super(FPN, self).__init__()
        self.in_dim = 64
        # 处理出入C1模块
        self.conv2 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        # 自下而上搭建C2,C3,C4,C5,stride1和2的区别
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], 2)
        self.layer3 = self._make_layer(256, layers[2], 2)
        self.layer4 = self._make_layer(512, layers[3], 2)
        # 对C5减少通道得到P5
        self.toplayer = nn.Conv2d(2048, 256, 1, 1, 0)
        # 3x3卷积融合
        self.smooth1 = nn.Conv2d(256, 256, 3, 1, 1)
        self.smooth2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.smooth3 = nn.Conv2d(256, 256, 3, 1, 1)
        # 横向连接，保证通道数相等
        self.latlayer1 = nn.Conv2d(1024, 256, 1, 1, 0)
        self.latlayer2 = nn.Conv2d(512, 256, 1, 1, 0)
        self.latlayer3 = nn.Conv2d(256, 256, 1, 1, 0)

        # 构建C
    def _make_layer (self, out_dim, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_dim != Bottleneck.expansion*out_dim :
            downsample = nn.Sequential(
                nn.Conv2d(self.in_dim, Bottleneck.expansion*out_dim, 1, stride, bias=False),
                nn.BatchNorm2d(Bottleneck.expansion*out_dim)
            )
        layers = []
        layers.append(Bottleneck(self.in_dim, out_dim, stride, downsample))
        self.in_dim = out_dim*Bottleneck.expansion
        for i in range(1, blocks):
            layers.append(Bottleneck(self.in_dim, out_dim))
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        _, _, H, W = y.shape
        return F.upsample(x, size=(H, W), mode='bilinear') + y

    def forward(self, x):
        c1 = self.maxpool(self.relu(self.bn1(self.conv1)))
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer1(c3))
        p2 = self._upsample_add(p3, self.latlayer1(c2))

        p4 = self.smooth1(p4)
        p3 = self.smooth1(p3)
        p2 = self.smooth1(p2)

        return p4, p3, p2, p5


