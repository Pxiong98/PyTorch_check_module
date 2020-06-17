import torch
import torch.nn as nn

class VGG(nn.Module):
    def __init__(self, num_classes=1000)://1000个分类
        super(VGG, self).__init__()
        layers = []
        in_dim = 3
        out_dim = 64
        for i in range(13):
            layers += [nn.Conv2d(in_dim, out_dim, 3, 1, 1), nn.ReLU(inplace=True)]
            in_dim = out_dim
            if i == 1 or i==3 or i==6 or i==9 or i==12:
                layers += [nn.MaxPool2d(2, 2)]
                if i != 9 :#除了第10个卷积核和前面维度一样，其他的加倍
                    out_dim*=2
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        self.features(x)
        self.view(x.size(0), -1)
        self.classifier(x)
        return x
