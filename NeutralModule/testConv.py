import torch
import torch.nn as nn

input = torch.randn(1, 1, 6, 6)
a = nn.Conv2d(1, 1, 3, 2, bias=False)
output = a(input)
print(output.shape)