import torch
from study_pytorch.NeutralModule.FPN import FPN

Inception = FPN([3, 4, 6, 3]).cuda()
print(Inception)

