import torch.nn as nn
import torch
import torch.nn.functional as F


class FSRCNN(nn.Module):
    def __init__(self, d=56, s=12, m=4, k=2):
        super().__init__()

        self.prelu = nn.PReLU()
        self.feature_extract = nn.Conv2d(1, d, kernel_size=5, padding=2, padding_mode="replicate")
        self.shrink = nn.Conv2d(d, s, kernel_size=1)
        self.map = nn.ModuleList()
        for i in range(m):
            self.map.append(nn.Conv2d(s, s, kernel_size=3, padding=1, padding_mode="zeros"))
        self.expand = nn.Conv2d(s, d, kernel_size=1)
        self.deconv = nn.ConvTranspose2d(d, 1, kernel_size=9, stride=k)

    def forward(self, x):
        x = self.prelu(self.feature_extract(x))
        x = self.prelu(self.shrink(x))

        for layer in self.map:
            x = self.prelu(layer(x))

        x = self.prelu(self.expand(x))
        x = self.deconv(x)
        return x
