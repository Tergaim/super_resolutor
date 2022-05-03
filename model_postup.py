import torch.nn as nn
import torch
import torch.nn.functional as F


class FSRCNN(nn.Module):
    def __init__(self, d=56, s=12, m=4, k=2):
        super().__init__()

        self.prelu = nn.PReLU()
        self.feature_extract = nn.Conv2d(1, d, kernel_size=5, padding=2, padding_mode="zeros")
        self.shrink = nn.Conv2d(d, s, kernel_size=1)
        self.map = nn.ModuleList()
        for i in range(m):
            self.map.append(nn.Conv2d(s, s, kernel_size=3, padding=1, padding_mode="zeros"))
        self.expand = nn.Conv2d(s, d, kernel_size=1)
        self.deconv = nn.ConvTranspose2d(d, 1, kernel_size=9, stride=k, padding=4, padding_mode="zeros")

    def forward(self, x):
        x = self.prelu(self.feature_extract(x))
        x = self.prelu(self.shrink(x))

        for layer in self.map:
            x = self.prelu(layer(x))

        x = self.prelu(self.expand(x))
        x = self.deconv(x, output_size=(64,64))
        return x

class SuperResolutor(nn.Module):
    def __init__(self):
        super().__init__()

        self.frcnnr = FSRCNN()
        self.frcnng = FSRCNN()
        self.frcnnb = FSRCNN()
        # self.convfinal = nn.Conv2d(3,3, kernel_size=1)

    def forward(self, x):
        r, g, b = torch.tensor_split(x,3, dim=1)
        r = self.frcnnr(r)
        g = self.frcnng(g)
        b = self.frcnnb(b)
        x = torch.cat((r,g,b), dim=1)
        return x