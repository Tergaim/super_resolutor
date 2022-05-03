import torch.nn as nn
import torch
import torch.nn.functional as F

class SRCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1,128, kernel_size=9, padding=2, padding_mode="replicate")
        self.conv2 = nn.Conv2d(128,64, kernel_size=1, padding=2, padding_mode="replicate")
        self.conv3 = nn.Conv2d(64,1, kernel_size=5, padding=2, padding_mode="replicate")

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x

class VDSR(nn.Module):
    def __init__(self, stages=20):
        super().__init__()

        self.convtower = nn.ModuleList()
        for i in range(stages):
            self.convtower.append(nn.Conv2d(64,64, kernel_size=3, padding=1, padding_mode="zeros"))
        self.convfirst = nn.Conv2d(1,64, kernel_size=1, padding=1, padding_mode="zeros")
        self.convlast = nn.Conv2d(64,1, kernel_size=3, padding=1, padding_mode="zeros")

    def forward(self, x):
        x = F.relu(self.convfirst(x))
        for layer in self.convtower:
            x = F.relu(layer(x))
        x = self.convlast(x)
        return x

class Residual(nn.Module):
    def __init__(self):
        super().__init__()

        self.SRCNN1 = SRCNN()
        self.SRCNN2 = SRCNN()
        self.SRCNN3 = SRCNN()

    def forward(self, x):
        x = self.SRCNN1(x)+x
        x = self.SRCNN2(x)+x
        x = self.SRCNN3(x)
        return x

class SuperResolutorUpscaled(nn.Module):
    def __init__(self):
        super().__init__()

        self.residualr = Residual()
        self.residualg = Residual()
        self.residualb = Residual()
        # self.convfinal = nn.Conv2d(3,3, kernel_size=1)

    def forward(self, x):
        r, g, b = torch.tensor_split(x,3, dim=1)
        r = self.residualr(r)
        g = self.residualg(g)
        b = self.residualb(b)
        x = torch.cat((r,g,b), dim=1)
        return x