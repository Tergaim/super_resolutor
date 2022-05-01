import torch.nn as nn
import torch.nn.functional as F

class SuperResolutor(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3,64, kernel_size=13, padding=4, padding_mode="replicate")
        self.conv2 = nn.Conv2d(64,32, kernel_size=1, padding=2, padding_mode="replicate")
        self.conv3 = nn.Conv2d(32,3, kernel_size=7, padding=3, padding_mode="replicate")

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x