import torch
import torch.nn as nn
import torch.nn.functional as F

class SharedConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.flatten = nn.Flatten() # 128, 3 3 - 1152

    def forward(self, x):
        x = self.conv(x)
        return self.flatten(x)



class DualHeadCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = SharedConvNet()
        self.head1 = self._make_head()
        self.head2 = self._make_head()

    def _make_head(self):
        return nn.Sequential(
            nn.Linear(128 * 3 * 3, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 10)
        )

    def forward(self, x, head=1):
        x = self.shared(x)
        return self.head1(x) if head == 1 else self.head2(x)
