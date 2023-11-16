import torch.nn as nn
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_max1 = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=7, padding=3),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv_max2 = nn.Sequential(
            nn.Conv2d(24, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv_max3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*3*3, 128),
            nn.ReLU(),
            nn.Linear(128, 5)
        )
        self.pool = nn.AvgPool2d(2)
    def forward(self, x):
        # print(x.size())
        out = self.conv_max1(x)
        # print(out.size())
        out = self.conv_max2(out)
        # out = self.conv_max3(out)
        out = self.fc(self.pool(out)+self.conv_max3(out))
        return out
