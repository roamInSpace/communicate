import torch
import torch.nn as nn

class CNN1(nn.Module):
    def __init__(self, channel, cla_num):
        super(CNN1, self).__init__()

        self.cla_num = cla_num

        self.conv1 = nn.Sequential(
            nn.Conv2d(channel, 32, 5, 1, 2), 
            nn.BatchNorm2d(32), 
            nn.ReLU(), 
            nn.MaxPool2d(2), 
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5, 1, 2), 
            nn.BatchNorm2d(64), 
            nn.ReLU(), 
            nn.MaxPool2d(2), 
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 5, 1, 2), 
            nn.BatchNorm2d(128), 
            nn.ReLU(), 
            nn.MaxPool2d(2), 
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 5, 1, 2), 
            nn.BatchNorm2d(256), 
            nn.ReLU(), 
            nn.MaxPool2d(2), 
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, 5, 1, 2), 
            nn.BatchNorm2d(512), 
            nn.ReLU(), 
            nn.MaxPool2d(2), 
        )

        self.out = nn.Sequential(
            nn.Linear(512*7*7, 5000), 
            nn.ReLU(), 
            nn.Linear(5000, cla_num), 
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output