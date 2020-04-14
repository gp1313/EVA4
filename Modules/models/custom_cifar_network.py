import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3) # 30
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, 3) # 28
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3) # 26
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 32, 1) # 26
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(32)

        self.maxpool1 = nn.MaxPool2d(2) # 13
        
        self.conv5 = nn.Conv2d(32, 64, 3) # 11
        self.relu5 = nn.ReLU()
        self.bn5 = nn.BatchNorm2d(64)

        self.conv6 = nn.Conv2d(64, 128, 3) # 9
        self.relu6 = nn.ReLU()
        self.bn6 = nn.BatchNorm2d(128)

        self.conv7 = nn.Conv2d(128, 32, 1) # 9
        self.relu7 = nn.ReLU()
        self.bn7 = nn.BatchNorm2d(32)

        self.conv8 = nn.Conv2d(32, 64, 3) # 7
        self.relu8 = nn.ReLU()
        self.bn8 = nn.BatchNorm2d(64)

        self.conv9 = nn.Conv2d(64, 128, 3) # 5
        self.relu9 = nn.ReLU()
        self.bn9 = nn.BatchNorm2d(128)

        self.conv10 = nn.Conv2d(128, 10, 1) # 5
        
        self.gap = nn.AdaptiveAvgPool2d(1) # 1
        
    def forward(self, x):
        
        x = self.bn1(self.relu1(self.conv1(x)))
        
        x = self.bn2(self.relu2(self.conv2(x)))
        x = self.bn3(self.relu3(self.conv3(x)))
        x = self.bn4(self.relu4(self.conv4(x)))

        x = self.maxpool1(x)
        
        x = self.bn5(self.relu5(self.conv5(x)))
        x = self.bn6(self.relu6(self.conv6(x)))
        x = self.bn7(self.relu7(self.conv7(x)))
        
        x = self.bn8(self.relu8(self.conv8(x)))
        x = self.bn9(self.relu9(self.conv9(x)))
        
        x = self.conv10(x)
        x = self.gap(x)
        
        x = x.view(-1, 10)
        
        return x