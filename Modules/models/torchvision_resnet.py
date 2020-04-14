import torch.nn as nn
from torchvision.models import resnet18

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.base = resnet18(pretrained=True)
        self.base.fc = nn.Linear(in_features=self.base.fc.in_features,
                                 out_features=10)
        self.layer4 = self.base.layer4
        
    def forward(self, x):
        x = self.base(x)
        x = x.view(-1, 10)
        return x