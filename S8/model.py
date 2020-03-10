
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        

        # CONVOLUTION BLOCK 1
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        ) # output_size = 30, rf = 3

        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        ) # output_size = 30, rf = 5

        # TRANSITION BLOCK 1
        self.pool1 = nn.MaxPool2d(2, 2) 
        # output_size = 15, rf = 6
        
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        ) # output_size = 15, rf = 10
        
        
        # CONVOLUTION BLOCK 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(32),            
            nn.ReLU(),
        ) # output_size = 13, rf = 14
        
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        ) # output_size = 13, rf = 18
        
        # TRANSITION BLOCK 2
        self.pool2 = nn.MaxPool2d(2, 2) 
        # output_size = 6, rf = 19
        
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        ) # output_size = 6, rf = 19
        
                
        # CONVOLUTION BLOCK 3
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(32),            
            nn.ReLU(),
        ) # output_size = 4, rf = 12
        
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        ) # output_size = 4, rf = 16

        
        # TRANSITION BLOCK 3
        self.pool3 = nn.MaxPool2d(2, 2) 
        # output_size = 2, rf = 17
        
        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        ) # output_size = 2, rf = 25
        
                
        # CONVOLUTION BLOCK 4
        self.convblock10 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2, 2), padding=0, bias=False),
            nn.BatchNorm2d(32),            
            nn.ReLU(),
        ) # output_size = 1, rf = 12

        
        # TRANSITION BLOCK 4
        
        self.convblock11 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        ) # output_size = 1, rf = 25
        
        # OUTPUT BLOCK               
#         self.gap = nn.Sequential(
#             nn.AvgPool2d(kernel_size=(6, 6)),
#             # nn.ReLU()
#         ) # output_size = 1, rf = 28
        
        self.convblock12 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            #nn.BatchNorm2d(10),
            #nn.ReLU()
        ) # output_size = 1, rf = 28


    def forward(self, x):
        
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.pool1(x)
        x = self.convblock3(x)
        
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.pool2(x)
        x = self.convblock6(x)
        
        x = self.convblock7(x)
        x = self.convblock8(x)
        x = self.pool3(x)
        x = self.convblock9(x)
        
        x = self.convblock10(x)
        x = self.convblock11(x)
        
        
        x = self.convblock12(x)
        x = x.view(-1, 10)
        
        return F.log_softmax(x, dim=-1)