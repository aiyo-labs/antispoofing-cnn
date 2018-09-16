"""
AntiSpoofing Depth model
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
#from ..weights_initializer import weights_init
from torchsummary import summary

class AntiSpoofing(nn.Module):
    def __init__(self,resolution_inp = 256, resolution_op = 256, channel = 6):
        super(AntiSpoofing,self).__init__()

        self.resolution_inp = resolution_inp
        self.resolution_op = resolution_op
        self.channel = channel
        # define layers
        self.zeropad2d = nn.ZeroPad2d((1,1,1,1))
        self.elu = nn.ELU()
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=64, kernel_size=3, stride=1,bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=128)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=196, kernel_size=3, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=196)
        self.conv4 = nn.Conv2d(in_channels=196, out_channels=128, kernel_size=3, stride=1, bias=False)
        self.bn4 = nn.BatchNorm2d(num_features=128)
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, bias=False)
        self.bn5 = nn.BatchNorm2d(num_features=128)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=196, kernel_size=3, stride=1, bias=False)
        self.bn6 = nn.BatchNorm2d(num_features=196)
        self.conv7 = nn.Conv2d(in_channels=196, out_channels=128, kernel_size=3, stride=1, bias=False)
        self.bn7 = nn.BatchNorm2d(num_features=128)
        self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2)

        self.conv8 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, bias=False)
        self.bn8 = nn.BatchNorm2d(num_features=128)
        self.conv9 = nn.Conv2d(in_channels=128, out_channels=196, kernel_size=3, stride=1, bias=False)
        self.bn9 = nn.BatchNorm2d(num_features=196)
        self.conv10 = nn.Conv2d(in_channels=196, out_channels=128, kernel_size=3, stride=1, bias=False)
        self.bn10 = nn.BatchNorm2d(num_features=128)
        self.pool3 = nn.MaxPool2d(kernel_size=2,stride=2)


        self.conv11 = nn.Conv2d(in_channels=384, out_channels=128, kernel_size=3, stride=1, bias=False)
        self.bn11 = nn.BatchNorm2d(num_features=128)
        self.conv12 = nn.Conv2d(in_channels=128, out_channels=3, kernel_size=3, stride=1, bias=False)
        self.bn12 = nn.BatchNorm2d(num_features=3)
        self.conv13 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, stride=1, bias=False)
        self.bn13 = nn.BatchNorm2d(num_features=1)



    def forward(self, x):
        
        x = self.bn1(self.elu(self.conv1(self.zeropad2d(x))))
        x = self.bn2(self.elu(self.conv2(self.zeropad2d(x))))
        x = self.bn3(self.elu(self.conv3(self.zeropad2d(x))))
        x = self.bn4(self.elu(self.conv4(self.zeropad2d(x))))
        x = self.pool1(x)
        resize1 = F.interpolate(x,size=32)

        x = self.bn5(self.elu(self.conv5(self.zeropad2d(x))))
        x = self.bn6(self.elu(self.conv6(self.zeropad2d(x))))
        x = self.bn7(self.elu(self.conv7(self.zeropad2d(x))))
        x = self.pool2(x)
        resize2 = F.interpolate(x,size=32)

        x = self.bn8(self.elu(self.conv8(self.zeropad2d(x))))
        x = self.bn9(self.elu(self.conv9(self.zeropad2d(x))))
        x = self.bn10(self.elu(self.conv10(self.zeropad2d(x))))
        x = self.pool3(x)
        
        x = torch.cat((resize1,resize2,x),1)

        x = self.bn11(self.elu(self.conv11(self.zeropad2d(x))))
        x = self.bn12(self.elu(self.conv12(self.zeropad2d(x))))
        x = self.bn13(self.elu(self.conv13(self.zeropad2d(x))))

        x = torch.sigmoid(x)