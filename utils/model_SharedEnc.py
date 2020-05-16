import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, in_ch, out_ch, task_num, mode_cond='vec'):
        super(Model, self).__init__()
        # Initial convolution layers
        self.conv1 = ConvLayer(in_ch, 32, kernel_size=9, stride=1)
        #self.film1 = film(32, task_num, mode_cond)
        self.in1 = nn.InstanceNorm2d(32, affine=True)
        
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        #self.film2 = film(64, task_num, mode_cond)
        self.in2 = nn.InstanceNorm2d(64, affine=True)
        
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        #self.film3 = film(128, task_num, mode_cond)
        self.in3 = nn.InstanceNorm2d(128, affine=True)
        
        # Residual layers
        self.res0 = Res(128)
        self.res1 = Res(128)
        self.res2 = Res(128)
        self.res3 = Res(128)
        self.res4 = Res(128)
        self.res5 = Res(128)
        self.res6 = Res(128)

        # Upsampling Layers
        self.dec0 = Decoder()
        self.dec1 = Decoder()
        self.dec2 = Decoder()
        self.dec3 = Decoder()
        self.dec4 = Decoder()
        self.dec5 = Decoder()
        self.dec6 = Decoder()

        # Non-linearities
        self.relu = torch.nn.ReLU()

    def forward(self, X, taskmap):
        idx = torch.where(taskmap[0]==1)
        if len(idx[0])==0:
            task = 0
        else:
            task = idx[0][0] + 1
        y = self.relu(self.in1(self.conv1(X)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.relu(self.in3(self.conv3(y)))
        if task==0:
            y = self.res0(y)
            y = self.dec0(y)
        elif task==1:
            y = self.res1(y)
            y = self.dec1(y)
        elif task==2:
            y = self.res2(y)
            y = self.dec2(y)
        elif task==3:
            y = self.res3(y)
            y = self.dec3(y)
        elif task==4:
            y = self.res4(y)
            y = self.dec4(y)
        elif task==5:
            y = self.res5(y)
            y = self.dec5(y)
        elif task==6:
            y = self.res6(y)
            y = self.dec6(y)
        return y


class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class ResidualBlock(torch.nn.Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = nn.InstanceNorm2d(channels, affine=True)
        
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = nn.InstanceNorm2d(channels, affine=True)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        return out
    
class Res(torch.nn.Module):
    def __init__(self, channels):
        super(Res, self).__init__()
        self.res1 = ResidualBlock(channels)
        self.res2 = ResidualBlock(channels)
        self.res3 = ResidualBlock(channels)
        self.res4 = ResidualBlock(channels)
        self.res5 = ResidualBlock(channels)

    def forward(self, x):
        y = self.res1(x)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)
        return y


class UpsampleConvLayer(torch.nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = torch.nn.functional.interpolate(x_in, mode='nearest', scale_factor=self.upsample)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out
    
class Decoder(torch.nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.in1 = nn.InstanceNorm2d(64, affine=True)
        
        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.in2 = nn.InstanceNorm2d(32, affine=True)
        
        self.deconv3 = ConvLayer(32, 3, kernel_size=9, stride=1)
        
        self.relu = torch.nn.ReLU()
        
    def forward(self, x):
        y = self.relu(self.in1(self.deconv1(x)))
        y = self.relu(self.in2(self.deconv2(y)))
        y = self.deconv3(y)
        return y