import torch
import torch.nn as nn
import numpy as np


class SymModel(nn.Module):
    def __init__(self, in_ch, out_ch, dim_label, bias=False):
        super(SymModel, self).__init__()
        # Initial convolution layers
        self.conv1 = ConvLayer(in_ch, 32, kernel_size=9, stride=1)
        self.gate_c1 = GateLayer(32, dim_label, bias)
        
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.gate_c2 = GateLayer(64, dim_label, bias)
        
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.gate_c3 = GateLayer(128, dim_label, bias)

        # Residual layers
        self.res1 = ResidualBlock(128)
        self.gate_r1 = GateLayer(128, dim_label, bias)
        
        self.res2 = ResidualBlock(128)
        self.gate_r2 = GateLayer(128, dim_label, bias)
        
        self.res3 = ResidualBlock(128)
        self.gate_r3 = GateLayer(128, dim_label, bias)
        
        self.res4 = ResidualBlock(128)
        self.gate_r4 = GateLayer(128, dim_label, bias)
        
        self.res5 = ResidualBlock(128)
        self.gate_r5 = GateLayer(128, dim_label, bias)
        
        # Upsampling Layers
        self.deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.gate_dc1 = GateLayer(64, dim_label, bias)
        
        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.gate_dc2 = GateLayer(32, dim_label, bias)
        
        self.deconv3 = ConvLayer(32, out_ch, kernel_size=9, stride=1)
        # Non-linearities
        self.relu = torch.nn.ReLU()
        
        

    def forward(self, X, label):
        y = self.relu(self.conv1(X))
        y = self.gate_c1(y, label)
        y = self.relu(self.conv2(y))
        y = self.gate_c2(y, label)
        y = self.relu(self.conv3(y))
        y = self.gate_c3(y, label)
        
        y = self.res1(y)
        y = self.gate_r1(y, label)
        y = self.res2(y)
        y = self.gate_r2(y, label)
        y = self.res3(y)
        y = self.gate_r3(y, label)
        y = self.res4(y)
        y = self.gate_r4(y, label)
        y = self.res5(y)
        y = self.gate_r5(y, label)

        y = self.relu(self.deconv1(y))
        y = self.gate_dc1(y, label)
        y = self.relu(self.deconv2(y))
        y = self.gate_dc2(y, label)
        y = self.deconv3(y)
        return y


class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.norm = nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        out = self.norm(out)
        return out


class ResidualBlock(torch.nn.Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.norm = nn.InstanceNorm2d(channels)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.norm(self.conv1(x)))
        out = self.norm(self.conv2(out))
        out = out + residual
        return out


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
        self.norm = nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = torch.nn.functional.interpolate(x_in, mode='nearest', scale_factor=self.upsample)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        out = self.norm(out)
        return out
    
    
def GateLayer(channel, cond_size, bias):
    if bias:
        return GateLayer_ScaleBias(channel, cond_size)
    else:
        return GateLayer_Scale(channel, cond_size)
    
    
class GateLayer_Scale(nn.Module): #avg pool
    def __init__(self, channel, cond_size, reduction=4):
        super(GateLayer_Scale, self).__init__()
        self.relu = torch.nn.ReLU()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel*2, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )
        
        self.l_c = nn.Linear(cond_size, channel)
        self.channel = channel
        
    def forward(self, x, cond):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        
        cond = self.relu(self.l_c(cond))
        y = torch.cat([y, cond], 1)

        y = self.fc(y).view(b, self.channel, 1, 1)
        y = y * x
        return y
    

class GateLayer_ScaleBias(nn.Module): #avg pool
    def __init__(self, channel, cond_size, reduction=4):
        super(GateLayer_ScaleBias, self).__init__()
        self.relu = torch.nn.ReLU()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel*2, (channel // reduction)*2),
            nn.ReLU(inplace=True),
            nn.Linear((channel // reduction)*2, channel*2),
        )
        self.sigmoid = nn.Sigmoid()
        self.l_c = nn.Linear(cond_size, channel)
        self.channel = channel
        
    def forward(self, x, cond):
        b, c, h, w = x.size()
        y = self.avg_pool(x).view(b, c)
        
        cond = self.relu(self.l_c(cond))
        y = torch.cat([y, cond], 1)
        y = self.fc(y)
        
        scale, bias = y.chunk(2, 1)
        scale = self.sigmoid(scale).view(b, self.channel, 1, 1)
        bias = bias.view(b, self.channel, 1, 1).repeat(1, 1, h, w)
        y = x * scale + bias
        return y
    