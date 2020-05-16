# https://lp-tech.net/articles/hzfn7/view?page=2
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class Model_UNet(nn.Module):
    def __init__(self, in_ch, out_ch, dim_label):
        super(Model_UNet, self).__init__()
        # Initial convolution layers
        self.conv1 = ConvLayer(in_ch, 32, kernel_size=9, stride=1)
        self.in1 = Conditional_InstanceNorm2d(32)
        self.param1 = fc_make_param(dim_label, 32)
        
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2 = Conditional_InstanceNorm2d(64)
        self.param2 = fc_make_param(dim_label, 64)
        
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3 = Conditional_InstanceNorm2d(128)
        self.param3 = fc_make_param(dim_label, 128)
        
        # Residual layers
        self.res1 = ResidualBlock(128, dim_label)
        self.res2 = ResidualBlock(128, dim_label)
        self.res3 = ResidualBlock(128, dim_label)
        self.res4 = ResidualBlock(128, dim_label)
        self.res5 = ResidualBlock(128, dim_label)
        
        # Upsampling Layers
        self.deconv1 = UpsampleConvLayer(256, 64, kernel_size=3, stride=1, upsample=2)
        self.in4 = Conditional_InstanceNorm2d(64)
        self.param4 = fc_make_param(dim_label, 64)
        
        self.deconv2 = UpsampleConvLayer(128, 32, kernel_size=3, stride=1, upsample=2)
        self.in5 = Conditional_InstanceNorm2d(32)
        self.param5 = fc_make_param(dim_label, 32)
        
        self.deconv3 = ConvLayer(64, out_ch, kernel_size=9, stride=1)
        # Non-linearities
        self.relu = torch.nn.ReLU()
        
        

    def forward(self, x, label):
        x1 = self.relu(self.in1(self.conv1(x), self.param1(label)))
        x2 = self.relu(self.in2(self.conv2(x1), self.param2(label)))
        x3 = self.relu(self.in3(self.conv3(x2), self.param3(label)))
        x = self.res1(x3, label)
        x = self.res2(x, label)
        x = self.res3(x, label)
        x = self.res4(x, label)
        x = self.res5(x, label)
        x = self.relu(self.in4(self.deconv1(x, x3), self.param4(label)))
        x = self.relu(self.in5(self.deconv2(x, x2), self.param5(label)))
        x = torch.cat([x, x1], dim=1)
        x = self.deconv3(x)
        return x

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

    def __init__(self, channels, dim_label):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = Conditional_InstanceNorm2d(channels)
        self.param1 = fc_make_param(dim_label, channels)
        
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = Conditional_InstanceNorm2d(channels)
        self.param2 = fc_make_param(dim_label, channels)
        self.relu = torch.nn.ReLU()

    def forward(self, x, label):
        residual = x
        out = self.relu(self.in1(self.conv1(x), self.param1(label)))
        out = self.in2(self.conv2(out), self.param2(label))
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

    def forward(self, x1,x2):
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        
        if self.upsample:
            x = torch.nn.functional.interpolate(x, mode='nearest', scale_factor=self.upsample)
        x = self.reflection_pad(x)
        
        out = self.conv2d(x)
        return out

    
class SLinear(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(SLinear, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        return self.linear(x)

    
class fc_make_param(nn.Module):
    def __init__(self, dim_latent, n_channel):
        super(fc_make_param, self).__init__()
        self.transform = SLinear(dim_latent, n_channel * 2)
        # "the biases associated with ys that we initialize to one"
        self.transform.linear.bias.data[:n_channel] = 1
        self.transform.linear.bias.data[n_channel:] = 0

    def forward(self, w):
        # Gain scale factor and bias with:
        param = self.transform(w).unsqueeze(2).unsqueeze(3)
        return param

    
class Conditional_InstanceNorm2d(nn.Module):
    def __init__(self, in_channels):
        super(Conditional_InstanceNorm2d, self).__init__()
        self.norm = nn.InstanceNorm2d(in_channels, affine=False)

    def forward(self, x, param):
        factor, bias = param.chunk(2, 1)
        result = self.norm(x)
        result = result * factor + bias  
        return result