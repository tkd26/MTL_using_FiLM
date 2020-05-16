import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

import csv

class Model(nn.Module):
    def __init__(self, in_ch, out_ch, task_num, mode_cond='map', show_feature=True):
        super(Model, self).__init__()
        
        self.show_feature = show_feature
        
        # Initial convolution layers
        self.conv1 = ConvLayer(in_ch, 32, kernel_size=9, stride=1)
        self.film1 = film(32, task_num, mode_cond)
        
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.film2 = film(64, task_num, mode_cond)
        
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.film3 = film(128, task_num, mode_cond)
        
        # Residual layers
        self.res1 = ResidualBlock(128, task_num, mode_cond)
        self.res2 = ResidualBlock(128, task_num, mode_cond)
        self.res3 = ResidualBlock(128, task_num, mode_cond)
        self.res4 = ResidualBlock(128, task_num, mode_cond)
        self.res5 = ResidualBlock(128, task_num, mode_cond)
        
        # Upsampling Layers
        self.deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.film4 = film(64, task_num, mode_cond)
        
        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.film5 = film(32, task_num, mode_cond)
        
        self.deconv3 = ConvLayer(32, out_ch, kernel_size=9, stride=1)
        # Non-linearities
        self.relu = torch.nn.ReLU()
        
        

    def forward(self, X, taskmap):
        y = self.relu(self.film1(self.conv1(X), taskmap))
        #print('conv1', y.shape)
        y = self.relu(self.film2(self.conv2(y), taskmap))
        #print('conv2', y.shape)
        y = self.relu(self.film3(self.conv3(y), taskmap))
        #if self.show_feature:
        #    with open('./digit_data/conv3.csv', 'a') as f:
        #        writer = csv.writer(f)
        #        writer.writerow(y.view(y.size()[0], -1).cpu().numpy()[0])
        #print('conv3', y.shape)
        y = self.res1(y, taskmap)
        #print('res1', y.shape)
        y = self.res2(y, taskmap)
        #print('res2', y.shape)
        y = self.res3(y, taskmap)
        #print('res3', y.shape)
        y = self.res4(y, taskmap)
        #print('res4', y.shape)
        y = self.res5(y, taskmap)
        #print('res5', y.shape)
        y = self.relu(self.film4(self.deconv1(y), taskmap))
        #print('deconv1', y.shape)
        y = self.relu(self.film5(self.deconv2(y), taskmap))
        #print('deconv2', y.shape)
        y = self.deconv3(y)
        #print('deconv3', y.shape)
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

    def __init__(self, channels, task_num, mode_cond):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.film1 = film(channels, task_num, mode_cond)
        
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.film2 = film(channels, task_num, mode_cond)
        self.relu = torch.nn.ReLU()

    def forward(self, x, taskmap):
        residual = x
        out = self.relu(self.film1(self.conv1(x), taskmap))
        #print('res1',out.shape)
        out = self.film2(self.conv2(out), taskmap)
        #print('res2',out.shape)
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

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = torch.nn.functional.interpolate(x_in, mode='nearest', scale_factor=self.upsample)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out
    
class film(nn.Module):
    def __init__(self, norm_nc, cond_nc, mode_cond='map'):
        super().__init__()
        
        self.mode_cond = mode_cond
        if self.mode_cond=='single':
            self.norm = nn.InstanceNorm2d(norm_nc, affine=True)
        else:
            self.norm = nn.InstanceNorm2d(norm_nc, affine=False)
            if mode_cond=='vec':
                self.transform = nn.Linear(cond_nc, norm_nc*2)
            elif mode_cond=='map':
                self.transform = nn.Conv2d(cond_nc, norm_nc*2, kernel_size=1)
            self.transform.bias.data[:norm_nc] = 1
            self.transform.bias.data[norm_nc:] = 0
        
    def forward(self, x, cond):
        # Part 1. generate parameter-free normalized activations
        normalized = self.norm(x)
        
        # Part 2. produce scaling and bias conditioned on semantic map
        if self.mode_cond=='single':
            out = normalized
        else:
            if self.mode_cond=='vec':
                param = self.transform(cond).unsqueeze(2).unsqueeze(3)
            elif self.mode_cond=='map':
                cond = F.interpolate(cond, size=x.size()[2:], mode='nearest')
                param = self.transform(cond)
            factor, bias = param.chunk(2, 1)

            # apply scale and bias
            out = normalized * factor + bias

        return out