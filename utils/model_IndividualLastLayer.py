import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, in_ch, out_ch, task_num, mode_cond='map', fc=1, fc_nc=64):
        super(Model, self).__init__()
        self.mode_cond = mode_cond
        self.nc_list = [32, 64, 128, 
                        128, 128, 128, 128, 128, 
                        128, 128, 128, 128, 128, 
                        64, 32]
        if self.mode_cond != 'single':
            self.film_generator = film_generator(sum(self.nc_list), task_num, mode_cond, fc, fc_nc)
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
        
        self.lastconv = nn.ModuleList([ConvLayer(32, out_ch, kernel_size=9, stride=1)])
        for _ in range(task_num-1):
            self.lastconv.append(ConvLayer(32, out_ch, kernel_size=9, stride=1))
        
        # Non-linearities
        self.relu = torch.nn.ReLU()
        
        

    def forward(self, X, taskmap):
        if self.mode_cond != 'single':
            factor, bias = self.film_generator(taskmap)
            factor_list, bias_list = [], []
            s = 0
            for nc in self.nc_list:
                factor_list.append(factor[:,s:s+nc,:,:])
                bias_list.append(bias[:,s:s+nc,:,:])
                s += nc
        else:
            factor_list = [0] * sum(self.nc_list)
            bias_list = [0] * sum(self.nc_list)
            
        y = self.relu(self.film1(self.conv1(X), factor_list[0], bias_list[0]))
        y = self.relu(self.film2(self.conv2(y), factor_list[1], bias_list[1]))
        y = self.relu(self.film3(self.conv3(y), factor_list[2], bias_list[2]))
        y = self.res1(y, factor_list[3], bias_list[3], factor_list[4], bias_list[4])
        y = self.res2(y, factor_list[5], bias_list[5], factor_list[6], bias_list[6])
        y = self.res3(y, factor_list[7], bias_list[7], factor_list[8], bias_list[8])
        y = self.res4(y, factor_list[9], bias_list[9], factor_list[10], bias_list[10])
        y = self.res5(y, factor_list[11], bias_list[11], factor_list[12], bias_list[12])
        y = self.relu(self.film4(self.deconv1(y), factor_list[13], bias_list[13]))
        y = self.relu(self.film5(self.deconv2(y), factor_list[14], bias_list[14]))
        
        if len(np.where(taskmap[0].cpu().numpy()==1)) == 1:
            task_idx = 0
        else:
            task_idx = np.where(taskmap[0].cpu().numpy()==1)[0][0] + 1
        y = self.lastconv[task_idx](y)
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

    def forward(self, x, factor1, bias1, factor2, bias2):
        residual = x
        out = self.relu(self.film1(self.conv1(x), factor1, bias1))
        out = self.film2(self.conv2(out), factor2, bias2)
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
    
class film_generator(nn.Module):
    def __init__(self, norm_nc, cond_nc, mode_cond='vec', fc=1, fc_nc=64):
        super().__init__()
        
        self.mode_cond = mode_cond
        self.fc = fc
        self.relu = torch.nn.ReLU()
        if mode_cond=='vec':
            if self.fc==1:
                self.transform = nn.Linear(cond_nc, norm_nc*2)
            if self.fc==3:
                self.transform1 = nn.Linear(cond_nc, fc_nc)
                self.transform2 = nn.Linear(fc_nc, fc_nc)
                self.transform = nn.Linear(fc_nc, norm_nc*2)
            if self.fc==5:
                self.transform1 = nn.Linear(cond_nc, fc_nc)
                self.transform2 = nn.Linear(fc_nc, fc_nc)
                self.transform3 = nn.Linear(fc_nc, fc_nc)
                self.transform4 = nn.Linear(fc_nc, fc_nc)
                self.transform = nn.Linear(fc_nc, norm_nc*2)
        elif mode_cond=='map':
            self.transform = nn.Conv2d(cond_nc, norm_nc*2, kernel_size=1)
        self.transform.bias.data[:norm_nc] = 1
        self.transform.bias.data[norm_nc:] = 0
        
    def forward(self, cond):
        if self.mode_cond=='vec':
            if self.fc==1:
                param = self.transform(cond).unsqueeze(2).unsqueeze(3)
            if self.fc==3:
                param = self.relu(self.transform1(cond))
                param = self.relu(self.transform2(param))
                param = self.transform(param).unsqueeze(2).unsqueeze(3)
            if self.fc==5:
                param = self.relu(self.transform1(cond))
                param = self.relu(self.transform2(param))
                param = self.relu(self.transform3(param))
                param = self.relu(self.transform4(param))
                param = self.transform(param).unsqueeze(2).unsqueeze(3)
                
        elif self.mode_cond=='map':
            cond = F.interpolate(cond, size=x.size()[2:], mode='nearest')
            param = self.transform(cond)
        factor, bias = param.chunk(2, 1)
        return factor, bias
    
    
    
class film(nn.Module):
    def __init__(self, norm_nc, cond_nc, mode_cond='map'):
        super().__init__()
        
        self.mode_cond = mode_cond
        if self.mode_cond=='single':
            self.norm = nn.InstanceNorm2d(norm_nc, affine=True)
        else:
            self.norm = nn.InstanceNorm2d(norm_nc, affine=False)
            
    def forward(self, x, factor, bias):
        # Part 1. generate parameter-free normalized activations
        normalized = self.norm(x)
        
        # Part 2. produce scaling and bias conditioned on semantic map
        if self.mode_cond=='single':
            out = normalized
        else:
            # apply scale and bias
            out = normalized * factor + bias

        return out