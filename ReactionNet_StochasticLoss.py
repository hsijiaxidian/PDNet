
import torch
import torch.nn as nn
import numpy as np
from pytorch_wavelets.dwt.transform2d import DWTForward,DWTInverse

def conv3x3(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,
                       padding=1, dilation=1, groups=1, bias=True)

class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Block, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(in_channels, out_channels)
        self.alpha = nn.Parameter(torch.Tensor(1).fill_(1),requires_grad=True)
        self.beta = nn.Parameter(torch.Tensor(1).fill_(1),requires_grad=True)

    def forward(self, x, reaction, alpha, beta):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.alpha*torch.tanh(-alpha)*out + \
              self.beta*beta*torch.tanh(beta)*(reaction-x) + x
        return out

class Interlayers(nn.Module):
    def __init__(self, layers_num, block, med_channels):
        super(Interlayers, self).__init__()
        self.layers_num = layers_num
        self.m = torch.distributions.half_normal.HalfNormal(0)
        self.alpha1 = nn.Parameter(torch.Tensor(1).fill_(0.5),requires_grad=True)
        self.beta1 = nn.Parameter(torch.Tensor(1).fill_(0.5),requires_grad=True)
        self.c1 = nn.Parameter(torch.Tensor(1).fill_(1),requires_grad=True)
        self.cc1 = nn.Parameter(torch.Tensor(1).fill_(1),requires_grad=True)

        layers = []
        for i in range(layers_num):
            layers.append(block(med_channels,med_channels))
        self.blocks = nn.Sequential(*layers)

    def forward(self,x,reaction):
        if self.training:
            prlayers_num = torch.floor(self.layers_num-self.m.sample()).numpy().astype(np.int8)
            if prlayers_num>self.layers_num:
                prlayers_num = self.layers_num
            elif prlayers_num<=0:
                prlayers_num = 1
        else:
            prlayers_num = self.layers_num

        out = x
        for i in range(prlayers_num):
            self.blocks[i].conv1.weight.required_grad = True
            self.blocks[i].conv1.bias.required_grad = True
            self.blocks[i].conv2.weight.required_grad = True
            self.blocks[i].conv2.bias.required_grad = True
            self.blocks[i].alpha.required_grad = True
            self.blocks[i].beta.required_grad = True
            phi = torch.tanh(self.c1) * (i + 1) ** (-torch.sigmoid(self.alpha1))
            psi = torch.tanh(self.cc1) * (i + 1) ** (-torch.sigmoid(self.beta1))
            out= self.blocks[i](out, reaction, phi, psi)

        for i in range(prlayers_num,self.layers_num):
            self.blocks[i].conv1.weight.required_grad = False
            self.blocks[i].conv1.bias.required_grad = False
            self.blocks[i].conv2.weight.required_grad = False
            self.blocks[i].conv2.bias.required_grad = False
            self.blocks[i].alpha.required_grad = False
            self.blocks[i].beta.required_grad = False

        return out

class Headtails(nn.Module):
    def __init__(self, input_features, middle_features, layers_num, Intermed = Interlayers):
        super(Headtails, self).__init__()
        self.in_channels = input_features
        self.mid_channels = middle_features
        if self.in_channels == 4:
            self.out_channels = 4 #Grayscale image
        elif self.in_channels == 15:
            self.out_channels = 12 #RGB image
        else:
            raise Exception('Invalid number of input features')
        self.num_layers = layers_num
        self.conv1 = nn.Conv2d(self.in_channels, self.mid_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(self.mid_channels, self.in_channels, kernel_size=3, padding=1)
        self.inmed = Intermed(self.num_layers, Block, self.mid_channels)
        self.bn1 = nn.BatchNorm2d(self.mid_channels)
        self.bn2 = nn.BatchNorm2d(self.mid_channels)

    def forward(self,x):
        out = self.bn1(self.conv1(x))
        out = self.inmed(out,out)
        out = self.conv2(self.bn2(out))
        return out

class ReactionNet(nn.Module):
    r"""Implements the FFDNet architecture
	"""
    def __init__(self, num_input_channels,test_mode=False):
        super(ReactionNet, self).__init__()
        self.num_input_channels = num_input_channels
        self.test_mode = test_mode
        if self.num_input_channels == 1:
# Grayscale image
            self.num_feature_maps = 96
            self.num_conv_layers = 35
            self.in_channels = 4
            self.out_channels = 4
        elif self.num_input_channels == 3:
# RGB image
            self.num_feature_maps = 64
            self.num_conv_layers = 30
            self.downsampled_channels = 15
            self.output_features = 12
        else:
            raise Exception('Invalid number of input features')

        self.MainNet = Headtails(\
				input_features=self.in_channels,\
				middle_features=self.num_feature_maps, \
                layers_num=self.num_conv_layers)
        self.dwt = DWTForward(J=1, wave='haar', mode='zero').cuda()
        self.idwt = DWTInverse(wave='haar', mode='zero').cuda()

    def forward(self, x):
        Yl,Yh = self.dwt(x)
        wtfeature = torch.cat((Yl,torch.squeeze(Yh[0],1)),1)
        auxi_res = self.MainNet(wtfeature)
        IYl = auxi_res[:, 0, :, :]
        IYh = auxi_res[:, 1:4, :, :]
        IYl = torch.unsqueeze(IYl, 1)
        IYh = torch.unsqueeze(IYh, 1)
        IYhi = []
        IYhi.append(IYh.contiguous())
        pred_noise=self.idwt((IYl, IYhi))

        return pred_noise

