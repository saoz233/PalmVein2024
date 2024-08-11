import torch.nn as nn
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, Sequential, Module
import torch, sys
from torchsummary import summary
import torch.nn.functional as F
sys.path.append('..')
from model.model_layers.attn_layer import *

class Flatten(Module):
    # return [bs, all_dim]
    def forward(self, x):
        return x.view(x.size(0), -1)
    
class Hswish(nn.Module):
    def __init__(self, num_parameters = 0):
        super(Hswish, self).__init__()
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out
    
class ConvBlock(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, activation = 'prelu'):
        super(ConvBlock, self).__init__()
        self.layers = nn.Sequential(
            Conv2d(in_c, out_c, kernel, groups=groups, stride=stride, padding=padding, bias=False),
            BatchNorm2d(num_features=out_c)
        )
        if activation == 'prelu':
            self.act = PReLU(num_parameters=out_c)
        elif activation == 'hswish':
            self.act = Hswish()
        else:
            self.act = nn.ReLU()

    def forward(self, x):
        x = self.layers(x)
        return self.act(x)


class LinearBlock(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(LinearBlock, self).__init__()
        self.layers = nn.Sequential(
            Conv2d(in_c, out_c, kernel, stride, padding, groups=groups, bias=False),
            BatchNorm2d(num_features=out_c)
        )

    def forward(self, x):
        return self.layers(x)

# 逆残差块
class DepthWise(Module):
    def __init__(self, in_c, out_c, residual=False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1, 
                 activation = 'prelu', attn_block = nn.Module|None):
        super(DepthWise, self).__init__()
        self.residual = residual
        self.attn = attn_block

        layers = []
        # 1x1 conv 降维
        layers += [ConvBlock(in_c, out_c=groups, kernel=(1, 1), padding=(0, 0), stride=(1, 1), activation=activation)]
        # depthwise conv
        layers += [ConvBlock(groups, groups, groups=groups, kernel=kernel, padding=padding, stride=stride, activation=activation)]
        if self.attn is not None:
            layers += [attn_block(groups, 16)]
        # pointwise conv
        layers += [LinearBlock(groups, out_c, kernel=(1, 1), padding=(0, 0), stride=(1, 1))]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        short_cut = None
        if self.residual:
            short_cut = x
        x = self.layers(x)
        if self.residual:
            output = short_cut + x
        else:
            output = x
        return output

# 只是把逆残差块封装成多个而已，但是降不了维
class Residual(Module):
    def __init__(self, c, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1), 
                 activation = 'prelu', attn_block = nn.Module|None):
        super(Residual, self).__init__()
        modules = []
        for _ in range(num_block):
            modules.append(DepthWise(c, c, True, kernel, stride, padding, groups, activation, attn_block))
        self.layers = Sequential(*modules)

    def forward(self, x):
        return self.layers(x)
    
class GDC(Module):
    def __init__(self, channel, embedding_size):
        super(GDC, self).__init__()
        # 7x7 depthwise conv, 其实图像大小就是7x7
        self.lb = LinearBlock(channel, channel, groups=channel, kernel=(7, 7), stride=(1, 1), padding=(0, 0))
        self.flatten = Flatten()
        self.fc = Linear(channel, embedding_size, bias=False)
        self.norm = BatchNorm1d(embedding_size)

    def forward(self, x):
        x = self.lb(x)
        x = self.flatten(x)
        x = self.fc(x)

        return self.norm(x)
    
'''
####################################################
class Conv_bn_act(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, activation_layer = None):
        if activation_layer == None:
            activation_layer = nn.PReLU
        super(Conv_bn_act, self).__init__()
        self.layers = nn.Sequential(
            Conv2d(in_c, out_c, kernel, groups=groups, stride=stride, padding=padding, bias=False),
            BatchNorm2d(num_features=out_c),
            activation_layer(num_parameters=out_c)
        )

    def forward(self, x):
        return self.layers(x)

    
class DepthWise(Module):
    def __init__(self, in_c, out_c, residual=True, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1, activation_layer = None):
        super(DepthWise, self).__init__()
        self.residual = residual and (in_c==out_c) and (stride==(1,1))
        self.layers = nn.Sequential(
            # 1x1 conv 降维
            # group 相当于中间expand的量
            Conv_bn_act(in_c, out_c=groups, kernel=(1, 1), padding=(0, 0), stride=(1, 1), activation_layer = activation_layer),
            # depthwise conv
            Conv_bn_act(groups, groups, groups=groups, kernel=kernel, padding=padding, stride=stride, activation_layer = activation_layer),
            # pointwise conv
            Conv_bn_act(groups, out_c, kernel=(1, 1), padding=(0, 0), stride=(1, 1), activation_layer = nn.Identity)
        )

    def forward(self, x):
        short_cut = self.layers(x)
        if self.residual:
            short_cut += x
        return short_cut



class Residual_Module(Module):
    def __init__(self, in_channel, out_channel, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1), activation_layer = None):
        super(Residual_Module, self).__init__()
        modules = []
        for _ in range(num_block):
            modules.append(DepthWise(in_channel, out_channel, True, kernel, stride, padding, groups, activation_layer = activation_layer))
        self.layers = Sequential(*modules)

    def forward(self, x):
        return self.layers(x)
        

class GDC(Module):
    #global pooling

    def __init__(self, channel, embedding_size):
        super(GDC, self).__init__()
        self.layers = nn.Sequential(
            # 7x7 depthwise conv, 其实图像大小就是7x7
            Conv_bn_act(channel, channel, groups=channel, kernel=(7, 7), stride=(1, 1), padding=(0, 0), activation_layer=nn.Identity),
            Flatten(),
            Linear(512, embedding_size, bias=False),
            BatchNorm1d(embedding_size))

    def forward(self, x):
        return self.layers(x)
'''