import torch.nn as nn
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, Sequential, Module
import torch
from torchsummary import summary
import math
from torch.nn import Parameter
import torch.nn.functional as F
from model.model_layers.model_layer import *
from model.model_layers.attn_layer import *
from model.model_layers.loss_layer import *

class MobileFaceNet(Module):
    def __init__(self, fp16=False, num_features=128, blocks=(1, 4, 6, 2), scale=2, activation = 'prelu', attn_block = SEblock):
        super(MobileFaceNet, self).__init__()
        self.scale = scale
        self.fp16 = fp16
        self.layers = nn.ModuleList()
        self.layers.append(
            ConvBlock(3, 64 * self.scale, kernel=(3, 3), stride=(2, 2), padding=(1, 1), activation=activation)
        )
        if blocks[0] == 1:
            self.layers.append(
                ConvBlock(64 * self.scale, 64 * self.scale, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, activation=activation)
            )
        else:
            self.layers.append(
                Residual(64 * self.scale, num_block=blocks[0], groups=128, kernel=(3, 3), stride=(1, 1), padding=(1, 1), 
                         activation=activation, attn_block=None),
            )
        
        self.layers.extend(
        [
            DepthWise(64 * self.scale, 64 * self.scale, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=128),
            Residual(64 * self.scale, num_block=blocks[1], groups=128, kernel=(3, 3), stride=(1, 1), padding=(1, 1)),
            
            DepthWise(64 * self.scale, 128 * self.scale, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=256, 
                      activation=activation, attn_block=None),
            Residual(128 * self.scale, num_block=blocks[2], groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1), 
                      activation=activation, attn_block=attn_block),
            
            DepthWise(128 * self.scale, 128 * self.scale, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=512, 
                      activation=activation, attn_block=None),
            Residual(128 * self.scale, num_block=blocks[3], groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1), 
                      activation=activation, attn_block=attn_block),
        ])

        self.conv_sep = ConvBlock(128 * self.scale, num_features* self.scale, kernel=(1, 1), stride=(1, 1), padding=(0, 0),
                                  activation=activation)
        self.features = GDC(num_features* self.scale, num_features)
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        with torch.cuda.amp.autocast(self.fp16):
            for func in self.layers:
                x = func(x)
        x = self.conv_sep(x.float() if self.fp16 else x)
        x = self.features(x)
        return x


class Full_Model(nn.Module):
    def __init__(self, fp16, num_features, blocks=(1, 4, 6, 2), scale=2, num_class=600):
        super(Full_Model, self).__init__()
        self.feature = get_mbf(fp16, num_features, blocks, scale=scale)
        # self.soft = Linear_Softmax(num_features, num_class)
        self.ArcMargin = ArcMarginProduct(num_features, num_class, s=32.0, m=0.50)

    def forward(self, img, y=None, classifier=None):
        x = self.feature(img)
        if classifier == 'arc' and y:
            x = self.ArcMargin(x, y)
        else:
            x = x
            # x = self.soft(x,y)
        return x



def get_mbf(fp16, num_features, blocks=(1, 4, 6, 2), scale=2):
    return MobileFaceNet(fp16, num_features, blocks, scale=scale, 
                         activation='hswish', attn_block=SEblock)



if __name__ == "__main__":
    model = get_mbf(False, 512)
    model.eval().cpu()
    print(model)
    summary(model, input_size=(3, 112, 112), batch_size=-1, device="cpu")