import sys
sys.path.append('..')
import torch.nn as nn
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, Sequential, Module
import torch
from model_layers.model_layer import *
from torchsummary import summary

class Layer_congif:
    def __init__(self,
                 in_ch, ex_ch, out_ch, num_block, stride) -> None:
        self.in_ch = in_ch
        self.ex_ch = ex_ch
        self.out_ch = out_ch
        self.num_block = num_block
        self.stride = stride

class MobileFaceNet(Module):
    def __init__(self, fp16=False, net_para = list[Layer_congif], num_features=128):
        super(MobileFaceNet, self).__init__()
        self.net_para = net_para
        #self.scale = scale
        self.fp16 = fp16
        self.layers = nn.ModuleList()
        
        first_l = net_para[0]
        if first_l.in_ch != 3:
            first_l.in_ch = 3
        self.layers.append(
            # [bs, 3, m,n] -> [bs, 64*s, m/2,n/2]
            Conv_bn_act(first_l.in_ch, first_l.out_ch, kernel=(3, 3), stride=first_l.stride, padding=(1, 1)))
        
        for layer_ch in self.net_para[1:]:
            if layer_ch.num_block==1:
                # 只有一层
                self.layers.append(
                    DepthWise(layer_ch.in_ch, layer_ch.out_ch, groups = layer_ch.ex_ch, 
                              kernel=(3, 3), stride=layer_ch.stride, padding=(1, 1))
                )
            else:
                self.layers.append(
                    Residual_Module(layer_ch.in_ch, layer_ch.out_ch, groups = layer_ch.ex_ch, 
                                    num_block=layer_ch.num_block, 
                                    kernel=(3, 3), stride=layer_ch.stride, padding=(1, 1))
                    )

        '''self.layers.extend(
        [
            DepthWise(64 * self.scale, 64 * self.scale, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=128),
            Residual_Module(64 * self.scale, num_block=blocks[1], groups=128, kernel=(3, 3), stride=(1, 1), padding=(1, 1)),
            DepthWise(64 * self.scale, 128 * self.scale, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=256),
            Residual_Module(128 * self.scale, num_block=blocks[2], groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1)),
            DepthWise(128 * self.scale, 128 * self.scale, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=512),
            Residual_Module(128 * self.scale, num_block=blocks[3], groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1)),
        ])'''
        
        last_layer = self.net_para[-1]
        self.conv_sep = Conv_bn_act(last_layer.out_ch, last_layer.out_ch*2, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        self.features = GDC(last_layer.out_ch*2, num_features)
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
                print(x.shape)
                x = func(x)
        x = self.conv_sep(x.float() if self.fp16 else x)
        x = self.features(x)
        return x


def get_mbf(fp16, num_features = 128):
    expand_size = 2
    net_para = [
        Layer_congif(3, 1, 16, 1, 2),
        Layer_congif(16, 32, 64, 1, 1),
        Layer_congif(64, 128, 128, 1, 1),
        
        Layer_congif(128, 128, 128, 1, 2),
        Layer_congif(128, 256, 128, 4, 1),
        
        Layer_congif(128, 256, 256, 1, 2),
        Layer_congif(256, 256, 256, 6, 1),
        
        Layer_congif(256, 512, 256, 1, 2),
        Layer_congif(256, 512, 256, 2, 1)
    ]
    return MobileFaceNet(fp16, net_para, num_features)




if __name__ == "__main__":
    model = get_mbf(False, 512)
    model.eval().cpu()
    print(model)
    summary(model, input_size=(3, 112, 112), batch_size=-1, device="cpu")