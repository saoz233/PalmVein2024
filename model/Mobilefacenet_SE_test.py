import torch.nn as nn
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, Sequential, Module
import torch
from torchsummary import summary
import math
from torch.nn import Parameter
import torch.nn.functional as F
from model.model_layers.model_layer import *
from model.model_layers.attn_layer import *

class MobileFaceNet(Module):
    def __init__(self, fp16=False, num_features=128, blocks=(1, 4, 6, 2), scale=2):
        super(MobileFaceNet, self).__init__()
        self.scale = scale
        self.fp16 = fp16
        self.layers = nn.ModuleList()
        self.layers.append(
            ConvBlock(3, 64 * self.scale, kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        )
        if blocks[0] == 1:
            self.layers.append(
                ConvBlock(64 * self.scale, 64 * self.scale, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=64)
            )
        else:
            self.layers.append(
                Residual(64 * self.scale, num_block=blocks[0], groups=128, kernel=(3, 3), stride=(1, 1), padding=(1, 1)),
            )
        
        self.layers.extend(
        [
            DepthWise(64 * self.scale, 64 * self.scale, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=128),
            Residual(64 * self.scale, num_block=blocks[1], groups=128, kernel=(3, 3), stride=(1, 1), padding=(1, 1)),
            DepthWise(64 * self.scale, 128 * self.scale, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=256,se = False),
            Residual(128 * self.scale, num_block=blocks[2], groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1), se= True),
            DepthWise(128 * self.scale, 128 * self.scale, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=512,se = False),
            Residual(128 * self.scale, num_block=blocks[3], groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1), se= True),
        ])

        self.conv_sep = ConvBlock(128 * self.scale, num_features* self.scale, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
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

# arcface损失，使用交叉熵共同作为损失函数
class ArcMarginProduct(nn.Module):
    def __init__(self, in_features=128, out_features=200, s=32.0, m=0.50, easy_margin=False, device = 'cuda'):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        # init.kaiming_uniform_()
        # self.weight.data.normal_(std=0.001)
        self.device = device
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x, label):
        # 相当于 内积/模=cos（模=1）
        #【第j项为 W_j * x：x分为j类的cos】
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        # z_i = cos(theta + m)
        # 这个m相当于希望分类之间有一个间隔（像SVM一样）
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

        # 把phi化成onehot形式
        one_hot = torch.zeros(cosine.size(), device=self.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        # s相当于是模长
        output *= self.s
        return output


class Linear_Softmax(nn.Module):
    def __init__(self, in_features=128, out_features=200):
        super(Linear_Softmax, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.soft = nn.Softmax(dim=-1)

    def forward(self, x, y):
        x = self.fc(x)
        return self.soft(x)


class Full_Model(nn.Module):
    def __init__(self, fp16, num_features, blocks=(1, 4, 6, 2), scale=2, num_class=600):
        super(Full_Model, self).__init__()
        self.feature = get_mbf(fp16, num_features, blocks, scale=scale)
        self.soft = Linear_Softmax(num_features, num_class)
        self.ArcMargin = ArcMarginProduct(num_features, num_class)

    def forward(self, img, y=None, classifier=None):
        x = self.feature(img)
        if classifier == 'arc' and y:
            x = self.ArcMargin(x, y)
        else:
            x = self.soft(x,y)
        return x



def get_mbf(fp16, num_features, blocks=(1, 4, 6, 2), scale=2):
    return MobileFaceNet(fp16, num_features, blocks, scale=scale)



if __name__ == "__main__":
    model = get_mbf(False, 512)
    model.eval().cpu()
    print(model)
    summary(model, input_size=(3, 112, 112), batch_size=-1, device="cpu")