import torch.nn as nn

class SEblock(nn.Module):  # 定义Squeeze and Excite注意力机制模块
    def __init__(self, channel, reduction = 16):  # 初始化方法
        super(SEblock, self).__init__()  # 继承初始化方法
 
        self.channel = channel  # 通道数
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # avgpool
        self.attention = nn.Sequential(  # 定义注意力模块
            nn.Linear(channel, channel // reduction, bias=False),  # 1x1conv，代替全连接
            nn.ReLU(inplace=True),  # relu
            nn.Linear(channel // reduction, channel, bias=False),  # 1x1conv，代替全连接
            nn.Hardswish(inplace=True)  # h-swish，此处原文图中为hard-alpha，未注明具体激活函数，这里使用h-swish
        )
 
    def forward(self, x):  # 前传函数
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.attention(y).view(b, c, 1, 1)
        return x * y.expand_as(x)