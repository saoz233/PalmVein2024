import time
import torch
from torch import nn, optim
import torch.nn.functional as F
from matplotlib_inline import backend_inline
from matplotlib import pyplot as plt
from IPython import display

def use_svg_display():
    """使用svg格式在Jupyter中显示绘图

    Defined in :numref:`sec_calculus`"""
    backend_inline.set_matplotlib_formats('svg')

def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """设置matplotlib的轴

    Defined in :numref:`sec_calculus`"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()

class Animator:
    """在动画中绘制数据"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        """Defined in :numref:`sec_softmax_scratch`"""
        # 增量地绘制多条线
        if legend is None:
            legend = []
        use_svg_display()
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        #display.display(self.fig)
        #display.clear_output(wait=True)

class Accumulator:
    """在n个变量上累加"""
    def __init__(self, n):
        """Defined in :numref:`sec_softmax_scratch`"""
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def accuracy(y_hat:torch.Tensor, y:torch.Tensor):
    """计算预测正确的数量

    Defined in :numref:`sec_softmax_scratch`"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(dim=1)
    cmp = y_hat.type(y.dtype) == y
    return float((cmp.type(y.dtype)).sum())

def evaluate_accuracy(data_iter, net, device=None):
    if isinstance(net, nn.Module):
        net.eval()# 评估模式, 这会关闭dropout
        if device is None and isinstance(net, torch.nn.Module):
            # 如果没指定device就使用net的device
            device = next(iter(net.parameters())).device
        # 正确预测的数量，总预测的数量: (acc_sum, n) = (0.0, 0)
        metric = Accumulator(2)
        with torch.no_grad():
            for X, y in data_iter:
                if isinstance(X, list):
                    # BERT微调所需的（之后将介绍）
                    X = [x.to(device) for x in X]
                else:
                    X = X.to(device)
                y = y.to(device)
                metric.add(accuracy(net(X), y), y.numel())

        net.train()
    return metric[0] / metric[1]

def train_cnn(net:nn.Module, train_iter, test_iter, 
              loss:nn.Module, optimizer:torch.optim.Optimizer, 
              device, num_epochs):
    def init_weight(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weight)
    net = net.to(device)
    print("training on ", device)
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs],
                        legend=['train loss', 'train acc', 'test acc'])
    start, num_batch = 0, len(train_iter)

    for epoch in range(num_epochs):
        now = time.time()
        # (train_l_sum, train_acc_sum, n_sample)
        metric = Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            optimizer.zero_grad()
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)

            l = loss(y_hat, y)
            l.backward()

            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], accuracy(y_hat, y), X.shape[0])
        
        train_loss = metric[0] / (metric[2]+1e-4)
        train_acc = metric[1] / (metric[2]+1e-4)
        test_acc = evaluate_accuracy(test_iter, net)
        if (i + 1) % (num_batch // 5) == 0 or i == num_batch - 1:
            animator.add(epoch + (i + 1) / num_batch,
                         (train_loss, train_acc, test_acc))
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, epoch time %.1f sec, total time %.1f min'
              % (epoch + 1, train_loss, train_acc, test_acc, time.time() - now, (time.time()-start)/60))
