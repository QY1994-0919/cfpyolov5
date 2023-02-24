# Activation functions

import torch
import torch.nn as nn
import torch.nn.functional as F


# SiLU https://arxiv.org/pdf/1606.08415.pdf ----------------------------------------------------------------------------
class SiLU(nn.Module):  # export-friendly version of nn.SiLU()
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x) #swish函数f(x)=x*sigmoid(x)

#在pytorch中自带torch.nn.Hardswish(),但作者这里没用而是自己实现
#为了支持导出torchscript, CoreML and ONNX 格式的文件
class Hardswish(nn.Module):  # export-friendly version of nn.Hardswish()
    @staticmethod
    def forward(x):
        # return x * F.hardsigmoid(x)  # for torchscript and CoreML
        return x * F.hardtanh(x + 3, 0., 6.) / 6.  # for torchscript, CoreML and ONNX 

# the Swish activation module implemented using custom ops
# 节省内存
class MemoryEfficientSwish(nn.Module):
    class F(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            # save_for_backward函数可以将对象保存起来，用于后续的backward函数
            ctx.save_for_backward(x) # 会保留此input的全部信息, 并提供避免in-place操作导致的input在backward被修改的情况.
            return x * torch.sigmoid(x)

        @staticmethod
        def backward(ctx, grad_output):
            #ctx.saved_tensors会返回forward函数内存储的对象
            x = ctx.saved_tensors[0]
            sx = torch.sigmoid(x)
            return grad_output * (sx * (1 + x * (1 - sx)))

    def forward(self, x):
        return self.F.apply(x)


# Mish https://github.com/digantamisra98/Mish --------------------------------------------------------------------------
# Mish函数定义：f(x)=x*tanh(softplus(x))
class Mish(nn.Module):
    @staticmethod
    def forward(x):
        return x * F.softplus(x).tanh()


class MemoryEfficientMish(nn.Module):
    class F(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return x.mul(torch.tanh(F.softplus(x)))  # x * tanh(ln(1 + exp(x)))

        @staticmethod
        def backward(ctx, grad_output):
            x = ctx.saved_tensors[0]
            sx = torch.sigmoid(x)
            fx = F.softplus(x).tanh()
            return grad_output * (fx + x * sx * (1 - fx * fx))

    def forward(self, x):
        return self.F.apply(x)


# FReLU https://arxiv.org/abs/2007.11824 -------------------------------------------------------------------------------
class FReLU(nn.Module):
    def __init__(self, c1, k=3):  # ch_in, kernel
        super().__init__()
        # nn.Con2d(in_channels, out_channels, kernel_size, stride, padding, dilation=1, groups=1, bias=True)
        self.conv = nn.Conv2d(c1, c1, k, 1, 1, groups=c1, bias=False)
        self.bn = nn.BatchNorm2d(c1)

    def forward(self, x):
        return torch.max(x, self.bn(self.conv(x)))
