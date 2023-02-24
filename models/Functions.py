import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function, Variable
from torch.nn import Module, parameter


import warnings
try:
    from queue import Queue
except ImportError:
    from Queue import Queue

from torch.nn.modules.batchnorm import _BatchNorm
from functools import partial


from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from timm.models.layers.helpers import to_2tuple
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


# ###############################################VC###########################################
class Encoding(nn.Module):
    def __init__(self, c1, num_codes):
        super(Encoding, self).__init__()
        # init codewords and smoothing factor
        self.c1, self.num_codes = c1, num_codes
        num_codes = 64
        std = 1. / ((num_codes * c1)**0.5)
        # [num_codes, channels]
        self.codewords = nn.Parameter(
            torch.empty(num_codes, c1, dtype=torch.float).uniform_(-std, std), requires_grad=True)
        # [num_codes]
        self.scale = nn.Parameter(torch.empty(num_codes, dtype=torch.float).uniform_(-1, 0), requires_grad=True)

    @staticmethod
    def scaled_l2(x, codewords, scale):
        num_codes, c1 = codewords.size()
        b = x.size(0)

        # ---处理特征向量x  b n c1
        expanded_x = x.unsqueeze(2).expand((b, x.size(1), num_codes, c1))
        #print(expanded_x.size(), "expanded_x_scaled_l2")

        # ---处理codebook (num_code, c1)
        reshaped_codewords = codewords.view((1, 1, num_codes, c1))
        #print(reshaped_codewords.size(), "reshaped_codewords_scaled_l2")

        # 把scale从1, num_code变成   batch, c2, N, num_codes
        reshaped_scale = scale.view((1, 1, num_codes))  # N, num_codes
        #print(reshaped_scale.size(), "reshaped_scale_scaled_l2")

        # ---计算rik = z1 - d  # b, N, num_codes
        scaled_l2_norm = reshaped_scale * (expanded_x - reshaped_codewords).pow(2).sum(dim=3)
        #print(scaled_l2_norm.size(), "scaled_l2_norm_scaled_l2")
        return scaled_l2_norm

    @staticmethod
    def aggregate(assignment_weights, x, codewords):
        num_codes, c1 = codewords.size()

        # ---处理codebook
        reshaped_codewords = codewords.view((1, 1, num_codes, c1))
        #print(reshaped_codewords.size(), "reshaped_codewords_aggregate")

        b = x.size(0)

        # ---处理特征向量x b, c1, N
        expanded_x = x.unsqueeze(2).expand((b, x.size(1), num_codes, c1))
        #print(expanded_x.size(), "expanded_x_aggregate")

        #变换rei  b, N, num_codes,-
        assignment_weights = assignment_weights.unsqueeze(3)  # b, N, num_codes,-
        #print(assignment_weights.size(), "assignment_weights_aggregate")

        # ---开始计算eik,必须在Rei计算完之后
        encoded_feat = (assignment_weights * (expanded_x - reshaped_codewords)).sum(1)
        #print(encoded_feat.size(), "encoded_feat_aggregate")
        return encoded_feat

    def forward(self, x):
        #print(x.size(), "xxx_farword")
        assert x.dim() == 4 and x.size(1) == self.c1
        b, c1, w, h = x.size()

        # [batch_size, height x width, channels]
        x = x.view(b, self.c1, -1).transpose(1, 2).contiguous()

        # assignment_weights: [batch_size, channels, num_codes]
        assignment_weights = F.softmax(self.scaled_l2(x, self.codewords, self.scale), dim=2)
        #print(assignment_weights.size(), "assignment_weights_forward")

        # aggregate
        encoded_feat = self.aggregate(assignment_weights, x, self.codewords)
        #print(encoded_feat.size(), "encoded_feat_farword")
        return encoded_feat

# ########################################ConvBlock组件##################################################
#  1*1 3*3 1*1
class ConvBlock(nn.Module):

    def __init__(self, c1, c2, stride=1, res_conv=False, act_layer=nn.ReLU, groups=1,
                 norm_layer=partial(nn.BatchNorm2d, eps=1e-6), drop_block=None, drop_path=None):
        super(ConvBlock, self).__init__()
        self.c1 = c1
        expansion = 4
        c = c2 // expansion

        self.conv1 = nn.Conv2d(c1, c, kernel_size=1, stride=1, padding=0, bias=False)  # [64, 256, 1, 1]
        self.bn1 = norm_layer(c)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(c, c, kernel_size=3, stride=stride, groups=groups, padding=1, bias=False)
        self.bn2 = norm_layer(c)
        self.act2 = act_layer(inplace=True)

        self.conv3 = nn.Conv2d(c, c2, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = norm_layer(c2)
        self.act3 = act_layer(inplace=True)

        if res_conv:
            self.residual_conv = nn.Conv2d(c1, c2, kernel_size=1, stride=1, padding=0, bias=False)
            self.residual_bn = norm_layer(c2)

        self.res_conv = res_conv
        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x, return_x_2=True):
        #print(x.size(), "ConvBlock__input_xx")  # [1, 64, 16, 16]
        residual = x
        #print(residual.size(), "residual")  # ([1, 64, 16, 16])

        x = self.conv1(x)
        #print(x.size(), "ConvBlock_conv000")
        x = self.bn1(x)
        #print(x.size(), "ConvBlock_bn000")
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)  # 前面特征图
        #print(x.size(), "ConvBlock_conv1")  # [1, 16, 16, 16])

        x = self.conv2(x) #if x_t_r is None else self.conv2(x + x_t_r)
        #print(x.size(), "ConvBlock_x + x_t_r_conv2")  # [1, 16, 16, 16])
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x2 = self.act2(x)
        #print(x2.size(), "ConvBlock_x2_conv2")  # [1, 16, 16, 16])

        x = self.conv3(x2)
       # print(x.size(), "ConvBlock_conv3")  # ([1, 64, 16, 16])
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.res_conv:
            residual = self.residual_conv(residual)
            #print(residual.size(), "ConvBlock_res_conv_residual1111")   # [1, 256, 16, 16])
            residual = self.residual_bn(residual)

        x += residual
        x = self.act3(x)

        if return_x_2:
            return x, x2
        else:
            return x


class Mean(nn.Module):
    def __init__(self, dim, keep_dim=False):
        super(Mean, self).__init__()
        self.dim = dim
        self.keep_dim = keep_dim

    def forward(self, input):
        return input.mean(self.dim, self.keep_dim)

# ########################################MLP组件##################################################
class Mlp(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions. Input: tensor with shape [B, C, H, W]
    """
    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        #print(x.size(), "Mlp_out")
        return x

class LayerNormChannel(nn.Module):
    """
    LayerNorm only for Channel Dimension.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_channels, eps=1e-05):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight.unsqueeze(-1).unsqueeze(-1) * x \
            + self.bias.unsqueeze(-1).unsqueeze(-1)
        #print(x.size(), "LayerNormChannel_out")
        return x

class GroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)