import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from .activations_jit import MishJit, SwishJit
from .activations import HardSwish, HardSigmoid

NON_LINEARITY = {
    'ReLU': nn.ReLU6(inplace=True),
    'Swish': HardSwish(inplace=True),
    'Sigmoid': HardSigmoid(inplace=True),
    'Mish': MishJit(inplace=True)
}

class SqueezeAndExcite(nn.Module):
    def __init__(self, channels, se_ratio):
        super(SqueezeAndExcite, self).__init__()

        squeeze_channels = channels * se_ratio
        if not squeeze_channels.is_integer():
            raise ValueError('channels must be divisible by 1/ratio')

        squeeze_channels = int(squeeze_channels)
        self.se_reduce = nn.Conv1d(
            channels, squeeze_channels, 1, 1, 0, bias=True)
        self.non_linear1 = NON_LINEARITY['Mish']
        self.se_expand = nn.Conv1d(
            squeeze_channels, channels, 1, 1, 0, bias=True)
        self.non_linear2 = NON_LINEARITY['Sigmoid']

    def forward(self, x):
        y = torch.mean(x, 2, keepdim=True)
        y = self.non_linear1(self.se_reduce(y))
        y = self.non_linear2(self.se_expand(y))
        y = x * y

        return y

class EfficientChannelAttention(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.gate_fn = NON_LINEARITY['Sigmoid']

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.transpose(-1, -2)).transpose(-1, -2)
        y = self.gate_fn(y)
        return x * y.expand_as(x)