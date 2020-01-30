import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._six import container_abcs

from itertools import repeat
from functools import partial
from typing import Union, List, Tuple, Optional, Callable
import numpy as np
import math
from .config import *

def _is_static_pad(kernel_size, stride=1, dilation=1, **_):
    return stride == 1 and (dilation * (kernel_size - 1)) % 2 == 0


def _get_padding(kernel_size, stride=1, dilation=1, **_):
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


def _calc_same_pad(i: int, k: int, s: int, d: int):
    return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

def pad_1d_same(pad_arg):
    def _pad(input):
        return F.pad(input, pad_arg)
    return _pad

def _same_pad_arg(input_size, kernel_size, stride, dilation):
    ih = input_size
    kh = kernel_size
    pad_h = _calc_same_pad(ih, kh, stride, dilation)
    return [pad_h // 2, pad_h - pad_h // 2]


def _split_channels(num_chan, num_groups):
    split = [num_chan // num_groups for _ in range(num_groups)]
    split[0] += num_chan - sum(split)
    return split


def conv1d_same(
        x, weight: torch.Tensor, bias: Optional[torch.Tensor] = None, stride: int = 1,
        padding: int = 0, dilation: int = 1, groups: int = 1):
    ih = x.size()[-1]
    kh = weight.size()[-1]
    pad_h = _calc_same_pad(ih, kh, stride[0], dilation[0])
    if pad_h > 0:
        x = F.pad(x, [pad_h // 2, pad_h - pad_h // 2])
    return F.conv1d(x, weight, bias, stride, 0, dilation, groups)


class Conv1dSame(nn.Conv1d):
    """ Tensorflow like 'SAME' convolution wrapper for 1D convolutions
    """

    # pylint: disable=unused-argument
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv1dSame, self).__init__(
            in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)

    def forward(self, x):
        return conv1d_same(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class Conv1dSameExport(nn.Conv1d):
    """ ONNX export friendly Tensorflow like 'SAME' convolution wrapper for 1D convolutions

    NOTE: This does not currently work with torch.jit.script
    """

    # pylint: disable=unused-argument
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv1dSameExport, self).__init__(
            in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
        self.pad = None
        self.pad_input_size = 0

    def forward(self, x):
        input_size = x.size()[-1]
        if self.pad is None:
            pad_arg = _same_pad_arg(input_size, self.weight.size()[-1], self.stride, self.dilation)
            self.pad = pad_1d_same(pad_arg)
            self.pad_input_size = input_size
        else:
            assert self.pad_input_size == input_size

        x = self.pad(x)
        return F.conv1d(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


def get_padding_value(padding, kernel_size, **kwargs):
    dynamic = False
    if isinstance(padding, str):
        # for any string padding, the padding will be calculated for you, one of three ways
        padding = padding.lower()
        if padding == 'same':
            # TF compatible 'SAME' padding, has a performance and GPU memory allocation impact
            if _is_static_pad(kernel_size, **kwargs):
                # static case, no extra overhead
                padding = _get_padding(kernel_size, **kwargs)
            else:
                # dynamic padding
                padding = 0
                dynamic = True
        elif padding == 'valid':
            # 'VALID' padding, same as padding=0
            padding = 0
        else:
            # Default to PyTorch style 'same'-ish symmetric padding
            padding = _get_padding(kernel_size, **kwargs)
    return padding, dynamic


def create_conv1d_pad(in_chs, out_chs, kernel_size, **kwargs):
    padding = kwargs.pop('padding', '')
    kwargs.setdefault('bias', False)
    padding, is_dynamic = get_padding_value(padding, kernel_size, **kwargs)
    if is_dynamic:
        if is_exportable():
            assert not is_scriptable()
            return Conv1dSameExport(in_chs, out_chs, kernel_size, **kwargs)
        else:
            return Conv1dSame(in_chs, out_chs, kernel_size, **kwargs)
    else:
        return nn.Conv1d(in_chs, out_chs, kernel_size, padding=padding, **kwargs)


class MDConv1d(nn.Module):
    """ Mixed Grouped Convolution
    Based on MDConv and GroupedConv in MixNet impl:
      https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mixnet/custom_layers.py

    NOTE: This does not currently work with torch.jit.script
    """

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding='same', dilation=1, depthwise=True, **kwargs):
        super(MDConv1d, self).__init__()

        kernel_size = kernel_size if isinstance(kernel_size, list) else [kernel_size]
        num_groups = len(kernel_size)
        in_splits = _split_channels(in_channels, num_groups)
        out_splits = _split_channels(out_channels, num_groups)
        for idx, (k, in_ch, out_ch) in enumerate(zip(kernel_size, in_splits, out_splits)):
            conv_groups = in_ch if depthwise else 1
            # use add_module to keep key space clean
            self.add_module(
                str(idx),
                create_conv1d_pad(
                    in_ch, out_ch, k, stride=stride,
                    padding=padding, dilation=dilation, groups=conv_groups, **kwargs)
            )
        self.splits = in_splits

    def forward(self, x):
        x_split = torch.split(x, self.splits, 1)
        x_out = [c(x) for x, c in zip(x_split, self._modules.values())]
        x = torch.cat(x_out, 1)
        return x

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)