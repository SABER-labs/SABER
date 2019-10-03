import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn_act(in_, out_, kernel_size,
                stride=1, groups=1, bias=True,
                eps=1e-3, momentum=0.01):
    return nn.Sequential(
        SamePadConv1d(in_, out_, kernel_size, stride, groups=groups, bias=bias),
        nn.BatchNorm1d(out_, eps, momentum),
        Swish()
    )

def split_layer(total_channels, num_groups):
    split = [int(math.ceil(total_channels / num_groups)) for _ in range(num_groups)]
    split[num_groups - 1] += total_channels - sum(split)
    return split


class DepthwiseConv2D(nn.Module):
    def __init__(self, in_channels, kernal_size, stride):
        super(DepthwiseConv2D, self).__init__()
        padding = (kernal_size - 1) // 2

        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=kernal_size, padding=padding, stride=stride, groups=in_channels)

    def forward(self, x):
        out = self.depthwise_conv(x)
        return out


class GroupConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, n_chunks=1):
        super(GroupConv2D, self).__init__()
        self.n_chunks = n_chunks
        self.split_in_channels = split_layer(in_channels, n_chunks)
        split_out_channels = split_layer(out_channels, n_chunks)

        if n_chunks == 1:
            self.group_conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size)
        else:
            self.group_layers = nn.ModuleList()
            for idx in range(n_chunks):
                self.group_layers.append(nn.Conv2d(self.split_in_channels[idx], split_out_channels[idx], kernel_size=kernel_size))

    def forward(self, x):
        if self.n_chunks == 1:
            return self.group_conv(x)
        else:
            split = torch.split(x, self.split_in_channels, dim=1)
            out = torch.cat([layer(s) for layer, s in zip(self.group_layers, split)], dim=1)
            return out


class MDConv(nn.Module):
    def __init__(self, out_channels, n_chunks, stride=1):
        super(MDConv, self).__init__()
        self.n_chunks = n_chunks
        self.split_out_channels = split_layer(out_channels, n_chunks)

        self.layers = nn.ModuleList()
        for idx in range(self.n_chunks):
            kernel_size = 2 * idx + 3
            self.layers.append(DepthwiseConv2D(self.split_out_channels[idx], kernal_size=kernel_size, stride=stride))

    def forward(self, x):
        split = torch.split(x, self.split_out_channels, dim=1)
        out = torch.cat([layer(s) for layer, s in zip(self.layers, split)], dim=1)
        return out


class SamePadConv1d(nn.Conv1d):
    """
    Conv with TF padding='same'
    https://github.com/pytorch/pytorch/issues/3867#issuecomment-349279036
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True, padding_mode="zeros"):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias, padding_mode)

    def get_pad_odd(self, in_, weight, stride, dilation):
        effective_filter_size_rows = (weight - 1) * dilation + 1
        out_rows = (in_ + stride - 1) // stride
        padding_needed = max(0, (out_rows - 1) * stride + effective_filter_size_rows - in_)
        padding_rows = max(0, (out_rows - 1) * stride + (weight - 1) * dilation + 1 - in_)
        rows_odd = (padding_rows % 2 != 0)
        return padding_rows, rows_odd

    def forward(self, x):
        padding_rows, rows_odd = self.get_pad_odd(x.shape[2], self.weight.shape[2], self.stride[0], self.dilation[0])

        if rows_odd:
            x = F.pad(x, [0, int(rows_odd)])

        return F.conv1d(x, self.weight, self.bias, self.stride,
                        padding=padding_rows // 2,
                        dilation=self.dilation, groups=self.groups)

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


class SEModule(nn.Module):
    def __init__(self, in_, squeeze_ch):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_, squeeze_ch, kernel_size=1, stride=1, padding=0, bias=True),
            Swish(),
            nn.Conv1d(squeeze_ch, in_, kernel_size=1, stride=1, padding=0, bias=True),
        )

    def forward(self, x):
        return x * torch.sigmoid(self.se(x))


class DropConnect(nn.Module):
    def __init__(self, ratio):
        super().__init__()
        self.ratio = 1.0 - ratio

    def forward(self, x):
        if not self.training:
            return x

        random_tensor = self.ratio
        random_tensor += torch.rand([x.shape[0], 1, 1], dtype=torch.float, device=x.device)
        random_tensor.requires_grad_(False)
        return x / self.ratio * random_tensor.floor()