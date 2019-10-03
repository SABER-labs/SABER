import math

import torch
import torch.nn as nn

from models.layers import conv_bn_act
from models.layers import SamePadConv1d
from models.layers import Flatten
from models.layers import SEModule
from models.layers import DropConnect


class MBConv(nn.Module):
    def __init__(self, in_, out_, expand,
                 kernel_size, stride, skip,
                 se_ratio, dc_ratio=0.2):
        super().__init__()
        mid_ = in_ * expand
        self.expand_conv = conv_bn_act(in_, mid_, kernel_size=1, bias=False) if expand != 1 else nn.Identity()

        self.depth_wise_conv = conv_bn_act(mid_, mid_,
                                           kernel_size=kernel_size, stride=stride,
                                           groups=mid_, bias=False)

        self.se = SEModule(mid_, int(in_ * se_ratio)) if se_ratio > 0 else nn.Identity()

        self.project_conv = nn.Sequential(
            SamePadConv1d(mid_, out_, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(out_, 1e-3, 0.01)
        )

        # if _block_args.id_skip:
        # and all(s == 1 for s in self._block_args.strides)
        # and self._block_args.input_filters == self._block_args.output_filters:
        self.skip = skip and (stride == 1) and (in_ == out_)

        # DropConnect
        self.dropconnect = DropConnect(dc_ratio) if dc_ratio > 0 else nn.Identity()
        # Original TF Repo not using drop_rate
        # https://github.com/tensorflow/tpu/blob/05f7b15cdf0ae36bac84beb4aef0a09983ce8f66/models/official/efficientnet/efficientnet_model.py#L408
        # self.dropconnect = nn.Identity()

    def forward(self, inputs):
        expand = self.expand_conv(inputs)
        x = self.depth_wise_conv(expand)
        x = self.se(x)
        x = self.project_conv(x)
        if self.skip:
            x = self.dropconnect(x)
            x = x + inputs
        return x


class MBBlock(nn.Module):
    def __init__(self, in_, out_, expand, kernel, stride, num_repeat, skip, se_ratio, drop_connect_ratio=0.2):
        super().__init__()
        layers = [MBConv(in_, out_, expand, kernel, stride, skip, se_ratio, drop_connect_ratio)]
        for i in range(1, num_repeat):
            layers.append(MBConv(out_, out_, expand, kernel, 1, skip, se_ratio, drop_connect_ratio))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class EfficientNet(nn.Module):
    def __init__(self, width_coeff, depth_coeff,
                 depth_div=8, min_depth=None,
                 dropout_rate=0.2, drop_connect_rate=0.2):
        super().__init__()
        min_depth = min_depth or depth_div

        def renew_ch(x):
            if not width_coeff:
                return x

            x *= width_coeff
            new_x = max(min_depth, int(x + depth_div / 2) // depth_div * depth_div)
            if new_x < 0.9 * x:
                new_x += depth_div
            return int(new_x)

        def renew_repeat(x):
            return int(math.ceil(x * depth_coeff))

        mul = 2.0
        chans = [x * mul for x in [32, 16, 24, 40, 80, 112, 192, 320]] + [1280 * 1.5]

        self.stem = conv_bn_act(80, renew_ch(chans[0]), kernel_size=3, stride=2, bias=False)

        self.blocks = nn.Sequential(
            #       input channel  output    expand  k  s                   skip  se
            MBBlock(renew_ch(chans[0]), renew_ch(chans[1]), 1, 3, 1, renew_repeat(1), True, 0.25, drop_connect_rate),
            MBBlock(renew_ch(chans[1]), renew_ch(chans[2]), 6, 3, 2, renew_repeat(2), True, 0.25, drop_connect_rate),
            MBBlock(renew_ch(chans[2]), renew_ch(chans[3]), 6, 5, 1, renew_repeat(2), True, 0.25, drop_connect_rate),
            MBBlock(renew_ch(chans[3]), renew_ch(chans[4]), 6, 3, 2, renew_repeat(3), True, 0.25, drop_connect_rate),
            MBBlock(renew_ch(chans[4]), renew_ch(chans[5]), 6, 5, 1, renew_repeat(3), True, 0.25, drop_connect_rate),
            MBBlock(renew_ch(chans[5]), renew_ch(chans[6]), 6, 5, 2, renew_repeat(4), True, 0.25, drop_connect_rate),
            MBBlock(renew_ch(chans[6]), renew_ch(chans[7]), 6, 3, 1, renew_repeat(1), True, 0.25, drop_connect_rate)
        )

        self.head = nn.Sequential(
            *conv_bn_act(renew_ch(chans[7]), renew_ch(chans[-1]), kernel_size=1, bias=False)
        )

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, SamePadConv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / math.sqrt(m.weight.shape[1])
                nn.init.uniform_(m.weight, -init_range, init_range)

    def forward(self, inputs):
        stem = self.stem(inputs)
        x = self.blocks(stem)
        head = self.head(x)
        return head


if __name__ == "__main__":
    print("Efficient B4 Summary")
    from torchsummary import summary
    from time import time
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = EfficientNet(width_coeff=1.4, depth_coeff=1.8, dropout_rate=0.4).to(device)
    summary(net, (80, 3001), batch_size=1, device=device)
    image = torch.randn(1, 80, 3001).to(device)
    start = time()
    y = net(image)
    print(y.shape)
    print(f"time taken is {time()-start:.3f}s")