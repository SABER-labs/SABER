import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

class HardSigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(HardSigmoid, self).__init__()
        self.relu6 = nn.ReLU6(inplace=inplace)
    def forward(self, x):
        return (self.relu6(x+3)) / 6

class HardSwish(nn.Module):
    def __init__(self, inplace=True):
        super(HardSwish, self).__init__()
        self.hsigmoid = HardSigmoid(inplace=inplace)
    def forward(self, x):
        return x * self.hsigmoid(x)

NON_LINEARITY = {
    'ReLU': nn.ReLU6(inplace=True),
    'Swish': HardSwish(),
    'Sigmoid': HardSigmoid()
}

def _RoundChannels(c, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_c = max(min_value, int(c + divisor / 2) // divisor * divisor)
    if new_c < 0.9 * c:
        new_c += divisor
    return new_c

def _SplitChannels(channels, num_groups):
    split_channels = [channels//num_groups for _ in range(num_groups)]
    split_channels[0] += channels - sum(split_channels)
    return split_channels

def Conv3x3Bn(in_channels, out_channels, stride, non_linear='ReLU'):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, 3, stride, 1, bias=False),
        nn.BatchNorm1d(out_channels),
        NON_LINEARITY[non_linear]
    )

def Conv1x1Bn(in_channels, out_channels, non_linear='ReLU'):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, 1, 1, 0, bias=False),
        nn.BatchNorm1d(out_channels),
        NON_LINEARITY[non_linear]
    )

class SqueezeAndExcite(nn.Module):
    def __init__(self, channels, se_ratio):
        super(SqueezeAndExcite, self).__init__()

        squeeze_channels = channels * se_ratio
        if not squeeze_channels.is_integer():
            raise ValueError('channels must be divisible by 1/ratio')

        squeeze_channels = int(squeeze_channels)
        self.se_reduce = nn.Conv1d(channels, squeeze_channels, 1, 1, 0, bias=True)
        self.non_linear1 = NON_LINEARITY['Swish']
        self.se_expand = nn.Conv1d(squeeze_channels, channels, 1, 1, 0, bias=True)
        self.non_linear2 = NON_LINEARITY['Sigmoid']

    def forward(self, x):
        y = torch.mean(x, 2, keepdim=True)
        y = self.non_linear1(self.se_reduce(y))
        y = self.non_linear2(self.se_expand(y))
        y = x * y

        return y

class MDConv(nn.Module):
    def __init__(self, channels, kernel_size, stride):
        super(MDConv, self).__init__()

        self.num_groups = len(kernel_size)
        self.split_channels = _SplitChannels(channels, self.num_groups)

        self.mixed_depthwise_conv = nn.ModuleList([])
        for i in range(self.num_groups):
            self.mixed_depthwise_conv.append(nn.Conv1d(
                self.split_channels[i],
                self.split_channels[i],
                kernel_size[i],
                stride=stride,
                padding=kernel_size[i]//2,
                groups=self.split_channels[i],
                bias=False
            ))

    def forward(self, x):
        if self.num_groups == 1:
            return self.mixed_depthwise_conv[0](x)

        x_split = torch.split(x, self.split_channels, dim=1)
        x = [conv(t) for conv, t in zip(self.mixed_depthwise_conv, x_split)]
        x = torch.cat(x, dim=1)

        return x

class MixNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, non_linear='ReLU', se_ratio=0.0):
        super(MixNetBlock, self).__init__()

        expand = (expand_ratio != 1)
        expand_channels = in_channels * expand_ratio
        se = (se_ratio != 0.0)
        self.residual_connection = (stride == 1 and in_channels == out_channels)

        conv = []

        if expand:
            # expansion phase
            pw_expansion = nn.Sequential(
                nn.Conv1d(in_channels, expand_channels, 1, 1, 0, bias=False),
                nn.BatchNorm1d(expand_channels),
                NON_LINEARITY[non_linear]
            )
            conv.append(pw_expansion)

        # depthwise convolution phase
        dw = nn.Sequential(
            MDConv(expand_channels, kernel_size, stride),
            nn.BatchNorm1d(expand_channels),
            NON_LINEARITY[non_linear]
        )
        conv.append(dw)

        if se:
            # squeeze and excite
            squeeze_excite = nn.Sequential(SqueezeAndExcite(expand_channels, se_ratio))
            conv.append(squeeze_excite)

        # projection phase
        pw_projection = nn.Sequential(
            nn.Conv1d(expand_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm1d(out_channels)
        )
        conv.append(pw_projection)

        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        if self.residual_connection:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MixNet(nn.Module):
    # [in_channels, out_channels, kernel_size, stride, expand_ratio, non_linear, se_ratio]
    mixnet_speech = [
        (128, 128, [11], 2, 1, 'ReLU', 0.0),
        (128, 128+32, [11, 13], 1, 2, 'ReLU', 0.0),
        (128+32, 128+32, [9, 11, 13], 1, 2, 'ReLU', 0.0),
        (128+32, 128+32, [9, 11, 13], 1, 2, 'Swish', 0.25),
        (128+32, 128+32, [9, 11, 13], 1, 2, 'Swish', 0.25),
        (128+32, 128+32*2, [9, 11, 13], 2, 2, 'Swish', 0.25),
        (128+32*2, 128+32*2, [9, 11, 13], 1, 2, 'Swish', 0.25),
        (128+32*2, 128+32*2, [9, 11, 13], 1, 2, 'Swish', 0.25),
        (128+32*2, 128+32*2, [9, 11, 13], 2, 2, 'Swish', 0.25),
        (128+32*2, 128+32*3, [13, 15, 17], 1, 2, 'Swish', 0.25),
        (128+32*3, 128+32*3, [13, 15, 17], 1, 2, 'Swish', 0.25),
        (128+32*3, 128+32*3, [13, 15, 17], 1, 2, 'Swish', 0.25),
        (128+32*3, 128+32*3, [13, 15, 17], 1, 2, 'Swish', 0.25),
        (128+32*3, 128+32*4, [13, 15, 17], 1, 2, 'Swish', 0.25),
        (128+32*4, 128+32*4, [15, 17, 19], 1, 2, 'Swish', 0.25),
        (128+32*4, 128+32*4, [15, 17, 19], 1, 2, 'Swish', 0.25),
        (128+32*4, 128+32*4, [15, 17, 19], 1, 2, 'Swish', 0.25),
        (128+32*4, 128+32*5, [17, 19, 21], 1, 2, 'Swish', 0.25),
        (128+32*5, 128+32*5, [17, 19, 21], 1, 2, 'Swish', 0.25),
        (128+32*5, 128+32*5, [17, 19, 21], 1, 2, 'Swish', 0.25),
        (128+32*5, 128+32*6, [19, 21, 23], 1, 2, 'Swish', 0.25),
        (128+32*6, 128+32*6, [19, 21, 23], 1, 2, 'Swish', 0.25),
        (128+32*6, 128+32*6, [19, 21, 23], 1, 2, 'Swish', 0.25),
        (128+32*6, 128+32*6, [19, 21, 23], 1, 2, 'Swish', 0.25),
        (128+32*6, 128+32*7, [21, 23, 25], 1, 2, 'Swish', 0.25),
        (128+32*7, 128+32*7, [21, 23, 25], 1, 2, 'Swish', 0.25),
        (128+32*7, 128+32*7, [21, 23, 25], 1, 2, 'Swish', 0.25),
        (128+32*7, 128+32*7, [21, 23, 25], 1, 2, 'Swish', 0.25),
        (128+32*7, 128+32*8, [23, 25, 27], 1, 2, 'Swish', 0.25),
        (128+32*8, 128+32*8, [23, 25, 27], 1, 2, 'Swish', 0.25),
        (128+32*8, 128+32*8, [23, 25, 27], 1, 2, 'Swish', 0.25),
        (128+32*8, 128+32*8, [23, 25, 27], 1, 2, 'Swish', 0.25),
        (128+32*8, 128+32*8, [23, 25, 27], 1, 2, 'Swish', 0.25),
        (128+32*8, 128+32*8, [23, 25, 27], 1, 2, 'Swish', 0.25),
        (128+32*8, 128+32*9, [25, 27, 29], 1, 2, 'Swish', 0.25),
        (128+32*9, 128+32*9, [25, 27, 29], 1, 2, 'Swish', 0.25),
        (128+32*9, 128+32*9, [25, 27, 29], 1, 2, 'Swish', 0.25),
        (128+32*9, 128+32*9, [25, 27, 29], 1, 2, 'Swish', 0.25),
        (128+32*9, 128+32*9, [25, 27, 29], 1, 2, 'Swish', 0.25)
    ]

    def __init__(self, num_classes=512, depth_multiplier=1.0):
        super(MixNet, self).__init__()
        config = self.mixnet_speech
        stem_channels = 128
        dropout_rate = 0.4

        self._stage_out_channels = int(1024)

        # depth multiplier
        stem_channels = _RoundChannels(stem_channels*depth_multiplier)

        for i, conf in enumerate(config):
            conf_ls = list(conf)
            conf_ls[0] = _RoundChannels(conf_ls[0]*depth_multiplier)
            conf_ls[1] = _RoundChannels(conf_ls[1]*depth_multiplier)
            config[i] = tuple(conf_ls)

        # stem convolution
        self.stem_conv = Conv3x3Bn(80, stem_channels, 2)

        # building MixNet blocks
        layers = []
        for in_channels, out_channels, kernel_size, stride, expand_ratio, non_linear, se_ratio in config:
            layers.append(MixNetBlock(in_channels, out_channels, kernel_size, stride, expand_ratio, non_linear, se_ratio))
        self.layers = nn.Sequential(*layers)

        # last several layers
        self.head_conv = Conv1x1Bn(config[-1][1], self._stage_out_channels, non_linear='Swish')
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self._stage_out_channels, num_classes)
        self._initialize_weights()

    def forward(self, x):
        x = self.stem_conv(x)
        x = self.layers(x)
        x = self.head_conv(x)
        x = x.permute(0, 2, 1)
        x = self.classifier(x)
        x = self.dropout(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                m.bias.data.zero_()


if __name__ == '__main__':
    # import time
    # model = MixNet(net_type='mixnet_l')
    # print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    # input = Variable(torch.randn(64, 80, 3001))
    # start = time.time()
    # y = model(input)
    # print(f"Time taken for model inference was {time.time() - start:.3f}s")
    # print(y.shape)
    print("Mixnet Summary")
    from torchsummary import summary
    from time import time
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = MixNet().to(device)
    summary(net, (80, 3001), batch_size=1, device=device)
    image = torch.randn(1, 80, 3001).to(device)
    start = time()
    y = net(image)
    print(y.shape)
    print(f"time taken is {time()-start:.3f}s")