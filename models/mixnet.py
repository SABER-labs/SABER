import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from .conv1dlayers import MDConv1d, create_conv1d_pad, PositionalEncoding


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


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # inlining this saves 1 second per epoch (V100 GPU) vs having a temp x and then returning x(!)
        return x * (torch.tanh(F.softplus(x)))


NON_LINEARITY = {
    'ReLU': nn.ReLU6(inplace=True),
    'Swish': HardSwish(),
    'Sigmoid': HardSigmoid(),
    'Mish': Mish()
}


def _RoundChannels(c, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_c = max(min_value, int(c + divisor / 2) // divisor * divisor)
    if new_c < 0.9 * c:
        new_c += divisor
    return new_c


def Conv1x1Bn(in_channels, out_channels, non_linear='Mish', kernel_size=1, dilation=1, stride=1, bias=False):
    return nn.Sequential(
        create_conv1d_pad(in_channels, out_channels, kernel_size, stride=stride,
                          padding='same', dilation=dilation, bias=bias),
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


class MixNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, non_linear='Mish', se_ratio=0.0, drop_connect_rate=0.00):
        super(MixNetBlock, self).__init__()

        expand = (expand_ratio != 1)
        expand_channels = in_channels * expand_ratio
        se = (se_ratio != 0.0)
        self.drop_connect_rate = drop_connect_rate
        self.residual_connection = (
            stride == 1 and in_channels == out_channels)

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
            MDConv1d(expand_channels, expand_channels, kernel_size, stride),
            nn.BatchNorm1d(expand_channels),
            NON_LINEARITY[non_linear]
        )
        conv.append(dw)

        if se:
            # squeeze and excite
            squeeze_excite = nn.Sequential(
                SqueezeAndExcite(expand_channels, se_ratio))
            conv.append(squeeze_excite)

        # projection phase
        pw_projection = nn.Sequential(
            nn.Conv1d(expand_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm1d(out_channels)
        )
        conv.append(pw_projection)

        self.conv = nn.Sequential(*conv)

    def _drop_connect(self, x):
        if not self.training:
            return x
        keep_prob = 1.0 - self.drop_connect_rate
        batch_size = x.size(0)
        random_tensor = keep_prob
        random_tensor += torch.rand(batch_size, 1, 1, device=x.device)
        binary_tensor = random_tensor.floor()
        return x.div(keep_prob) * binary_tensor

    def forward(self, x):
        if self.residual_connection:
            return x + self.conv(x)
        else:
            return self.conv(x)


def custom_range(start, num, inc):
    return list(range(start, start + (num * inc), inc))


def form_stage(in_channel, out_channel, start_kernel, stride, growth, non_linearity, squeeze_factor, repeats):
    params = [(in_channel, out_channel, custom_range(
        start_kernel, 5, 2), stride, growth, non_linearity, squeeze_factor)]
    for i in range(1, repeats):
        params.append((out_channel, out_channel, custom_range(
            start_kernel, 5, 2), 1, growth, non_linearity, squeeze_factor))
    return params


class ASRModel(nn.Module):
    # [in_channels, out_channels, kernel_size, stride, expand_ratio, non_linear, se_ratio]
    filters = [9, 11, 15, 19, 23]
    repeats = 3
    mixnet_speech = []
    add_channels = 128
    in_filter = 256
    for i, filter_i in enumerate(filters):
        stride = 2 if i < 2 else 1
        non_linearity = 'Mish' if i < 1 else 'Mish'
        growth = 1 if i < 1 else 4
        squeeze_factor = 0.0 if i < 1 else 0.25
        out_channel = in_filter + add_channels  # * (i+1)
        mixnet_speech.extend(form_stage(in_filter, out_channel, filter_i,
                                        stride, growth, non_linearity, squeeze_factor, repeats))
        in_filter = out_channel

    def __init__(self, input_features=80, num_classes=128, depth_multiplier=1.0):
        super(ASRModel, self).__init__()
        config = self.mixnet_speech
        stem_channels = 256
        dropout_rate = 0.3

        self._stage_out_channels = _RoundChannels(1024 * depth_multiplier)

        # depth multiplier
        stem_channels = _RoundChannels(stem_channels*depth_multiplier)

        for i, conf in enumerate(config):
            conf_ls = list(conf)
            conf_ls[0] = _RoundChannels(conf_ls[0]*depth_multiplier)
            conf_ls[1] = _RoundChannels(conf_ls[1]*depth_multiplier)
            config[i] = tuple(conf_ls)

        # stem convolution
        self.stem_conv = Conv1x1Bn(
            input_features, stem_channels, kernel_size=9, stride=2)

        # building MixNet blocks
        layers = []
        for in_channels, out_channels, kernel_size, stride, expand_ratio, non_linear, se_ratio in config:
            layers.append(MixNetBlock(in_channels, out_channels, kernel_size, stride, expand_ratio, non_linear, se_ratio))
            
        self.layers = nn.Sequential(*layers)

        # last several layers
        self.head_conv1 = Conv1x1Bn(
            config[-1][1], self._stage_out_channels, kernel_size=29, non_linear='Mish', dilation=2)
        self.head_conv2 = Conv1x1Bn(
            self._stage_out_channels, self._stage_out_channels, non_linear='Mish')
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self._stage_out_channels, num_classes)
        decoder_layers = nn.TransformerEncoderLayer(self._stage_out_channels, 8, dim_feedforward=max(1024, int(1.5 * self._stage_out_channels)), dropout=0.1, activation='gelu')
        self.decoder = nn.TransformerEncoder(decoder_layers, num_layers=3)
        self.pos_encoding = PositionalEncoding(self._stage_out_channels, max_len=1000, dropout=0.1)
        self._initialize_weights()

    def forward(self, x):
        x = self.stem_conv(x)
        x = self.layers(x)
        x = self.head_conv1(x)
        x = self.head_conv2(x)

        
        x = x.permute(2, 0, 1)
        x = self.pos_encoding(x)
        x = self.decoder(x)
        x = x.permute(1, 2, 0)

        x = self.dropout(x)
        x = x.permute(0, 2, 1)
        x = self.classifier(x)
        x = x.permute(0, 2, 1)
        return x

    def _initialize_weights(self, n=''):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                fan_out = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                fan_out = m.weight.size(0)  # fan-out
                fan_in = 0
                if 'routing_fn' in n:
                    fan_in = m.weight.size(1)
                init_range = 1.0 / math.sqrt(fan_in + fan_out)
                m.weight.data.uniform_(-init_range, init_range)
                m.bias.data.zero_()


if __name__ == '__main__':
    print("ASRModel Summary")
    from torchsummary import summary
    from time import time
    from utils import config
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = ASRModel(input_features=config.num_mel_banks,
                   num_classes=config.vocab_size).to(device)
    with torch.no_grad():
        summary(net, (config.num_mel_banks, 500), batch_size=1, device=device)
        image = torch.randn(1, config.num_mel_banks, 500).to(device)
        start = time()
        y = net(image)
        print(y.shape)
        print(f"time taken is {time()-start:.3f}s")
