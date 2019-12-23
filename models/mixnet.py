import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from .conv1dlayers import MDConv1d, create_conv1d_pad, PositionalEncoding
from .activations_jit import MishJit, SwishJit
from .activations import HardSwish, HardSigmoid
from .excite_layers import EfficientChannelAttention

NON_LINEARITY = {
    'ReLU': nn.ReLU6(inplace=True),
    'Swish': HardSwish(inplace=True),
    'Sigmoid': HardSigmoid(inplace=True),
    'Mish': MishJit(inplace=True)
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


class MixNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, non_linear='Mish', sq_ex=False, drop_connect_rate=0.1):
        super(MixNetBlock, self).__init__()

        expand = (expand_ratio != 1)
        expand_channels = in_channels * expand_ratio
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

        if sq_ex:
            # squeeze and excite
            squeeze_excite = nn.Sequential(
                EfficientChannelAttention(expand_channels))
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
            return x + self._drop_connect(self.conv(x))
        else:
            return self.conv(x)


def custom_range(start, num, inc):
    return list(range(start, start + (num * inc), inc))


def form_stage(in_channel, out_channel, start_kernel, stride, growth, non_linearity, sq_ex, repeats):
    params = []
    for i in range(repeats - 1):
        params.append((in_channel, in_channel, custom_range(
            start_kernel, 5, 2), 1, growth, non_linearity, sq_ex))
    params.append((in_channel, out_channel, custom_range(
        start_kernel, 5, 2), stride, growth, non_linearity, sq_ex))
    return params


class ASRModel(nn.Module):
    # [in_channels, out_channels, kernel_size, stride, growth_factor_for_inv_res, non_linear, squeeze_excite]
    filters = [9, 11, 13, 15, 17]
    strides = [2, 1, 2, 1, 1]
    growths = [1, 6, 6, 6, 6]
    squeeze_excites = [False, True, True, True, True]
    repeats = [2, 2, 3, 3, 3]
    non_linearities = ['ReLU', 'ReLU', 'Mish', 'Mish', 'Mish']
    in_channels =  [24, 56, 152, 344, 568]
    out_channels = [56, 152, 344, 568, 568]
    mixnet_speech = []
    for i, filter_i in enumerate(filters):
        growth = growths[i]
        sq_ex = squeeze_excites[i]
        stride = strides[i]
        repeat = repeats[i]
        out_channel = out_channels[i]
        in_channel = in_channels[i]
        non_linearity = non_linearities[i]
        mixnet_speech.extend(form_stage(in_channel, out_channel, filter_i,
                                        stride, growth, non_linearity, sq_ex, repeat))
        in_filter = out_channel

    def __init__(self, input_features=80, num_classes=128, width_multiplier=1.0):
        super(ASRModel, self).__init__()
        config = self.mixnet_speech
        stem_channels = 24
        final_channels = 728
        dropout_rate = 0.1

        # depth multiplier
        stem_channels = _RoundChannels(stem_channels*width_multiplier)
        self._stage_out_channels = _RoundChannels(final_channels * width_multiplier)

        for i, conf in enumerate(config):
            conf_ls = list(conf)
            conf_ls[0] = _RoundChannels(conf_ls[0]*width_multiplier)
            conf_ls[1] = _RoundChannels(conf_ls[1]*width_multiplier)
            config[i] = tuple(conf_ls)

        # stem convolution
        self.stem_conv = Conv1x1Bn(
            input_features, stem_channels, kernel_size=9, stride=2)

        # building MixNet blocks
        layers = []
        for in_channels, out_channels, kernel_size, stride, expand_ratio, non_linear, se_ratio in config:
            layers.append(MixNetBlock(in_channels, out_channels,
                                      kernel_size, stride, expand_ratio, non_linear, se_ratio))

        self.layers = nn.Sequential(*layers)

        # last several layers
        self.head_conv1 = Conv1x1Bn(
            config[-1][1], self._stage_out_channels, kernel_size=27, non_linear='Mish', dilation=2)
        self.head_conv2 = Conv1x1Bn(
            self._stage_out_channels, self._stage_out_channels, non_linear='Mish')
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Conv1d(self._stage_out_channels, num_classes, 1, 1, bias=False)
        # decoder_layers = nn.TransformerEncoderLayer(self._stage_out_channels, 16, dim_feedforward=2048, dropout=0.1, activation='gelu')
        # self.decoder = nn.TransformerEncoder(decoder_layers, num_layers=2)
        # self.pos_encoding = PositionalEncoding(self._stage_out_channels, max_len=750, dropout=0.1)
        self._initialize_weights()

    def forward(self, x):
        x = self.stem_conv(x)
        x = self.layers(x)
        x = self.head_conv1(x)
        x = self.head_conv2(x)

        # x = x.permute(2, 0, 1)
        # x = self.pos_encoding(x)
        # x = self.decoder(x)
        # x = x.permute(1, 2, 0)

        x = self.dropout(x)
        x = self.classifier(x)
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
                fan_out = m.weight.size(0)
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
    from torchviz import make_dot
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = ASRModel(input_features=config.num_mel_banks,
                   num_classes=config.vocab_size).to(device)
    print('Total params: %.2fM' % (sum(p.numel()
                                       for p in net.parameters())/1000000.0))
    with torch.no_grad():
        # summary(net, (config.num_mel_banks, 500), batch_size=1, device=device)
        image = torch.randn(1, config.num_mel_banks, 500).to(device)
        start = time()
        y = net(image)
        make_dot(y)
        print(y.shape)
        print(f"time taken is {time()-start:.3f}s")
