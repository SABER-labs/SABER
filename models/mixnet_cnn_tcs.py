import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from .conv1dlayers import MDConv1d, create_conv1d_pad, PositionalEncoding
from .activations_jit import MishJit, SwishJit
from .activations import HardSwish, HardSigmoid
from .excite_layers import EfficientChannelAttention
from .mixnet import NON_LINEARITY, _RoundChannels, Conv1x1Bn, MixNetBlock, form_stage


class ASRModel(nn.Module):
    # [in_channels, out_channels, kernel_size, stride, growth_factor_for_inv_res, non_linear, squeeze_excite]
    filters = [9, 11, 13, 15, 17]
    strides = [1, 1, 1, 1, 1]
    growths = [1, 6, 6, 6, 6]
    squeeze_excites = [False, True, True, True, True]
    repeats = [2, 3, 3, 4, 4]
    non_linearities = ['Mish', 'Mish', 'Mish', 'Mish', 'Mish']
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

    def __init__(self, input_features=80, num_classes=128, width_multiplier=1.8):
        super(ASRModel, self).__init__()
        config = self.mixnet_speech
        stem_channels = 24
        final_channels = 768
        dropout_rate = 0.2

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
            input_features, stem_channels, kernel_size=9, stride=3)

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
        self.classifier = nn.Conv1d(self._stage_out_channels, num_classes, 1, 1, bias=True)
        self._initialize_weights()

    def forward(self, x):
        x = self.stem_conv(x) # bct
        x = self.layers(x)
        x = self.head_conv1(x)
        x = self.head_conv2(x)
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
                if m.bias is not None:
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