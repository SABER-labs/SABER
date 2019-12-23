import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from .conv1dlayers import MDConv1d, create_conv1d_pad, PositionalEncoding
from .activations_jit import MishJit
from .activations import HardSwish, HardSigmoid

NON_LINEARITY = {
    'ReLU': nn.ReLU6(inplace=True),
    'Swish': HardSwish(inplace=True),
    'Sigmoid': HardSigmoid(inplace=True),
    'Mish': MishJit(inplace=True)
}

def Conv1x1Bn(in_channels, out_channels, non_linear='Mish', kernel_size=1, dilation=1, stride=1, bias=False):
    return nn.Sequential(
        create_conv1d_pad(in_channels, out_channels, kernel_size, stride=stride,
                          padding='same', dilation=dilation, bias=bias),
        nn.BatchNorm1d(out_channels),
        NON_LINEARITY[non_linear]
    )

# Based on the paper: https://arxiv.org/pdf/1901.10055.pdf
class ASRModel(nn.Module):

    def __init__(self, input_features=80, num_classes=128):
        super().__init__()
        self.downsampler = nn.AvgPool1d(8, 8)
        dhead = 512
        dff = 2048
        nheads = 8
        num_layers = 10
        self.pos_encoder = PositionalEncoding(dhead, max_len=750, dropout=0.1)
        self.embedder = Conv1x1Bn(input_features, dhead, non_linear='Mish')
        encoder_layers = nn.TransformerEncoderLayer(dhead, nheads, dff, activation='relu')
        self.san_layers = nn.TransformerEncoder(encoder_layers, num_layers)
        self.dropout = nn.Dropout(0.2)
        self.point_wise_proj = nn.Conv1d(dhead, num_classes, 1, 1, bias=True)

    def forward(self, x):
        x = self.downsampler(x)
        x = self.embedder(x)
        x = x.permute(2, 0, 1)
        x = self.pos_encoder(x)
        x = self.san_layers(x)
        x = x.permute(1, 2, 0)
        x = self.dropout(x)
        x = self.point_wise_proj(x)
        return x

if __name__ == '__main__':
    print("ASRModel Summary")
    from torchsummary import summary
    from time import time
    from utils import config
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
        print(y.shape)
        print(f"time taken is {time()-start:.3f}s")