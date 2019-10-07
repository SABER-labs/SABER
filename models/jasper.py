# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import random


jasper_activations = {
    "hardtanh": nn.Hardtanh,
    "relu": nn.ReLU,
    "selu": nn.SELU,
}

def init_weights(m, mode='xavier_uniform'):
    if type(m) == nn.Conv1d or type(m) == MaskedConv1d:
        if mode == 'xavier_uniform':
            nn.init.xavier_uniform_(m.weight, gain=1.0)
        elif mode == 'xavier_normal':
            nn.init.xavier_normal_(m.weight, gain=1.0)
        elif mode == 'kaiming_uniform':
            nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
        elif mode == 'kaiming_normal':
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        else:
            raise ValueError("Unknown Initialization mode: {0}".format(mode))
    elif type(m) == nn.BatchNorm1d:
        if m.track_running_stats:
            m.running_mean.zero_()
            m.running_var.fill_(1)
            m.num_batches_tracked.zero_()
        if m.affine:
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

def get_same_padding(kernel_size, stride, dilation):
    if stride > 1 and dilation > 1:
        raise ValueError("Only stride OR dilation may be greater than 1")
    return (kernel_size // 2) * dilation

class JasperEncoder(nn.Module):

    """Jasper encoder
    """
    def __init__(self, **kwargs):
        cfg = {}
        for key, value in kwargs.items():
            cfg[key] = value

        nn.Module.__init__(self)
        self._cfg = cfg

        activation = jasper_activations[cfg['encoder']['activation']]()
        self.use_conv_mask = cfg['encoder'].get('convmask', False)
        feat_in = 80
        init_mode = cfg.get('init_mode', 'xavier_uniform')

        residual_panes = []
        encoder_layers = []
        self.dense_residual = False
        for lcfg in cfg['jasper']:
            dense_res = []
            if lcfg.get('residual_dense', False):
                residual_panes.append(feat_in)
                dense_res = residual_panes
                self.dense_residual = True
            print(f"JasperBlock create in in_feat: {feat_in} and output_feat: {lcfg['filters']}")
            encoder_layers.append(
                JasperBlock(feat_in, lcfg['filters'], repeat=lcfg['repeat'],
                                        kernel_size=lcfg['kernel'], stride=lcfg['stride'],
                                        dilation=lcfg['dilation'], dropout=lcfg['dropout'],
                                        residual=lcfg['residual'], activation=activation,
                                        residual_panes=dense_res, use_conv_mask=self.use_conv_mask))
            feat_in = lcfg['filters']

        self.encoder = nn.Sequential(*encoder_layers)
        self.apply(lambda x: init_weights(x, mode=init_mode))

    def num_weights(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        return self.encoder([x])

class JasperDecoderForCTC(nn.Module):
    """Jasper decoder
    """
    def __init__(self, **kwargs):
        nn.Module.__init__(self)
        self._feat_in = kwargs.get("feat_in")
        self._num_classes = kwargs.get("num_classes")
        init_mode = kwargs.get('init_mode', 'xavier_uniform')

        self.decoder_layers = nn.Sequential(
            nn.Conv1d(self._feat_in, self._num_classes, kernel_size=1, bias=True),)
        self.apply(lambda x: init_weights(x, mode=init_mode))

    def num_weights(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, encoder_output):
        out = self.decoder_layers(encoder_output[-1]).transpose(1, 2)
        return nn.functional.log_softmax(out, dim=2)


class JasperAcousticModel(nn.Module):
    def __init__(self, enc, dec, transpose_in=False):
        nn.Module.__init__(self)
        self.jasper_encoder = enc
        self.jasper_decoder = dec
        self.transpose_in = transpose_in
    def forward(self, x):               
        t_encoded_t = self.jasper_encoder(x)
        print(t_encoded_t.shape)
        out = self.jasper_decoder(encoder_output=t_encoded_t)
        print(out.shape)
        if self.jasper_encoder.use_conv_mask:
            return out, t_encoded_len_t
        else:
            return out

class Jasper(nn.Module):
    """Contains data preprocessing, spectrogram augmentation, jasper encoder and decoder
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.jasper_encoder = JasperEncoder(**kwargs.get("jasper_model_definition"))
        self.jasper_decoder = JasperDecoderForCTC(feat_in=kwargs.get("feat_in"),
                                                  num_classes=kwargs.get("num_classes"))
        self.acoustic_model = JasperAcousticModel(self.jasper_encoder, self.jasper_decoder)

    def num_weights(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):

        # Apply optional preprocessing
            
        if (self.jasper_encoder.use_conv_mask):
            a_inp = (t_processed_signal, p_length_t)
        else:
            a_inp = x
        # Forward Pass through Encoder-Decoder
        return self.acoustic_model.forward(a_inp)

class MaskedConv1d(nn.Conv1d):
    """1D convolution with sequence masking
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                             padding=0, dilation=1, groups=1, bias=False, use_conv_mask=True):
        super(MaskedConv1d, self).__init__(in_channels, out_channels, kernel_size,
                                                                             stride=stride,
                                                                             padding=padding, dilation=dilation,
                                                                             groups=groups, bias=bias)
        self.use_conv_mask = use_conv_mask

    def get_seq_len(self, lens):
        return ((lens + 2 * self.padding[0] - self.dilation[0] * (
            self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)

    def forward(self, inp):
        if self.use_conv_mask:
            x, lens = inp
            max_len = x.size(2)
            idxs = torch.arange(max_len).to(lens.dtype).to(lens.device).expand(len(lens), max_len)
            mask = idxs >= lens.unsqueeze(1)
            x = x.masked_fill(mask.unsqueeze(1).to(device=x.device), 0)
            del mask
            del idxs
            lens = self.get_seq_len(lens)
        else:
            x = inp
        out = super(MaskedConv1d, self).forward(x)

        if self.use_conv_mask:
            return out, lens
        else:
            return out

class JasperBlock(nn.Module):
    """Jasper Block. See https://arxiv.org/pdf/1904.03288.pdf
    """
    def __init__(self, inplanes, planes, repeat=3, kernel_size=11, stride=1,
                             dilation=1, padding='same', dropout=0.2, activation=None,
                             residual=True, residual_panes=[], use_conv_mask=False):
        super(JasperBlock, self).__init__()

        if padding != "same":
            raise ValueError("currently only 'same' padding is supported")

        padding_val = get_same_padding(kernel_size[0], stride[0], dilation[0])
        self.use_conv_mask = use_conv_mask
        self.conv = nn.ModuleList()
        inplanes_loop = inplanes
        for _ in range(repeat - 1):
            self.conv.extend(
                self._get_conv_bn_layer(inplanes_loop, planes, kernel_size=kernel_size,
                                                                stride=stride, dilation=dilation,
                                                                padding=padding_val))
            self.conv.extend(
                self._get_act_dropout_layer(drop_prob=dropout, activation=activation))
            inplanes_loop = planes
        self.conv.extend(
            self._get_conv_bn_layer(inplanes_loop, planes, kernel_size=kernel_size,
                                                            stride=stride, dilation=dilation,
                                                            padding=padding_val))

        self.res = nn.ModuleList() if residual else None
        res_panes = residual_panes.copy()
        self.dense_residual = residual
        if residual:
            if len(residual_panes) == 0:
                res_panes = [inplanes]
                self.dense_residual = False
            for ip in res_panes:
                self.res.append(nn.ModuleList(
                    modules=self._get_conv_bn_layer(ip, planes, kernel_size=1)))
        self.out = nn.Sequential(
            *self._get_act_dropout_layer(drop_prob=dropout, activation=activation))

    def _get_conv_bn_layer(self, in_channels, out_channels, kernel_size=11,
                                                 stride=1, dilation=1, padding=0, bias=False):
        layers = [
            MaskedConv1d(in_channels, out_channels, kernel_size, stride=stride,
                                     dilation=dilation, padding=padding, bias=bias,
                                     use_conv_mask=self.use_conv_mask),
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.1)
        ]
        return layers

    def _get_act_dropout_layer(self, drop_prob=0.2, activation=None):
        if activation is None:
            activation = nn.Hardtanh(min_val=0.0, max_val=20.0)
        layers = [
            activation,
            nn.Dropout(p=drop_prob)
        ]
        return layers

    def num_weights(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, input_):
        if self.use_conv_mask:
            xs, lens_orig = input_
        else:
            xs = input_
            lens_orig = 0
        # compute forward convolutions
        out = xs[-1]
        lens = lens_orig
        for i, l in enumerate(self.conv):
            if self.use_conv_mask and isinstance(l, MaskedConv1d):
                out, lens = l((out, lens))
            else:
                out = l(out)
        # compute the residuals
        if self.res is not None:
            for i, layer in enumerate(self.res):
                res_out = xs[i]
                for j, res_layer in enumerate(layer):
                    if j == 0 and self.use_conv_mask:
                        res_out, _ = res_layer((res_out, lens_orig))
                    else:
                        res_out = res_layer(res_out)
                out += res_out

        # compute the output
        out = self.out(out)
        if self.res is not None and self.dense_residual:
            out = xs + [out]
        else:
            out = [out]

        if self.use_conv_mask:
            return out, lens
        else:
            return out

if __name__ == "__main__":
    import toml
    import utils.config as config
    print("Jasper Summary")
    from torchsummary import summary
    from time import time
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = Jasper(jasper_model_definition=toml.load(config.jasper_model_definition), feat_in=1024, num_classes=config.vocab_size)
    # summary(net, (80, 1600), batch_size=1, device=device)
    image = torch.randn(1, 80, 1600).to(device)
    start = time()
    y = net(image)
    print(y.shape)
    print(f"time taken is {time()-start:.3f}s")