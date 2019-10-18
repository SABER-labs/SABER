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


def Conv3x3Bn(in_channels, out_channels, stride, non_linear='ReLU', kernel_size=3):
	return nn.Sequential(
		nn.Conv1d(in_channels, out_channels, kernel_size,
				  stride, kernel_size//2, bias=False),
		nn.BatchNorm1d(out_channels),
		NON_LINEARITY[non_linear]
	)


def Conv1x1Bn(in_channels, out_channels, non_linear='ReLU', kernel_width=1, dilation=1, stride=1, bias=False):
	p = kernel_width//2 if dilation == 1 else kernel_width - 1
	return nn.Sequential(
		nn.Conv1d(in_channels, out_channels, kernel_width, stride,
				  padding=p, dilation=dilation, bias=bias),
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
		self.non_linear1 = NON_LINEARITY['Swish']
		self.se_expand = nn.Conv1d(
			squeeze_channels, channels, 1, 1, 0, bias=True)
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
			MDConv(expand_channels, kernel_size, stride),
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
	start = 128
	filters = [29, 35, 47, 59, 71]
	repeats = 3
	mixnet_speech = []
	in_filter = start
	for i, filter_i in enumerate(filters):
		stride = 2 if i < 2 else 1
		non_linearity = 'ReLU' if i == 0 else 'Swish'
		growth = 1 if i == 0 else 5
		squeeze_factor = 0.0 if i == 0 else 0.5
		mixnet_speech.extend(form_stage(in_filter, start * (i + 1), filter_i,
							 stride, growth, non_linearity, squeeze_factor, repeats))
		in_filter = start * (i + 1)

	def __init__(self, input_features=80, num_classes=128, depth_multiplier=1.2):
		super(ASRModel, self).__init__()
		config = self.mixnet_speech
		stem_channels = 128
		dropout_rate = 0.1

		self._stage_out_channels = int(1024)

		# depth multiplier
		stem_channels = _RoundChannels(stem_channels*depth_multiplier)

		for i, conf in enumerate(config):
			conf_ls = list(conf)
			conf_ls[0] = _RoundChannels(conf_ls[0]*depth_multiplier)
			conf_ls[1] = _RoundChannels(conf_ls[1]*depth_multiplier)
			config[i] = tuple(conf_ls)

		# stem convolution
		self.stem_conv = Conv3x3Bn(
			input_features, stem_channels, 2, kernel_size=27)

		# building MixNet blocks
		layers = []
		for in_channels, out_channels, kernel_size, stride, expand_ratio, non_linear, se_ratio in config:
			layers.append(MixNetBlock(in_channels, out_channels,
						  kernel_size, stride, expand_ratio, non_linear, se_ratio))
		self.layers = nn.Sequential(*layers)

		# last several layers
		self.head_conv1 = Conv1x1Bn(
			config[-1][1], self._stage_out_channels, kernel_width=87, non_linear='Swish', dilation=2)
		self.head_conv2 = Conv1x1Bn(
			self._stage_out_channels, self._stage_out_channels, non_linear='Swish')
		# self.dropout = nn.Dropout(dropout_rate)

		self.classifier = nn.Conv1d(
			self._stage_out_channels, num_classes, 1, 1, bias=True)
		self._initialize_weights()

	def forward(self, x):
		x = self.stem_conv(x)
		x = self.layers(x)
		x = self.head_conv1(x)
		x = self.head_conv2(x)
		# x = self.dropout(x)
		x = self.classifier(x)
		return x

	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv1d):
				nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
			elif isinstance(m, nn.BatchNorm1d):
				if m.track_running_stats:
					m.running_mean.zero_()
					m.running_var.fill_(1)
					m.num_batches_tracked.zero_()
				if m.affine:
					nn.init.ones_(m.weight)
					nn.init.zeros_(m.bias)


if __name__ == '__main__':
	print("ASRModel Summary")
	from torchsummary import summary
	from time import time
	from utils import config
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	net = ASRModel(input_features=config.num_mel_banks ,num_classes=config.vocab_size).to(device)
	with torch.no_grad():
		# summary(net, (config.num_mel_banks, 1000), batch_size=1, device=device)
		image = torch.randn(1, config.num_mel_banks, 500).to(device)
		start = time()
		y = net(image)
		print(y.shape)
		print(f"time taken is {time()-start:.3f}s")
