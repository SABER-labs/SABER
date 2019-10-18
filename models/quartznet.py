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
	'ReLU': nn.ReLU(inplace=True),
	'Swish': HardSwish(),
	'Sigmoid': HardSigmoid()
}


def ConvBnNonLinearity(in_channels, out_channels, kernel_width, stride, non_linear='ReLU', dilation=1, dropout=0, bias=False):
	p = kernel_width//2 if dilation == 1 else kernel_width - 1
	if dropout > 0:
		return nn.Sequential(
			DepthWiseSeperableConv(in_channels, out_channels, kernel_width, stride, dilation=dilation, bias=bias),
			nn.BatchNorm1d(out_channels),
			NON_LINEARITY[non_linear],
			nn.Dropout(dropout)
		)

	else:
		return nn.Sequential(
			DepthWiseSeperableConv(in_channels, out_channels, kernel_width, stride, dilation=dilation, bias=bias),
			nn.BatchNorm1d(out_channels),
			NON_LINEARITY[non_linear]
		)


def Conv1x1Bn(in_channels, out_channels, non_linear=None, bias=False):
	if non_linear is not None:
		return [
			nn.Conv1d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias),
			nn.BatchNorm1d(out_channels),
			NON_LINEARITY[non_linear]
		]
	else:
		return [
			nn.Conv1d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias),
			nn.BatchNorm1d(out_channels),
		]

class DepthWiseSeperableConv(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, dilation=1, bias=False):
		super(DepthWiseSeperableConv, self).__init__()
		p = kernel_size//2 if dilation == 1 else kernel_size - 1
		self.conv1 = nn.Conv1d(in_channels, in_channels, kernel_size,
							   stride, p, dilation, groups=in_channels, bias=bias)
		self.pointwise = nn.Conv1d(
			in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

	def forward(self, x):
		x = self.conv1(x)
		x = self.pointwise(x)
		return x


class QuartzBlock(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride, non_linear='ReLU', n_repeats=5, dropout=0.2):
		super(QuartzBlock, self).__init__()
		conv = []
		for i in range(n_repeats):
			if i == 0:
				conv.extend(self._get_standard_quartz_arm(
					in_channels, out_channels, kernel_size, stride, non_linear=non_linear, dropout=dropout))
			else:
				conv.extend(self._get_standard_quartz_arm(
					out_channels, out_channels, kernel_size, stride, non_linear=non_linear, dropout=dropout))

		self.pointwise_batchnorm = nn.Sequential(
			*Conv1x1Bn(in_channels, out_channels))
		if dropout > 0:
			self.non_linearity = nn.Sequential(NON_LINEARITY[non_linear], nn.Dropout(dropout))
		else:
			self.non_linearity = NON_LINEARITY[non_linear]
		self.conv = nn.Sequential(*conv, nn.Conv1d(out_channels, out_channels, kernel_size, stride, kernel_size//2) , nn.BatchNorm1d(out_channels))

	def _get_standard_quartz_arm(self, in_channels, out_channels, kernel_size, stride, non_linear='ReLU', dropout=0.2):
		if dropout > 0:
			return [
				DepthWiseSeperableConv(in_channels, out_channels, kernel_size, stride),
				nn.BatchNorm1d(out_channels),
				NON_LINEARITY[non_linear],
				nn.Dropout(dropout)
			]
		else:
			return [
				DepthWiseSeperableConv(in_channels, out_channels, kernel_size, stride),
				nn.BatchNorm1d(out_channels),
				NON_LINEARITY[non_linear]
			]

	def forward(self, input):
		arm1 = self.conv(input)
		arm2 = self.pointwise_batchnorm(input)
		return self.non_linearity(arm1 + arm2)


class QuartzRepeats(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride, dropout=0.0, non_linear='ReLU', n_branches=3):
		super(QuartzRepeats, self).__init__()
		conv = []
		for i in range(n_branches):
			if i == 0:
				conv.append(QuartzBlock(in_channels, out_channels,
					kernel_size, stride, non_linear=non_linear, n_repeats=5, dropout=dropout))
			else:
				conv.append(QuartzBlock(out_channels, out_channels,
					kernel_size, stride, non_linear=non_linear, n_repeats=5, dropout=dropout))
		self.conv = nn.Sequential(*conv)

	def forward(self, input):
		return self.conv(input)


class ASRModel(nn.Module):
	def __init__(self, input_features=80, num_classes=128):
		super().__init__()
		self.conv_block1 = ConvBnNonLinearity(
			input_features, 256, 33, 2, non_linear='Swish', dropout=0)
		self.quartz_block1 = nn.Sequential(QuartzRepeats(
			256, 256, 33, 1, dropout=0, non_linear='Swish'), nn.AvgPool1d(2, 2))
		self.quartz_block2 = nn.Sequential(QuartzRepeats(
			256, 256, 39, 1, dropout=0, non_linear='Swish'), nn.AvgPool1d(2, 2))
		self.quartz_block3 = nn.Sequential(QuartzRepeats(
			256, 512, 51, 1, dropout=0, non_linear='Swish'))
		self.quartz_block4 = nn.Sequential(QuartzRepeats(
			512, 512, 63, 1, dropout=0, non_linear='Swish'))
		self.quartz_block5 = nn.Sequential(QuartzRepeats(
			512, 512, 75, 1, dropout=0, non_linear='Swish'))
		self.conv_block2 = ConvBnNonLinearity(
			512, 512, 87, 1, non_linear='Swish', dilation=2, dropout=0)
		self.conv_block3 = ConvBnNonLinearity(
			512, 1024, 1, 1, non_linear='Swish', dropout=0)
		self.conv_block4 = nn.Conv1d(1024, num_classes, 1, 1, bias=True)
		self._initialize_weights()

	def forward(self, x):
		x = self.conv_block1(x)
		x = self.quartz_block1(x)
		x = self.quartz_block2(x)
		x = self.quartz_block3(x)
		x = self.quartz_block4(x)
		x = self.quartz_block5(x)
		x = self.conv_block2(x)
		x = self.conv_block3(x)
		x = self.conv_block4(x)
		return x

	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv1d):
				n = m.kernel_size[0] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(1.0 / n))
				if m.bias is not None:
					m.bias.data.zero_()
			elif isinstance(m, nn.BatchNorm1d):
				if m.track_running_stats:
					m.running_mean.zero_()
					m.running_var.fill_(1)
					m.num_batches_tracked.zero_()
				if m.affine:
					nn.init.ones_(m.weight)
					nn.init.zeros_(m.bias)
			elif isinstance(m, nn.Linear):
				n = m.weight.size(1)
				m.weight.data.normal_(0, math.sqrt(1.0 / n))
				m.bias.data.zero_()


if __name__ == '__main__':
	print("ASRModel Summary")
	from torchsummary import summary
	from time import time
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	input_features = 80
	net = ASRModel(input_features=input_features).to(device)
	summary(net, (input_features, 400), batch_size=1, device=device)
	image = torch.randn(1, input_features, 400).to(device)
	start = time()
	y = net(image)
	print(y.shape)
	print(f"time taken is {time()-start:.3f}s")