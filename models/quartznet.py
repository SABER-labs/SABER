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


def ConvBnNonLinearity(in_channels, out_channels, kernel_width, stride, non_linear='ReLU', dilation=1, dropout=0):
	p = kernel_width//2 if dilation == 1 else kernel_width - 1
	if dropout > 0:
		return nn.Sequential(
			nn.Conv1d(in_channels, out_channels, kernel_width,
					  stride, p, bias=False, dilation=dilation),
			nn.BatchNorm1d(out_channels),
			NON_LINEARITY[non_linear],
			nn.Dropout(dropout)
		)

	else:
		return nn.Sequential(
			nn.Conv1d(in_channels, out_channels, kernel_width,
					  stride, p, bias=False, dilation=dilation),
			nn.BatchNorm1d(out_channels),
			NON_LINEARITY[non_linear]
		)


def Conv1x1Bn(in_channels, out_channels, non_linear='ReLU'):
	if non_linear is not None:
		return [
			nn.Conv1d(in_channels, out_channels, 1, 1, 0, bias=False),
			nn.BatchNorm1d(out_channels),
			NON_LINEARITY[non_linear]
		]
	else:
		return [
			nn.Conv1d(in_channels, out_channels, 1, 1, 0, bias=False),
			nn.BatchNorm1d(out_channels),
		]


class DepthWiseSeperableConv(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
		super(DepthWiseSeperableConv, self).__init__()

		self.conv1 = nn.Conv1d(in_channels, in_channels, kernel_size,
							   stride, kernel_size//2, dilation, groups=in_channels, bias=bias)
		self.pointwise = nn.Conv1d(
			in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

	def forward(self, x):
		x = self.conv1(x)
		x = self.pointwise(x)
		return x


class QuartzBlock(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride, non_linear='ReLU', n_repeats=5):
		super(QuartzBlock, self).__init__()
		conv = []
		for i in range(n_repeats):
			if i == 0:
				conv.extend(self._get_standard_quartz_arm(
					in_channels, out_channels, kernel_size, stride))
			else:
				conv.extend(self._get_standard_quartz_arm(
					out_channels, out_channels, kernel_size, stride))

		self.pointwise_batchnorm = nn.Sequential(
			*Conv1x1Bn(in_channels, out_channels))
		self.non_linearity = nn.Sequential(nn.BatchNorm1d(out_channels), NON_LINEARITY[non_linear])
		self.conv = nn.Sequential(*conv)

	def _get_standard_quartz_arm(self, in_channels, out_channels, kernel_size, stride, non_linear='ReLU'):
		return [
			DepthWiseSeperableConv(
				in_channels, out_channels, kernel_size, stride),
			nn.BatchNorm1d(out_channels),
			NON_LINEARITY[non_linear]
		]

	def forward(self, input):
		arm1 = self.conv(input)
		arm2 = self.pointwise_batchnorm(input)
		return self.non_linearity(arm1 + arm2)


class QuartzRepeats(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride, non_linear='ReLU', n_branches=2):
		super(QuartzRepeats, self).__init__()
		conv = []
		for i in range(n_branches):
			if i == 0:
				conv.append(QuartzBlock(in_channels, out_channels,
					kernel_size, stride, non_linear=non_linear, n_repeats=5))
			else:
				conv.append(QuartzBlock(out_channels, out_channels,
									kernel_size, stride, non_linear=non_linear, n_repeats=5))
		self.conv = nn.Sequential(*conv)

	def forward(self, input):
		return self.conv(input)


class QuartzNet(nn.Module):
	def __init__(self, num_classes=512):
		super().__init__()
		self.conv_block1 = ConvBnNonLinearity(
			80, 256, 11, 2, non_linear='Swish', dropout=0.2)
		self.quartz_block1 = nn.Sequential(QuartzRepeats(
			256, 256, 11, 1, non_linear='Swish'), nn.Dropout(0.2))
		self.quartz_block2 = nn.Sequential(QuartzRepeats(
			256, 384, 13, 1, non_linear='Swish'), nn.Dropout(0.2))
		self.quartz_block3 = nn.Sequential(QuartzRepeats(
			384, 512, 17, 1, non_linear='Swish'), nn.Dropout(0.2))
		self.quartz_block4 = nn.Sequential(QuartzRepeats(
			512, 640, 21, 1, non_linear='Swish'), nn.Dropout(0.3))
		self.quartz_block5 = nn.Sequential(QuartzRepeats(
			640, 768, 25, 1, non_linear='Swish'), nn.Dropout(0.3))
		self.conv_block2 = ConvBnNonLinearity(
			768, 896, 29, 1, non_linear='Swish', dilation=2, dropout=0.4)
		self.conv_block3 = ConvBnNonLinearity(
			896, 1024, 1, 1, non_linear='Swish', dropout=0.4)
		self.conv_block4 = ConvBnNonLinearity(
			1024, num_classes, 1, 1, non_linear='Swish')

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
		return x.permute(0, 2, 1)


if __name__ == '__main__':
	print("QuartzNet Summary")
	from torchsummary import summary
	from time import time
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	net = QuartzNet().to(device)
	summary(net, (80, 3001), batch_size=1, device=device)
	image = torch.randn(1, 80, 3001).to(device)
	start = time()
	y = net(image)
	print(y.shape)
	print(f"time taken is {time()-start:.3f}s")
