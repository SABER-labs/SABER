import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from .mixnet import NON_LINEARITY, Conv1x1Bn
from .conv1dlayers import create_conv1d_pad

class TimeChannelSeperableConv(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size=1, bias=False, dilation=1):
		super().__init__()
		self.conv1 = create_conv1d_pad(in_channels, in_channels, kernel_size, dilation=dilation, padding='same', bias=bias, groups=in_channels)
		self.pointwise = create_conv1d_pad(
			in_channels, out_channels, 1, padding='same', bias=bias, dilation=1)

	def forward(self, x):
		x = self.conv1(x)
		x = self.pointwise(x)
		return x

def Conv1x1SepBn(in_channels, out_channels, kernel_size=1, dilation=1, bias=False, non_linear='Mish'):
        return nn.Sequential(
            TimeChannelSeperableConv(in_channels, out_channels, kernel_size, dilation=dilation, bias=bias),
            nn.BatchNorm1d(out_channels),
			NON_LINEARITY[non_linear]
        )

class QuartzBlock(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride, non_linear='Mish', n_repeats=5):
		super(QuartzBlock, self).__init__()
		conv = []
		for i in range(n_repeats - 1):
			conv.extend(self._get_standard_quartz_arm(
					in_channels, in_channels, kernel_size, non_linear=non_linear))
		conv.extend(self._get_standard_quartz_arm(
					in_channels, out_channels, kernel_size, non_linear=non_linear))

		self.pointwise_batchnorm = Conv1x1Bn(in_channels, out_channels, non_linearity=False)
		self.non_linearity = NON_LINEARITY[non_linear]
		self.conv = nn.Sequential(*conv, TimeChannelSeperableConv(out_channels, out_channels, kernel_size), nn.BatchNorm1d(out_channels))

	def _get_standard_quartz_arm(self, in_channels, out_channels, kernel_size, non_linear='Mish'):
		return [
				TimeChannelSeperableConv(in_channels, out_channels, kernel_size),
				nn.BatchNorm1d(out_channels),
				NON_LINEARITY[non_linear]
		]

	def forward(self, input):
		arm1 = self.conv(input)
		arm2 = self.pointwise_batchnorm(input)
		return self.non_linearity(arm1 + arm2)

class QuartzRepeats(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride=1, non_linear='Mish', n_branches=3):
		super(QuartzRepeats, self).__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		conv = []
		for i in range(n_branches-1):
			conv.append(QuartzBlock(in_channels, in_channels,
					kernel_size, stride, non_linear=non_linear, n_repeats=5))
		conv.append(QuartzBlock(in_channels, out_channels,
					kernel_size, stride, non_linear=non_linear, n_repeats=5))
		self.conv_layers = nn.ModuleList(conv)

	def forward(self, input):
		conv_output = input
		for layer in self.conv_layers[:-1]:
			conv_output = layer(conv_output) + input
		if self.in_channels == self.out_channels:
			return self.conv_layers[-1](conv_output) + input
		else:
			return self.conv_layers[-1](conv_output)


# class QuartzRepeats(nn.Module):
# 	def __init__(self, in_channels, out_channels, kernel_size, stride=1, non_linear='Mish', n_branches=3):
# 		super(QuartzRepeats, self).__init__()
# 		self.in_channels = in_channels
# 		self.out_channels = out_channels
# 		conv = []
# 		for i in range(n_branches-1):
# 			conv.append(QuartzBlock(in_channels, in_channels,
# 					kernel_size, stride, non_linear=non_linear, n_repeats=5))
# 		self.conv_last = QuartzBlock(in_channels, out_channels,
# 					kernel_size, stride, non_linear=non_linear, n_repeats=5)
# 		self.conv_first = nn.Sequential(*conv)

# 	def forward(self, input):
# 		conv1 = self.conv_first(input)
# 		if self.in_channels == self.out_channels:
# 			return self.conv_last(conv1) + input
# 		else:
# 			return self.conv_last(conv1 + input)


class ASRModel(nn.Module):
	def __init__(self, input_features=80, num_classes=128):
		super().__init__()
		self.conv_block1 = Conv1x1Bn(
			input_features, 256, kernel_size=33, stride=2, non_linear='Mish')
		self.quartz_block1 = QuartzRepeats(
			256, 256, 33, 1, non_linear='Mish')
		self.quartz_block2 = QuartzRepeats(
			256, 256, 39, 1, non_linear='Mish')
		self.quartz_block3 = QuartzRepeats(
			256, 512, 51, 1, non_linear='Mish')
		self.quartz_block4 = QuartzRepeats(
			512, 512, 63, 1, non_linear='Mish')
		self.quartz_block5 = QuartzRepeats(
			512, 512, 75, 1, non_linear='Mish')
		self.conv_block2 = Conv1x1SepBn(
			512, 512, kernel_size=87, non_linear='Mish', dilation=2)
		self.conv_block3 = Conv1x1Bn(
			512, 1024, non_linear='Mish')
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
	net = ASRModel(input_features=input_features, num_classes=128).to(device)
	print(f'Total params: {sum([p.numel() for p in net.parameters()]) / 1000000.0 :.2f} M')
	# summary(net, (input_features, 500), batch_size=1, device=device)
	with torch.no_grad():
		image = torch.randn(1, input_features, 500).to(device)
		start = time()
		y = net(image)
		print(y.shape)
		print(f"time taken is {time()-start:.3f}s")