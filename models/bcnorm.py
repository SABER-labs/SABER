import torch
import torch.nn as nn
import torch.nn.functional as F

class BCNorm(nn.Module):

    def __init__(self, num_channels, num_groups=8, eps=1e-8, estimate=False):
        super(BCNorm, self).__init__()
        self.num_channels = num_channels
        self.num_groups = num_groups
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, num_groups, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_groups, 1))
        if estimate:
            self.bn = EstBN(num_channels)
        else:
            self.bn = nn.BatchNorm1d(num_channels)

    def forward(self, inp):
        out = self.bn(inp)
        out = out.view(1, inp.size(0) * self.num_groups, -1)
        out = torch.batch_norm(out, None, None, None, None, True, 0, self.eps, True)
        out = out.view(inp.size(0), self.num_groups, -1)
        out = self.weight * out + self.bias
        out = out.view_as(inp)
        return out

class EstBN(nn.Module):

    def __init__(self, num_features):
        super(EstBN, self).__init__()
        self.num_features = num_features
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        self.register_buffer('estbn_moving_speed', torch.zeros(1))

    def forward(self, inp):
        ms = self.estbn_moving_speed.item()
        if self.training:
            with torch.no_grad():
                inp_t = inp.transpose(0, 1).contiguous().view(self.num_features, -1)
                running_mean = inp_t.mean(dim=1)
                inp_t = inp_t - self.running_mean.view(-1, 1)
                running_var = torch.mean(inp_t * inp_t, dim=1)
                self.running_mean.data.mul_(1 - ms).add_(ms * running_mean.data)
                self.running_var.data.mul_(1 - ms).add_(ms * running_var.data)
        out = inp - self.running_mean.view(1, -1, 1)
        out = out / torch.sqrt(self.running_var + 1e-5).view(1, -1, 1)
        weight = self.weight.view(1, -1, 1)
        bias = self.bias.view(1, -1, 1)
        out = weight * out + bias
        return out