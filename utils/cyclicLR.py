# code from https://github.com/bluesky314/Cyclical_LR_Scheduler_With_Decay_Pytorch

import math
from bisect import bisect_right, bisect_left

import torch
import numpy as np
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer


class CyclicCosAnnealingLR(_LRScheduler):
    r"""

    Implements reset on milestones inspired from CosineAnnealingLR pytorch

    Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr and
    :math:`T_{cur}` is the number of epochs since the last restart in SGDR:
    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 +
        \cos(\frac{T_{cur}}{T_{max}}\pi))
    When last_epoch > last set milestone, lr is automatically set to \eta_{min}
    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_. Note that this only
    implements the cosine annealing part of SGDR, and not the restarts.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        milestones (list of ints): List of epoch indices. Must be increasing.
        decay_milestones(list of ints):List of increasing epoch indices. Ideally,decay values should overlap with milestone points
        gamma (float): factor by which to decay the max learning rate at each decay milestone
        eta_min (float): Minimum learning rate. Default: 1e-6
        last_epoch (int): The index of last epoch. Default: -1.


    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(self, optimizer, milestones=[10, 25, 60, 80, 120, 180, 240, 320, 400, 480, 1000], decay_milestones=[60, 120, 240, 480, 960], epoch_length=1000, gamma=0.5, eta_min=1e-6, last_epoch=-1):
        if not list(milestones) == sorted(milestones):
            raise ValueError('Milestones should be a list of'
                             ' increasing integers. Got {}', milestones)
        self.eta_min = eta_min

        self.milestones = (np.array(milestones) *
                           epoch_length / 1000).astype(np.int32)
        self.milestones2 = (np.array(decay_milestones) *
                            epoch_length / 1000).astype(np.int32)

        self.gamma = gamma
        super(CyclicCosAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):

        if self.last_epoch >= self.milestones[-1]:
            return [self.eta_min for base_lr in self.base_lrs]

        idx = bisect_right(self.milestones, self.last_epoch)

        left_barrier = 0 if idx == 0 else self.milestones[idx-1]
        right_barrier = self.milestones[idx]

        width = right_barrier - left_barrier
        curr_pos = self.last_epoch - left_barrier

        if isinstance(self.milestones2, np.ndarray):
            return [self.eta_min + (base_lr * self.gamma ** bisect_right(self.milestones2, self.last_epoch) - self.eta_min) *
                    (1 + math.cos(math.pi * curr_pos / width)) / 2
                    for base_lr in self.base_lrs]
        else:
            return [self.eta_min + (base_lr - self.eta_min) *
                    (1 + math.cos(math.pi * curr_pos / width)) / 2
                    for base_lr in self.base_lrs]


class CyclicLinearLR(_LRScheduler):
    r"""
    Implements reset on milestones inspired from Linear learning rate decay

    Set the learning rate of each parameter group using a linear decay
    schedule, where :math:`\eta_{max}` is set to the initial lr and
    :math:`T_{cur}` is the number of epochs since the last restart:
    .. math::
        \eta_t = \eta_{min} + (\eta_{max} - \eta_{min})(1 -\frac{T_{cur}}{T_{max}})
    When last_epoch > last set milestone, lr is automatically set to \eta_{min}

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        milestones (list of ints): List of epoch indices. Must be increasing.
        decay_milestones(list of ints):List of increasing epoch indices. Ideally,decay values should overlap with milestone points
        gamma (float): factor by which to decay the max learning rate at each decay milestone
        eta_min (float): Minimum learning rate. Default: 1e-6
        last_epoch (int): The index of last epoch. Default: -1.
    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(self, optimizer, milestones=[10, 25, 60, 80, 120, 180, 240, 320, 400, 480, 1000], decay_milestones=[60, 120, 240, 480, 960], epoch_length=1000, gamma=0.5, eta_min=1e-6, last_epoch=-1):
        if not list(milestones) == sorted(milestones):
            raise ValueError('Milestones should be a list of'
                             ' increasing integers. Got {}', milestones)
        self.eta_min = eta_min

        self.gamma = gamma
        self.milestones = (np.array(milestones) *
                           epoch_length / 1000).astype(np.int32)
        self.milestones2 = (np.array(decay_milestones) *
                            epoch_length / 1000).astype(np.int32)
        super(CyclicLinearLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):

        if self.last_epoch >= self.milestones[-1]:
            return [self.eta_min for base_lr in self.base_lrs]

        idx = bisect_right(self.milestones, self.last_epoch)

        left_barrier = 0 if idx == 0 else self.milestones[idx-1]
        right_barrier = self.milestones[idx]

        width = right_barrier - left_barrier
        curr_pos = self.last_epoch - left_barrier

        if isinstance(self.milestones2, np.ndarray):
            return [self.eta_min + (base_lr * self.gamma ** bisect_right(self.milestones2, self.last_epoch) - self.eta_min) *
                    (1. - 1.0*curr_pos / width)
                    for base_lr in self.base_lrs]

        else:
            return [self.eta_min + (base_lr - self.eta_min) *
                    (1. - 1.0*curr_pos / width)
                    for base_lr in self.base_lrs]


if __name__ == "__main__":
    from crnn_pytorch.models.grcnn_v3 import CRNN_CNN_MixNet
    import matplotlib.pyplot as plt
    model = CRNN_CNN_MixNet(50)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    epochs = 1000
    scheduler = CyclicLinearLR(
        optimizer, epoch_length=epochs, eta_min=5e-5)
    current_lr = []
    for i in range(epochs):
        current_lr.append(scheduler.get_lr())
        scheduler.step()
    plt.plot(list(range(epochs)), current_lr)
    plt.xlabel("epoch")
    plt.ylabel("Learning rate")
    plt.savefig("cyclic_annealing.png")