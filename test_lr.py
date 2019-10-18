# code from https://github.com/davidtvs/pytorch-lr-finder

import torch
from torch_lr_finder import LRFinder
from datasets.librispeech import allign_collate
import os
from utils.lmdb import lmdbDataset
from utils.aggloss import ACELoss, CustomCTCLoss, FocalACELoss
from models.mixnet import ASRModel
from torch.optim.lr_scheduler import _LRScheduler
from utils.radam import RAdam
from utils import config
from functools import partial
from tqdm import tqdm
import matplotlib.pyplot as plt

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

class LinearLR(_LRScheduler):
    """Linearly increases the learning rate between two boundaries over a number of
    iterations.

    Arguments:
        optimizer (torch.optim.Optimizer): wrapped optimizer.
        end_lr (float, optional): the initial learning rate which is the lower
            boundary of the test. Default: 10.
        num_iter (int, optional): the number of iterations over which the test
            occurs. Default: 100.
        last_epoch (int): the index of last epoch. Default: -1.

    """

    def __init__(self, optimizer, end_lr, num_iter, last_epoch=-1):
        self.end_lr = end_lr
        self.num_iter = num_iter
        super(LinearLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        curr_iter = self.last_epoch + 1
        r = curr_iter / self.num_iter
        return [base_lr + r * (self.end_lr - base_lr) for base_lr in self.base_lrs]


class ExponentialLR(_LRScheduler):
    """Exponentially increases the learning rate between two boundaries over a number of
    iterations.

    Arguments:
        optimizer (torch.optim.Optimizer): wrapped optimizer.
        end_lr (float, optional): the initial learning rate which is the lower
            boundary of the test. Default: 10.
        num_iter (int, optional): the number of iterations over which the test
            occurs. Default: 100.
        last_epoch (int): the index of last epoch. Default: -1.

    """

    def __init__(self, optimizer, end_lr, num_iter, last_epoch=-1):
        self.end_lr = end_lr
        self.num_iter = num_iter
        super(ExponentialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        curr_iter = self.last_epoch + 1
        r = curr_iter / self.num_iter
        return [base_lr * (self.end_lr / base_lr) ** r for base_lr in self.base_lrs]

class NewLRFinder(LRFinder):

    def range_test(
        self,
        train_loader,
        val_loader=None,
        end_lr=10,
        num_iter=100,
        step_mode="exp",
        smooth_f=0.05,
        diverge_th=5,
    ):
        self.history = {"lr": [], "loss": []}
        self.best_loss = None

        self.model.to(self.device)

        if step_mode.lower() == "exp":
            lr_schedule = ExponentialLR(self.optimizer, end_lr, num_iter)
        elif step_mode.lower() == "linear":
            lr_schedule = LinearLR(self.optimizer, end_lr, num_iter)
        else:
            raise ValueError("expected one of (exp, linear), got {}".format(step_mode))

        if smooth_f < 0 or smooth_f >= 1:
            raise ValueError("smooth_f is outside the range [0, 1[")

        iterator = iter(train_loader)
        for iteration in tqdm(range(num_iter)):
            try:
                inputs, labels, label_lengths = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                inputs, labels, label_lengths = next(iterator)

            # Train on batch and retrieve loss
            loss = self._train_batch(inputs, labels, label_lengths)
            if val_loader:
                loss = self._validate(val_loader)

            # Update the learning rate
            lr_schedule.step()
            self.history["lr"].append(lr_schedule.get_lr()[0])

            # Track the best loss and smooth it if smooth_f is specified
            if iteration == 0:
                self.best_loss = loss
            else:
                if smooth_f > 0:
                    loss = smooth_f * loss + (1 - smooth_f) * self.history["loss"][-1]
                if loss < self.best_loss:
                    self.best_loss = loss

            # Check if the loss has diverged; if it has, stop the test
            self.history["loss"].append(loss)
            if loss > diverge_th * self.best_loss:
                print("Stopping early, the loss has diverged")
                break

        print("Learning rate search finished. See the graph with {finder_name}.plot()")

    def _train_batch(self, inputs, labels, label_lengths):
        # Set model to training mode
        self.model.train()

        # Move data to the correct device
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        # Forward pass
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels, torch.tensor([outputs.size(1)]*outputs.size(0), dtype=torch.long), label_lengths)

        # Backward pass
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def _validate(self, dataloader):
        # Set model to evaluation mode and disable gradient computation
        running_loss = 0
        self.model.eval()
        with torch.no_grad():
            for inputs, labels, label_lengths in dataloader:
                # Move data to the correct device
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # Forward pass and loss computation
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels, torch.tensor([outputs.size(1)]*outputs.size(0), dtype=torch.long), label_lengths)
                running_loss += loss.item() * inputs.size(0)

        return running_loss / len(dataloader.dataset)

    def plot(self, skip_start=10, skip_end=5, log_lr=True):
        """Plots the learning rate range test.

        Arguments:
            skip_start (int, optional): number of batches to trim from the start.
                Default: 10.
            skip_end (int, optional): number of batches to trim from the start.
                Default: 5.
            log_lr (bool, optional): True to plot the learning rate in a logarithmic
                scale; otherwise, plotted in a linear scale. Default: True.

        """

        if skip_start < 0:
            raise ValueError("skip_start cannot be negative")
        if skip_end < 0:
            raise ValueError("skip_end cannot be negative")

        # Get the data to plot from the history dictionary. Also, handle skip_end=0
        # properly so the behaviour is the expected
        lrs = self.history["lr"]
        losses = self.history["loss"]
        if skip_end == 0:
            lrs = lrs[skip_start:]
            losses = losses[skip_start:]
        else:
            lrs = lrs[skip_start:-skip_end]
            losses = losses[skip_start:-skip_end]

        # Plot loss as a function of the learning rate
        plt.plot(lrs, losses)
        if log_lr:
            plt.xscale("log")
        plt.xlabel("Learning rate")
        plt.ylabel("Loss")
        plt.savefig("1cycle_policy.png")


device = 'cuda' if torch.cuda.is_available() else 'cpu'
allign_collate_partial = partial(allign_collate, device=device)
trainCleanPath = os.path.join(config.lmdb_root_path, 'train-labelled')
train_clean = lmdbDataset(root=trainCleanPath)
trainloader = torch.utils.data.DataLoader(train_clean, batch_size=4, shuffle=True, num_workers=20, collate_fn=allign_collate_partial)

model = ASRModel(input_features=config.num_mel_banks ,num_classes=config.vocab_size).to(device)
criterion = ACELoss()
optimizer = RAdam(model.parameters(), lr=1e-7, eps=1e-5)
lr_finder = NewLRFinder(model, optimizer, criterion, device="cuda")
lr_finder.range_test(trainloader, end_lr=100, num_iter=200)
lr_finder.plot()

'''
from cyclicLR import CyclicCosAnnealingLR
import torch

figure max_lr, min_lr

optimizer = torch.optim.SGD(model.parameters(),lr=1e-3)
scheduler = CyclicCosAnnealingLR(optimizer,milestones=[10,25,60,80,120,180,240,320,400,480],decay_milestones=[60, 120, 240, 480, 960],eta_min=1e-6)
for epoch in range(500):
  scheduler.step()
  train(..)
  validate(..)
'''