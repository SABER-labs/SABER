import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ACELoss(nn.Module):

    def forward(self, probs, targets, input_lengths, target_lengths):
        bs, class_size, T_ = probs.size()
        probs = torch.softmax(probs, dim=1)
        tagets_split = list(torch.split(targets, target_lengths.tolist()))
        targets_padded = torch.nn.utils.rnn.pad_sequence(tagets_split, batch_first=True, padding_value=0)
        targets_padded = F.one_hot(targets_padded.long(), num_classes=class_size) # batch, seq, class
        targets_padded = torch.sum(targets_padded, 1).float().cuda() # sum across seq, to get batch * class
        targets_padded[:,0] = T_ - target_lengths
        probs = torch.sum(probs, 2) # sum across seq, to get batch * class
        probs = probs/T_
        targets_padded = targets_padded/T_
        return (-torch.sum(torch.log(probs)*targets_padded)) / bs
        # return F.kl_div(torch.log(probs), targets_padded, reduction="batchmean")

    def update_epoch(self, epoch):
        self.epoch = epoch

class FocalACELoss(ACELoss):

    def __init__(self, alpha=0.25, gamma=0.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, probs, targets, input_lengths, target_lengths):
        loss = super().forward(probs, targets, input_lengths, target_lengths)
        p = torch.exp(-loss)
        return self.alpha * torch.pow((1-p), self.gamma) * loss

class UDALoss(nn.Module):

    def forward(self, probs_imgs, probs_aug_imgs):
        bs, class_size, T_ = probs_imgs.size()
        probs1 = torch.softmax(probs_imgs, dim=1)
        probs2 = torch.softmax(probs_aug_imgs, dim=1)
        probs1 = torch.sum(probs1, 2)
        probs2 = torch.sum(probs2, 2)
        probs1 = probs1/T_
        probs2 = probs2/T_
        return F.kl_div(torch.log(probs2), probs1, reduction="batchmean")

class CustomCTCLoss(nn.Module):

    def forward(self, probs, targets, input_lengths, target_lengths):
        probs = probs.permute(2, 0, 1)
        log_probs = torch.log_softmax(probs, dim=2)
        return F.ctc_loss(log_probs, targets, input_lengths, target_lengths, zero_infinity=True)

class FocalUDALoss(UDALoss):

    def __init__(self, alpha=0.25, gamma=0.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, probs_imgs, probs_aug_imgs):
        loss = super().forward(probs_imgs, probs_aug_imgs)
        p = torch.exp(-loss)
        return self.alpha * torch.pow((1-p), self.gamma) * loss
