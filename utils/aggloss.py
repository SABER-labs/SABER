import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .config import temperature_softmax


class ACELoss(nn.Module):

    def __init__(self, label_smoothing=0.1):
        super().__init__()
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets, input_lengths, target_lengths):
        logits = logits.permute(2, 0, 1)
        T_, bs, class_size = logits.size()
        tagets_split = list(torch.split(targets, target_lengths.tolist()))
        targets_padded = torch.nn.utils.rnn.pad_sequence(
            tagets_split, batch_first=True, padding_value=0)
        targets_padded = F.one_hot(targets_padded.long(
        ), num_classes=class_size)  # batch, seq, class
        targets_padded = targets_padded.mul(1.0 - self.label_smoothing) + (
            1 - targets_padded).mul(self.label_smoothing / (class_size - 1))
        # sum across seq, to get batch * class
        targets_padded = torch.sum(targets_padded, 1).float().cuda()
        targets_padded[:, 0] = T_ - target_lengths
        probs = torch.softmax(logits, dim=2)  # softmax on class
        probs = torch.sum(probs, 0)  # sum across seq, to get batch * class
        probs = probs/T_
        targets_padded = targets_padded/T_
        loss1 =  -torch.sum(targets_padded * torch.log(probs)) / bs
        # targets_padded = F.normalize(targets_padded, p=1, dim=1)
        # loss2 = F.kl_div(torch.log(probs), targets_padded, reduction='batchmean')
        return loss1


class FocalACELoss(ACELoss):

    def __init__(self, alpha=0.5, gamma=1.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, probs, targets, input_lengths, target_lengths):
        loss = super().forward(probs, targets, input_lengths, target_lengths)
        p = torch.exp(-loss)
        return self.alpha * torch.pow((1-p), self.gamma) * loss


class CustomCTCLoss(nn.Module):

    def forward(self, logits, targets, target_lengths, input_lengths):
        log_probs = torch.log_softmax(logits, dim=1)
        log_probs = log_probs.permute(2, 0, 1).contiguous()
        # input_lengths = torch.full(size=(log_probs.size(1),), fill_value=log_probs.size(0), dtype=torch.int32)
        return F.ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=0, reduction='mean', zero_infinity=False)


class CustomFocalCTCLoss(CustomCTCLoss):

    def __init__(self, alpha=1.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, probs, targets, target_lengths):
        loss = super().forward(probs, targets, target_lengths)
        p = torch.exp(-loss)
        return self.alpha * torch.pow((1-p), self.gamma) * loss


class UDALoss(nn.Module):

    def forward(self, probs_imgs, probs_aug_imgs):
        bs, class_size, T_ = probs_imgs.size()
        probs1 = torch.softmax(probs_imgs / temperature_softmax, dim=1)
        probs2 = torch.softmax(probs_aug_imgs, dim=1)
        probs1 = torch.sum(probs1, 2)
        probs2 = torch.sum(probs2, 2)
        probs1 = probs1/T_
        probs2 = probs2/T_
        return F.kl_div(torch.log(probs2), probs1, reduction="batchmean")


class FocalUDALoss(UDALoss):

    def __init__(self, alpha=0.25, gamma=0.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, probs_imgs, probs_aug_imgs):
        loss = super().forward(probs_imgs, probs_aug_imgs)
        p = torch.exp(-loss)
        return self.alpha * torch.pow((1-p), self.gamma) * loss
