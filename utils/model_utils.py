from datasets.librispeech import get_sentence
import torch.nn as nn

def get_most_probable(tensor):
    values, preds_idx = tensor.max(2)
    sentences = [get_sentence(seq) for seq in preds_idx.tolist()]
    return sentences

class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        b = torch.softmax(x, dim=1) * torch.log_softmax(x, dim=1)
        b = -1.0 * b.sum()
        return b