from datasets.librispeech import sequence_to_string, get_sentence
import torch.nn as nn
import torch

def get_most_probable(tensor):
    values, preds_idx = tensor.max(1)
    sentences = [get_sentence(seq) for seq in preds_idx.tolist()]
    return sentences

def ctc_greedy_decoder_topk(probs, blank=0):
    top_kprobs, top_kindexes = torch.topk(torch.softmax(probs.permute(1, 0), dim=1), k=2, dim=1)
    letter_miss_cond = (top_kprobs[:, 0] - top_kprobs[:, 1]) < 0.1
    is_index_zero = top_kindexes[:, 0] == 0
    confusion = ((letter_miss_cond) & is_index_zero).type(torch.long)

    max_indexes = top_kindexes[torch.arange(top_kindexes.shape[0]), confusion]
    max_probs = top_kprobs[:, 0]
    mask = torch.cat([
        torch.tensor([1], dtype=torch.bool, device=probs.device),
        ((max_indexes[:-1] - max_indexes[1:]).abs() > 0)
    ])
    mask = mask * (max_indexes != blank)
    return max_probs[mask].mean(), max_indexes[mask]

def get_most_probable_topk(tensor):
    values, preds_idx = ctc_greedy_decoder_topk(tensor[0])
    sentences = sequence_to_string(preds_idx.tolist())
    return sentences

def get_model_size(model):
    return sum(p.numel() for p in model.parameters())/1000000.0

class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        b = torch.softmax(x, dim=1) * torch.log_softmax(x, dim=1)
        b = -1.0 * b.sum()
        return b