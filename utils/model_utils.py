from datasets.librispeech import sequence_to_string, get_sentence, get_vocab_list
import torch.nn as nn
import torch
from utils.config import alpha_lm, beta_lm, lm_model_path, beam_width, cpus_for_beam_search, sentencepiece_model
# from ctc_decoders import ctc_beam_search_decoder_batch, Scorer
import os

# import pdb; pdb.set_trace()
# scorer = Scorer(alpha_lm, beta_lm, lm_model_path, sentencepiece_model, get_vocab_list())

def get_most_probable(tensor):
    values, preds_idx = tensor.max(1)
    sentences = [get_sentence(seq) for seq in preds_idx.tolist()]
    return sentences

# def get_most_probable_beam(tensor):
#     probs_list = tensor.permute(0, 2, 1).cpu().tolist()
#     res = ctc_beam_search_decoder_batch(probs_list, get_vocab_list(), beam_size=beam_width, num_processes=cpus_for_beam_search, ext_scoring_func=scorer)
#     return [re[0][0] for re in res]

def get_model_size(model):
    return sum(p.numel() for p in model.parameters())/1000000.0

class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        b = torch.softmax(x, dim=1) * torch.log_softmax(x, dim=1)
        b = -1.0 * b.sum()
        return b