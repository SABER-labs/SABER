import Levenshtein as Lev
import numpy as np
from utils.model_utils import get_most_probable
from ignite.metrics import Metric, Accuracy
from ignite.metrics.metric import reinit__is_reduced
from datasets.librispeech import get_sentence, get_vocab_list
import torch

def clean_gt(s2):
    s2 = ''.join([s for s in s2 if s in get_vocab_list()])
    return s2.strip()

def werCalc(s1, s2):
    """
    Computes the Word Error Rate, defined as the edit distance between the
    two provided sentences after tokenizing to words.
    Arguments:
        s1 (string): space-separated sentence
        s2 (string): space-separated sentence
    """
    s1 = s1.lower()
    s2 = clean_gt(s2.lower())
    # build mapping of words to integers
    b = set(s1.split() + s2.split())
    word2char = dict(zip(b, range(len(b))))

    # map the words to a char array (Levenshtein packages only accepts
    # strings)
    w1 = [chr(word2char[w]) for w in s1.split()]
    w2 = [chr(word2char[w]) for w in s2.split()]
    return Lev.distance(''.join(w1), ''.join(w2)) / len(s2.split(' '))

def cerCalc(s1, s2):
    """
    Computes the Character Error Rate, defined as the edit distance.
    Arguments:
        s1 (string): space-separated sentence
        s2 (string): space-separated sentence
    """
    s1 = s1.lower()
    s2 = clean_gt(s2.lower())
    s1, s2 = s1.replace(' ', ''), s2.replace(' ', '')
    return Lev.distance(s1, s2) / len(s2)

def batch_wer_accuracy(preds, labels, label_lengths):
    pred_sentences = get_most_probable(preds)
    labels_list = labels.tolist()
    idx = 0
    wer = []
    for i, length in enumerate(label_lengths.cpu().tolist()):
        pred_sentence = pred_sentences[i]
        gt_sentence = get_sentence(labels_list[idx:idx+length])
        wer.append(werCalc(pred_sentence, gt_sentence))
        idx += length
    return np.sum(wer)

def batch_cer_accuracy(preds, labels, label_lengths):
    pred_sentences = get_most_probable(preds)
    labels_list = labels.tolist()
    idx = 0
    cer = []
    for i, length in enumerate(label_lengths.cpu().tolist()):
        pred_sentence = pred_sentences[i]
        gt_sentence = get_sentence(labels_list[idx:idx+length])
        cer.append(cerCalc(pred_sentence, gt_sentence))
        idx += length
    return np.sum(cer)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n

    def get_average():
        return self.sum / self.count

class WordErrorRate(Accuracy):

    @reinit__is_reduced
    def update(self, output):
        with torch.no_grad():
            y_pred, labels, label_lengths = output
            wer = batch_wer_accuracy(y_pred, labels, label_lengths)
        self._num_correct += wer
        self._num_examples += label_lengths.shape[0]

class CharacterErrorRate(Accuracy):

    @reinit__is_reduced
    def update(self, output):
        with torch.no_grad():
            y_pred, labels, label_lengths = output
            cer = batch_cer_accuracy(y_pred, labels, label_lengths)
        self._num_correct += cer
        self._num_examples += label_lengths.shape[0]

if __name__ == "__main__":
    s1 = "WTF WHO ARE YOu ? $!"
    s2 = "wff who are you"
    wer = werCalc(s2, s1)
    cer = cerCalc(s2, s1)
    print(wer, cer)