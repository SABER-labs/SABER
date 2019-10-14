import warnings
import torch
import numpy as np


class TrainingSignalAnnealing:

    def __init__(self, num_steps, mode='log', num_class=512):
        assert mode in ("linear", "log", "exp")
        alphas = {
            'log': 1 - np.exp(-np.arange(num_steps)/num_steps * 5),
            'linear': np.arange(num_steps)/num_steps,
            'exp': np.exp((np.arange(num_steps)/num_steps - 1) * 5)
        }

        self.thresholds = (alphas[mode] * (1 - 1/num_class)) + (1/num_class)
        self._step = 0
        self.preds_as_probas = preds_as_probas

    def __call__(self, y_pred, y, step=None):
        step = self._step if step is None else step
        self._step += 1

        if step >= len(self.thresholds) or step < 0:
            warnings.warn("Step {} is out of bounds".format(step))
            return y_pred, y

        t = self.thresholds[step]
        tmp_y_pred = y_pred.detach()
        res = tmp_y_pred.gather(dim=1, index=y.unsqueeze(dim=1))
        mask = (res < t).squeeze(dim=1)
        if mask.sum() > 0:            
            return y_pred[mask], y[mask]

        warnings.warn("Threshold {} is too low, all predictions are discarded.\n".format(t) +
                      "y_pred.min/max: {}, {}".format(tmp_y_pred.min(), tmp_y_pred.max()))
        return y_pred, y