import torch
from utils.logger import logger
import utils.config as config
import os
from torch.utils.data import Dataset
from ignite.utils import convert_tensor
import numpy as np
from ignite.engine import Events, Engine
from ignite._utils import _to_hours_mins_secs
import time

def _prepare_batch(batch, device, non_blocking):
    x, y = batch
    return (convert_tensor(x, device=device, non_blocking=non_blocking),
            convert_tensor(y, device=device, non_blocking=non_blocking))

def cycle(iterable):
    while True:
        for i in iterable:
            yield i

def get_learning_rate(optimizer):
    lr = 0
    for param_group in optimizer.param_groups:
        lr += param_group['lr']
    return lr

def set_optimizer_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def save_checkpoint(model, optimizer, best_meter, wer, cer, epoch):
    state = {'model_state_dict': model.state_dict(),
             'optimizer_state_dict': optimizer.state_dict(),
             'optimizer_lr': get_learning_rate(optimizer),
             'best_stats': best_meter,
             'epoch': epoch}
    os.makedirs(config.checkpoint_root, exist_ok=True)
    checkpoint_path = os.path.join(config.checkpoint_root, f"saber_w{wer:.3f}_c{cer:.3f}_e{epoch}.pth")
    torch.save(state, checkpoint_path)
    logger.info(f'models saved to {checkpoint_path}')
    if wer < best_meter.best_wer:
        checkpoint_path = os.path.join(config.checkpoint_root, config.best_model_version)
        torch.save(state, checkpoint_path)
        logger.info(f'Best Model Saved to {checkpoint_path}')

def load_state_dict_model(model, pre_trainmodel):
    model_dict = model.state_dict()
    for k, _ in model_dict.items():
        pkeys = [k.replace('module.', '') for k in list(pre_trainmodel.keys())]
        if k in list(pre_trainmodel.keys()):
            model_dict[k] = pre_trainmodel[k]
        elif k in pkeys:
            model_dict[k] = pre_trainmodel['module.' + k]
    model.load_state_dict(model_dict)

def load_checkpoint(model, optimizer, params):
    version = config.best_model_version
    if config.checkpoint_version != '':
        version = config.checkpoint_version
    checkpoint_path = os.path.join(config.checkpoint_root, version)
    if os.path.exists(checkpoint_path):
        loader = torch.load(checkpoint_path, map_location='cpu')
        load_state_dict_model(model, loader['model_state_dict'])
        optimizer.load_state_dict(loader['optimizer_state_dict'])
        if torch.cuda.is_available():
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()
        set_optimizer_learning_rate(optimizer, loader['optimizer_lr'])
        # set_optimizer_learning_rate(optimizer, config.lr)
        params['best_stats'] = loader['best_stats']
        params['start_epoch'] = loader['epoch']
        logger.info(f'Checkpoint and Optimizer loaded from {checkpoint_path}')
    else:
        logger.info(f'Couldn\'t load from {checkpoint_path}')

class BestMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.best_epoch = 0
        self.best_wer = 10000
        self.best_cer = 10000

    def update(self, wer, cer, epoch):
        if wer < self.best_wer:
            self.best_wer = wer
            self.best_cer = cer
            self.best_epoch = epoch

    def __str__(self):
        return f"Epoch: {self.best_epoch}, WER: {self.best_wer:.3f}, CER: {self.best_cer:.3f}"