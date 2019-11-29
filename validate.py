from utils.model_utils import get_most_probable
from datasets.librispeech import get_sentence

from utils import config
import os
import torch
from models.mixnet import ASRModel
# from models.wav2letter import ASRModel
from utils.logger import logger
from functools import partial
from datasets.librispeech import allign_collate, align_collate_unlabelled, allign_collate_val
from utils.lmdb import lmdbDataset, lmdbDoubleDataset
from utils.training_utils import save_checkpoint, BestMeter
from utils.config import lmdb_root_path, workers, train_batch_size, unsupervision_warmup_epoch, log_path, epochs
import ignite
from ignite.engine import Events, Engine
from ignite.metrics import Loss, RunningAverage
from utils.metrics import WordErrorRate, CharacterErrorRate
from ignite.handlers import ModelCheckpoint, Timer
from ignite.contrib.handlers.tensorboard_logger import *
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from utils.optimizers import RAdam, NovoGrad, Ranger
from utils.cyclicLR import CyclicCosAnnealingLR
from utils.loss_scaler import DynamicLossScaler
from utils.aggloss import ACELoss, UDALoss, CustomCTCLoss, FocalACELoss, FocalUDALoss
from utils.training_utils import load_checkpoint
import numpy as np
import toml

torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


def init_parms():
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get(
        'CUDA_VISIBLE_DEVICES', config.gpu_id)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    params = {
        'device': device,
        'start_epoch': -1
    }
    return params


def main():
    params = init_parms()
    device = params.get('device')
    model = ASRModel(input_features=config.num_mel_banks,
                     num_classes=config.vocab_size).to(device)
    model = torch.nn.DataParallel(model)
    model.eval()
    optimizer = Ranger(model.parameters(), lr=config.lr, eps=1e-5)
    load_checkpoint(model, optimizer, params)

    testCleanPath = os.path.join(lmdb_root_path, 'test-clean')
    testOtherPath = os.path.join(lmdb_root_path, 'test-other')

    test_clean = lmdbDataset(root=testCleanPath)
    test_other = lmdbDataset(root=testOtherPath)

    logger.info(
        f'Loaded Test Datasets, test_clean={len(test_clean)} & test_other={len(test_other)} examples')

    @torch.no_grad()
    def validate_update_function(engine, batch):
        img, labels, label_lengths = batch
        y_pred = model(img.to(device))
        if np.random.rand() > 0.99:
            pred_sentences = get_most_probable(y_pred)
            labels_list = labels.tolist()
            idx = 0
            for i, length in enumerate(label_lengths.cpu().tolist()):
                pred_sentence = pred_sentences[i]
                gt_sentence = get_sentence(labels_list[idx:idx+length])
                idx += length
                print(f"Pred sentence: {pred_sentence}, GT: {gt_sentence}")
        return (y_pred, labels, label_lengths)

    allign_collate_partial = partial(allign_collate, device=device)
    align_collate_unlabelled_partial = partial(
        align_collate_unlabelled, device=device)
    allign_collate_val_partial = partial(allign_collate_val, device=device)

    test_loader_clean = torch.utils.data.DataLoader(
        test_clean, batch_size=train_batch_size, shuffle=False, num_workers=config.workers, pin_memory=False, collate_fn=allign_collate_val_partial)
    test_loader_other = torch.utils.data.DataLoader(
        test_other, batch_size=train_batch_size, shuffle=False, num_workers=config.workers, pin_memory=False, collate_fn=allign_collate_val_partial)
    evaluator_clean = Engine(validate_update_function)
    evaluator_other = Engine(validate_update_function)
    metrics = {'wer': WordErrorRate(), 'cer': CharacterErrorRate()}
    for name, metric in metrics.items():
        metric.attach(evaluator_clean, name)
        metric.attach(evaluator_other, name)

    evaluator_clean.run(test_loader_clean)
    evaluator_other.run(test_loader_other)

    metrics_clean = evaluator_clean.state.metrics
    metrics_other = evaluator_other.state.metrics

    print(f"Clean wer: {metrics_clean['wer']} Clean cer: {metrics_clean['cer']}")
    print(f"Other wer: {metrics_other['wer']} Other cer: {metrics_other['cer']}")

if __name__ == "__main__":
    main()