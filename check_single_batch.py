from utils.model_utils import get_most_probable, get_model_size
from datasets.librispeech import sequence_to_string

from utils import config
import os
import torch
from models.quartznet import ASRModel
from utils.logger import logger
from functools import partial
from datasets.librispeech import allign_collate, image_train_transform, image_val_transform
from utils.lmdb_tester import lmdbMultiDatasetTester
from utils.training_utils import save_checkpoint, BestMeter
from utils.config import workers, \
    train_batch_size, unsupervision_warmup_epoch, \
    log_path, epochs, \
    lmdb_root_path, lmdb_commonvoice_root_path, \
    lmdb_airtel_root_path, lmdb_airtel_payments_root_path, \
    lmdb_airtel_hinglish_root_path
import ignite
from ignite.engine import Events, Engine
from ignite.metrics import Loss
from utils.metrics import WordErrorRate, CharacterErrorRate
from ignite.handlers import ModelCheckpoint, Timer
from ignite.contrib.handlers.tensorboard_logger import *
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from utils.optimizers import RAdam, NovoGrad, Ranger
from utils.aggloss import ACELoss, UDALoss, CustomCTCLoss, FocalACELoss, FocalUDALoss, CustomFocalCTCLoss
from utils.training_utils import load_checkpoint
import numpy as np
np.random.bit_generator = np.random._bit_generator

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def get_alpha(epoch):
    return np.clip(epoch / unsupervision_warmup_epoch, 0.0, 0.5)


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
    # Init state params
    params = init_parms()
    device = params.get('device')

    # Loading the model, optimizer & criterion
    model = ASRModel(input_features=config.num_mel_banks,
                     num_classes=config.vocab_size).to(device)
    model = torch.nn.DataParallel(model)
    logger.info(f'Model initialized with {get_model_size(model):.3f}M parameters')
    optimizer = Ranger(model.parameters(), lr=config.lr, eps=1e-5)
    load_checkpoint(model, optimizer, params)
    start_epoch = params['start_epoch']
    sup_criterion = CustomCTCLoss()

    # Validation progress bars defined here.
    pbar = ProgressBar(persist=True, desc="Loss")
    pbar_valid = ProgressBar(persist=True, desc="Validate")

    # load timer and best meter to keep track of state params
    timer = Timer(average=True)

    # load all the train data
    logger.info('Begining to load Datasets')
    trainAirtelPaymentsPath = os.path.join(lmdb_airtel_payments_root_path, 'train-labelled-en')

    # form data loaders
    train = lmdbMultiDatasetTester(roots=[trainAirtelPaymentsPath], transform=image_val_transform)

    logger.info(
        f'loaded train & test dataset = {len(train)}')

    def train_update_function(engine, _):
        optimizer.zero_grad()
        imgs_sup, labels_sup, label_lengths, input_lengths = next(
            engine.state.train_loader_labbeled)
        imgs_sup = imgs_sup.to(device)
        labels_sup = labels_sup
        probs_sup = model(imgs_sup)
        sup_loss = sup_criterion(probs_sup, labels_sup, label_lengths, input_lengths)
        sup_loss.backward()
        optimizer.step()
        return sup_loss.item()

    @torch.no_grad()
    def validate_update_function(engine, batch):
        img, labels, label_lengths, image_lengths = batch
        y_pred = model(img.to(device))
        if np.random.rand() > 0.99:
            pred_sentences = get_most_probable(y_pred)
            labels_list = labels.tolist()
            idx = 0
            for i, length in enumerate(label_lengths.cpu().tolist()):
                pred_sentence = pred_sentences[i]
                gt_sentence = sequence_to_string(labels_list[idx:idx+length])
                idx += length
                print(f"Pred sentence: {pred_sentence}, GT: {gt_sentence}")
        return (y_pred, labels, label_lengths)

    train_loader = torch.utils.data.DataLoader(
        train, batch_size=train_batch_size, shuffle=True, num_workers=config.workers, pin_memory=True, collate_fn=allign_collate)
    trainer = Engine(train_update_function)
    evaluator = Engine(validate_update_function)
    metrics = {'wer': WordErrorRate(), 'cer': CharacterErrorRate()}
    iteration_log_step = int(0.33 * len(train_loader))
    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=config.lr_gamma, patience=int(
        config.epochs * 0.05), verbose=True, threshold_mode="abs", cooldown=int(config.epochs * 0.025), min_lr=1e-5)

    pbar.attach(trainer, output_transform=lambda x: {'loss': x})
    pbar_valid.attach(evaluator, [
                      'wer', 'cer'], event_name=Events.EPOCH_COMPLETED, closing_event_name=Events.COMPLETED)
    timer.attach(trainer)

    @trainer.on(Events.STARTED)
    def set_init_epoch(engine):
        engine.state.epoch = params['start_epoch']
        logger.info(f'Initial epoch for trainer set to {engine.state.epoch}')

    @trainer.on(Events.EPOCH_STARTED)
    def set_model_train(engine):
        if hasattr(engine.state, 'train_loader_labbeled'):
            del engine.state.train_loader_labbeled
        engine.state.train_loader_labbeled = iter(train_loader)

    @trainer.on(Events.ITERATION_COMPLETED)
    def iteration_completed(engine):
        if (engine.state.iteration % iteration_log_step == 0) and (engine.state.iteration > 0):
            engine.state.epoch += 1
            train.set_epochs(engine.state.epoch)
            model.eval()
            logger.info('Model set to eval mode')
            evaluator.run(train_loader)
            model.train()
            logger.info('Model set back to train mode')

    @trainer.on(Events.EPOCH_COMPLETED)
    def after_complete(engine):
        logger.info('Epoch {} done. Time per batch: {:.3f}[s]'.format(
            engine.state.epoch, timer.value()))
        timer.reset()

    trainer.run(train_loader, max_epochs=epochs)
    tb_logger.close()


if __name__ == "__main__":
    main()