from utils import config
import os
import torch
from models.quartznet import QuartzNet
from utils.logger import logger
from functools import partial
from datasets.librispeech import allign_collate, align_collate_unlabelled, allign_collate_val
from utils.lmdb import lmdbDataset
from utils.training_utils import MixDatasets, save_checkpoint, BestMeter
from utils.config import lmdb_root_path, workers, train_batch_size, unsupervision_warmup_epoch, log_path, epochs
import ignite
from ignite.engine import Events, Engine
from ignite.metrics import Loss, RunningAverage
from utils.metrics import WordErrorRate, CharacterErrorRate
from ignite.handlers import ModelCheckpoint, Timer
from ignite.contrib.handlers.tensorboard_logger import *
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from utils.radam import RAdam
from utils.aggloss import ACELoss, UDALoss, CustomCTCLoss, FocalACELoss, FocalUDALoss
from utils.training_utils import load_checkpoint
import numpy as np
import toml

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

def get_alpha(epoch):
    return np.clip(epoch / unsupervision_warmup_epoch, 0.0, 0.5)

def init_parms():
    os.environ['CUDA_VISIBLE_DEVICES'] =  os.environ.get('CUDA_VISIBLE_DEVICES', config.gpu_id)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    params = {
        'device': device,
        'start_epoch': -1
    }
    return params

def main():
    params = init_parms()
    start_epoch = params['start_epoch']
    device = params.get('device')
    model = QuartzNet(num_classes=config.vocab_size).to(device)
    model = torch.nn.DataParallel(model)
    optimizer = RAdam(model.parameters(), lr=config.lr)
    load_checkpoint(model, optimizer, params)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.lr_decay_step, gamma=config.lr_gamma, last_epoch=start_epoch)
    # sup_criterion = FocalACELoss()
    sup_criterion = CustomCTCLoss()
    unsup_criterion = UDALoss()
    tb_logger = TensorboardLogger(log_dir=log_path)
    pbar = ProgressBar(persist=True, desc="Training")
    pbar_valid = ProgressBar(persist=True, desc="Validation Clean")
    pbar_valid_other = ProgressBar(persist=True, desc="Validation Other")
    timer = Timer(average=True)
    best_meter = BestMeter()

    trainCleanPath = os.path.join(lmdb_root_path, 'train-labelled')
    trainOtherPath = os.path.join(lmdb_root_path, 'train-unlabelled')
    testCleanPath = os.path.join(lmdb_root_path, 'test-clean')
    testOtherPath = os.path.join(lmdb_root_path, 'test-other')
    devOtherPath = os.path.join(lmdb_root_path, 'dev-other')

    train_clean_libri = lmdbDataset(root=trainCleanPath)
    train_other_libri = lmdbDataset(root=trainOtherPath)
    test_clean = lmdbDataset(root=testCleanPath)
    test_other = lmdbDataset(root=testOtherPath)
    dev_other_libri = lmdbDataset(root=devOtherPath)

    train_clean = MixDatasets([train_clean_libri, train_other_libri])
    train_other = dev_other_libri

    # train_clean = train_clean_libri
    # train_other = MixDatasets([train_other_libri, dev_other_libri])

    logger.info(f'Loaded Train & Test Datasets, train_labbeled={len(train_clean)}, train_unlabbeled={len(train_other)}, test_clean={len(test_clean)} & test_other={len(test_other)} examples')

    def train_update_function(engine, _):
        optimizer.zero_grad()
        model.zero_grad()

        # Supervised gt, pred
        imgs_sup, labels_sup, label_lengths = next(engine.state.train_loader_labbeled)
        imgs_sup = imgs_sup.to(device)
        labels_sup = labels_sup.to(device)
        probs_sup = model(imgs_sup)

        # Unsupervised gt, pred
        # imgs_unsup, augmented_imgs_unsup = next(engine.state.train_loader_unlabbeled)
        # with torch.no_grad():
        #     probs_unsup = model(imgs_unsup.to(device))
        #     probs_aug_unsup = model(augmented_imgs_unsup.to(device))

        sup_loss = sup_criterion(probs_sup, labels_sup, torch.tensor([probs_sup.size(1)]*probs_sup.size(0), dtype=torch.long), label_lengths)
        # unsup_loss = unsup_criterion(probs_unsup, probs_aug_unsup)

        # Blend supervised and unsupervised losses till unsupervision_warmup_epoch
        # alpha = get_alpha(engine.state.epoch)
        # final_loss = ((1 - alpha) * sup_loss) + (alpha * unsup_loss)

        final_loss = sup_loss

        final_loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        # torch.nn.utils.clip_grad_value_(model.parameters(), 5.0)
        optimizer.step()
        return final_loss.item()

    @torch.no_grad()
    def validate_update_function(engine, batch):
        img, labels, label_lengths = batch
        y_pred = model(img.to(device))
        return (y_pred, labels, label_lengths)

    train_loader_labbeled_loader = torch.utils.data.DataLoader(train_clean, batch_size=train_batch_size, shuffle=True, num_workers=config.workers, pin_memory=False, collate_fn=partial(allign_collate, device=device))
    train_loader_unlabbeled_loader = torch.utils.data.DataLoader(train_other, batch_size=train_batch_size, shuffle=True, num_workers=config.workers, pin_memory=False, collate_fn=partial(align_collate_unlabelled, device=device))
    test_loader_clean = torch.utils.data.DataLoader(test_clean, batch_size=train_batch_size, shuffle=False, num_workers=config.workers, pin_memory=False, collate_fn=partial(allign_collate_val, device=device))
    test_loader_other = torch.utils.data.DataLoader(test_other, batch_size=train_batch_size, shuffle=False, num_workers=config.workers, pin_memory=False, collate_fn=partial(allign_collate_val, device=device))
    trainer = Engine(train_update_function)
    evaluator_clean = Engine(validate_update_function)
    evaluator_other = Engine(validate_update_function)
    metrics = {'wer': WordErrorRate(), 'cer': CharacterErrorRate()}
    for name, metric in metrics.items():
        metric.attach(evaluator_clean, name)
        metric.attach(evaluator_other, name)

    tb_logger.attach(trainer, log_handler=OutputHandler(tag="training", output_transform=lambda loss: {'loss': loss}),
                 event_name=Events.ITERATION_COMPLETED)
    tb_logger.attach(trainer,
                 log_handler=OptimizerParamsHandler(optimizer),
                 event_name=Events.ITERATION_STARTED)
    tb_logger.attach(trainer,
                 log_handler=WeightsHistHandler(model),
                 event_name=Events.EPOCH_COMPLETED)
    tb_logger.attach(trainer,
                 log_handler=GradsScalarHandler(model),
                 event_name=Events.ITERATION_COMPLETED)
    tb_logger.attach(trainer,
                 log_handler=GradsHistHandler(model),
                 event_name=Events.EPOCH_COMPLETED)
    tb_logger.attach(evaluator_clean, 
                 log_handler=OutputHandler(tag="validation_clean", metric_names=["wer", "cer"], another_engine=trainer),
                 event_name=Events.EPOCH_COMPLETED)
    tb_logger.attach(evaluator_other, 
                 log_handler=OutputHandler(tag="validation_other", metric_names=["wer", "cer"], another_engine=trainer),
                 event_name=Events.EPOCH_COMPLETED)
    pbar.attach(trainer, output_transform=lambda x: {'loss': x})
    pbar_valid.attach(evaluator_clean, ['wer', 'cer'], event_name=Events.EPOCH_COMPLETED, closing_event_name=Events.COMPLETED)
    pbar_valid_other.attach(evaluator_other, ['wer', 'cer'], event_name=Events.EPOCH_COMPLETED, closing_event_name=Events.COMPLETED)
    timer.attach(trainer)

    @trainer.on(Events.STARTED)
    def set_init_epoch(engine):
        engine.state.epoch = params['start_epoch']
        logger.info(f'Initial epoch for trainer set to {engine.state.epoch}')

    @trainer.on(Events.EPOCH_STARTED)
    def set_model_train(engine):
        model.train()
        logger.info('Model set to train mode')
        engine.state.train_loader_labbeled = iter(train_loader_labbeled_loader)
        engine.state.train_loader_unlabbeled = iter(train_loader_unlabbeled_loader)

    @trainer.on(Events.EPOCH_COMPLETED)
    def after_complete(engine):
        logger.info('Epoch {} done. Time per batch: {:.3f}[s]'.format(
            engine.state.epoch, timer.value()))
        timer.reset()
        train_clean.set_epochs(engine.state.epoch)
        train_other.set_epochs(engine.state.epoch)
        model.eval()
        logger.info('Model set to eval mode')
        evaluator_clean.run(test_loader_clean)
        evaluator_other.run(test_loader_other)

    @evaluator_other.on(Events.EPOCH_COMPLETED)
    def save_checkpoints(engine):
        scheduler.step()
        metrics = engine.state.metrics
        wer = metrics['wer']
        cer = metrics['cer']
        epoch = trainer.state.epoch
        best_meter.update(wer, cer, epoch)
        save_checkpoint(model, optimizer, best_meter, wer, cer, epoch)

    trainer.run(train_loader_labbeled_loader, max_epochs=epochs)
    tb_logger.close()

if __name__ == "__main__":
    main()
