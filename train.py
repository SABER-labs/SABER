from utils.model_utils import get_most_probable, get_model_size
from datasets.librispeech import get_sentence

from utils import config
import os
import torch
from models.quartznet import ASRModel
from utils.logger import logger
from functools import partial
from datasets.librispeech import allign_collate, image_train_transform, image_val_transform
from utils.lmdb import lmdbMultiDataset
from utils.training_utils import save_checkpoint, BestMeter
from utils.config import lmdb_root_path, workers, train_batch_size, unsupervision_warmup_epoch, log_path, epochs, lmdb_commonvoice_root_path, lmdb_airtel_root_path, lmdb_airtel_payments_root_path
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
    params = init_parms()
    device = params.get('device')
    model = ASRModel(input_features=config.num_mel_banks,
                     num_classes=config.vocab_size).to(device)
    model = torch.nn.DataParallel(model)
    logger.info(f'Model initialized with {get_model_size(model):.3f}M parameters')
    optimizer = Ranger(model.parameters(), lr=config.lr, eps=1e-5)
    load_checkpoint(model, optimizer, params)
    start_epoch = params['start_epoch']
    sup_criterion = CustomCTCLoss()
    unsup_criterion = UDALoss()
    tb_logger = TensorboardLogger(log_dir=log_path)
    pbar = ProgressBar(persist=True, desc="Training")
    pbar_valid = ProgressBar(persist=True, desc="Validation Clean")
    pbar_valid_other = ProgressBar(persist=True, desc="Validation Other")
    pbar_valid_airtel = ProgressBar(persist=True, desc="Validation Airtel")
    pbar_valid_airtel_payments = ProgressBar(persist=True, desc="Validation Airtel Payments")
    timer = Timer(average=True)
    best_meter = params.get('best_stats', BestMeter())

    logger.info('Begining to load Datasets')

    trainCleanPath = os.path.join(lmdb_root_path, 'train-labelled')
    trainOtherPath = os.path.join(lmdb_root_path, 'train-unlabelled')
    trainCommonVoicePath = os.path.join(
        lmdb_commonvoice_root_path, 'train-labelled-en')
    trainAirtelPath = os.path.join(lmdb_airtel_root_path, 'train-labelled-en')
    trainAirtelPaymentsPath = os.path.join(lmdb_airtel_payments_root_path, 'train-labelled-en')
    testCleanPath = os.path.join(lmdb_root_path, 'test-clean')
    testOtherPath = os.path.join(lmdb_root_path, 'test-other')
    testAirtelPath = os.path.join(lmdb_airtel_root_path, 'test-labelled-en')
    testAirtelPaymentsPath = os.path.join(lmdb_airtel_payments_root_path, 'test-labelled-en')
    devOtherPath = os.path.join(lmdb_root_path, 'dev-other')

    train_clean = lmdbMultiDataset(
        roots=[trainCleanPath, trainOtherPath, trainCommonVoicePath, trainAirtelPath, trainAirtelPaymentsPath], transform=image_train_transform)
    train_other = lmdbMultiDataset(roots=[devOtherPath], transform=image_train_transform)

    test_clean = lmdbMultiDataset(roots=[testCleanPath], transform=image_val_transform)
    test_other = lmdbMultiDataset(roots=[testOtherPath], transform=image_val_transform)
    test_airtel = lmdbMultiDataset(roots=[testAirtelPath], transform=image_val_transform)
    test_payments_airtel = lmdbMultiDataset(roots=[testAirtelPaymentsPath], transform=image_val_transform)

    logger.info(
        f'Loaded Train & Test Datasets, train_labbeled={len(train_clean)}, train_unlabbeled={len(train_other)}, test_clean={len(test_clean)}, test_other={len(test_other)}, test_airtel={len(test_airtel)}, test_payments_airtel={len(test_payments_airtel)} examples')

    def train_update_function(engine, _):
        optimizer.zero_grad()

        # Supervised gt, pred
        imgs_sup, labels_sup, label_lengths = next(
            engine.state.train_loader_labbeled)

        imgs_sup = imgs_sup.to(device)
        labels_sup = labels_sup
        probs_sup = model(imgs_sup)
        
        # Unsupervised gt, pred
        # imgs_unsup, augmented_imgs_unsup = next(engine.state.train_loader_unlabbeled)
        # with torch.no_grad():
        #     probs_unsup = model(imgs_unsup.to(device))
        # probs_aug_unsup = model(augmented_imgs_unsup.to(device))
        sup_loss = sup_criterion(probs_sup, labels_sup, label_lengths)
        # unsup_loss = unsup_criterion(probs_unsup, probs_aug_unsup)

        # Blend supervised and unsupervised losses till unsupervision_warmup_epoch
        # alpha = get_alpha(engine.state.epoch)
        # final_loss = ((1 - alpha) * sup_loss) + (alpha * unsup_loss)

        # final_loss = sup_loss
        sup_loss.backward()
        optimizer.step()

        return sup_loss.item()

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

    train_loader_labbeled_loader = torch.utils.data.DataLoader(
        train_clean, batch_size=train_batch_size, shuffle=True, num_workers=config.workers, pin_memory=True, collate_fn=allign_collate)
    train_loader_unlabbeled_loader = torch.utils.data.DataLoader(
        train_other, batch_size=train_batch_size * 4, shuffle=True, num_workers=config.workers, pin_memory=True, collate_fn=allign_collate)
    test_loader_clean = torch.utils.data.DataLoader(
        test_clean, batch_size=torch.cuda.device_count(), shuffle=False, num_workers=config.workers, pin_memory=True, collate_fn=allign_collate)
    test_loader_other = torch.utils.data.DataLoader(
        test_other, batch_size=torch.cuda.device_count(), shuffle=False, num_workers=config.workers, pin_memory=True, collate_fn=allign_collate)
    test_loader_airtel = torch.utils.data.DataLoader(
        test_airtel, batch_size=torch.cuda.device_count(), shuffle=False, num_workers=config.workers, pin_memory=True, collate_fn=allign_collate)
    test_loader_airtel_payments = torch.utils.data.DataLoader(
        test_payments_airtel, batch_size=torch.cuda.device_count(), shuffle=False, num_workers=config.workers, pin_memory=True, collate_fn=allign_collate)
    trainer = Engine(train_update_function)
    evaluator_clean = Engine(validate_update_function)
    evaluator_other = Engine(validate_update_function)
    evaluator_airtel = Engine(validate_update_function)
    evaluator_airtel_payments = Engine(validate_update_function)
    metrics = {'wer': WordErrorRate(), 'cer': CharacterErrorRate()}
    iteration_log_step = int(0.33 * len(train_loader_labbeled_loader))
    for name, metric in metrics.items():
        metric.attach(evaluator_clean, name)
        metric.attach(evaluator_other, name)
        metric.attach(evaluator_airtel, name)
        metric.attach(evaluator_airtel_payments, name)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=config.lr_gamma, patience=int(
        config.epochs * 0.05), verbose=True, threshold_mode="abs", cooldown=int(config.epochs * 0.025), min_lr=1e-5)

    tb_logger.attach(trainer, log_handler=OutputHandler(tag="training", output_transform=lambda loss: {'loss': loss}),
                     event_name=Events.ITERATION_COMPLETED)
    tb_logger.attach(trainer,
                     log_handler=OptimizerParamsHandler(optimizer),
                     event_name=Events.ITERATION_STARTED)
    tb_logger.attach(trainer,
                     log_handler=WeightsHistHandler(model),
                     event_name=Events.EPOCH_COMPLETED)
    tb_logger.attach(trainer,
                     log_handler=WeightsScalarHandler(model),
                     event_name=Events.ITERATION_COMPLETED)
    tb_logger.attach(trainer,
                     log_handler=GradsScalarHandler(model),
                     event_name=Events.ITERATION_COMPLETED)
    tb_logger.attach(trainer,
                     log_handler=GradsHistHandler(model),
                     event_name=Events.EPOCH_COMPLETED)
    tb_logger.attach(evaluator_clean,
                     log_handler=OutputHandler(tag="validation_clean", metric_names=[
                                               "wer", "cer"], another_engine=trainer),
                     event_name=Events.EPOCH_COMPLETED)
    tb_logger.attach(evaluator_other,
                     log_handler=OutputHandler(tag="validation_other", metric_names=[
                                               "wer", "cer"], another_engine=trainer),
                     event_name=Events.EPOCH_COMPLETED)
    tb_logger.attach(evaluator_airtel,
                     log_handler=OutputHandler(tag="validation_airtel", metric_names=[
                                               "wer", "cer"], another_engine=trainer),
                     event_name=Events.EPOCH_COMPLETED)
    tb_logger.attach(evaluator_airtel_payments,
                     log_handler=OutputHandler(tag="validation_airtel_payments", metric_names=[
                                               "wer", "cer"], another_engine=trainer),
                     event_name=Events.EPOCH_COMPLETED)
    pbar.attach(trainer, output_transform=lambda x: {'loss': x})
    pbar_valid.attach(evaluator_clean, [
                      'wer', 'cer'], event_name=Events.EPOCH_COMPLETED, closing_event_name=Events.COMPLETED)
    pbar_valid_other.attach(evaluator_other, [
                            'wer', 'cer'], event_name=Events.EPOCH_COMPLETED, closing_event_name=Events.COMPLETED)
    pbar_valid_airtel.attach(evaluator_airtel, [
                            'wer', 'cer'], event_name=Events.EPOCH_COMPLETED, closing_event_name=Events.COMPLETED)
    pbar_valid_airtel_payments.attach(evaluator_airtel_payments, [
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
        engine.state.train_loader_labbeled = iter(train_loader_labbeled_loader)
        # engine.state.train_loader_unlabbeled = iter(train_loader_unlabbeled_loader)

    @trainer.on(Events.ITERATION_COMPLETED)
    def iteration_completed(engine):
        if (engine.state.iteration % iteration_log_step == 0) and (engine.state.iteration > 0):
            engine.state.epoch += 1
            train_clean.set_epochs(engine.state.epoch)
            train_other.set_epochs(engine.state.epoch)
            model.eval()
            logger.info('Model set to eval mode')
            evaluator_clean.run(test_loader_clean)
            evaluator_other.run(test_loader_other)
            evaluator_airtel.run(test_loader_airtel)
            evaluator_airtel_payments.run(test_loader_airtel_payments)
            model.train()
            logger.info('Model set back to train mode')

    @trainer.on(Events.EPOCH_COMPLETED)
    def after_complete(engine):
        logger.info('Epoch {} done. Time per batch: {:.3f}[s]'.format(
            engine.state.epoch, timer.value()))
        timer.reset()

    @evaluator_other.on(Events.EPOCH_COMPLETED)
    def save_checkpoints(engine):
        metrics = engine.state.metrics
        wer = metrics['wer']
        cer = metrics['cer']
        epoch = trainer.state.epoch
        scheduler.step(wer)
        save_checkpoint(model, optimizer, best_meter, wer, cer, epoch)
        best_meter.update(wer, cer, epoch)

    trainer.run(train_loader_labbeled_loader, max_epochs=epochs)
    tb_logger.close()


if __name__ == "__main__":
    main()