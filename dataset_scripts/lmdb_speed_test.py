from audtorch import datasets, transforms
from utils.config import lmdb_root_path, workers, train_batch_size
from utils.lmdb import lmdbDataset
from utils.logger import logger
from functools import partial
from datasets.librispeech import allign_collate, label_re_transform, align_collate_unlabelled
import os
from tqdm import tqdm
import torch
import time
import numpy as np

def test_data(loader, name, epoch):
    max_length = 0
    start = time.time()
    min_ratios = []
    audio_size = []
    for i, x in enumerate(tqdm(loader)):
        img, label, label_lengths = zip(*x)
        max_length = max(max_length, max(list(map(len, label))))
        ratios = [img_b.shape[1] / len(label_b) for img_b, label_b in zip(img, label)]
        audio_size.extend([img_b.shape[1] for img_b in img])
        # random_index = np.random.choice(range(len(label_lengths)))
        # sample_label_index_start = label_lengths[:random_index].sum().item()
        # sample_label_index_end = sample_label_index_start + label_lengths[random_index].item()
        # sample_label = label[sample_label_index_start:sample_label_index_end].cpu().tolist()
        # sample_text = label_re_transform(sample_label)
        # logger.info(f"Sample text is: {sample_text}")
        min_ratios.extend(ratios)
    end = time.time()
    logger.info(f"Average size, min, max, median of audio is {np.mean(audio_size)}, {np.min(audio_size)}, {np.max(audio_size)}, {np.median(audio_size)}")
    logger.info(f"Longest length of text in {name} dataset is {max_length}")
    logger.info(f"min_ratio text in {np.min(min_ratios)} and max_ratio is {np.max(min_ratios)}, avg is {np.mean(min_ratios)}, median is {np.median(min_ratios)}")
    logger.info(f"Time taken for epoch {epoch} was {end-start:.3f}s")


if __name__ == '__main__':
    trainLabbeledPath = os.path.join(lmdb_root_path, 'train-labelled')
    # trainUnLabelledPath = os.path.join(lmdb_root_path, 'train-unlabelled')
    data_labbeled = lmdbDataset(root=trainLabbeledPath)
    # data_unlabbeled = lmdbDataset(root=trainUnLabelledPath)
    for epoch in range(1):
        data_loader_labbeled = torch.utils.data.DataLoader(data_labbeled, batch_size=train_batch_size, shuffle=True, num_workers=5, pin_memory=False, collate_fn=lambda x: x)
        # data_loader_unlabbeled = torch.utils.data.DataLoader(data_unlabbeled, batch_size=train_batch_size, shuffle=True, num_workers=5, pin_memory=False, collate_fn=lambda x: x)
        test_data(data_loader_labbeled, 'labelled', epoch)
        # test_data(data_loader_unlabbeled, 'unlabelled', epoch)
