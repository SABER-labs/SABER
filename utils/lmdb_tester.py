import sys
import os
import lmdb # install lmdb by "pip install lmdb"
import cv2
import numpy as np
import msgpack
from bisect import bisect_right
import msgpack_numpy as m
from torch.utils.data import Dataset
from tqdm import tqdm
from utils.logger import logger
from joblib import Parallel, delayed
from utils.config import num_cores, train_batch_size
m.patch()

class lmdbMultiDatasetTester(Dataset):

    def __init__(self, roots=[], transform=None, target_transform=None):
        super().__init__()
        self.nSamples = 0
        self.cutoffs = [0]
        for i, root in enumerate(roots):
            setattr(self, f'env{i}', lmdb.open(
                root,
                max_readers=100,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False))
            with getattr(self, f'env{i}').begin(write=False) as txn:
                nSamples_dataset = int(txn.get('num-samples'.encode('ascii')))
            self.nSamples += nSamples_dataset
            self.cutoffs.append(self.nSamples)

        self.transform = transform
        self.target_transform = target_transform
        self.epoch = 0
        self.idx = np.random.choice(self.nSamples, train_batch_size)

    def __len__(self):
        return self.nSamples

    def set_epochs(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index):
        index = np.random.choice(self.idx)
        assert index <= len(self), 'index range error'
        bisect_index = bisect_right(self.cutoffs, index) - 1
        index -= self.cutoffs[bisect_index]
        env = getattr(self, f'env{bisect_index}')
        with env.begin(write=False) as txn:
            data_key = f'data-{index:09d}'.encode('ascii')
            data_enc = txn.get(data_key)
            if not data_enc:
                return self.__getitem__(np.random.choice(range(len(self))))
            data = msgpack.unpackb(data_enc, object_hook=m.decode, raw=False)

            img = data['img']
            label = data['label']

            if self.transform is not None:
                img = self.transform(img, self.epoch)

            if self.target_transform is not None:
                label = self.target_transform(label)

            return (img, label)