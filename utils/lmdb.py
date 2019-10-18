import sys
import os
import lmdb # install lmdb by "pip install lmdb"
import cv2
import numpy as np
import msgpack
import msgpack_numpy as m
from torch.utils.data import Dataset
from tqdm import tqdm
from .logger import logger
from joblib import Parallel, delayed
from .config import num_cores
m.patch()

def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, value in cache.items():
            key = k.encode('ascii')
            txn.put(key, value)

def createDataset_parallel(outputPath, dataset, img_transform=None, label_transform=None, exclude_func=None, n_jobs=num_cores):
    nSamples = len(dataset)
    env = lmdb.open(outputPath, map_size=1099511627776)
    logger.info(f'Begining to create dataset at {outputPath}')
    cache = {}
    done = 0
    ignored = 0
    with Parallel(n_jobs=n_jobs, require='sharedmem') as parallel:
        while done < nSamples - ignored:
            num_left_or_batch_size = min(100, nSamples - done)
            parallel(delayed(fillCache)(done + i, dataset, cache, img_transform=img_transform, label_transform=label_transform, exclude_func=exclude_func) for i in range(num_left_or_batch_size))
            writeCache(env, cache)
            done_batch_size = len(cache.items())
            done += done_batch_size
            cache = {}
            ignored += (num_left_or_batch_size - done_batch_size)
            logger.info(f'Written {done:d} / {nSamples - ignored:d}')
    nSamples = done
    cache['num-samples'] = str(nSamples).encode('ascii')
    writeCache(env, cache)
    logger.info(f'Created dataset with {nSamples:d} samples')

def fillCache(index, dataset, cache, img_transform=None, label_transform=None, exclude_func=None):
    img, label = dataset[index]
    dataKey = f'data-{index:09d}'
    if exclude_func is not None:
        if exclude_func(img, label):
            return
    if img_transform is not None:
        img = img_transform(img)
    if label_transform is not None:
        label_transformed = label_transform(label)
    cache[dataKey] = msgpack.packb({'img': img, 'label': label_transformed}, default=m.encode, use_bin_type=True)
    
def createDataset_single(outputPath, dataset, img_transform=None, label_transform=None, exclude_func=None):
    nSamples = len(dataset)
    env = lmdb.open(outputPath, map_size=1099511627776)
    logger.info(f'Begining to create dataset at {outputPath}')
    cache = {}
    cnt = 0
    for i, (img, label) in enumerate(dataset):
        dataKey = f'data-{cnt:09d}'
        if exclude_func is not None:
            if exclude_func(img, label):
                continue
        if img_transform is not None:
            img = img_transform(img)
        if label_transform is not None:
            label_transformed = label_transform(label)
        cache[dataKey] = msgpack.packb({'img': img, 'label': label_transformed}, default=m.encode, use_bin_type=True)
        if cnt % 100 == 0 and cnt != 0:
            writeCache(env, cache)
            cache = {}
            logger.info(f'Written {cnt:d} / {nSamples:d}')
        cnt += 1
    nSamples = cnt
    cache['num-samples'] = str(nSamples).encode('ascii')
    writeCache(env, cache)
    logger.info(f'Created dataset with {nSamples:d} samples')

class lmdbDataset(Dataset):

    def __init__(self, root=None, transform=None, target_transform=None):
        super().__init__()
        self.env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)

        if not self.env:
            logger.info(f'cannot open lmdb from {root}')
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode('ascii')))
            self.nSamples = nSamples

        self.transform = transform
        self.target_transform = target_transform
        self.epochs = 0

    def __len__(self):
        return self.nSamples

    def set_epochs(self, epoch):
        self.epochs = epoch

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        with self.env.begin(write=False) as txn:
            data_key = f'data-{index:09d}'.encode('ascii')
            data_enc = txn.get(data_key)
            if not data_enc:
                return self.__getitem__(np.random.choice(range(len(self))))
            data = msgpack.unpackb(data_enc, object_hook=m.decode, raw=False)

            img = data['img']
            label = data['label']

            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                label = self.target_transform(label)

            return (img, label, self.epochs)

class lmdbDoubleDataset(Dataset):

    def __init__(self, root1=None, root2=None, transform=None, target_transform=None):
        super().__init__()
        self.env1 = lmdb.open(
            root1,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)

        self.env2 = lmdb.open(
            root2,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)

        if not self.env1 or not self.env2:
            logger.info(f'cannot open lmdb from {root1} or {root2}')
            sys.exit(0)

        with self.env1.begin(write=False) as txn:
            nSamples1 = int(txn.get('num-samples'.encode('ascii')))

        with self.env2.begin(write=False) as txn:
            nSamples2 = int(txn.get('num-samples'.encode('ascii')))
            
        self.nSamples = nSamples1 + nSamples2

        self.cutoff = nSamples1

        self.transform = transform
        self.target_transform = target_transform
        self.epochs = 0

    def __len__(self):
        return self.nSamples

    def set_epochs(self, epoch):
        self.epochs = epoch

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        if index >= self.cutoff:
            index = index - self.cutoff
            env = self.env2
        else:
            env = self.env1
        with env.begin(write=False) as txn:
            data_key = f'data-{index:09d}'.encode('ascii')
            data_enc = txn.get(data_key)
            if not data_enc:
                return self.__getitem__(np.random.choice(range(len(self))))
            data = msgpack.unpackb(data_enc, object_hook=m.decode, raw=False)

            img = data['img']
            label = data['label']

            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                label = self.target_transform(label)

            return (img, label, self.epochs)