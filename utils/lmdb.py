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

def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, value in cache.items():
            key = k.encode('ascii')
            txn.put(key, value)

def createDataset_parallel(outputPath, dataset, image_transform=None, label_transform=None, exclude_func=None, n_jobs=num_cores):
    nSamples = len(dataset)
    env = lmdb.open(outputPath, map_size=1099511627776)
    logger.info(f'Begining to create dataset at {outputPath}')
    cache = {}
    done = 0
    ignored = 0
    with Parallel(n_jobs=n_jobs, require='sharedmem') as parallel:
        while done < nSamples - ignored:
            num_left_or_batch_size = min(100, nSamples - done)
            parallel(delayed(fillCache)(done + i, dataset, cache, image_transform=image_transform, label_transform=label_transform, exclude_func=exclude_func) for i in range(num_left_or_batch_size))
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

def fillCache(index, dataset, cache, image_transform=None, label_transform=None, exclude_func=None):
    img, label = dataset[index]
    dataKey = f'data-{index:09d}'
    if exclude_func is not None:
        if exclude_func(img, label):
            return
    if image_transform is not None:
        img = image_transform(img)
    if label_transform is not None:
        label_transformed = label_transform(label)
    cache[dataKey] = msgpack.packb({'img': img, 'label': label_transformed}, default=m.encode, use_bin_type=True)
    
def createDataset_single(outputPath, dataset, image_transform=None, label_transform=None, exclude_func=None):
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
        if image_transform is not None:
            img = image_transform(img)
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

class lmdbMultiDataset(Dataset):

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

    def __len__(self):
        return self.nSamples

    def set_epochs(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index):
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

if __name__ == "__main__":
    from utils.config import lmdb_root_path
    from datasets.librispeech import sequence_to_string
    lmdb_commonvoice_root_path = "lmdb-databases-common_voice"
    lmdb_airtel_root_path = "lmdb-databases-airtel"
    trainCleanPath = os.path.join(lmdb_root_path, 'train-labelled')
    trainOtherPath = os.path.join(lmdb_root_path, 'train-unlabelled')    
    trainCommonVoicePath = os.path.join(lmdb_commonvoice_root_path, 'train-labelled-en')
    testAirtelPath = os.path.join(lmdb_airtel_root_path, 'test-labelled-en')
    roots = [trainCleanPath, trainOtherPath, trainCommonVoicePath]
    dataset = lmdbMultiDataset(roots=[testAirtelPath])
    print(sequence_to_string(dataset[np.random.choice(len(dataset))][1].tolist()))