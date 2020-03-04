from audtorch import datasets, transforms
from utils.config import libri_dataset_root, libri_labelled_data_sets, libri_unlabeled_data_sets, lmdb_root_path, max_audio_length_in_secs, min_audio_length_in_secs, sampling_rate, max_label_length, libri_test_clean_data_sets, libri_test_other_data_sets, libri_dev_data_sets
from utils.lmdb import createDataset_parallel as createDataset
from utils.logger import logger
from datasets.librispeech import convert_to_mel, label_transform
import os
from functools import partial

def exclude_func(img, label):
    audio_size = img.squeeze(0).shape[0]
    label_size = len(label)
    return not ((min_audio_length_in_secs <= (audio_size / sampling_rate) <= max_audio_length_in_secs) and (label_size <= max_label_length))

if __name__ == '__main__':
    trainPath = os.path.join(lmdb_root_path, 'train-labelled')
    valPath = os.path.join(lmdb_root_path, 'train-unlabelled')
    testCleanPath = os.path.join(lmdb_root_path, 'test-clean')
    testOtherPath = os.path.join(lmdb_root_path, 'test-other')
    devPath = os.path.join(lmdb_root_path, 'dev-other')
    os.makedirs(lmdb_root_path, exist_ok=True)
    os.makedirs(trainPath, exist_ok=True)
    os.makedirs(valPath, exist_ok=True)
    os.makedirs(testCleanPath, exist_ok=True)
    os.makedirs(testOtherPath, exist_ok=True)
    os.makedirs(devPath, exist_ok=True)
    logger.info('Loading datasets')
    convert_to_mel_val = partial(convert_to_mel, train=False)

    data_labbeled = datasets.LibriSpeech(root=libri_dataset_root, sets=libri_labelled_data_sets, download=False)
    logger.info('Labelled dataset loaded')
    createDataset(trainPath, data_labbeled, convert_to_mel, label_transform, exclude_func)
    logger.info(f"Num of labelled examples {len(data_labbeled)}")
    del data_labbeled

    data_unlabbeled = datasets.LibriSpeech(root=libri_dataset_root, sets=libri_unlabeled_data_sets, download=False)
    logger.info('Unlabelled dataset loaded')
    createDataset(valPath, data_unlabbeled, convert_to_mel, label_transform, exclude_func)    
    logger.info(f"Num of unlabelled examples {len(data_unlabbeled)}")
    del data_unlabbeled

    data_test_clean= datasets.LibriSpeech(root=libri_dataset_root, sets=libri_test_clean_data_sets, download=False)
    logger.info('test clean dataset loaded')
    createDataset(testCleanPath, data_test_clean, convert_to_mel_val, label_transform, exclude_func)
    logger.info(f"Num of test clean examples {len(data_test_clean)}")
    del data_test_clean
    
    data_test_other = datasets.LibriSpeech(root=libri_dataset_root, sets=libri_test_other_data_sets, download=False)
    logger.info('test other dataset loaded')
    createDataset(testOtherPath, data_test_other, convert_to_mel_val, label_transform, exclude_func)
    logger.info(f"Num of test other examples {len(data_test_other)}")
    del data_test_other
    
    data_dev = datasets.LibriSpeech(root=libri_dataset_root, sets=libri_dev_data_sets, download=False)
    logger.info('dev dataset loaded')
    createDataset(devPath, data_dev, convert_to_mel_val, label_transform, exclude_func)
    logger.info(f"Num of dev examples {len(data_dev)}")
    del data_dev