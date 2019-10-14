from audtorch import datasets, transforms
from utils.config import max_audio_length_in_secs, sampling_rate, max_label_length
from utils.lmdb import createDataset_parallel as createDataset
from utils.logger import logger
from datasets.librispeech import convert_to_mel, label_transform
import os
import pandas as pd

def exclude_func(img, label):
    return (img.squeeze(0).shape[0] > (max_audio_length_in_secs * sampling_rate)) or (len(label) > max_label_length)

if __name__ == '__main__':
    lmdb_root_path = "lmdb-databases-common_voice"
    commonvoice_dataset_root = "/tts_data/asrdata/common_voice_v3"
    language_codes = ["en"]

    for language in language_codes:
        trainPath = os.path.join(lmdb_root_path, f'train-labelled-{language}')
        testPath = os.path.join(lmdb_root_path, f'test-labelled-{language}')
        os.makedirs(lmdb_root_path, exist_ok=True)
        os.makedirs(trainPath, exist_ok=True)
        os.makedirs(testPath, exist_ok=True)
        logger.info('Loading datasets')
        audio_root = os.path.join(commonvoice_dataset_root, language, "clips")

        labelled_root = os.path.join(commonvoice_dataset_root, language, "validated.tsv")
        labelled_df = pd.read_csv(labelled_root, sep='\t', usecols=["path", "sentence"]).rename(columns={"path": "audio_path", "sentence": "transcription"})
        data_labbeled = datasets.LibriSpeech(root=audio_root, sets=None, dataframe=labelled_df)
        logger.info('Labelled dataset loaded')
        createDataset(trainPath, data_labbeled, convert_to_mel, label_transform, exclude_func)
        logger.info(f"Num of labelled examples {len(data_labbeled)}")
        del data_labbeled

        test_root = os.path.join(commonvoice_dataset_root, language, "test.tsv")
        test_df = pd.read_csv(test_root, sep='\t', usecols=["path", "sentence"]).rename(columns={"path": "audio_path", "sentence": "transcription"})
        data_test = datasets.LibriSpeech(root=audio_root, sets=None, dataframe=test_df)
        logger.info('test dataset loaded')
        createDataset(testPath, data_test, convert_to_mel, label_transform, exclude_func)
        logger.info(f"Num of test examples {len(data_test)}")
        del data_test