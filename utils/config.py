# Dataset configs
libri_dataset_root = '/tts_data/asrdata/librispeech'
libri_labelled_data_sets = ['train-clean-100']
libri_unlabeled_data_sets = ['train-clean-360', 'train-other-500']
libri_test_clean_data_sets = ['test-clean']
libri_test_other_data_sets = ['test-other']
libri_dev_data_sets = ['dev-clean', 'dev-other']
sentencepiece_model = '/tts_data/saber/dataset_scripts/sp_librispeech.model'
lmdb_root_path = 'lmdb-databases-librispeech'
log_path = "checkpoints_logs/exp-sp-nonfocal-vocab512"

# Mel feature configs
sampling_rate = 16000
ms_in_one_sec = 1000
window_in_ms = 25
hop_in_ms = 10
num_mel_banks = 80
max_audio_length_in_secs = 20
max_label_length = 150
samples_per_ms = int(sampling_rate / ms_in_one_sec)
window_length = int(samples_per_ms * window_in_ms)
hop_length = int(samples_per_ms * hop_in_ms)
n_fft = int(window_length * 2)
ref_db = 20
max_db = 100
num_cores = 12
vocab_size = 512

# Training configs
gpu_id = '0,1,2'
workers = 30
train_batch_size = 16 * len(gpu_id.split(","))
start_epoch = 0
epochs = 600
lr = 1e-3
lr_decay_step = [int(epochs * 0.25), int(epochs * 0.75)]
lr_gamma = 0.1
checkpoint_root = 'checkpoints'
checkpoint_version = ''
best_model_version = 'best_saber.pth'

augment_warmup_epoch = int(epochs * 0.2)
unsupervision_warmup_epoch = int(epochs * 0.075)
max_sprinkles_percent = 0.25
max_sprinkles = 30