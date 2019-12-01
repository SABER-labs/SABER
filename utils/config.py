import math

# Dataset configs
libri_dataset_root = '/tts_data/asrdata/librispeech'
libri_labelled_data_sets = ['train-clean-100']
libri_unlabeled_data_sets = ['train-clean-360', 'train-other-500']
libri_test_clean_data_sets = ['test-clean']
libri_test_other_data_sets = ['test-other']
libri_dev_data_sets = ['dev-clean', 'dev-other']
sentencepiece_model = 'dataset_scripts/sp_librispeech_128.model'
lmdb_root_path = 'lmdb-databases-librispeech_128'
log_path = "checkpoints_logs/exp-sp-ctc-vocab128-radam"

# Mel feature configs
sampling_rate = 16000
ms_in_one_sec = 1000
window_in_ms = 25
hop_in_ms = 10
num_mel_banks = 80
max_audio_length_in_secs = 17
min_audio_length_in_secs = 2
max_label_length = 350
samples_per_ms = int(sampling_rate / ms_in_one_sec)
window_length = int(samples_per_ms * window_in_ms)
hop_length = int(samples_per_ms * hop_in_ms)
n_fft = 2 ** math.ceil(math.log2(window_length))
ref_db = 20
max_db = 100

# Model training configs
num_cores = 24
vocab_size = 128

# Training configs
gpu_id = '0,1,2'
workers = 30
train_batch_size = 16 * len(gpu_id.split(","))
epochs = 300
lr = 1e-3
cyclic_lr_min = 1e-5
lr_gamma = 0.1
lr_decay_step = [2, 70]
checkpoint_root = 'checkpoints_5x3_radam'
# checkpoint_version = 'saber_w0.359_c0.175_e29.pth'
checkpoint_version = ''
best_model_version = 'best_saber.pth'

# UDA hyper-params
augment_warmup_epoch = int(epochs * 0.5)
unsupervision_warmup_epoch = int(epochs * 0.4)
temperature_softmax = 0.4

# Cutout hyper-params
max_sprinkles_percent = 0.125
max_sprinkles = 20

# Spec augment hyper-params
config_time_warp = 80
config_freq_width = 20
config_time_width = 20
