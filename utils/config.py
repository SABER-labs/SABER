import math

# Dataset configs
libri_dataset_root = '/tts_data/asrdata/librispeech'
libri_labelled_data_sets = ['train-clean-100']
libri_unlabeled_data_sets = ['train-clean-360', 'train-other-500']
libri_test_clean_data_sets = ['test-clean']
libri_test_other_data_sets = ['test-other']
libri_dev_data_sets = ['dev-clean', 'dev-other']
vocab_size = 128
sentencepiece_model = f'dataset_scripts/sp_librispeech_{vocab_size}.model'
lmdb_root_path = f'lmdb-databases-librispeech_{vocab_size}'
lmdb_commonvoice_root_path = f'lmdb-databases-common_voice_{vocab_size}'
lmdb_airtel_root_path = f'lmdb-databases-airtel_{vocab_size}'
lmdb_airtel_payments_root_path = f'lmdb-databases-airtel_payments_{vocab_size}'

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
num_cores = 30

# Training configs
gpu_id = '0,1,2'
workers = 50
train_batch_size = 12 * len(gpu_id.split(","))
epochs = 400
lr = 1e-3
lr_gamma = 0.1
checkpoint_root = f'checkpoints_saber_{vocab_size}'
log_path = f"checkpoints_logs/exp-{checkpoint_root}"
checkpoint_version = ''
best_model_version = 'best_saber.pth'
min_frac = 0.6
max_frac = 0.9

# UDA hyper-params
augment_warmup_epoch = int(epochs * 0.15)
unsupervision_warmup_epoch = int(epochs * 0.15)
temperature_softmax = 0.4

# Cutout hyper-params
max_sprinkles_percent = 0.125
max_sprinkles = 20

# Spec augment hyper-params
config_time_warp = 80
config_freq_width = 40
config_time_width = 120

# Common hyper-params
min_spec_perc = 0.5
max_spec_perc = 1.0

# Beam search params
alpha_lm = 1.6
beta_lm = 0
lm_model_path = "downloads/wiki_en_with_indian_payments.trie.klm"
beam_width = 50
cpus_for_beam_search = 30

#Server configs
host_ip = '0.0.0.0'
port = 8088
server_checkpoint = 'checkpoint/best_saber.pth'