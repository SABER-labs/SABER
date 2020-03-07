from audtorch import datasets, transforms
import utils.config as config
from transforms import *
from torchvision.transforms import *
import sentencepiece as spm
import torch
from utils.logger import logger
from utils.vocab import Vocab

sp = spm.SentencePieceProcessor()
sp.Load(config.sentencepiece_model)
logger.info(f'{config.sentencepiece_model} has been loaded!')

def convert_to_mel(signal, frac_to_apply=0.25, train=True):
    data = {'samples': signal.squeeze(0),
            'sample_rate': config.sampling_rate}
    stretch = RandomApply([StretchAudio()], p=frac_to_apply)
    to_stft = ToSTFT(n_fft=config.n_fft, hop_length=config.hop_length,
                     win_length=config.window_length)
    stft_to_mel = ToMelSpectrogramFromSTFT(n_mels=config.num_mel_banks)
    interim_transforms = [ to_stft, stft_to_mel, DeleteSTFT(), ToAudioTensor(['mel_spectrogram']) ]
    if train:
        interim_transforms = [ stretch ] + interim_transforms
    return Compose(interim_transforms)(data)

def image_train_transform(spec, epoch):
    data = {'mel_spectrogram': spec,
            'sample_rate': config.sampling_rate, 'epoch': epoch}
    epoch_index = min(epoch, config.augment_warmup_epoch - 1)
    frac_to_apply = np.linspace(config.min_frac, config.max_frac, config.augment_warmup_epoch)[epoch_index]
    random_apply_specaugment = RandomApply([RandomChoice([SpecAugmentOnMel(), SpecSprinkleOnMel()])], p=frac_to_apply)
    transforms = Compose([random_apply_specaugment, ToAudioTensor(
        ['mel_spectrogram'])])
    return transforms(data)

def image_val_transform(spec, epoch):
    data = {'mel_spectrogram': spec,
            'sample_rate': config.sampling_rate, 'epoch': epoch}
    transforms = Compose([ToAudioTensor(
        ['mel_spectrogram'])])
    return transforms(data)

def label_transform(label):
    return np.array([i for i in sp.EncodeAsIds(label.lower()) if i != 0], dtype=np.int32)

def label_re_transform(classes):
    return sp.DecodeIds(classes)

def get_vocab_list():
    return [sp.IdToPiece(id) for id in range(sp.GetPieceSize())][1:]

def allign_collate(batch, device='cpu'):
    img_list, label_list = zip(*batch)
    mel_spec_sizes = [img.shape[1] for img in img_list]
    max_size = max(mel_spec_sizes)
    mel_output_prob_sizes = [np.ceil(size / 2) for size in mel_spec_sizes]
    length_list = [label.shape[0] for label in label_list]

    for out_shape, length_l, i in zip(mel_output_prob_sizes, length_list, range(len(length_list))):
        if out_shape < length_l:
            mel_output_prob_sizes[i] = length_l

    lengths = np.array(length_list).astype(np.int32)
    imgs = np.zeros((len(img_list), config.num_mel_banks, max_size), dtype=np.float32)
    for i, img in enumerate(img_list):
        imgs[i, :, :img.shape[1]] = img
    flat_label_list = np.concatenate(label_list).astype(np.int32)
    imgs = torch.from_numpy(imgs)
    labels = torch.from_numpy(flat_label_list)
    label_lengths = torch.from_numpy(lengths)
    image_lengths = torch.tensor(mel_output_prob_sizes, dtype=torch.int32) # model stride is 2
    return (imgs, labels, label_lengths, image_lengths)

def align_collate_unlabelled(batch, device='cpu'):
    img_list, img_aug_list = zip(*batch)
    imgs = torch.nn.utils.rnn.pad_sequence([img.permute(1, 0) for img in img_list], batch_first=True, padding_value=0).permute(0, 2, 1)
    augmented_imgs = torch.nn.utils.rnn.pad_sequence([img.permute(1, 0) for img in img_aug_list], batch_first=True, padding_value=0).permute(0, 2, 1)
    return (imgs, augmented_imgs)

def process_seq(sequence):
    size = len(sequence)
    final_sequence = []
    for i in range(size):
        char = sequence[i]
        if char not in [0, 1]:
            if i != 0 and char == sequence[i - 1]:
                pass
            else:
                final_sequence.append(char)
    return final_sequence


def sequence_to_string(sequence):
    return label_re_transform(sequence)


def get_sentence(sequence):
    return sequence_to_string(process_seq(sequence))