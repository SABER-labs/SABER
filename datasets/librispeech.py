from audtorch import datasets, transforms
import utils.config as config
from transforms import *
from torchvision.transforms import *
import sentencepiece as spm
import torch
from utils.logger import logger

sp = spm.SentencePieceProcessor()
sp.Load(config.sentencepiece_model)
logger.info(f'{config.sentencepiece_model} has been loaded!')

def convert_to_mel(signal, frac_to_apply=0.8):
    data = {'samples': signal.squeeze(0),
            'sample_rate': config.sampling_rate}
    stretch = RandomApply([StretchAudio()], p=frac_to_apply)
    to_stft = ToSTFT(n_fft=config.n_fft, hop_length=config.hop_length,
                     win_length=config.window_length)
    stft_to_mel = ToMelSpectrogramFromSTFT(n_mels=config.num_mel_banks)
    transforms = [ stretch, to_stft, stft_to_mel, DeleteSTFT(), ToAudioTensor(['mel_spectrogram']) ]
    return Compose(transforms)(data)

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
    return np.array(sp.EncodeAsIds(label.lower()), dtype=np.int32)

def label_re_transform(classes):
    return sp.DecodeIds(classes)

def allign_collate(batch, device='cpu'):
    img_list, label_list = zip(*batch)
    imgs = np.zeros((len(img_list), config.num_mel_banks, max([img.shape[1] for img in img_list])), dtype=np.float32)
    for i, img in enumerate(img_list):
        imgs[i, :, :img.shape[1]] = img
    lengths = np.array([label.shape[0] for label in label_list]).astype(np.int32)
    flat_label_list = np.concatenate(label_list).astype(np.int32)
    imgs = torch.from_numpy(imgs)
    labels = torch.from_numpy(flat_label_list)
    label_lengths = torch.from_numpy(lengths)
    return (imgs, labels, label_lengths)

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

def get_vocab_list():
    return [sp.IdToPiece(id) for id in range(sp.GetPieceSize())][1:]