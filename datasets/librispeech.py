from audtorch import datasets, transforms
import utils.config as config
from transforms import *
from torchvision.transforms import *
import sentencepiece as spm
import torch
from utils.logger import logger

sp = spm.SentencePieceProcessor()
sp.Load(config.sentencepiece_model)

def convert_to_mel(signal):
    data = {'samples': signal.squeeze(0), 'sample_rate': config.sampling_rate}
    to_stft = ToSTFT(n_fft=config.n_fft, hop_length=config.hop_length, win_length=config.window_length)
    stft_to_mel = ToMelSpectrogramFromSTFT(n_mels=config.num_mel_banks)
    transforms = Compose([to_stft, stft_to_mel, DeleteSTFT(), ToAudioTensor(['mel_spectrogram'])])
    return transforms(data)

def image_train_transform(spec, epoch):
    data = {'mel_spectrogram': spec, 'sample_rate': config.sampling_rate, 'epoch': epoch}
    frac_to_apply = np.clip(epoch / config.augment_warmup_epoch, 0, 1)
    random_apply_specaugment = RandomApply([RandomChoice([SpecAugmentOnMel(), SpecSprinkleOnMel()])], p=frac_to_apply)
    transforms = Compose([random_apply_specaugment, ToAudioTensor(['mel_spectrogram']), torch.from_numpy])
    return transforms(data).unsqueeze(0)

def image_val_transform(spec):
    data = {'mel_spectrogram': spec, 'sample_rate': config.sampling_rate}
    transforms = Compose([ToAudioTensor(['mel_spectrogram']), torch.from_numpy])
    return transforms(data).unsqueeze(0)

def label_transform(label):
    return np.array(sp.EncodeAsIds(label.lower()), dtype=np.int32)

def label_re_transform(classes):
    return sp.DecodeIds(classes)

def allign_collate(batch, device='cpu'):
    img_list, label_list, epochs = zip(*batch)
    epoch = epochs[0]
    imgs = torch.nn.utils.rnn.pad_sequence([image_train_transform(img, epoch=epoch).squeeze(0).permute(1, 0) for img in img_list], batch_first=True, padding_value=1e-8).permute(0, 2, 1)
    flat_label_list = np.concatenate(label_list).astype(np.int32)
    labels = torch.from_numpy(flat_label_list)
    label_lengths = torch.tensor([label.shape[0] for label in label_list], dtype=torch.int32)
    return (imgs, labels, label_lengths)

def allign_collate_val(batch, device='cpu'):
    img_list, label_list, _ = zip(*batch)
    imgs = torch.nn.utils.rnn.pad_sequence([image_val_transform(img).squeeze(0).permute(1, 0) for img in img_list], batch_first=True, padding_value=1e-8).permute(0, 2, 1)
    flat_label_list = np.concatenate(label_list).astype(np.int32)
    labels = torch.from_numpy(flat_label_list)
    label_lengths = torch.tensor([label.shape[0] for label in label_list], dtype=torch.int32)
    return (imgs, labels, label_lengths)

def align_collate_unlabelled(batch, device='cpu'):
    img_list, label_list, epochs = zip(*batch)
    imgs = torch.nn.utils.rnn.pad_sequence([image_val_transform(img).squeeze(0).permute(1, 0) for img in img_list], batch_first=True, padding_value=1e-8).permute(0, 2, 1)
    # imgs = torch.stack([image_val_transform(img) for img in img_list])
    # augmented_imgs = torch.stack([image_train_transform(img, epoch=config.augment_warmup_epoch) for img in img_list])
    augmented_imgs = torch.nn.utils.rnn.pad_sequence([image_train_transform(img, epoch=config.augment_warmup_epoch).squeeze(0).permute(1, 0) for img in img_list], batch_first=True, padding_value=1e-8).permute(0, 2, 1)
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