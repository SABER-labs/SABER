"""Transforms on the short time fourier transforms of wav samples."""

import random
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset
from utils.config import augment_warmup_epoch, max_sprinkles_percent, max_sprinkles, ref_db, max_db, n_fft, hop_length, window_length, config_time_warp, config_freq_width, config_time_width
from .spec_augment_utils import spec_augment, cutout
from imgaug import augmenters as iaa
from torchvision import transforms


class ToSTFT(object):
    """Applies on an audio the short time fourier transform."""

    def __init__(self, n_fft=n_fft, hop_length=hop_length, win_length=window_length):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

    def __call__(self, data):
        samples = data['samples']
        sample_rate = data['sample_rate']
        data['n_fft'] = self.n_fft
        data['hop_length'] = self.hop_length
        data['win_length'] = self.win_length
        samples, index = librosa.effects.trim(samples, top_db=60, frame_length=self.n_fft, hop_length=self.hop_length)
        data['stft'] = librosa.stft(
            samples, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length)
        data['stft_shape'] = data['stft'].shape
        return data


class StretchAudioOnSTFT(object):
    """Stretches an audio on the frequency domain."""

    def __init__(self, max_scale=0.2):
        self.max_scale = max_scale

    def __call__(self, data):
        stft = data['stft']
        sample_rate = data['sample_rate']
        hop_length = data['hop_length']
        scale = random.uniform(-self.max_scale, self.max_scale)
        stft_stretch = librosa.core.phase_vocoder(
            stft, 1+scale, hop_length=hop_length)
        data['stft'] = stft_stretch
        return data


class TimeshiftAudioOnSTFT(object):
    """A simple timeshift on the frequency domain without multiplying with exp."""

    def __init__(self, max_shift=8):
        self.max_shift = max_shift

    def __call__(self, data):
        stft = data['stft']
        shift = random.randint(-self.max_shift, self.max_shift)
        a = -min(0, shift)
        b = max(0, shift)
        stft = np.pad(stft, ((0, 0), (a, b)), "constant")
        if a == 0:
            stft = stft[:, b:]
        else:
            stft = stft[:, 0:-a]
        data['stft'] = stft
        return data


class AddBackgroundNoiseOnSTFT(Dataset):
    """Adds a random background noise on the frequency domain."""

    def __init__(self, bg_dataset, max_percentage=0.45):
        self.bg_dataset = bg_dataset
        self.max_percentage = max_percentage

    def __call__(self, data):
        noise = random.choice(self.bg_dataset)['stft']
        percentage = random.uniform(0, self.max_percentage)
        data['stft'] = data['stft'] * (1 - percentage) + noise * percentage
        return data


class SpecAugmentOnMel(object):

    def __init__(self, no_time_wrap=False):
        self.no_time_wrap = no_time_wrap

    def __call__(self, data):
        tensor = data['mel_spectrogram'].astype(np.float32)
        percentage = np.clip(data['epoch'] / augment_warmup_epoch, 0, 1)
        max_time_warp = max(1, int(percentage * config_time_warp))
        max_freq_width = max(1, int(percentage * config_freq_width))
        max_time_width = max(1, int(percentage * config_time_width))
        tensor = tensor.swapaxes(1, 0)
        data['mel_spectrogram'] = spec_augment(tensor, max_time_warp=max_time_warp, max_freq_width=max_freq_width,
                                               max_time_width=max_time_width, n_freq_mask=2, n_time_mask=2, no_time_wrap=self.no_time_wrap).swapaxes(1, 0)
        return data


class SpecSprinkleOnMel(object):

    def __call__(self, data):
        tensor = data['mel_spectrogram'].astype(np.float32)
        warmup_percen = np.clip(data['epoch'] / augment_warmup_epoch, 0, 1)
        percentage = warmup_percen * max_sprinkles_percent
        max_sprinkles_cuts = max(2, int(warmup_percen * max_sprinkles))
        num_cuts = np.random.randint(1, max_sprinkles_cuts)
        tensor = cutout(tensor, percentage, num_cuts)
        data['mel_spectrogram'] = tensor
        return data

class SpecBlurring(object):

    def __call__(self, data):
        tensor = data['mel_spectrogram'].astype(np.float32)
        tensor = tensor[np.newaxis, :]
        average_blur = iaa.AverageBlur(k=(1, 7)).augment_image
        median_blur = iaa.MedianBlur(k=(1, 7)).augment_image
        motion_blur = iaa.MotionBlur(k=(3, 7)).augment_image
        random_blur = transforms.RandomChoice([average_blur, median_blur, motion_blur])
        data['mel_spectrogram'] = random_blur(tensor).squeeze(0)
        return data


class FixSTFTDimension(object):
    """Either pads or truncates in the time axis on the frequency domain, applied after stretching, time shifting etc."""

    def __call__(self, data):
        stft = data['stft']
        t_len = stft.shape[1]
        orig_t_len = data['stft_shape'][1]
        if t_len > orig_t_len:
            stft = stft[:, 0:orig_t_len]
        elif t_len < orig_t_len:
            stft = np.pad(stft, ((0, 0), (0, orig_t_len-t_len)), "constant")

        data['stft'] = stft
        return data


class ToMelSpectrogramFromSTFT(object):
    """Creates the mel spectrogram from the short time fourier transform of a file. The result is a 32x32 matrix."""

    def __init__(self, n_mels=80):
        self.n_mels = n_mels

    def __call__(self, data):
        stft = data['stft']
        sample_rate = data['sample_rate']
        n_fft = data['n_fft']
        mel_basis = librosa.filters.mel(sample_rate, n_fft, self.n_mels)
        mag = np.abs(stft)
        mel = np.dot(mel_basis, mag)
        mel = 20 * np.log10(np.maximum(1e-5, mel))
        mel = np.clip((mel - ref_db + max_db) / max_db, 1e-8, 1).astype(np.float32)
        data['mel_spectrogram'] = mel
        data['n_mels'] = self.n_mels
        return data

class PerChannelNormalization(object):

    def __call__(self, data):
        spec = data['mel_spectrogram']
        channels = data['n_mels']
        means = np.mean(spec, axis=1)
        means = np.expand_dims(means, axis=1)
        stds = np.std(spec, axis=1) + 1e-15
        stds = np.expand_dims(stds, axis=1)
        spec -= means
        spec /= stds
        data['mel_spectrogram'] = spec
        return data


class DeleteSTFT(object):
    """Pytorch doesn't like complex numbers, use this transform to remove STFT after computing the mel spectrogram."""

    def __call__(self, data):
        del data['stft']
        return data


class AudioFromSTFT(object):
    """Inverse short time fourier transform."""

    def __call__(self, data):
        stft = data['stft']
        data['istft_samples'] = librosa.core.istft(
            stft, dtype=data['samples'].dtype)
        return data


class ToPCEN(object):

    def __call__(self, data):
        stft = data['stft']
        sample_rate = data['sample_rate']
        n_fft = data['n_fft']
        hop_length = data['hop_length']
        pcen = librosa.core.pcen(
            np.abs(stft), sr=sample_rate, hop_length=hop_length)
        data['pcen'] = pcen
        return data
