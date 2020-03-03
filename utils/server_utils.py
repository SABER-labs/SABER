from models.quartznet import ASRModel
from utils import config
import torch
import os
import soundfile as sf
from datasets.librispeech import image_val_transform
from transforms import *
from torchvision.transforms import *
from utils.model_utils import get_most_probable
from collections import OrderedDict
from utils.logger import logger
import numpy as np
import torch.nn.functional as F

def load_checkpoint(model):
    checkpoint_path = os.path.abspath(config.server_checkpoint)
    if os.path.exists(checkpoint_path):
        loader = torch.load(checkpoint_path, map_location='cpu')
        old_state_dict = loader['model_state_dict']
        new_state_dict = OrderedDict()
        for k, v in old_state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

def load_model():
    model = ASRModel(input_features=config.num_mel_banks, num_classes=config.vocab_size)
    logger.info('Intialized ASRModel.')
    load_checkpoint(model)
    logger.info(f'ASRModel weights set from {config.server_checkpoint}.')
    model.eval()
    logger.info('Set ASRModel to eval.')
    return model

def get_mel_spec(signal):
    data = {'samples': signal,
            'sample_rate': config.sampling_rate}
    to_stft = ToSTFT(n_fft=config.n_fft, hop_length=config.hop_length,
                     win_length=config.window_length)
    stft_to_mel = ToMelSpectrogramFromSTFT(n_mels=config.num_mel_banks)
    transforms = [to_stft, stft_to_mel, DeleteSTFT(), ToAudioTensor(['mel_spectrogram']), torch.from_numpy]
    return Compose(transforms)(data)

@torch.no_grad()
def infer(audio):
    spec = get_mel_spec(audio)
    spec = F.pad(input=spec, pad=(0, int(spec.shape[1] * 0.2)), mode='constant', value=0)
    spec = spec.unsqueeze(0)
    y_pred = model(spec)
    pred_sentence = get_most_probable(y_pred)[0]
    return pred_sentence

model = load_model()