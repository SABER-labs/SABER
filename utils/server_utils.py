from models.mixnet import ASRModel
from utils import config
import torch
import os
import soundfile as sf
from datasets.librispeech import convert_to_mel, image_val_transform, label_re_transform
from utils.model_utils import get_most_probable_topk
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

@torch.no_grad()
def infer(audio):
    audio = audio[np.newaxis, :]
    spec = convert_to_mel(audio)
    model_transforms = image_val_transform(spec)
    model_transforms = F.pad(input=model_transforms, pad=(int(model_transforms.shape[1] * 0.05), int(model_transforms.shape[1] * 0.4)), mode='constant', value=0)
    y_pred = model(model_transforms)
    pred_sentence = get_most_probable_topk(y_pred)
    return pred_sentence

model = load_model()