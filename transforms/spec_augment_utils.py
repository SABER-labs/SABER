import random
import numpy
from PIL import Image
from PIL.Image import BICUBIC

def time_warp(x, max_time_warp=80, inplace=False, mode="PIL"):
    window = max_time_warp
    if mode == "PIL":
        t = x.shape[0]
        if t - window <= window:
            return x
        # NOTE: randrange(a, b) emits a, a + 1, ..., b - 1
        center = random.randrange(window, t - window)
        warped = random.randrange(center - window, center + window) + 1  # 1 ... t - 1

        left = Image.fromarray(x[:center]).resize((x.shape[1], warped), BICUBIC)
        right = Image.fromarray(x[center:]).resize((x.shape[1], t - warped), BICUBIC)
        if inplace:
            x[:warped] = left
            x[warped:] = right
            return x
        return numpy.concatenate((left, right), 0)
    else:
        raise NotImplementedError("unknown resize mode: " + mode + ", choose one from (PIL, sparse_image_warp).")


def freq_mask(x, F=30, n_mask=2, replace_with_zero=True, inplace=False):
    """freq mask for spec agument

    :param numpy.ndarray x: (time, freq)
    :param int n_mask: the number of masks
    :param bool inplace: overwrite
    :param bool replace_with_zero: pad zero on mask if true else use mean
    """
    if inplace:
        cloned = x
    else:
        cloned = x.copy()

    mean_value = cloned.mean()
    num_mel_channels = cloned.shape[1]
    fs = numpy.random.randint(0, F, size=(n_mask, 2))

    for f, mask_end in fs:
        f_zero = random.randrange(0, num_mel_channels - f)
        mask_end += f_zero

        # avoids randrange error if values are equal and range is empty
        if f_zero == f_zero + f:
            continue

        if replace_with_zero:
            cloned[:, f_zero:mask_end] = 0
        else:
            cloned[:, f_zero:mask_end] = mean_value
    return cloned


def time_mask(spec, T=40, n_mask=2, replace_with_zero=True, inplace=False):
    """freq mask for spec agument

    :param numpy.ndarray spec: (time, freq)
    :param int n_mask: the number of masks
    :param bool inplace: overwrite
    :param bool replace_with_zero: pad zero on mask if true else use mean
    """
    if inplace:
        cloned = spec
    else:
        cloned = spec.copy()
    mean_value = cloned.mean()
    len_spectro = cloned.shape[0]
    ts = numpy.random.randint(0, T, size=(n_mask, 2))
    for t, mask_end in ts:
        # avoid randint range error
        if len_spectro - t <= 0:
            continue
        t_zero = random.randrange(0, len_spectro - t)

        # avoids randrange error if values are equal and range is empty
        if t_zero == t_zero + t:
            continue

        mask_end += t_zero
        if replace_with_zero:
            cloned[t_zero:mask_end] = 0
        else:
            cloned[t_zero:mask_end] = mean_value
    return cloned


def spec_augment(x, resize_mode="PIL", max_time_warp=80,
                 max_freq_width=27, n_freq_mask=2,
                 max_time_width=100, n_time_mask=2, inplace=False, replace_with_zero=False, no_time_wrap=False):
    """spec agument

    apply random time warping and time/freq masking
    default setting is based on LD (Librispeech double) in Table 2 https://arxiv.org/pdf/1904.08779.pdf

    :param numpy.ndarray x: (time, freq)
    :param str resize_mode: "PIL" (fast, nondifferentiable) or "sparse_image_warp" (slow, differentiable)
    :param int max_time_warp: maximum frames to warp the center frame in spectrogram (W)
    :param int freq_mask_width: maximum width of the random freq mask (F)
    :param int n_freq_mask: the number of the random freq mask (m_F)
    :param int time_mask_width: maximum width of the random time mask (T)
    :param int n_time_mask: the number of the random time mask (m_T)
    :param bool inplace: overwrite intermediate array
    :param bool replace_with_zero: pad zero on mask if true else use mean
    """
    assert isinstance(x, numpy.ndarray)
    assert x.ndim == 2
    if not no_time_wrap:
        x = time_warp(x, max_time_warp, inplace=inplace, mode=resize_mode)
    x = freq_mask(x, max_freq_width, n_freq_mask, inplace=inplace, replace_with_zero=replace_with_zero)
    x = time_mask(x, max_time_width, n_time_mask, inplace=inplace, replace_with_zero=replace_with_zero)
    return x

def cutout(tensor, percen, num_cuts, replace_with_zero=False):
    image = tensor.copy()
    mean_value = tensor.mean()
    mask_size_height = int(percen * min(image.shape))
    mask_size_width = int(percen * max(image.shape))
    mask_size_half_height = mask_size_height // 2
    mask_size_half_width = mask_size_width // 2
    offset_x = 1 if mask_size_width % 2 == 0 else 0
    offset_y = 1 if mask_size_height % 2 == 0 else 0
    h, w = image.shape[:2]
    for i in range(num_cuts):
        cxmin, cxmax = mask_size_half_width, w + offset_x - mask_size_half_width
        cymin, cymax = mask_size_half_height, h + offset_y - mask_size_half_height
        cx = numpy.random.randint(cxmin, cxmax)
        cy = numpy.random.randint(cymin, cymax)
        xmin = cx - mask_size_half_width
        ymin = cy - mask_size_half_height
        xmax = xmin + mask_size_width
        ymax = ymin + mask_size_height
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(w, xmax)
        ymax = min(h, ymax)
        if replace_with_zero:
            image[ymin:ymax, xmin:xmax] = 0
        else:
            image[ymin:ymax, xmin:xmax] = mean_value
    return image