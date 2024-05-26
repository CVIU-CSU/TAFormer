# Obtained from: https://github.com/vikolss/DACS
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import random
import collections
import cv2

import kornia
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


def strong_transform(param, strong_aug_nums=3, data=None, target=None):
    assert ((data is not None) or (target is not None))
    data, target = one_mix(mask=param['mix'], data=data, target=target)
    data = color_jitter(
        color_jitter=param['color_jitter'],
        s=param['color_jitter_s'],
        p=param['color_jitter_p'],
        mean=param['mean'],
        std=param['std'],
        data=data)
    data = gaussian_blur(blur=param['blur'], data=data)
    return data, target


def get_mean_std(data_preprocessor):
    mean = data_preprocessor.mean
    std = data_preprocessor.std
    return mean, std


def denorm(img, mean, std):
    return img.mul(std).add(mean) / 255.0


def denorm_(img, mean, std):
    img.mul_(std).add_(mean).div_(255.0)


def renorm(img, mean, std):
    return img.mul(255.0).sub(mean).div(std)


def renorm_(img, mean, std):
    img.mul_(255.0).sub_(mean).div_(std)


def color_jitter(color_jitter, mean, std, data=None, s=.25, p=.2):
    # s is the strength of colorjitter
    if not (data is None):
        if data.shape[1] == 3:
            if color_jitter > p:
                if isinstance(s, dict):
                    seq = nn.Sequential(kornia.augmentation.ColorJitter(**s))
                else:
                    seq = nn.Sequential(
                        kornia.augmentation.ColorJitter(
                            brightness=s, contrast=s, saturation=s, hue=s))
                data = denorm(data, mean, std)
                data = seq(data)
                data = renorm(data, mean, std)
    return data


def gaussian_blur(blur, data=None):
    if not (data is None):
        if data.shape[1] == 3:
            if blur > 0.5:
                sigma = np.random.uniform(0.15, 1.15)
                kernel_size_y = int(
                    np.floor(
                        np.ceil(0.1 * data.shape[2]) - 0.5 +
                        np.ceil(0.1 * data.shape[2]) % 2))
                kernel_size_x = int(
                    np.floor(
                        np.ceil(0.1 * data.shape[3]) - 0.5 +
                        np.ceil(0.1 * data.shape[3]) % 2))
                kernel_size = (kernel_size_y, kernel_size_x)
                seq = nn.Sequential(
                    kornia.filters.GaussianBlur2d(
                        kernel_size=kernel_size, sigma=(sigma, sigma)))
                data = seq(data)
    return data


def gray_scale(mean, std, data=None, p=.2):
    if not (data is None):
        if data.shape[1] == 3:
            seq = nn.Sequential(
                kornia.augmentation.RandomGrayscale(p=p))
            data = denorm(data, mean, std)
            data = seq(data)
            data = renorm(data, mean, std)
    return data


def get_class_masks(labels):
    class_masks = []
    for label in labels:
        classes = torch.unique(labels)
        nclasses = classes.shape[0]
        class_choice = np.random.choice(
            nclasses, int((nclasses + nclasses % 2) / 2), replace=False)
        classes = classes[torch.Tensor(class_choice).long()]
        class_masks.append(generate_class_mask(label, classes).unsqueeze(0))
    return class_masks


def generate_class_mask(label, classes):
    label, classes = torch.broadcast_tensors(label,
                                             classes.unsqueeze(1).unsqueeze(2))
    class_mask = label.eq(classes).sum(0, keepdims=True)
    return class_mask

def get_cut_masks(img_shape, random_aspect_ratio=True, within_bounds=True, mask_props=0.4, device='cuda:0'):
    n, _, h, w = img_shape
    if random_aspect_ratio:
        y_props = np.exp(np.random.uniform(low=0.0, high=1, size=(n, 1)) * np.log(mask_props))
        x_props = mask_props / y_props
    else:
        y_props = x_props = np.sqrt(mask_props)

    sizes = np.round(np.stack([y_props, x_props], axis=2) * np.array((h, w))[None, None, :])
    if within_bounds:
        positions = np.round(
            (np.array((h, w)) - sizes) * np.random.uniform(low=0.0, high=1.0, size=sizes.shape))
        rectangles = np.append(positions, positions + sizes, axis=2)
    else:
        centres = np.round(np.array((h, w)) * np.random.uniform(low=0.0, high=1.0, size=sizes.shape))
        rectangles = np.append(centres - sizes * 0.5, centres + sizes * 0.5, axis=2)

    mask = torch.zeros((n, 1, h, w), device=device).long()
    for i, sample_rectangles in enumerate(rectangles):
        y0, x0, y1, x1 = sample_rectangles[0]
        mask[i, 0, int(y0):int(y1), int(x0):int(x1)] = 1
    return mask

def one_mix(mask, data=None, target=None):
    '''
        mask: (B, 1, H, W)
        data: (B, _, H, W)
        target(B, C, H, W)
    '''
    if mask is None:
        return data, target
    shifts = random.choice([1,2])
    if not (data is None):
        shifted_data = torch.roll(data, shifts=shifts , dims=0)
        stackedMask0, _ = torch.broadcast_tensors(mask, data)
        data = (stackedMask0 * shifted_data +
                (1 - stackedMask0) * data)
    if not (target is None):
        shifted_target = torch.roll(target, shifts=shifts, dims=0)
        stackedMask0, _ = torch.broadcast_tensors(mask, target)
        target = (stackedMask0 * shifted_target +
                  (1 - stackedMask0) * target)
    return data, target


# # # # # # # # # # # # # # # # # # # # # # # #
# # # 2. Strong Augmentation for image only
# # # # # # # # # # # # # # # # # # # # # # # #

def img_aug_identity(img, scale=None):
    return img


def img_aug_autocontrast(img, scale=None):
    return ImageOps.autocontrast(img)


def img_aug_equalize(img, scale=None):
    return ImageOps.equalize(img)


def img_aug_invert(img, scale=None):
    return ImageOps.invert(img)


def img_aug_blur(img, scale=[0.1, 2.0]):
    assert scale[0] < scale[1]
    sigma = np.random.uniform(scale[0], scale[1])
    # print(f"sigma:{sigma}")
    return img.filter(ImageFilter.GaussianBlur(radius=sigma))


def img_aug_contrast(img, scale=[0.05, 0.95]):
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v) * random.random()
    v = max_v - v
    # # print(f"final:{v}")
    # v = np.random.uniform(scale[0], scale[1])
    return ImageEnhance.Contrast(img).enhance(v)


def img_aug_brightness(img, scale=[0.05, 0.95]):
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v) * random.random()
    v = max_v - v
    # print(f"final:{v}")
    return ImageEnhance.Brightness(img).enhance(v)


def img_aug_color(img, scale=[0.05, 0.95]):
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v) * random.random()
    v = max_v - v
    # print(f"final:{v}")
    return ImageEnhance.Color(img).enhance(v)


def img_aug_sharpness(img, scale=[0.05, 0.95]):
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v) * random.random()
    v = max_v - v
    # print(f"final:{v}")
    return ImageEnhance.Sharpness(img).enhance(v)


def img_aug_hue(img, scale=[0, 0.5]):
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v) * random.random()
    v += min_v
    if np.random.random() < 0.5:
        hue_factor = -v
    else:
        hue_factor = v
    # print(f"Final-V:{hue_factor}")
    input_mode = img.mode
    if input_mode in {"L", "1", "I", "F"}:
        return img
    h, s, v = img.convert("HSV").split()
    np_h = np.array(h, dtype=np.uint8)
    # uint8 addition take cares of rotation across boundaries
    with np.errstate(over="ignore"):
        np_h += np.uint8(hue_factor * 255)
    h = Image.fromarray(np_h, "L")
    img = Image.merge("HSV", (h, s, v)).convert(input_mode)
    return img


def img_aug_posterize(img, scale=[4, 8]):
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v) * random.random()
    # print(min_v, max_v, v)
    v = int(np.ceil(v))
    v = max(1, v)
    v = max_v - v
    # print(f"final:{v}")
    return ImageOps.posterize(img, v)


def img_aug_solarize(img, scale=[1, 256]):
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v) * random.random()
    # print(min_v, max_v, v)
    v = int(np.ceil(v))
    v = max(1, v)
    v = max_v - v
    # print(f"final:{v}")
    return ImageOps.solarize(img, v)


def get_augment_list(flag_using_wide=False):
    if flag_using_wide:
        l = [
            (img_aug_identity, None),
            (img_aug_autocontrast, None),
            (img_aug_equalize, None),
            (img_aug_blur, [0.1, 2.0]),
            (img_aug_contrast, [0.1, 1.8]),
            (img_aug_brightness, [0.1, 1.8]),
            (img_aug_color, [0.1, 1.8]),
            (img_aug_sharpness, [0.1, 1.8]),
            (img_aug_posterize, [2, 8]),
            (img_aug_solarize, [1, 256]),
            (img_aug_hue, [0, 0.5])
        ]
    else:
        l = [
            (img_aug_identity, None),
            (img_aug_autocontrast, None),
            (img_aug_equalize, None),
            (img_aug_blur, [0.1, 2.0]),
            (img_aug_contrast, [0.05, 0.95]),
            (img_aug_brightness, [0.05, 0.95]),
            (img_aug_color, [0.05, 0.95]),
            (img_aug_sharpness, [0.05, 0.95]),
            (img_aug_posterize, [4, 8]),
            (img_aug_solarize, [1, 256]),
            (img_aug_hue, [0, 0.5])
        ]
    return l


class strong_img_aug:
    def __init__(self, num_augs, flag_using_random_num=False):
        assert 1 <= num_augs <= 11
        self.n = num_augs
        self.augment_list = get_augment_list(flag_using_wide=False)
        self.flag_using_random_num = flag_using_random_num

    def __call__(self, img):
        if self.flag_using_random_num:
            max_num = np.random.randint(1, high=self.n + 1)
        else:
            max_num = self.n
        ops = random.choices(self.augment_list, k=max_num)
        for op, scales in ops:
            # print("="*20, str(op))
            img = op(img, scales)
        return img
