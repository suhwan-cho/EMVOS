import numpy as np
import torch
from PIL import Image
import random
import math


def load_image_in_PIL(path, mode):
    img = Image.open(path)
    img.load()
    return img.convert(mode)


class LabelToLongTensor(object):
    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            label = torch.from_numpy(pic).long()
        elif pic.mode == '1':
            label = torch.from_numpy(np.array(pic, np.uint8, copy=False)).long().view(1, pic.size[1], pic.size[0])
        else:
            label = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            if pic.mode == 'LA':
                label = label.view(pic.size[1], pic.size[0], 2)
                label = label.transpose(0, 1).transpose(0, 2).contiguous().long()[0]
                label = label.view(1, label.size(0), label.size(1))
            else:
                label = label.view(pic.size[1], pic.size[0], -1)
                label = label.transpose(0, 1).transpose(0, 2).contiguous().long()
        label[label == 255] = 0
        return label


def random_affine_params(degree, translate, scale_ranges, shear, img_size):
    angle = random.uniform(-degree, degree)
    max_dx = translate * img_size[0]
    max_dy = translate * img_size[1]
    translations = (np.round(random.uniform(-max_dx, max_dx)),
                    np.round(random.uniform(-max_dy, max_dy)))
    scale = random.uniform(scale_ranges[0], scale_ranges[1])
    shear = [random.uniform(-shear, shear), random.uniform(-shear, shear)]
    return angle, translations, scale, shear


def random_crop_params(img, scale, ratio=(3/4, 4/3)):
    width, height = img.size
    area = height * width

    for attempt in range(10):
        target_area = random.uniform(*scale) * area
        log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
        aspect_ratio = math.exp(random.uniform(*log_ratio))

        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))

        if 0 < w <= width and 0 < h <= height:
            i = random.randint(0, height - h)
            j = random.randint(0, width - w)
            return i, j, h, w

    in_ratio = float(width) / float(height)
    if in_ratio < min(ratio):
        w = width
        h = int(round(w / min(ratio)))
    elif in_ratio > max(ratio):
        h = height
        w = int(round(h * max(ratio)))
    else:
        w = width
        h = height
    i = (height - h) // 2
    j = (width - w) // 2
    return i, j, h, w
