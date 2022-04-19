import torch
import random


def rand_crop(image, label, crop_size):
    _, _, t, _, _ = image.size()

    new_t = random.randint(0, t - crop_size)

    image = image[:, :, new_t: new_t + crop_size :, :]
    label = label[:, :, new_t: new_t + crop_size, :, :]

    return image, label



