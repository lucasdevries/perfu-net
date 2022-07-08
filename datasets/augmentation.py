import torch
import numpy as np
import matplotlib.pyplot as plt
from einops.einops import rearrange
from scipy.ndimage import rotate
import time
import albumentations as A

class RandomClip(object):
    """Padd ndarrays in sample."""
    def __init__(self, clip_length):
        assert isinstance(clip_length, int)
        self.clip_length = clip_length

    def __call__(self, sample):
        clip, mask = sample['ctp'], sample['mask']
        start = np.random.randint(0, clip.shape[2] - self.clip_length)
        clip = clip[:, :, start:start+self.clip_length, :]
        return {'ctp': clip.copy(), 'mask': mask.copy()}

class ShiftScaleRotate(object):
    def __init__(self, probability):
        assert isinstance(probability, float)
        self.probability = probability
        self.transform = A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=10, border_mode=1,
                                            p=self.probability)
    def __call__(self, sample):
        image, mask = sample['ctp'], sample['mask']
        image = rearrange(image, 'h w t c -> h w (c t)')
        transformed = self.transform(image=image, mask=mask)
        transformed_image = transformed['image']
        transformed_mask = transformed['mask']
        transformed_image = rearrange(transformed_image, 'h w (c t) -> h w t c', c=2)
        return {'ctp': transformed_image.copy(), 'mask': transformed_mask.copy()}

class RotateAngleA(object):
    def __init__(self, probability):
        assert isinstance(probability, float)
        self.probability = probability
        self.transform = A.Rotate(limit=10, interpolation=1, p=self.probability, border_mode=1)

    def __call__(self, sample):
        image, mask = sample['ctp'], sample['mask']
        c = image.shape[3]
        image = rearrange(image, 'h w t c -> h w (c t)')
        transformed = self.transform(image=image, mask=mask)
        transformed_image = transformed['image']
        transformed_mask = transformed['mask']
        transformed_image = rearrange(transformed_image, 'h w (c t) -> h w t c', c=c)
        return {'ctp': transformed_image.copy(), 'mask': transformed_mask.copy()}
class VerticalFlip(object):
    """Flip ndarrays upward direction."""

    def __init__(self, probability):
        assert isinstance(probability, float)
        self.probability = probability

    def __call__(self, sample):

        image, mask = sample['ctp'], sample['mask']
        if float(torch.rand(1, dtype=torch.float64)) < self.probability:
            image = np.flipud(image)
            mask = np.flipud(mask)
        else:
            image, mask = image, mask

        return {'ctp': image.copy(), 'mask': mask.copy()}


class HorizontalFlip(object):
    """Flip ndarrays horizontal direction."""

    def __init__(self, probability):
        assert isinstance(probability, float)
        self.probability = probability

    def __call__(self, sample):

        image, mask = sample['ctp'], sample['mask']
        if float(torch.rand(1, dtype=torch.float64)) < self.probability:
            image = np.fliplr(image)
            mask = np.fliplr(mask)
        else:
            image, mask = image, mask
        return {'ctp': image.copy(), 'mask': mask.copy()}

# class WindowSelection(object):
#     """Padd ndarrays in sample."""
#     def __init__(self, clip_length):
#         assert isinstance(clip_length, int)
#         self.clip_length = clip_length
#
#     def __call__(self, sample):
#         clip, mask = sample['ctp'], sample['mask']
#         channel0 = clip[:, :, :, 0]
#         clip_sum = np.sum(channel0, axis=(0, 1))
#         peak = np.argmax(clip_sum)
#         interval_start = peak - self.clip_length // 2
#         interval_end = interval_start + self.clip_length
#         clip_result = clip[:, :, interval_start:interval_end, :]
#         h, w, t, c = clip_result.shape
#         if interval_start < 0:
#             clip_result = np.pad(clip, [(0, 0), (0, 0), (abs(interval_start), 0), (0, 0)], mode='edge')
#         elif t < self.clip_length:
#             clip_result = np.pad(clip, [(0, 0), (0, 0), (0, abs(interval_end - t)), (0, 0)], mode='edge')
#         clip_result = clip_result[:, :, 0:self.clip_length, :]
#
#         return {'ctp': clip_result.copy(), 'mask': mask.copy()}

class WindowSelection(object):
    """Padd ndarrays in sample."""
    def __init__(self, clip_length):
        assert isinstance(clip_length, int)
        self.clip_length = clip_length

    def __call__(self, sample):
        clip, mask = sample['ctp'], sample['mask']
        channel0 = clip[:, :, :, 0]
        clip_sum = np.sum(channel0, axis=(0, 1))
        peak = np.argmax(clip_sum)
        interval_start = peak - self.clip_length // 2
        interval_end = interval_start + self.clip_length
        if interval_start < 0:
            clip_result = np.pad(clip, [(0, 0), (0, 0), (abs(interval_start), 0), (0, 0)], mode='edge')
            clip_result = clip_result[:, :, 0:self.clip_length, :]
        elif interval_end > len(clip_sum):
            clip_result = np.pad(clip, [(0, 0), (0, 0), (0, abs(len(clip_sum) - interval_end)), (0, 0)], mode='edge')
            clip_result = clip_result[:, :, interval_start:interval_end, :]
        else:
            clip_result = clip[:, :, interval_start:interval_end, :]
        return {'ctp': clip_result.copy(), 'mask': mask.copy()}