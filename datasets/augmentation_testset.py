import torch
import numpy as np
import matplotlib.pyplot as plt
from einops.einops import rearrange
from scipy.ndimage import rotate
import time
import albumentations as A

class WindowSelection(object):
    """Padd ndarrays in sample."""
    def __init__(self, clip_length):
        assert isinstance(clip_length, int)
        self.clip_length = clip_length

    def __call__(self, sample):
        clip = sample['ctp']
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
        return {'ctp': clip_result.copy()}

# class WindowSelection(object):
#     """Padd ndarrays in sample."""
#     def __init__(self, clip_length):
#         assert isinstance(clip_length, int)
#         self.clip_length = clip_length
#
#     def __call__(self, sample):
#         clip = sample['ctp']
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
#         return {'ctp': clip_result.copy()}