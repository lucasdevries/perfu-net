import torch
import numpy as np
import albumentations as A




class RotateAngleA(object):
    def __init__(self, probability):
        assert isinstance(probability, float)
        self.probability = probability
        self.transform = A.Rotate(limit=10, interpolation=1, p=self.probability, border_mode=1)

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        # Channels last
        transformed = self.transform(image=image, mask=mask)
        transformed_image = transformed['image']
        transformed_mask = transformed['mask']
        return {'image': transformed_image.copy(), 'mask': transformed_mask.copy()}

class HorizontalFlip(object):
    """Flip ndarrays horizontal direction."""

    def __init__(self, probability):
        assert isinstance(probability, float)
        self.probability = probability

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        if float(torch.rand(1, dtype=torch.float64)) < self.probability:
            image = np.fliplr(image)
            mask = np.fliplr(mask)
        else:
            image, mask = image, mask
        return {'image': image.copy(), 'mask': mask.copy()}
