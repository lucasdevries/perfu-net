import math
from scipy.ndimage.morphology import distance_transform_edt
import numpy as np
import torch
import random
import os
def adjust_learning_rate(optimizer, epoch, config, batch=None, nBatch=None, method='cosine'):
    if method == 'cosine':
        T_total = config.max_epoch * nBatch
        T_cur = (epoch % config.max_epoch) * nBatch + batch
        lr = 0.5 * config.learning_rate * (1 + math.cos(math.pi * T_cur / T_total))
    elif method == 'decay':
        """Sets the learning rate to the initial LR decayed by 10 every X epochs"""
        lr = config.learning_rate * (config.gamma ** (epoch // config.lr_decay_epoch))
    else:
        lr = config.learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def adjust_alpha(start_alpha, epoch, start_epoch, config) -> object:
    if epoch < start_epoch:
        alpha = start_alpha
    else:
        epoch = epoch - start_epoch
        T_total = config.max_epoch
        T_cur = (epoch % config.max_epoch)
        alpha = 0.5 * start_alpha * (1 + math.cos(math.pi * T_cur / T_total))
    return alpha

def read_file(filepath):
    with open(filepath, 'rb') as f:
        return np.load(f)
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print('*** Setting seed to {}, deterministic behaviour on...'.format(seed))

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def weight_map(labels, w0=0.5, sigma=50):
    """
    Generate weight maps as in Automatic Ischemic Stroke Lesion Segmentation
    from Computed Tomography Perfusion Images by Image Synthesis and Attention-
    Based Deep Neural Networks

    Parameters
    ----------
    labels: Numpy array
        2D array of shape (image_height, image_width) representing binary mask
        of objects.
    w0: int
        Default weight weight parameter.
    sigma: int
        Decay regulator parameter.

    Returns
    -------
    Numpy array
        Training weights. A 2D array of shape (image_height, image_width).
    """

    no_labels = labels == 0
    distances = np.zeros((labels.shape[0], labels.shape[1], 1))
    distances[:, :, 0] = distance_transform_edt(labels.cpu() != 1)
    distances = np.sort(distances, axis=2)
    d1 = distances[:, :, 0]

    w = torch.Tensor((w0 + np.exp(-(d1) / sigma) / (np.exp(-((d1) / sigma)) + 1))).to(torch.device("cuda:0")) * no_labels + labels
    return w


def weight_map_batch(batch):
    return torch.stack([weight_map(batch[i]) for i in range(batch.shape[0])])