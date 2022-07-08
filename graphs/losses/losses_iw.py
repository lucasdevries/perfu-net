# """
# loss file for all the loss functions used in this model
# name: loss.py
# date: April 2018
# """
# import torch
#
# from dpipe.torch.utils import sequence_to_var, to_np
# from dpipe.torch.model import optimizer_step
# import numpy as np
# import matplotlib.pyplot as plt
#
# def mask2weight(mask, type='ln'):
#     voxel_area = torch.sum(mask, dim=(1,2))
#     if type == 'ln':
#         return 1/torch.log(voxel_area)
#     elif type == 'xln':
#         return 1/(voxel_area*torch.log(voxel_area))
#     else:
#         return torch.ones(mask.shape[0])
# # ================ Dice Loss ==========================================================================================
# def cc2weight(cc, w_min: float = 1., w_max: float = 2e5):
#     weight = np.zeros_like(cc, dtype='float32')
#     cc_items = np.unique(cc)
#     K = len(cc_items) - 1
#     N = np.prod(cc.shape)
#     for i in cc_items:
#         weight[cc == i] = N / ((K + 1) * np.sum(cc == i))
#     return np.clip(weight, w_min, w_max)
#
# # ======= Train Step to pass weights (or old: connected components) through ===========================================
# def dice_loss(preds: torch.Tensor, target: torch.Tensor, weight: torch.Tensor = None):
#     """
#     References
#     ----------
#     `Dice Loss <https://arxiv.org/abs/1606.04797>`_
#     """
#     if not (target.size() == preds.size()):
#         raise ValueError("Target size ({}) must be the same as logit size ({})".format(target.size(), logit.size()))
#
#     sum_dims = list(range(1, preds.dim()))
#
#     if weight is None:
#         dice = 2 * torch.sum(preds * target, dim=sum_dims) / torch.sum(preds ** 2 + target ** 2, dim=sum_dims)
#     else:
#         dice = 2 * torch.sum(preds * target, dim=sum_dims) / torch.sum((preds ** 2 + target ** 2), dim=sum_dims)
#         dice = dice * weight
#
#     loss = 1 - dice
#
#     return loss.mean()
#
#
# def train_step_with_cc(*inputs, criterion, optimizer, with_cc=False, **optimizer_params):
#     if with_cc:
#         n_inputs = len(inputs) - 2  # target and weight
#         # inputs = sequence_to_var(*inputs, device='cuda')
#         outputs, target, weight = inputs[0], inputs[-2], inputs[-1]
#         loss = criterion(outputs, target, weight)
#     else:
#         n_inputs = len(inputs) - 1  # target
#         inputs = sequence_to_var(*inputs, device='cuda')
#         outputs, target = inputs[:n_inputs], inputs[-1]
#         loss = criterion(outputs, target)
#
#     optimizer_step(optimizer, loss, **optimizer_params)
#     return loss