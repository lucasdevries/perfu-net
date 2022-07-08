"""
loss file for all the loss functions used in this model
name: loss.py
date: April 2018
"""
import torch
from torch import nn
from torch import einsum
import torch.nn.functional as F
import numpy as np

class CompoundWCEFocal(nn.Module):
    def __init__(self, gamma=2, apply_nonlin=torch.nn.Softmax(dim=1)):
        super(CompoundWCEFocal, self).__init__()
        # super().__init__()
        self.CE = CrossEntropyLoss2d()
        self.Focal = FocalLoss(gamma=gamma, apply_nonlin=apply_nonlin)

    def forward(self, logits, labels, weights, alpha):
        if weights.device != logits.device:
            weights = weights.to(logits.device)

        loss_ce = self.CE(logits, labels, weights)
        loss_focal = self.Focal(logits, labels, weights)
        loss = alpha * loss_ce + (1-alpha) * loss_focal
        return loss

class CompoundDiceFocal(nn.Module):
    def __init__(self, gamma=2, apply_nonlin=torch.nn.Softmax(dim=1)):
        super(CompoundDiceFocal, self).__init__()
        # super().__init__()
        self.Dice = GDiceLoss(apply_nonlin=apply_nonlin)
        self.Focal = FocalLoss(gamma=gamma, apply_nonlin=apply_nonlin)

    def forward(self, logits, labels, weights, alpha):
        if weights.device != logits.device:
            weights = weights.to(logits.device)

        loss_dice = self.Dice(logits, labels)
        loss_focal = self.Focal(logits, labels, weights)
        loss = alpha * loss_dice + (1-alpha) * loss_focal
        return loss
class CompoundDiceCE(nn.Module):
    def __init__(self, apply_nonlin=torch.nn.Softmax(dim=1)):
        super(CompoundDiceCE, self).__init__()
        # super().__init__()
        self.Dice = GDiceLoss(apply_nonlin=apply_nonlin)
        self.CE = CrossEntropyLoss2d()

    def forward(self, logits, labels, weights, alpha=0.5):
        if weights is None:
            loss_ce = self.CE(logits, labels, None)
        else:
            if weights.device != logits.device:
                weights = weights.to(logits.device)
            loss_ce = self.CE(logits, labels, weights)

        loss_dice = self.Dice(logits, labels)
        loss = alpha * loss_dice + (1-alpha) * loss_ce
        return loss
class CrossEntropyLoss2d(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss2d, self).__init__()
        # super().__init__()

    def forward(self, logits, labels, weights):
        if weights is None:
            loss = F.cross_entropy(logits, labels)
        else:
            if weights.device != logits.device:
                weights = weights.to(logits.device)
            loss = F.cross_entropy(logits, labels, weights)
        return loss


class GDiceLoss(nn.Module):
    def __init__(self, apply_nonlin=None, smooth=1e-5):
        """
        Generalized Dice;
        Copy from: https://github.com/LIVIAETS/surface-loss/blob/108bd9892adca476e6cdf424124bc6268707498e/losses.py#L29
        paper: https://arxiv.org/pdf/1707.03237.pdf
        tf code: https://github.com/NifTK/NiftyNet/blob/dev/niftynet/layer/loss_segmentation.py#L279
        """
        super(GDiceLoss, self).__init__()

        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, net_output, gt):

        shp_x = net_output.shape  # (batch size,class_num,x,y,z)
        shp_y = gt.shape  # (batch size,1,x,y,z)
        # one hot code for gt
        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                gt = gt.view((shp_y[0], 1, *shp_y[1:]))

            if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = gt
            else:
                gt = gt.long()
                y_onehot = torch.zeros(shp_x)
                if net_output.device.type == "cuda":
                    y_onehot = y_onehot.cuda(net_output.device.index)
                y_onehot.scatter_(1, gt, 1)

        if self.apply_nonlin is not None:
            net_output = self.apply_nonlin(net_output)

        # copy from https://github.com/LIVIAETS/surface-loss/blob/108bd9892adca476e6cdf424124bc6268707498e/losses.py#L29
        net_output = net_output.double()
        y_onehot = y_onehot.double()
        w: torch.Tensor = 1 / (einsum("bcxy->bc", y_onehot).type(torch.float32) + 1e-10) ** 2
        intersection: torch.Tensor = w * einsum("bcxy, bcxy->bc", net_output, y_onehot)
        union: torch.Tensor = w * (einsum("bcxy->bc", net_output) + einsum("bcxy->bc", y_onehot))
        divided: torch.Tensor = - 2 * (einsum("bc->b", intersection) + self.smooth) / (
                einsum("bc->b", union) + self.smooth)
        gdc = divided.mean()

        return gdc


class FocalLoss(nn.Module):
    """
    copy from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/FocalLoss.py
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, apply_nonlin=None, gamma=2, balance_index=0, smooth=1e-5, size_average=True):
        super(FocalLoss, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, logit, target, alpha):
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)
        num_class = logit.shape[1]

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = torch.squeeze(target, 1)
        target = target.view(-1, 1)

        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray, torch.Tensor)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            alpha = torch.ones(num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha
        else:
            raise TypeError('Not support alpha type')

        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)
        idx = target.cpu().long()

        one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)
        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth / (num_class - 1), 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + self.smooth
        logpt = pt.log()
        gamma = self.gamma

        alpha = alpha[idx]
        alpha = torch.squeeze(alpha)
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


def loss_module(config):
    if config.criterion == "wCE":
        criterion = CrossEntropyLoss2d()
    elif config.criterion == "wFocal":
        criterion = FocalLoss(gamma=config.focal_gamma, apply_nonlin=torch.nn.Softmax(dim=1))
    elif config.criterion == "gDice":
        criterion = GDiceLoss(apply_nonlin=torch.nn.Softmax(dim=1))
    elif config.criterion == "gDicewCE":
        criterion = CompoundDiceCE(apply_nonlin=torch.nn.Softmax(dim=1))
    else:
        raise ValueError('Loss not implemented!!!')
    return criterion
