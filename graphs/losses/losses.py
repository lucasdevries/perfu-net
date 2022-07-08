"""
loss file for all the loss functions used in this model
name: loss.py
date: April 2018
"""
import torch
from torch import nn
from torch import einsum
import torch.nn.functional as F
from utils.train_utils import weight_map_batch


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight):
        super(CrossEntropyLoss2d, self).__init__()
        # super().__init__()
        self.loss = nn.CrossEntropyLoss(weight=weight)

    def forward(self, logits, labels):
        loss = self.loss(logits, labels)
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
    Do Not forget to initialize the bias according to the original paper. Bias = -2 for the final layer when
    performing binary segmentation.
    """

    def __init__(self, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, labels):
        log_prob = F.log_softmax(logits, dim=1)
        prob = torch.exp(log_prob)
        loss_func = (1 - prob) ** self.gamma * log_prob
        nll_loss = F.nll_loss(loss_func, labels, weight=self.weight, reduction='mean')
        return nll_loss


class DoubleWeightedCE(nn.Module):
    """
    Should not yet be trusted.
    """

    def __init__(self, weight):
        super(DoubleWeightedCE, self).__init__()
        self.loss = nn.CrossEntropyLoss(weight=weight, reduction='none')
        self.weight = weight

    def forward(self, logits, labels):
        loss = self.loss(logits, labels)
        w0 = self.weight[0]  # 0.03
        w1 = self.weight[1]  # 0.97
        result = loss * labels * w1 + loss * (1 - labels) * w0
        sum_weights = (labels * w1 + (1 - labels) * w0).sum()
        result = result.sum(dim=(1, 2)).type(torch.float32)
        scaling = 1 / labels.sum(dim=(1, 2)).type(torch.float32)
        new_loss = torch.dot(result, scaling)
        return new_loss


class PixelWiseCrossEntropy(nn.Module):
    def __init__(self, class_weight=None, smooth=1e-5):
        super(PixelWiseCrossEntropy, self).__init__()
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.smooth = smooth
        self.class_weight = class_weight

    def forward(self, net_output, gt):

        weight = weight_map_batch(gt)
        assert gt.size() == weight.size()
        log_probabilities = self.log_softmax(net_output)

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
        weight = weight.unsqueeze(1)
        weight = weight.expand_as(net_output)

        if self.class_weight == None:
            class_weight = torch.ones(net_output.size()).float().cuda(net_output.device.index)
        else:
            class_weight = self.class_weight
            class_weight = class_weight.view(1, -1, 1, 1)

        weight = class_weight * weight

        result = -weight * y_onehot * log_probabilities

        return result.mean()


def create_loss_module(config, class_weights):
    if config.criterion == "CrossEntropy":
        criterion = CrossEntropyLoss2d(weight=class_weights)

    elif config.criterion == "Focal":
        criterion = FocalLoss(weight=class_weights)

    elif config.criterion == "Dice":
        criterion = GDiceLoss(apply_nonlin=torch.nn.Softmax(dim=1))

    elif config.criterion == "DoubleWeightedCE":
        criterion = DoubleWeightedCE(weight=class_weights)

    elif config.criterion == "PixelWiseCrossEntropy":
        criterion = PixelWiseCrossEntropy(class_weight=class_weights)

    return criterion
