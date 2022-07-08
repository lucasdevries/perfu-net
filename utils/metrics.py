"""
This file will contain the metrics of the framework
"""
import numpy as np
import torch
import torch.nn as nn
from torch import einsum


class DiceMetric(nn.Module):
    def __init__(self, apply_nonlin=None, smooth=1e-5):
        """
        Generalized Dice;
        Copy from: https://github.com/LIVIAETS/surface-loss/blob/108bd9892adca476e6cdf424124bc6268707498e/losses.py#L29
        paper: https://arxiv.org/pdf/1707.03237.pdf
        tf code: https://github.com/NifTK/NiftyNet/blob/dev/niftynet/layer/loss_segmentation.py#L279
        """
        super(DiceMetric, self).__init__()

        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, net_output, gt):
        # net_output = net_output[torch.sum(gt, dim=(1,2)) > 0]
        # gt = gt[torch.sum(gt, dim=(1,2)) > 0]
        net_output, gt = preprocessor(net_output, gt, self.apply_nonlin)

        intersection: torch.Tensor = einsum("bcxy, bcxy->bc", net_output, gt)
        summed: torch.Tensor = (einsum("bcxy->bc", net_output) + einsum("bcxy->bc", gt))
        divided: torch.Tensor = 2 * (einsum("bc->b", intersection) + self.smooth) / (
                einsum("bc->b", summed) + self.smooth)
        gdc = divided.mean()
        return gdc


class IoUMetric(nn.Module):
    def __init__(self, apply_nonlin=None, smooth=1e-5):
        """
        Intersection over Union based on Generalized Dice from;
        Copy from: https://github.com/LIVIAETS/surface-loss/blob/108bd9892adca476e6cdf424124bc6268707498e/losses.py#L29
        paper: https://arxiv.org/pdf/1707.03237.pdf
        tf code: https://github.com/NifTK/NiftyNet/blob/dev/niftynet/layer/loss_segmentation.py#L279
        """
        super(IoUMetric, self).__init__()

        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, net_output, gt):
        net_output, gt = preprocessor(net_output, gt, self.apply_nonlin)

        intersection: torch.Tensor = einsum("bcxy, bcxy->bc", net_output, gt)
        union: torch.Tensor = (einsum("bcxy->bc", net_output) + einsum("bcxy->bc", gt) - intersection)
        divided: torch.Tensor = (einsum("bc->b", intersection) + self.smooth) / (
                einsum("bc->b", union) + self.smooth)
        iou = divided.mean()

        return iou


class PrecisionMetric(nn.Module):
    """
    Precision or Positive Predictive Value.
    """

    def __init__(self, apply_nonlin=None, smooth=1e-5):
        super(PrecisionMetric, self).__init__()

        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, net_output, gt):
        net_output, gt = preprocessor(net_output, gt, self.apply_nonlin)
        # tp = net_output_binary * gt
        # fp = net_output_binary * (1 - gt)
        # fn = (1 - net_output_binary) * gt

        tp: torch.Tensor = einsum("bcxy, bcxy->bc", net_output, gt)
        fp: torch.Tensor = einsum("bcxy, bcxy->bc", net_output, (1 - gt))
        divided: torch.Tensor = (einsum("bc->b", tp) + self.smooth) / (
                einsum("bc->b", tp) + einsum("bc->b", fp) + self.smooth)
        precision = divided.mean()

        return precision


class RecallMetric(nn.Module):
    def __init__(self, apply_nonlin=None, smooth=1e-5):
        """
        Recall or Sensitivity.
        """
        super(RecallMetric, self).__init__()

        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, net_output, gt):
        net_output, gt = preprocessor(net_output, gt, self.apply_nonlin)
        # tp = net_output_binary * gt
        # fp = net_output_binary * (1 - gt)
        # fn = (1 - net_output_binary) * gt

        tp: torch.Tensor = einsum("bcxy, bcxy->bc", net_output, gt)
        fn: torch.Tensor = einsum("bcxy, bcxy->bc", (1 - net_output), gt)

        divided: torch.Tensor = (einsum("bc->b", tp) + self.smooth) / (
                einsum("bc->b", tp) + einsum("bc->b", fn) + self.smooth)
        recall = divided.mean()

        return recall
class Prec_1d(nn.Module):
    def __init__(self, apply_nonlin=None, smooth=1e-5):
        """
        Recall or Sensitivity.
        """
        super(Prec_1d, self).__init__()

        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, net_output, gt):
        # net_output, gt = preprocessor(net_output, gt, self.apply_nonlin)
        # net_output = self.apply_nonlin(net_output, dim=1)
        net_output = net_output.argmax(dim=1)
        net_output = net_output.unsqueeze(dim=1)
        # tp = net_output_binary * gt
        # fp = net_output_binary * (1 - gt)
        # fn = (1 - net_output_binary) * gt

        tp: torch.Tensor = einsum("bc, bc->bc", net_output, gt)
        fn: torch.Tensor = einsum("bc, bc->bc", (1 - net_output), gt)
        fp: torch.Tensor = einsum("bc, bc->bc", net_output, (1 - gt))

        divided: torch.Tensor = (einsum("bc->b", tp).sum() + self.smooth) / (
                einsum("bc->b", tp).sum() + einsum("bc->b", fp).sum() + self.smooth)
        precision = divided
        return precision

class F1(nn.Module):
    def __init__(self, apply_nonlin=None, smooth=1e-5):
        """
        Recall or Sensitivity.
        """
        super(F1, self).__init__()

        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, net_output, gt):
        # net_output, gt = preprocessor(net_output, gt, self.apply_nonlin)
        # net_output = self.apply_nonlin(net_output, dim=1)
        net_output = net_output.argmax(dim=1)
        net_output = net_output.unsqueeze(dim=1)
        # tp = net_output_binary * gt
        # fp = net_output_binary * (1 - gt)
        # fn = (1 - net_output_binary) * gt

        tp: torch.Tensor = einsum("bc, bc->bc", net_output, gt)
        fn: torch.Tensor = einsum("bc, bc->bc", (1 - net_output), gt)
        fp: torch.Tensor = einsum("bc, bc->bc", net_output, (1 - gt))

        divided: torch.Tensor = (einsum("bc->b", tp).sum() + self.smooth) / (
                einsum("bc->b", tp).sum() + einsum("bc->b", fp).sum() + self.smooth)
        precision = divided.mean()

        divided2: torch.Tensor = (einsum("bc->b", tp).sum() + self.smooth) / (
                einsum("bc->b", tp).sum() + einsum("bc->b", fn).sum() + self.smooth)
        recall = divided2.mean()


        f1 = (2* precision*recall)/ (precision+recall)
        # f1 = tp / (tp + 0.5(fp + fn))
        # divided: torch.Tensor = (einsum("bc->b", tp) + self.smooth) / (
        #         einsum("bc->b", tp) + 0.5*(einsum("bc->b", fn) + einsum("bc->b", fp)) + self.smooth )
        # f1 = divided.mean()
        return f1


class SpecificityMetric(nn.Module):
    def __init__(self, apply_nonlin=None, smooth=1e-5):
        """
        Specificity.
        """
        super(SpecificityMetric, self).__init__()

        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, net_output, gt):
        net_output, gt = preprocessor(net_output, gt, self.apply_nonlin)

        # tp = net_output_binary * gt
        # fp = net_output_binary * (1 - gt)
        # fn = (1 - net_output_binary) * gt
        # tn = (1-gt) * (1-net_output_binary)

        tn: torch.Tensor = einsum("bcxy, bcxy->bc", (1 - gt), (1 - net_output))
        fp: torch.Tensor = einsum("bcxy, bcxy->bc", net_output, (1 - gt))

        divided: torch.Tensor = (einsum("bc->b", tn) + self.smooth) / (
                einsum("bc->b", tn) + einsum("bc->b", fp) + self.smooth)
        recall = divided.mean()

        return recall


class AverageMeter:
    def __init__(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.value = val * n
        self.count += n
        self.avg = self.sum / self.count

    @property
    def val(self):
        return self.value


def preprocessor(net_output, gt, apply_nonlin):
    shp_x = net_output.shape  # (batch size,class_num,x,y,z)
    shp_y = gt.shape  # (batch size,1,x,y,z)
    # one hot code for gt
    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

    if apply_nonlin is not None:
        net_output = apply_nonlin(net_output)

    # copy from https://github.com/LIVIAETS/surface-loss/blob/108bd9892adca476e6cdf424124bc6268707498e/losses.py#L29
    net_output = net_output[:, 1, :, :]
    net_output_binary = torch.where(net_output > 0.5, 1, 0)
    net_output_binary = net_output_binary.view((net_output_binary.shape[0], 1, *net_output_binary.shape[1:]))

    if net_output.device.type == "cuda":
        net_output_binary = net_output_binary.cuda(net_output.device.index)

    gt = gt.type(torch.float64)
    net_output_binary = net_output_binary.type(torch.float64)
    return net_output_binary, gt
