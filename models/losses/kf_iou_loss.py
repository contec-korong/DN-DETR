"""
Most code is copied from mmrotate
https://github.com/open-mmlab/mmrotate/blob/main/mmrotate/models/losses/kf_iou_loss.py
"""
from torch import nn
import torch
import torch.nn.functional as F

from util.box_ops import box_decode


def reduce_loss(loss, reduction):
    """Reduce loss as specified.
    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".
    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def xy_wh_r_2_xy_sigma(xywhr):
    """Convert oriented bounding box to 2-D Gaussian distribution.

    Args:
        xywhr (torch.Tensor): rbboxes with shape (N, 5).

    Returns:
        xy (torch.Tensor): center point of 2-D Gaussian distribution
            with shape (N, 2).
        sigma (torch.Tensor): covariance matrix of 2-D Gaussian distribution
            with shape (N, 2, 2).
    """
    _shape = xywhr.shape
    assert _shape[-1] == 5
    xy = xywhr[..., :2]
    wh = xywhr[..., 2:4].clamp(min=1e-7, max=1e7).reshape(-1, 2)
    r = xywhr[..., 4]
    cos_r = torch.cos(r)
    sin_r = torch.sin(r)
    R = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)
    S = 0.5 * torch.diag_embed(wh)

    sigma = R.bmm(S.square()).bmm(R.permute(0, 2,
                                            1)).reshape(_shape[:-1] + (2, 2))

    return xy, sigma


def kfiou_match_loss(pred,
                     target,
                     fun=None,
                     reduction='mean',
                     eps=1e-6):
    """Kalman filter IoU loss.

    Args:
        pred (torch.Tensor): Predicted decoded bboxes. (M, 5)
        target (torch.Tensor): Corresponding decoded gt bboxes. (N, 5)
        fun (str): The function applied to distance. Defaults to None.
        reduction (str): Reduction function. Defaults to mean.
        eps (float): Defaults to 1e-6.

    Returns:
        loss (torch.Tensor): KFIoU loss. (M, N)
    """
    _, Sigma_p = xy_wh_r_2_xy_sigma(pred)       # sigma_p: M, 2, 2
    _, Sigma_t = xy_wh_r_2_xy_sigma(target)     # sigma_t: N, 2, 2

    Vb_p = 4 * abs(Sigma_p.det()).sqrt()        # M,
    Vb_t = 4 * abs(Sigma_t.det()).sqrt()        # N,

    len_p = Vb_p.shape[0]
    len_t = Vb_t.shape[0]

    Vb_p = Vb_p.repeat(len_t, 1).mT     # M, N
    Vb_t = Vb_t.repeat(len_p, 1)        # M, N

    Sigma_p = Sigma_p.repeat(len_t, 1, 1, 1).transpose(1, 0)   # M, N, 2, 2
    Sigma_t = Sigma_t.repeat(len_p, 1, 1, 1)                   # M, N, 2, 2
    K = Sigma_p.matmul((Sigma_p + Sigma_t).inverse())          # M, N, 2, 2
    Sigma = Sigma_p - K.matmul(Sigma_p)

    Vb = 4 * abs(Sigma.det()).sqrt()    # M, N
    # Vb = torch.where(torch.isnan(Vb), torch.full_like(Vb, 0), Vb)
    KFIoU = Vb / (Vb_p + Vb_t - Vb + eps)

    # print('KFIoU match 1: ', KFIoU)  # Value is too high. It should be < 1/3 ???

    if fun == 'ln':
        kf_loss = -torch.log(KFIoU + eps)
    elif fun == 'exp':
        kf_loss = torch.exp(1 - KFIoU) - 1
    else:
        kf_loss = 1 - KFIoU

    loss = kf_loss.clamp(0)
    loss = reduce_loss(loss, reduction)

    return loss


def kfiou_loss(pred,
               target,
               fun=None,
               reduction='none',
               eps=1e-6):
    """Kalman filter IoU loss.

    Args:
        pred (torch.Tensor): Predicted decoded bboxes.
        target (torch.Tensor): Corresponding gt decoded bboxes.
        fun (str): The function applied to distance. Defaults to None.
        reduction (str): Reduction function. Defaults to None.
        eps (float): Defaults to 1e-6.

    Returns:
        loss (torch.Tensor)
    """
    _, Sigma_p = xy_wh_r_2_xy_sigma(pred)       # Variance of predicted boxes
    _, Sigma_t = xy_wh_r_2_xy_sigma(target)     # Variance of target boxes

    Vb_p = 4 * abs(Sigma_p.det()).sqrt()        # Area of predicted boxes
    Vb_t = 4 * abs(Sigma_t.det()).sqrt()        # Area of target boxes
    K = Sigma_p.bmm((Sigma_p + Sigma_t).inverse())  # Kalman gain
    Sigma = Sigma_p - K.bmm(Sigma_p)            # Variance of overlapped area between the predicted and target boxes
    Vb = 4 * abs(Sigma.det()).sqrt()            # Overlapped area
    # Vb = torch.where(torch.isnan(Vb), torch.full_like(Vb, 0), Vb)
    KFIoU = Vb / (Vb_p + Vb_t - Vb + eps)

    if fun == 'ln':
        kf_loss = -torch.log(KFIoU + eps)
    elif fun == 'exp':
        kf_loss = torch.exp(1 - KFIoU) - 1
    else:
        kf_loss = 1 - KFIoU

    loss = kf_loss.clamp(0)
    loss = reduce_loss(loss, reduction)

    return loss


class KFLoss(nn.Module):
    """Kalman filter based loss.

    Args:
        fun (str, optional): The function applied to distance.
            Defaults to 'none'.
        reduction (str, optional): The reduction method of the
            loss. Defaults to 'mean'.
        match (bool, optional): Calculation for HungarianMatcher. Defaults to False.
        loss_weight (float, optional): The weight of loss. Defaults to 1.0.

    Returns:
        loss (torch.Tensor)
    """

    def __init__(self,
                 fun='none',
                 reduction='mean',
                 match=False,
                 loss_weight=1.0):
        super(KFLoss, self).__init__()
        assert fun in ['none', 'ln', 'exp']
        self.fun = fun
        self.reduction = reduction
        self.match = match
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target):
        """Forward function.

        Args:
            pred (torch.Tensor): Predicted convexes.
            target (torch.Tensor): Corresponding gt convexes.

        Returns:
            loss (torch.Tensor)
        """
        pred = box_decode(pred)
        target = box_decode(target)

        if self.match:
            return kfiou_match_loss(
                pred,
                target,
                fun=self.fun,
                reduction=self.reduction) * self.loss_weight
        else:
            return kfiou_loss(
                pred,
                target,
                fun=self.fun,
                reduction=self.reduction) * self.loss_weight
