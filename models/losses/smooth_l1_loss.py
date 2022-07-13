from torch import nn
import torch
import torch.nn.functional as F


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


def smooth_l1_match_loss(pred,
                         target,
                         reduction='mean',
                         beta=1./9.):
    """Smooth L1 loss for center of bbox.

    Args:
        pred (torch.Tensor): Predicted bboxes normalized by image size. (M, 5)
        target (torch.Tensor): Corresponding gt bboxes  normalized by image size. (N, 5)
        reduction (str): Reduction function. Defaults to mean.
        beta (float): Defaults to 1./9.

    Returns:
        loss (torch.Tensor): Smooth L1 loss. (M, N)
    """
    xy_p = pred[:, :2]     # (M, 2)
    xy_t = target[:, :2]    # (N, 2)

    loss = torch.cdist(xy_p, xy_t, p=1)  # L1-norm
    loss = torch.where(loss < beta,
                       0.5 * loss * loss / beta,
                       loss - 0.5 * beta)
    loss = reduce_loss(loss, reduction)

    return loss


def smooth_l1_loss(pred,
                   target,
                   reduction='none',
                   beta=1./9.):
    """Smooth L1 loss for center of bbox.

    Args:
        pred (torch.Tensor): Predicted bboxes normalized by image size. (N, 5)
        target (torch.Tensor): Corresponding gt bboxes normalized by image size. (N, 5)
        reduction (str): Reduction function. Defaults to None.
        beta (float): Defaults to 1./9.

    Returns:
        loss (torch.Tensor): Smooth L1 loss (N, 5)
    """
    xy_p = pred[:, :2]
    xy_t = target[:, :2]

    diff = torch.abs(xy_p - xy_t)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                       diff - 0.5 * beta).sum(dim=-1)
    loss = reduce_loss(loss, reduction)

    return loss


class SmoothLoss(nn.Module):
    """Smooth L1 loss.

    Args:
        reduction (str, optional): The reduction method of the loss. Defaults to 'mean'.
        match (bool, optional): Calculation for HungarianMatcher. Defaults to False.

    Returns:
        loss (torch.Tensor)
    """

    def __init__(self,
                 reduction='mean',
                 match=False):
        super(SmoothLoss, self).__init__()
        self.reduction = reduction
        self.match = match

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
        if self.match:
            return smooth_l1_match_loss(
                pred,
                target,
                reduction=self.reduction)
        else:
            return smooth_l1_loss(
                pred,
                target,
                reduction=self.reduction)
