import torch.nn as nn
import torch.nn.functional as F
import torch

from .utils import weight_reduce_loss
from ..builder import LOSSES



@LOSSES.register_module()
class CriterionPixelWiseLossPPM(nn.Module):
    """ PPM pixel wise loss calculation module.

    Args:
        tau (float, optional): Temperature coefficient. Defaults to 1.0.
        loss_weight (float, optional): Weight of loss.Defaults to 1.0.
        in_channels (int, optional): .Number of input channels.
        Defaults to None.
        out_channels (int, optional): .Number of output channels.
        Defaults to None.
    """
    def __init__(self,
                 tau=1.0,
                 loss_weight=1.0,
                 in_channels = None,
                 out_channels = None):
        super(CriterionPixelWiseLossPPM, self).__init__()
        self.tau = tau
        self.loss_weight = loss_weight
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=1,bias = False)

    def forward(self,
                preds_S,
                preds_T):
        """Forward function."""
        if  not self.in_channels == self.out_channels:
            self.conv1.to(preds_S.device)
            preds_S = self.conv1(preds_S)
        N,C,W,H = preds_S.shape
        softmax_pred_T = F.softmax(preds_T.view(-1,W*H)/self.tau, dim=1)
        logsoftmax = torch.nn.LogSoftmax(dim=1)
        loss = torch.sum( - softmax_pred_T * logsoftmax(preds_S.view(-1,W*H)/self.tau))
        return self.loss_weight*loss / (C * N)


@LOSSES.register_module()
class CriterionPixelWiseLossLogits(nn.Module):
    """ Logits pixel wise loss calculation module.

    Args:
        tau (float, optional): Temperature coefficient. Defaults to 1.0.
        loss_weight (float, optional): Weight of loss.Defaults to 1.0.
    """
    def __init__(self,
                 tau=1.0,
                 loss_weight=1.0):
        super(CriterionPixelWiseLossLogits, self).__init__()
        self.tau = tau
        self.loss_weight = loss_weight

    def forward(self,
                preds_S,
                preds_T):
        """Forward function."""
        assert preds_S.shape == preds_T.shape,'the output dim of teacher and student differ'
        N,C,W,H = preds_S.shape
        softmax_pred_T = F.softmax(preds_T.view(-1,W*H)/self.tau, dim=1)
        logsoftmax = torch.nn.LogSoftmax(dim=1)
        loss = torch.sum( - softmax_pred_T * logsoftmax(preds_S.view(-1,W*H)/self.tau))
        return self.loss_weight*loss / (C * N)