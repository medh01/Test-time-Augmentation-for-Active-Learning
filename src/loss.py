import torch
import torch.nn as nn
from torch.nn.functional import F


class TverskyLossNoBG(nn.Module):
    def __init__(self, alpha, beta, smooth=1e-6, bg_idx=4):
        super().__init__()
        self.alpha, self.beta = alpha, beta
        self.smooth = smooth
        self.bg_idx = bg_idx

    def forward(self, logits, target):
        B,C,H,W = logits.shape
        probs   = F.softmax(logits, dim=1)
        oh      = F.one_hot(target.clamp(0,C-1), C).permute(0,3,1,2).float()
        dims    = (0,2,3)
        TP = (probs * oh).sum(dims)
        FP = (probs * (1 - oh)).sum(dims)
        FN = ((1 - probs) * oh).sum(dims)
        # exclude background
        mask = torch.ones(C, dtype=torch.bool, device=logits.device)
        mask[self.bg_idx] = False
        TP,FP,FN = TP[mask], FP[mask], FN[mask]
        TI = (TP + self.smooth) / (TP + self.alpha*FP + self.beta*FN + self.smooth)
        return (1 - TI).mean()