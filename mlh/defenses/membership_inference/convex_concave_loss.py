import torch.nn as nn
import torch
import torch.nn.functional as F
def focal_loss(input_values, gamma, reduction="mean"):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values

    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        raise ValueError("Invalid reduction option. Use 'none', 'mean', or 'sum'.")
class FocalLoss(nn.Module):
    def __init__(self, gamma=0., beta = 1,reduction='mean'):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.beta = beta
        self.reduction = reduction
    def forward(self, input, target):
        return self.beta *focal_loss(F.cross_entropy(input, target, reduction="none"), self.gamma, reduction = self.reduction)
    
def taylor_exp(input_values, alpha, reduction="mean"):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    loss = alpha*input_values - (1-alpha)*(p+torch.pow(p,2)/2)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        raise ValueError("Invalid reduction option. Use 'none', 'mean', or 'sum'.")
    
def ce_concave_exp_loss(input_values, alpha, beta, reduction="mean"):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    
    loss = alpha * input_values - beta *torch.exp(p)

    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        raise ValueError("Invalid reduction option. Use 'none', 'mean', or 'sum'.")
def concave_exp_loss(input_values, gamma =1, reduction="mean"):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    loss = torch.exp(gamma*p)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        raise ValueError("Invalid reduction option. Use 'none', 'mean', or 'sum'.")


class CCEL(nn.Module):
    def __init__(self, alpha = 1, beta = 1, gamma=1.0, tau =1,reduction='mean'):
        super(CCEL, self).__init__()
        assert gamma >= 1e-7
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction
    def forward(self, input, target):
        return self.beta*ce_concave_exp_loss(F.cross_entropy(input, target, reduction="none"), self.alpha, (1-self.alpha), self.gamma, reduction = self.reduction)


class CCQL(nn.Module):
    def __init__(self, alpha = 1,reduction='mean'):
        super(CCQL, self).__init__()
        #assert gamma >= 1e-7
        self.gamma = 2
        self.alpha = alpha
        self.reduction = reduction
    def forward(self, input, target):
        ce = F.cross_entropy(input, target, reduction="none")
        return taylor_exp(ce, self.alpha, self.reduction)

class ConcaveTaylor(nn.Module):
    def __init__(self, alpha = 1, beta = 1, gamma=2, tau =1,reduction='mean'):
        super(ConcaveTaylor, self).__init__()
        self.gamma = 2
        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction
    def forward(self, input, target):
        ce = F.cross_entropy(input, target, reduction="none")
        p = torch.exp(-ce)
        loss = self.alpha * ce 
        for n in range(self.gamma):
            loss = self.alpha*loss - (1-self.alpha)*(torch.pow(p , n+1))
        return self.beta*loss.mean()



class FocalCCEL(nn.Module):
    def __init__(self, alpha = 1, gamma=1.0,reduction='mean'):
        super(FocalCCEL, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
    def forward(self, input, target):
        ce = F.cross_entropy(input, target, reduction="none")
        losses = focal_loss(ce, gamma = self.gamma, reduction="none")
        cel = concave_exp_loss(ce,reduction="none")
        loss = self.alpha * losses + (1-self.alpha)*cel
        if self.reduction == "none":
            return loss
        elif self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            raise ValueError("Invalid reduction option. Use 'none', 'mean', or 'sum'.")