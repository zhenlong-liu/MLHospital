import sys
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../..")
from functools import partial
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import math
from torch.autograd.function import Function
from torch.autograd import Variable
from utils import CrossEntropy_soft, one_hot_embedding

def get_loss(loss_type, device, args, train_loader = None, num_classes = 10, reduction = "mean"):
    CIFAR10_CONFIG = {
        "smape" : SMAPELoss(num_classes=num_classes, scale = 5*args.temp, reduction = reduction),
        "logitclip_o": LogitClip(device, temp =args.temp, reduction = reduction),
        "ereg": EntropyRegularizedLoss(alpha = 0.1*args.temp, reduction = reduction),
        "ce": nn.CrossEntropyLoss(reduction = reduction),
        "ce_ls": nn.CrossEntropyLoss(label_smoothing= 0.1*args.temp, reduction = reduction),
        "focal": FocalLoss(gamma=args.temp, reduction = reduction),
        "mae": MAELoss(num_classes=num_classes, reduction = reduction),
        "gce": GCE(device, k=num_classes, alpha = args.alpha, q=args.temp, reduction = reduction),
        "sce": SCE(alpha=0.5, beta=args.temp, num_classes=num_classes, reduction = reduction),
        "ldam": LDAMLoss(device=device),
        "logit_norm": LogitNormLoss(device, args.temp, p=args.lp, reduction = reduction),
        "normreg": NormRegLoss(device, args.temp, p=args.lp),
        "logneg": logNegLoss(device, t=args.temp),
        "logit_clip": LogitClipLoss(device, threshold=args.temp, reduction = reduction),
        "cnorm": CNormLoss(device, args.temp),
        "tlnorm": TLogitNormLoss(device, args.temp, m=10),
        # "nlnl": NLNL(device, train_loader=train_loader, num_classes=num_classes),
        "nce": NCELoss(num_classes=num_classes, reduction = reduction),
        "ael": AExpLoss(num_classes=10, a=2.5),
        "aul": AUELoss(num_classes=10, a=5.5, q=3),
        "phuber": PHuberCE(tau=10),
        "taylor": TaylorCE(device=device, series=args.series),
        "cores": CoresLoss(device=device),
        "ncemae": NCEandMAE(alpha=1, beta=1, num_classes=10),
        "ngcemae": NGCEandMAE(alpha=1, beta=1, num_classes=10),
        "ncerce": NGCEandMAE(alpha=1, beta=1.0, num_classes=10),
        "nceagce": NCEandAGCE(alpha=1, beta=4, a=6, q=1.5, num_classes=10),
        "flood": FloodLoss(device=device, t = 0.1, reduction = reduction),
        "logit_cliping": LogitClipingLoss(device=device, tau= args.temp, p=args.lp, reduction = reduction),
        "concave_exp": ConcaveExpLoss(alpha= 1, beta =1 ) ,
        "concave_log": ConcaveLogLoss(alpha= 1, beta =1, gamma = 1),
        "concave_loss": ConcaveLoss(alpha= 1, beta =1, gamma = 1, tau = 0.5),
        "mixup_py": MixupPy(alpha= 0.05, beta =0.05, gamma = 1, tau = 0.5,device = args.device),
        "concave_taylor":ConcaveTaylor(alpha = args.alpha, beta = args.temp, gamma = args.gamma, tau = args.tau),
    }
    CIFAR100_CONFIG = {
        "ce": nn.CrossEntropyLoss(),
        "ce_ls": nn.CrossEntropyLoss(label_smoothing= 0.1*args.temp, reduction = reduction),
        "ereg": EntropyRegularizedLoss(alpha = 0.1*args.temp, reduction = reduction),
        "focal": FocalLoss(gamma=0.5),
        "mae": MAELoss(num_classes=num_classes),
        "gce": GCE(device, alpha = 0.1, q=0.2,k=num_classes),
        "sce": SCE(alpha=0.1, beta=0.1, num_classes=100),
        "ldam": LDAMLoss(device=device),
        "logit_clip": LogitClipLoss(device, threshold=args.temp),
        "logit_norm": LogitNormLoss(device, args.temp, p=args.lp),
        "normreg": NormRegLoss(device, args.temp, p=args.lp),
        "tlnorm": TLogitNormLoss(device, args.temp, m=100),
        "cnorm": CNormLoss(device, args.temp),
        # "nlnl": NLNL(device, train_loader=train_loader, num_classes=num_classes),
        "nce": NCELoss(num_classes=num_classes),
        "ael": AExpLoss(num_classes=100, a=2.5),
        "aul": AUELoss(num_classes=100, a=5.5, q=3),
        "phuber": PHuberCE(tau=30),
        "taylor": TaylorCE(device=device, series=args.series),
        "cores": CoresLoss(device=device),
        "ncemae": NCEandMAE(alpha=50*args.alpha, beta=1*args.temp, num_classes=100),
        "ngcemae": NGCEandMAE(alpha=50*args.alpha, beta=1*args.temp, num_classes=100),
        "ncerce": NGCEandMAE(alpha=50, beta=1.0, num_classes=100),
        "nceagce": NCEandAGCE(alpha=50*args.alpha, beta=0.1*args.temp, a=1.8, q=3.0, num_classes=100),
        "flood": FloodLoss(device=device, t = 0.1*args.temp, reduction = reduction),
        "concave_exp": ConcaveExpLoss(alpha= 0.05, beta =0.05 ),
        "concave_log": ConcaveLogLoss(alpha= 0.05, beta =0.05, gamma = 1),
        "concave_loss": ConcaveLoss(alpha= 0.05, beta =0.05, gamma = 1, tau = 0.5),
        "mixup_py": MixupPy(alpha= 0.05, beta =0.05, gamma = 1, tau = 0.5,device = args.device),
        "concave_taylor":ConcaveTaylor(alpha = args.alpha, beta = args.temp, gamma = args.gamma, tau = args.tau),
    }
    
    Imagenet_R_CONFIG = {
        "ce": nn.CrossEntropyLoss(),
        "ce_ls": nn.CrossEntropyLoss(label_smoothing= 0.1*args.temp, reduction = reduction),
        "ereg": EntropyRegularizedLoss(alpha = 0.1*args.temp, reduction = reduction),
        "focal": FocalLoss(gamma=0.5),
        "mae": MAELoss(num_classes=num_classes),
        "gce": GCE(device, alpha = 0.1, q=0.2,k=num_classes),
        "sce": SCE(alpha=0.1, beta=0.1, num_classes=100),
        "ldam": LDAMLoss(device=device),
        "logit_clip": LogitClipLoss(device, threshold=args.temp),
        "logit_norm": LogitNormLoss(device, args.temp, p=args.lp),
        "normreg": NormRegLoss(device, args.temp, p=args.lp),
        "tlnorm": TLogitNormLoss(device, args.temp, m=100),
        "cnorm": CNormLoss(device, args.temp),
        # "nlnl": NLNL(device, train_loader=train_loader, num_classes=num_classes),
        "nce": NCELoss(num_classes=num_classes),
        "ael": AExpLoss(num_classes=100, a=2.5),
        "aul": AUELoss(num_classes=100, a=5.5, q=3),
        "phuber": PHuberCE(tau=30),
        "taylor": TaylorCE(device=device, series=args.series),
        "cores": CoresLoss(device=device),
        "ncemae": NCEandMAE(alpha=50*args.alpha, beta=1*args.temp, num_classes=100),
        "ngcemae": NGCEandMAE(alpha=50*args.alpha, beta=1*args.temp, num_classes=100),
        "ncerce": NGCEandMAE(alpha=50, beta=1.0, num_classes=100),
        "nceagce": NCEandAGCE(alpha=50*args.alpha, beta=0.1*args.temp, a=1.8, q=3.0, num_classes=100),
        "flood": FloodLoss(device=device, t = 0.1*args.temp, reduction = reduction),
        "concave_exp": ConcaveExpLoss(alpha= 0.05, beta =0.05 ),
        "concave_log": ConcaveLogLoss(alpha= 0.05, beta =0.05, gamma = 1),
        "concave_loss": ConcaveLoss(alpha= 0.05, beta =0.05, gamma = 1, tau = 0.5),
        "mixup_py": MixupPy(alpha= 0.05, beta =0.05, gamma = 1, tau = 0.5,device = args.device),
        "concave_taylor":ConcaveTaylor(alpha = args.alpha, beta = args.temp, gamma = args.gamma, tau = args.tau),
    }
    
    Imagenet_CONFIG = {
        "ce": nn.CrossEntropyLoss(),
        "ce_ls": nn.CrossEntropyLoss(label_smoothing= 0.1*args.temp, reduction = reduction),
        "ereg": EntropyRegularizedLoss(alpha = 0.1*args.temp, reduction = reduction),
        "focal": FocalLoss(gamma=0.5),
        "mae": MAELoss(num_classes=num_classes),
        "gce": GCE(device, alpha = 0.1, q=0.1,k=num_classes),
        "sce": SCE(alpha=0.1*args.alpha, beta=0.1*args.temp, num_classes=num_classes),
        "ldam": LDAMLoss(device=device),
        "logit_clip": LogitClipLoss(device, threshold=args.temp),
        "logit_norm": LogitNormLoss(device, args.temp, p=args.lp),
        "normreg": NormRegLoss(device, args.temp, p=args.lp),
        "tlnorm": TLogitNormLoss(device, args.temp, m=num_classes),
        "cnorm": CNormLoss(device, args.temp),
        # "nlnl": NLNL(device, train_loader=train_loader, num_classes=num_classes),
        "nce": NCELoss(num_classes=num_classes),
        "ael": AExpLoss(num_classes=100, a=2.5),
        "aul": AUELoss(num_classes=100, a=5.5, q=3),
        "phuber": PHuberCE(tau=30),
        "taylor": TaylorCE(device=device, series=args.series),
        "cores": CoresLoss(device=device),
        "ncemae": NCEandMAE(alpha=50*args.alpha, beta=1*args.temp, num_classes=100),
        "ngcemae": NGCEandMAE(alpha=50*args.alpha, beta=1*args.temp, num_classes=100),
        "ncerce": NGCEandMAE(alpha=50, beta=1.0, num_classes=100),
        "nceagce": NCEandAGCE(alpha=50*args.alpha, beta=0.1*args.temp, a=1.8, q=3.0, num_classes=100),
        "flood": FloodLoss(device=device, t = 0.1*args.temp, reduction = reduction),
        "concave_log": ConcaveLogLoss(alpha= 1, beta =1, gamma = 1),
        "concave_loss": ConcaveLoss(alpha= 1, beta =1, gamma = 1, tau = 0.5),
        "mixup_py": MixupPy(alpha= 0.05, beta =0.05, gamma = 1, tau = 0.5,device = args.device),
        "concave_taylor":ConcaveTaylor(alpha = args.alpha, beta = args.temp, gamma = args.gamma, tau = args.tau),
    }
    
    TinyImagenet_CONFIG = {
        "ce": nn.CrossEntropyLoss(),
        "ce_ls": nn.CrossEntropyLoss(label_smoothing= 0.1*args.temp, reduction = reduction),
        "ereg": EntropyRegularizedLoss(alpha = 0.1*args.temp, reduction = reduction),
        "focal": FocalLoss(gamma=0.5),
        "mae": MAELoss(num_classes=num_classes),
        "gce": GCE(device, alpha = args.alpha, q=0.1*args.temp,k=num_classes),
        "sce": SCE(alpha=0.1*args.alpha , beta=0.1*args.temp, num_classes=num_classes),
        "ldam": LDAMLoss(device=device),
        "logit_clip": LogitClipLoss(device, threshold=args.temp),
        "logit_norm": LogitNormLoss(device, args.temp, p=args.lp),
        "normreg": NormRegLoss(device, args.temp, p=args.lp),
        "tlnorm": TLogitNormLoss(device, args.temp, m=num_classes),
        "cnorm": CNormLoss(device, args.temp),
        # "nlnl": NLNL(device, train_loader=train_loader, num_classes=num_classes),
        "nce": NCELoss(num_classes=num_classes),
        "ael": AExpLoss(num_classes=100, a=2.5),
        "aul": AUELoss(num_classes=100, a=5.5, q=3),
        "phuber": PHuberCE(tau=30),
        "taylor": TaylorCE(device=device, series=args.series),
        "cores": CoresLoss(device=device),
        "ncemae": NCEandMAE(alpha=50*args.alpha, beta=1*args.temp, num_classes=100),
        "ngcemae": NGCEandMAE(alpha=50*args.alpha, beta=1*args.temp, num_classes=100),
        "ncerce": NGCEandMAE(alpha=50, beta=1.0, num_classes=100),
        "nceagce": NCEandAGCE(alpha=50*args.alpha, beta=0.1*args.temp, a=1.8, q=3.0, num_classes=100),
        "flood": FloodLoss(device=device, t = 0.1*args.temp, reduction = reduction),
        "concave_log": ConcaveLogLoss(alpha= 1, beta =1, gamma = 1),
        "concave_loss": ConcaveLoss(alpha= 1, beta =1, gamma = 1, tau = 0.5),
        "mixup_py": MixupPy(alpha= 0.05, beta =0.05, gamma = 1, tau = 0.5,device = args.device),
        "concave_taylor":ConcaveTaylor(alpha = args.alpha, beta = args.temp, gamma = args.gamma, tau = args.tau),
    }
    
    WEB_CONFIG = {
        "ce": nn.CrossEntropyLoss(),
        "focal": FocalLoss(gamma=0.5),
        "mae": MAELoss(num_classes=num_classes),
        "gce": GCE(device, k=num_classes),
        "sce": SCE(alpha=0.5, beta=1.0, num_classes=num_classes),
        "ldam": LDAMLoss(device=device),
        "logit_norm": LogitNormLoss(device, args.temp, p=args.lp),
        "logit_clip": LogitClipLoss(device, threshold=args.temp),
        "normreg": NormRegLoss(device, args.temp, p=args.lp),
        "cnorm": CNormLoss(device,args.temp),
        "tlnorm": TLogitNormLoss(device, args.temp, m=50),
        # "nlnl": NLNL(device, train_loader=train_loader, num_classes=num_classes),
        "nce": NCELoss(num_classes=num_classes),
        "ael": AExpLoss(num_classes=50, a=2.5),
        "aul": AUELoss(num_classes=50, a=5.5, q=3),
        "phuber": PHuberCE(tau=30),
        "taylor": TaylorCE(device=device, series=args.series),
        "cores": CoresLoss(device=device),
        "ncemae": NCEandMAE(alpha=50, beta=0.1, num_classes=50),
        "ngcemae": NGCEandMAE(alpha=50, beta=0.1, num_classes=50),
        "ncerce": NGCEandMAE(alpha=50, beta=0.1, num_classes=50),
        "nceagce": NCEandAGCE(alpha=50, beta=0.1, a=2.5, q=3.0, num_classes=50),
        "concave_log": ConcaveLogLoss(alpha= 1, beta =1, gamma = 1),
        "concave_loss": ConcaveLoss(alpha= 1, beta =1, gamma = 1, tau = 0.5),
        "mixup_py": MixupPy(alpha= 0.05, beta =0.05, gamma = 1, tau = 0.5,device = args.device),
        "concave_taylor":ConcaveTaylor(alpha = args.alpha, beta = args.temp, gamma = args.gamma, tau = args.tau),
    }
    
    FashionMNIST = {
        "smape" : SMAPELoss(num_classes=num_classes, scale = 5*args.temp, reduction = reduction),
        "logitclip_o": LogitClip(device, temp =args.temp, reduction = reduction),
        "ereg": EntropyRegularizedLoss(alpha = 0.1*args.temp, reduction = reduction),
        "ce": nn.CrossEntropyLoss(reduction = reduction),
        "ce_ls": nn.CrossEntropyLoss(label_smoothing= 0.1*args.temp, reduction = reduction),
        "focal": FocalLoss(gamma=args.temp, reduction = reduction),
        "mae": MAELoss(num_classes=num_classes, reduction = reduction),
        "gce": GCE(device, k=num_classes, alpha = args.alpha, q=args.temp, reduction = reduction),
        "sce": SCE(alpha=0.5, beta=args.temp, num_classes=num_classes, reduction = reduction),
        "ldam": LDAMLoss(device=device),
        "logit_norm": LogitNormLoss(device, args.temp, p=args.lp, reduction = reduction),
        "normreg": NormRegLoss(device, args.temp, p=args.lp),
        "logneg": logNegLoss(device, t=args.temp),
        "logit_clip": LogitClipLoss(device, threshold=args.temp, reduction = reduction),
        "cnorm": CNormLoss(device, args.temp),
        "tlnorm": TLogitNormLoss(device, args.temp, m=10),
        # "nlnl": NLNL(device, train_loader=train_loader, num_classes=num_classes),
        "nce": NCELoss(num_classes=num_classes, reduction = reduction),
        "ael": AExpLoss(num_classes=10, a=2.5),
        "aul": AUELoss(num_classes=10, a=5.5, q=3),
        "phuber": PHuberCE(tau=10),
        "taylor": TaylorCE(device=device, series=args.series),
        "cores": CoresLoss(device=device),
        "ncemae": NCEandMAE(alpha=1, beta=1, num_classes=10),
        "ngcemae": NGCEandMAE(alpha=1, beta=1, num_classes=10),
        "ncerce": NGCEandMAE(alpha=1, beta=1.0, num_classes=10),
        "nceagce": NCEandAGCE(alpha=1, beta=4, a=6, q=1.5, num_classes=10),
        "flood": FloodLoss(device=device, t = 0.1, reduction = reduction),
        "logit_cliping": LogitClipingLoss(device=device, tau= args.temp, p=args.lp, reduction = reduction),
        "concave_exp": ConcaveExpLoss(alpha= 1, beta =1 ),
        "concave_log": ConcaveLogLoss(alpha= 1, beta =1, gamma = 1),
        "concave_loss": ConcaveLoss(alpha= 1, beta =1, gamma = 1, tau = 0.5),
        "mixup_py": MixupPy(alpha= 0.05, beta =0.05, gamma = 1, tau = 0.5,device = args.device),
        "concave_exp_one": ConcaveExpOneLoss(alpha= 1, beta =1, gamma = 1, tau = 0.5),
        "concave_taylor":ConcaveTaylor(alpha = args.alpha, beta = args.temp, gamma = args.gamma, tau = args.tau),
    }
    if args.dataset.lower() == "cifar10":
        return CIFAR10_CONFIG[loss_type]
    elif args.dataset.lower() == "cifar100":
        return CIFAR100_CONFIG[loss_type]
    elif args.dataset.lower() == "fashionmnist":
        return FashionMNIST[loss_type]
    elif args.dataset.lower() == "webvision":
        return WEB_CONFIG[loss_type]
    elif args.dataset.lower() == "imagenet":
        return Imagenet_CONFIG[loss_type]
    elif args.dataset.lower() == "tinyimagenet":
        return TinyImagenet_CONFIG[loss_type]
    elif args.dataset.lower() == "imagenet_r":
        return Imagenet_R_CONFIG[loss_type]
    else:
        raise ValueError("Dataset not implemented yet :P")


def get_loss_adj(loss_type, device, args, train_loader = None, num_classes = 10, reduction = "mean"):
    CONFIG = {
        "ce": nn.CrossEntropyLoss(),
        "ce_ls": nn.CrossEntropyLoss(label_smoothing= args.temp, reduction = reduction),
        "ereg": EntropyRegularizedLoss(alpha = args.alpha, reduction = reduction),
        "focal": FocalLoss(gamma=args.gamma,beta=args.temp ,reduction = reduction),
        "mae": MAELoss(num_classes=num_classes),
        "gce": GCE(device, alpha = args.alpha, q=args.temp, k=num_classes),
        "sce": SCE(alpha=args.alpha, beta=args.temp, num_classes=num_classes),
        "ldam": LDAMLoss(device=device),
        "logit_clip": LogitClipLoss(device, threshold=args.temp),
        "logit_norm": LogitNormLoss(device, args.temp, p=args.lp),
        "normreg": NormRegLoss(device, args.temp, p=args.lp),
        "tlnorm": TLogitNormLoss(device, args.temp, m=num_classes),
        "cnorm": CNormLoss(device, args.temp),
        # "nlnl": NLNL(device, train_loader=train_loader, num_classes=num_classes),
        "nce": NCELoss(num_classes=num_classes),
        "ael": AExpLoss(num_classes=num_classes, a=2.5),
        "aul": AUELoss(num_classes=num_classes, a=5.5, q=3),
        "phuber": PHuberCE(tau=args.tau),
        "taylor": TaylorCE(device=device, series=args.alpha),
        "cores": CoresLoss(device=device),
        "ncemae": NCEandMAE(alpha=args.alpha, beta=args.temp, num_classes=num_classes),
        "ngcemae": NGCEandMAE(alpha=args.alpha, beta=args.temp, num_classes=num_classes),
        "ncerce": NGCEandMAE(alpha=args.alpha, beta=args.temp, num_classes=num_classes),
        "nceagce": NCEandAGCE(alpha=args.alpha, beta=args.temp, a=1.8, q=3.0, num_classes=num_classes),
        "flood": FloodLoss(device=device, t = args.temp, reduction = reduction),
        "concave_exp": ConcaveExpLoss(alpha= args.alpha, beta =args.temp, gamma = args.gamma),
        "concave_log": ConcaveLogLoss(alpha= args.alpha, beta =args.temp, gamma = args.gamma), 
        "concave_loss": ConcaveLoss(alpha= args.alpha, beta =args.temp, gamma = args.gamma, tau = args.tau),
        "mixup_py": MixupPy(alpha= args.alpha, beta =args.temp, gamma = args.gamma, tau = args.tau, device = args.device),
        "gce_mixup": GCE(device, alpha = args.alpha, q=args.temp, k=num_classes, mixup_beta = args.tau, mixup= True),
        "concave_exp_one": ConcaveExpOneLoss(alpha = args.alpha, beta = args.temp, gamma = args.gamma, tau = args.tau),
        "concave_qua":ConcaveQ(alpha = args.alpha, beta = args.temp, gamma = args.gamma, tau = args.tau),
        "concave_taylor":ConcaveTaylor(alpha = args.alpha, beta = args.temp, gamma = args.gamma, tau = args.tau),
        "concave_taylor_n":ConcaveTaylorN(alpha = args.alpha, beta = args.temp, gamma = args.gamma, tau = args.tau),
        "variance_penalty": VariancePenalty(alpha = args.alpha, beta = args.temp, gamma = args.gamma, tau = args.tau),
        "focal_exp":FocalExp(alpha = args.alpha, beta = args.temp, gamma = args.gamma, tau = args.tau),
        "csce":CustomSoftmaxCrossEntropyLoss(C = args.alpha,reduction = reduction),
    }

    return CONFIG[loss_type]



class RelaxLoss(nn.Module):
    def __init__(self, alpha, epochs, num_classes):
        super(RelaxLoss, self).__init__()
        self.alpha = alpha
        self.epochs = epochs
        self.softmax =nn.Softmax(dim=1)
        self.crossentropy_soft = partial(CrossEntropy_soft, reduction='none')
        self.num_classes = num_classes
    def forward(self, logits, label):
        loss_ce_full = nn.CrossEntropyLoss(reduction='none')(logits, label)
        loss_ce = torch.mean(loss_ce_full)
        if self.epochs %2 ==0:
            loss = (loss_ce - self.alpha).abs()
        else: 
            if loss_ce > self.alpha:  # normal gradient descent
                loss = loss_ce
            else:
                pred = torch.argmax(logits, dim=1)
                correct = torch.eq(pred, label).float()
                confidence_target = self.softmax(logits)[torch.arange(label.size(0)), label]
                confidence_target = torch.clamp(confidence_target, min=0., max=1)
                confidence_else = (1.0 - confidence_target) / (self.num_classes - 1)
                onehot = one_hot_embedding(label, num_classes=self.num_classes)
                soft_targets = onehot * confidence_target.unsqueeze(-1).repeat(1, self.num_classes) \
                                + (1 - onehot) * confidence_else.unsqueeze(-1).repeat(1, self.num_classes)
                loss = (1 - correct) * self.crossentropy_soft(logits, soft_targets) - 1. * loss_ce_full
                loss = torch.mean(loss)
        return loss




class EntropyRegularizedLoss(nn.Module):
    def __init__(self, alpha, reduction='mean'):
        """
        初始化熵正则化损失函数
        :param alpha: 熵正则化的权重系数 (float)
        :param reduction: 损失的减少方式，可选值为 'mean', 'sum', 或 'none'
        """
        super(EntropyRegularizedLoss, self).__init__()
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, outputs, targets):
        """
        前向传播计算带熵正则化的损失
        :param outputs: 模型的预测输出 (tensor)
        :param targets: 真实标签 (tensor)
        :return: 带熵正则化的总损失 (tensor)
        """
        # 计算交叉熵损失
        cross_entropy_loss = F.cross_entropy(outputs, targets, reduction=self.reduction)

        # 计算输出概率分布的熵
        prob_dist = F.softmax(outputs, dim=1)
        entropy = -torch.sum(prob_dist * torch.log(prob_dist + 1e-9), dim=1)

        if self.reduction == 'mean':
            entropy = entropy.mean()
        elif self.reduction == 'sum':
            entropy = entropy.sum()

        # 计算带熵正则化的总损失
        return cross_entropy_loss - self.alpha * entropy



class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels, t=1.0):
        softmaxes = F.softmax(logits/t, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

class LDAMLoss(nn.Module):

    def __init__(self, device, max_m=0.5, s=30):
        super(LDAMLoss, self).__init__()
        cls_num_list = [5000] * 10
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list, device=device)
        self.device = device
        self.m_list = m_list
        assert s > 0
        self.s = s

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.type(torch.cuda.FloatTensor).to(self.device)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m

        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s * output, target)





class RingLoss(nn.Module):
    """
    Refer to paper
    Ring loss: Convex Feature Normalization for Face Recognition
    """

    def __init__(self, type='L2', loss_weight=1.0):
        super(RingLoss, self).__init__()
        self.radius = nn.Parameter(torch.Tensor(1))
        self.radius.data.fill_(-1)
        self.loss_weight = loss_weight
        self.type = type

    def forward(self, x):
        x = x.pow(2).sum(dim=1).pow(0.5)
        if self.radius.data[0] < 0:  # Initialize the radius with the mean feature norm of first iteration
            self.radius.data.fill_(x.mean().data[0])
        if self.type == 'L1':  # Smooth L1 Loss
            loss1 = F.smooth_l1_loss(x, self.radius.expand_as(x)).mul_(self.loss_weight)
            loss2 = F.smooth_l1_loss(self.radius.expand_as(x), x).mul_(self.loss_weight)
            ringloss = loss1 + loss2
        elif self.type == 'auto':  # Divide the L2 Loss by the feature's own norm
            diff = x.sub(self.radius.expand_as(x)) / (x.mean().detach().clamp(min=0.5))
            diff_sq = torch.pow(torch.abs(diff), 2).mean()
            ringloss = diff_sq.mul_(self.loss_weight)
        else:  # L2 Loss, if not specified
            diff = x.sub(self.radius.expand_as(x))
            diff_sq = torch.pow(torch.abs(diff), 2).mean()
            ringloss = diff_sq.mul_(self.loss_weight)
        return ringloss


class COCOLoss(nn.Module):
    """
        Refer to paper:
        Yu Liu, Hongyang Li, Xiaogang Wang
        Rethinking Feature Discrimination and Polymerization for Large scale recognition. NIPS workshop 2017
        re-implement by yirong mao
        2018 07/02
        """

    def __init__(self, num_classes, feat_dim, alpha=6.25):
        super(COCOLoss, self).__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.alpha = alpha
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, feat):
        norms = torch.norm(feat, p=2, dim=-1, keepdim=True)
        nfeat = torch.div(feat, norms)
        snfeat = self.alpha * nfeat

        norms_c = torch.norm(self.centers, p=2, dim=-1, keepdim=True)
        ncenters = torch.div(self.centers, norms_c)

        logits = torch.matmul(snfeat, torch.transpose(ncenters, 0, 1))

        return logits


class LMCL_loss(nn.Module):
    """
        Refer to paper:
        Hao Wang, Yitong Wang, Zheng Zhou, Xing Ji, Dihong Gong, Jingchao Zhou,Zhifeng Li, and Wei Liu
        CosFace: Large Margin Cosine Loss for Deep Face Recognition. CVPR2018
        re-implement by yirong mao
        2018 07/02
        """

    def __init__(self, num_classes, feat_dim, s=7.00, m=0.2):
        super(LMCL_loss, self).__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, feat, label):
        batch_size = feat.shape[0]
        norms = torch.norm(feat, p=2, dim=-1, keepdim=True)
        nfeat = torch.div(feat, norms)

        norms_c = torch.norm(self.centers, p=2, dim=-1, keepdim=True)
        ncenters = torch.div(self.centers, norms_c)
        logits = torch.matmul(nfeat, torch.transpose(ncenters, 0, 1))

        # y_onehot = torch.FloatTensor(batch_size, self.num_classes)
        # y_onehot.zero_()
        # y_onehot = Variable(y_onehot).cuda()
        # y_onehot.scatter_(1, torch.unsqueeze(label, dim=-1), self.m)

        y_onehot = F.one_hot(label, self.num_classes) * self.m
        margin_logits = self.s * (logits - y_onehot)

        return logits, margin_logits


class LGMLoss(nn.Module):
    """
    Refer to paper:
    Weitao Wan, Yuanyi Zhong,Tianpeng Li, Jiansheng Chen
    Rethinking Feature Distribution for Loss Functions in Image Classification. CVPR 2018
    re-implement by yirong mao
    2018 07/02
    """

    def __init__(self, num_classes, feat_dim, alpha):
        super(LGMLoss, self).__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.alpha = alpha

        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        self.log_covs = nn.Parameter(torch.zeros(num_classes, feat_dim))

    def forward(self, feat, label):
        batch_size = feat.shape[0]
        log_covs = torch.unsqueeze(self.log_covs, dim=0)

        covs = torch.exp(log_covs)  # 1*c*d
        tcovs = covs.repeat(batch_size, 1, 1)  # n*c*d
        diff = torch.unsqueeze(feat, dim=1) - torch.unsqueeze(self.centers, dim=0)
        wdiff = torch.div(diff, tcovs)
        diff = torch.mul(diff, wdiff)
        dist = torch.sum(diff, dim=-1)  # eq.(18)

        y_onehot = torch.FloatTensor(batch_size, self.num_classes)
        y_onehot.zero_()
        y_onehot = Variable(y_onehot).cuda()
        y_onehot.scatter_(1, torch.unsqueeze(label, dim=-1), self.alpha)
        y_onehot = y_onehot + 1.0
        margin_dist = torch.mul(dist, y_onehot)

        slog_covs = torch.sum(log_covs, dim=-1)  # 1*c
        tslog_covs = slog_covs.repeat(batch_size, 1)
        margin_logits = -0.5 * (tslog_covs + margin_dist)  # eq.(17)
        logits = -0.5 * (tslog_covs + dist)

        cdiff = feat - torch.index_select(self.centers, dim=0, index=label.long())
        cdist = cdiff.pow(2).sum(1).sum(0) / 2.0

        slog_covs = torch.squeeze(slog_covs)
        reg = 0.5 * torch.sum(torch.index_select(slog_covs, dim=0, index=label.long()))
        likelihood = (1.0 / batch_size) * (cdist + reg)

        return logits, margin_logits, likelihood


class LGMLoss_v0(nn.Module):
    """
    LGMLoss whose covariance is fixed as Identity matrix
    """

    def __init__(self, num_classes, feat_dim, alpha):
        super(LGMLoss_v0, self).__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.alpha = alpha

        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, feat, label):
        batch_size = feat.shape[0]

        diff = torch.unsqueeze(feat, dim=1) - torch.unsqueeze(self.centers, dim=0)
        diff = torch.mul(diff, diff)
        dist = torch.sum(diff, dim=-1)

        y_onehot = torch.FloatTensor(batch_size, self.num_classes)
        y_onehot.zero_()
        y_onehot = Variable(y_onehot).cuda()
        y_onehot.scatter_(1, torch.unsqueeze(label, dim=-1), self.alpha)
        y_onehot = y_onehot + 1.0
        margin_dist = torch.mul(dist, y_onehot)
        margin_logits = -0.5 * margin_dist
        logits = -0.5 * dist

        cdiff = feat - torch.index_select(self.centers, dim=0, index=label.long())
        likelihood = (1.0 / batch_size) * cdiff.pow(2).sum(1).sum(0) / 2.0
        return logits, margin_logits, likelihood


class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        self.centerlossfunction = CenterlossFunction.apply

    def forward(self, y, feat):
        # To squeeze the Tenosr
        batch_size = feat.size(0)
        feat = feat.view(batch_size, 1, 1, -1).squeeze()
        # To check the dim of centers and features
        if feat.size(1) != self.feat_dim:
            raise ValueError(
                "Center's dim: {0} should be equal to input feature's dim: {1}".format(self.feat_dim, feat.size(1)))
        return self.centerlossfunction(feat, y, self.centers)


class CenterlossFunction(Function):

    @staticmethod
    def forward(ctx, feature, label, centers):
        ctx.save_for_backward(feature, label, centers)
        centers_pred = centers.index_select(0, label.long())
        return (feature - centers_pred).pow(2).sum(1).sum(0) / 2.0

    @staticmethod
    def backward(ctx, grad_output):
        feature, label, centers = ctx.saved_variables
        grad_feature = feature - centers.index_select(0, label.long())  # Eq. 3

        # init every iteration
        counts = torch.ones(centers.size(0))
        grad_centers = torch.zeros(centers.size())
        if feature.is_cuda:
            counts = counts.cuda()
            grad_centers = grad_centers.cuda()
        # print counts, grad_centers

        # Eq. 4 || need optimization !! To be vectorized, but how?
        for i in range(feature.size(0)):
            j = int(label[i].data[0])
            counts[j] += 1
            grad_centers[j] += (centers.data[j] - feature.data[i])
        # print counts
        grad_centers = Variable(grad_centers / counts.view(-1, 1))

        return grad_feature * grad_output, None, grad_centers



class LogitNormLoss(nn.Module):

    def __init__(self, device, t=1.0, p=2, reduction = "mean"):
        super(LogitNormLoss, self).__init__()
        self.device = device
        self.t = t
        self.p = p
        self.reduction = reduction
    def forward(self, x, target):
        norms = torch.norm(x, p=self.p, dim=-1, keepdim=True) + 1e-7
        logit_norm = torch.div(x, norms) / self.t
        return F.cross_entropy(logit_norm, target, reduction = self.reduction)



class FloodingLoss(nn.Module):
    def __init__(self, device, b=0.1, reduction='mean'):
        super(FloodingLoss, self).__init__()
        self.device = device
        self.b = b
        self.reduction = reduction

    def forward(self, x, target):
        ce_loss = F.cross_entropy(x, target)

        if self.reduction == 'mean':
            return (ce_loss - self.b).abs().mean() + self.b
        elif self.reduction == 'sum':
            return (ce_loss - self.b).abs().sum() + self.b
        elif self.reduction == 'none':
            return (ce_loss - self.b).abs() + self.b
        else:
            raise ValueError("Invalid reduction option. Use 'mean', 'sum', or 'none'.")

class NormRegLoss(nn.Module):

    def __init__(self, device, t=1.0, p=2):
        super(NormRegLoss, self).__init__()
        self.device = device
        self.t = t
        self.p = p

    def forward(self, x, target):
        norms = torch.norm(x, p=self.p, dim=-1)
        return F.cross_entropy(x, target) + self.t * norms.mean()

class CNormLoss(nn.Module):

    def __init__(self, device, t=1.0, p=2):
        super(CNormLoss, self).__init__()
        self.device = device
        self.t = t
        self.p = p
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x, target):
        norms = torch.norm(x, p=self.p, dim=-1, keepdim=True) + 1e-7
        norms_condition = torch.norm(x, p=self.p, dim=-1)
        x_norm = torch.div(x, norms) / self.t

        p = self.softmax(x)
        p = p[torch.arange(p.shape[0]), target]

        p_norm = self.softmax(x_norm)
        p_norm = p_norm[torch.arange(p_norm.shape[0]), target]

        loss = torch.empty_like(p)
        clip = norms_condition > self.t

        loss[clip] = -torch.log(p_norm[clip])
        loss[~clip] = -torch.log(p[~clip])

        return torch.mean(loss)

class TLogitNormLoss(nn.Module):

    def __init__(self, device, t=1.0, m=10):
        super(TLogitNormLoss, self).__init__()
        self.device = device
        self.t = t
        self.m = m
        # Probability threshold for the clipping
        self.prob_thresh = 1 / self.m
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, target):
        norms = torch.norm(x, p=2, dim=-1, keepdim=True) + 1e-7
        x_norm = torch.div(x, norms) / self.t

        p = self.softmax(x)
        p = p[torch.arange(p.shape[0]), target]

        p_norm = self.softmax(x_norm)
        p_norm = p_norm[torch.arange(p_norm.shape[0]), target]

        loss = torch.empty_like(p)
        clip = p <= self.prob_thresh

        loss[clip] = -torch.log(p_norm[clip])
        loss[~clip] = -torch.log(p[~clip])

        return torch.mean(loss)


class logNegLoss(nn.Module):

    def __init__(self, device, t=1.0):
        super(logNegLoss, self).__init__()
        self.device = device
        self.t = t
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, target):
        p = self.softmax(x)
        p = p[torch.arange(p.shape[0]), target]
        # loss = torch.log(p**(-self.t)*(1-p)**(1-self.t))
        loss = -self.t * torch.log(p) + (1 - self.t) * torch.log(1 - p)
        return torch.mean(loss)

class FloodLoss(nn.Module):
    def __init__(self, device, t=0.01, reduction='mean'):
        super(FloodLoss, self).__init__()
        self.device = device
        self.t = t
        self.reduction = reduction

    def forward(self, x, target):
        losses = F.cross_entropy(x, target, reduction="none")
        losses = (losses - self.t).abs() + self.t

        if self.reduction == 'mean':
            return losses.mean()
        elif self.reduction == 'sum':
            return losses.sum()
        elif self.reduction == 'none':
            return losses
        else:
            raise ValueError("Invalid reduction option. Use 'mean', 'sum', or 'none'.")

class DoubleSoftLoss(nn.Module):

    def __init__(self, device, t=1.0):
        super(DoubleSoftLoss, self).__init__()
        self.device = device
        self.t = t

    def forward(self, x, target):
        logit_norm = F.softmax(x, dim=-1)
        return F.cross_entropy(logit_norm/self.t, target)



class LogitClipLoss(nn.Module):
    def __init__(self, device, threshold=1.0, reduction='mean'):
        super(LogitClipLoss, self).__init__()
        self.device = device
        self.min = -threshold
        self.max = threshold
        self.reduction = reduction
    def forward(self, x, target):
        x = torch.clamp(x, self.min, self.max)
        if self.reduction == 'mean':
            return F.cross_entropy(x, target, reduction='mean')
        elif self.reduction == 'sum':
            return F.cross_entropy(x, target, reduction='sum')
        elif self.reduction == 'none':
            return F.cross_entropy(x, target, reduction='none')
        else:
            raise ValueError("Invalid reduction option. Use 'mean', 'sum', or 'none'.")
class CustomSoftmaxCrossEntropyLoss(nn.Module):
    def __init__(self, C=1.0, reduction='mean'):
        super(CustomSoftmaxCrossEntropyLoss, self).__init__()
        self.C = C
        self.reduction = reduction
    
    def forward(self, logits, targets):
        # Adjust logits to improve numerical stability
        max_logits = logits.max(dim=1, keepdim=True)[0]
        stabilized_logits = logits - max_logits
        
        # Manually compute the modified softmax with C added to the denominator
        exp_logits = torch.exp(stabilized_logits)
        modified_softmax_denominator = exp_logits.sum(dim=1, keepdim=True) + torch.exp(torch.tensor(self.C) - max_logits)
        log_probs = stabilized_logits - modified_softmax_denominator.log()
        
        # Compute the negative log likelihood loss
        return F.nll_loss(log_probs, targets, reduction=self.reduction)


class LogitClip(nn.Module):
    def __init__(self, device, temp=1.0, lp=2, reduction='mean'):
        super(LogitClip, self).__init__()
        self.lp = lp
        self.threshold = temp
        self.reduction =reduction
    def forward(self, x, target):
        norms = torch.norm(x, p=self.lp, dim=-1, keepdim=True) + 1e-7
        logits_norm = torch.div(x, norms) * self.threshold
        clip = (norms > self.threshold).expand(-1, x.shape[-1])
        logits_final = torch.where(clip, logits_norm, x)
        
        
        if self.reduction == 'mean':
            return F.cross_entropy(logits_final, target, reduction='mean')
        elif self.reduction == 'sum':
            return F.cross_entropy(logits_final, target, reduction='sum')
        elif self.reduction == 'none':
            return F.cross_entropy(logits_final, target, reduction='none')
        else:
            raise ValueError("Invalid reduction option. Use 'mean', 'sum', or 'none'.")


class LogitClipingLoss(nn.Module):
    def __init__(self, device, tau=1.0, p=2, reduction='mean'):
        super(LogitClipingLoss, self).__init__()
        self.tau = tau
        self.delta = tau
        self.device = device
        self.p = p
        self.reduction = reduction

    def forward(self, x, target):
        # [N, C]
        norms = torch.norm(x, p=self.p, dim=-1, keepdim=True) + 1e-7  # [N,]      
        identify = (norms >= self.tau)
        logit_clip = self.delta * torch.div(x, norms) * identify + x * (~identify)

        if self.reduction == 'mean':
            return F.cross_entropy(logit_clip, target, reduction='mean')
        elif self.reduction == 'sum':
            return F.cross_entropy(logit_clip, target, reduction='sum')
        elif self.reduction == 'none':
            return F.cross_entropy(logit_clip, target, reduction='none')
        else:
            raise ValueError("Invalid reduction option. Use 'mean', 'sum', or 'none'.")


class LogitTempLoss(nn.Module):

    def __init__(self, device, t=1.0):
        super(LogitTempLoss, self).__init__()
        self.device = device
        self.t = t

    def forward(self, x, target):
        return F.cross_entropy(x / self.t, target)



class LogitPenLoss(nn.Module):

    def __init__(self, device, beta=0):
        super(LogitPenLoss, self).__init__()
        self.device = device
        self.beta = beta

    def forward(self, x, target):
        norms = torch.norm(x, p=2)

        return F.cross_entropy(x, target) + self.beta * norms

class SigmoidCE(nn.Module):

    def __init__(self, device, num_class):
        super(SigmoidCE, self).__init__()
        self.device = device
        self.num_class = num_class
    def forward(self, x, target):
        label_one_hot = F.one_hot(target, self.num_class).float()
        return F.binary_cross_entropy(x * torch.sigmoid(x), label_one_hot)

class SquaredLoss(nn.Module):

    def __init__(self, device, num_class=10, k=9, m=60):
        super(SquaredLoss, self).__init__()
        self.device = device
        self.num_class = num_class
        self.k = k
        self.m = m

    def forward(self, x, target):
        label_one_hot = F.one_hot(target, self.num_class)
        self.k * label_one_hot * x
        return F.binary_cross_entropy(x * torch.sigmoid(x), target)


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

class PHuberCE(nn.Module):
    def __init__(self, tau=10):
        super(PHuberCE, self).__init__()
        
        try:
            if tau == 0:
                raise ValueError("tau should not be 0.")
            self.tau = tau
        except ValueError as e:
            #print(e)
            self.tau = 1e-10  # 
        

        # Probability threshold for the clipping
        self.prob_thresh = 1 / self.tau
        # Negative of the Fenchel conjugate of base loss at tau
        self.boundary_term = math.log(self.tau) + 1

        self.softmax = nn.Softmax(dim=1)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = self.softmax(input)
        p = p[torch.arange(p.shape[0]), target]

        loss = torch.empty_like(p)
        clip = p <= self.prob_thresh
        loss[clip] = -self.tau * p[clip] + self.boundary_term
        loss[~clip] = -torch.log(p[~clip])

        return torch.mean(loss)

def ce_concave_log_loss(input_values, alpha, beta, gamma =1, reduction="mean"):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    #gamma = torch.clamp(gamma , min=1e-7)
    loss = alpha * input_values - beta *torch.log(1-p+gamma)

    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        raise ValueError("Invalid reduction option. Use 'none', 'mean', or 'sum'.")



def mixup_py(softmax_input, alpha, beta,gamma =1.0, tau=1.0, reduction="mean", use_cuda=True, device ="cuda:0"):
    if beta > 0.:
        lam = np.random.beta(beta, beta)
    else:
        lam = 1.
    batch_size = softmax_input.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).to(device)
    else:
        index = torch.randperm(batch_size)
    mixed_p = lam * softmax_input + (1 - lam) * softmax_input[index]
    return mixed_p

def ce_mixup_py_loss(input_values, alpha, beta, gamma =1.0, tau=1.0, reduction="mean", use_cuda=True, device ="cuda:0"):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    if beta > 0.:
        lam = np.random.beta(beta, beta)
    else:
        lam = 1.
    batch_size = p.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).to(device)
    else:
        index = torch.randperm(batch_size)
    mixed_p = lam * p + (1 - lam) * p[index]
    loss = - alpha *torch.log(mixed_p)
    
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        raise ValueError("Invalid reduction option. Use 'none', 'mean', or 'sum'.")


class MixupPy(nn.Module):
    def __init__(self, alpha = 1, beta = 1, gamma=1.0,tau=0.5,reduction='mean',device ="cuda:0" ):
        super(MixupPy, self).__init__()
        # assert tau >= 1e-7
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.tau = tau
        self.reduction = reduction
        self.device = device
    def forward(self, input, target):
        return ce_mixup_py_loss(F.cross_entropy(input, target, reduction="none"), self.alpha, self.beta, self.gamma, self.tau, reduction = self.reduction,device =self.device)


def ce_concave_loss(input_values, alpha, beta, gamma =1.0, tau=1.0, reduction="mean"):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    
    loss = alpha * input_values - beta *(1+tau-p)**gamma

    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        raise ValueError("Invalid reduction option. Use 'none', 'mean', or 'sum'.")

class ConcaveLoss(nn.Module):
    def __init__(self, alpha = 1, beta = 1, gamma=1.0,tau=0.5,reduction='mean'):
        super(ConcaveLoss, self).__init__()
        # assert tau >= 1e-7
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.tau = tau
        self.reduction = reduction
    def forward(self, input, target):
        return ce_concave_loss(F.cross_entropy(input, target, reduction="none"), self.alpha, self.beta, self.gamma, self.tau, reduction = self.reduction)
def ce_concave_quadratic_loss(input_values, alpha, gamma =1, reduction="mean"):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    
    loss = (1-alpha) * input_values -  alpha*torch.pow(p, 2)

    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        raise ValueError("Invalid reduction option. Use 'none', 'mean', or 'sum'.")

    
    
def ce_concave_exp_loss(input_values, alpha, beta, gamma =1, reduction="mean"):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    
    loss = alpha * input_values - beta *torch.exp(gamma*p)

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


    
    
class ConcaveExpLoss(nn.Module):
    def __init__(self, alpha = 1, beta = 1, gamma=1.0,reduction='mean'):
        super(ConcaveExpLoss, self).__init__()
        assert gamma >= 1e-7
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction
    def forward(self, input, target):
        return ce_concave_exp_loss(F.cross_entropy(input, target, reduction="none"), self.alpha, self.beta, self.gamma, reduction = self.reduction)

class ConcaveExpOneLoss(nn.Module):
    def __init__(self, alpha = 1, beta = 1, gamma=1.0, tau =1,reduction='mean'):
        super(ConcaveExpOneLoss, self).__init__()
        assert gamma >= 1e-7
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction
    def forward(self, input, target):
        return self.beta*ce_concave_exp_loss(F.cross_entropy(input, target, reduction="none"), self.alpha, (1-self.alpha), self.gamma, reduction = self.reduction)

class ConcaveQ(nn.Module):
    def __init__(self, alpha = 1, beta = 1, gamma=1.0, tau =1,reduction='mean'):
        super(ConcaveQ, self).__init__()
        assert gamma >= 1e-7
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction
    def forward(self, input, target):
        return self.beta*ce_concave_quadratic_loss(F.cross_entropy(input, target, reduction="none"),self.alpha)
    
class ConcaveTaylor(nn.Module):
    def __init__(self, alpha = 1, beta = 1, gamma=2, tau =1,reduction='mean'):
        super(ConcaveTaylor, self).__init__()
        #assert gamma >= 1e-7
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


def taylor_exp(input_values, alpha, beta, gamma =2, reduction="mean"):
    """Computes the focal loss"""
    gamma = int(gamma)
    p = torch.exp(-input_values)
    loss = alpha*input_values
    for n in range(gamma):
        loss = loss - (1-alpha)*(torch.pow(p , n+1))/math.factorial(n+1)

    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        raise ValueError("Invalid reduction option. Use 'none', 'mean', or 'sum'.")

class ConcaveTaylorN(nn.Module):
    def __init__(self, alpha = 1, beta = 1, gamma=2, tau =1,reduction='mean'):
        super(ConcaveTaylorN, self).__init__()
        #assert gamma >= 1e-7
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction
    def forward(self, input, target):
        loss = taylor_exp(F.cross_entropy(input, target, reduction="none"),self.alpha, self.beta, self.gamma) 
        return self.beta*loss


class VariancePenalty(nn.Module):
    def __init__(self, alpha = 1, beta = 1, gamma=1.0, tau =1,reduction='mean'):
        super(VariancePenalty, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction
    def forward(self, input, target):
        losses  = F.cross_entropy(input, target, reduction="none")
        penalty = torch.var(losses)
        loss = losses+self.alpha *penalty
        if self.reduction == "none":
            return loss
        elif self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            raise ValueError("Invalid reduction option. Use 'none', 'mean', or 'sum'.")
            
class FocalExp(nn.Module):
    def __init__(self, alpha = 1, beta = 1, gamma=1.0, tau =1,reduction='mean'):
        super(FocalExp, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.tau =tau
        self.reduction = reduction
    def forward(self, input, target):
        ce = F.cross_entropy(input, target, reduction="none")
        losses = focal_loss(ce, gamma = self.tau, reduction="none")
        
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





class ConcaveLogLoss(nn.Module):
    def __init__(self, alpha = 1, beta = 1, gamma=1.0,reduction='mean'):
        super(ConcaveLogLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction
    def forward(self, input, target):
        return ce_concave_log_loss(F.cross_entropy(input, target, reduction="none"), self.alpha, self.beta, self.gamma, reduction = self.reduction)






def loss_sce(y, labels_one_hot, alpha, beta, reduction='mean'):
    pred = F.softmax(y, dim=1)
    pred = torch.clamp(pred, min=1e-7, max=1.0)
    label_one_hot = torch.clamp(labels_one_hot, min=1e-4, max=1.0)
    
    ce = (-1 * torch.sum(label_one_hot * torch.log(pred), dim=1))
    rce = (-1 * torch.sum(pred * torch.log(label_one_hot), dim=1))

    if reduction == 'mean':
        ce = ce.mean()
        rce = rce.mean()
    elif reduction == 'sum':
        ce = ce.sum()
        rce = rce.sum()
    elif reduction == 'none':
        pass
    else:
        raise ValueError("Invalid reduction option. Use 'mean', 'sum', or 'none'.")

    return alpha * ce + beta * rce


    
    
class SCE(nn.Module):
    def __init__(self, alpha=0.5, beta=1.0, num_classes=10, reduction='mean'):
        super(SCE, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.reduction = reduction
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        labels_one_hot = torch.zeros(target.shape[0], self.num_classes).to(input.device).scatter_(1,
                                                                                            target.unsqueeze(1), 1)

        return loss_sce(input, labels_one_hot, self.alpha, self.beta, reduction=self.reduction)





class GCE(nn.Module):
    def __init__(self, device, q=0.7, k=10, alpha=1, mixup_beta =1,reduction='mean',mixup = False):
        super(GCE, self).__init__()
        self.q = q
        self.k = k
        self.device = device
        self.reduction = reduction
        self.alpha = alpha
        self.mixup= mixup
        self.beta =mixup_beta
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        soft_max = nn.Softmax(dim=1)
        sm_outputs = soft_max(input)
        if self.mixup: 
            sm_outputs =mixup_py(sm_outputs, alpha=1, beta =self.beta,device =self.device)
        label_one_hot = nn.functional.one_hot(target, self.k).float().to(self.device)
        sm_out = torch.pow((sm_outputs * label_one_hot).sum(dim=1), self.q)
        target = torch.ones_like(target)
        loss_vec = self.alpha * (target - sm_out) / self.q

        if self.reduction == 'mean':
            return loss_vec.mean()
        elif self.reduction == 'sum':
            return loss_vec.sum()
        elif self.reduction == 'none':
            return loss_vec
        else:
            raise ValueError("Invalid reduction option. Use 'mean', 'sum', or 'none'.")


    




class AGCELoss(nn.Module):
    def __init__(self, num_classes=10, a=1, q=2, eps=1e-7, scale=1.):
        super(AGCELoss, self).__init__()
        self.a = a
        self.q = q
        self.num_classes = num_classes
        self.eps = eps
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        loss = ((self.a+1)**self.q - torch.pow(self.a + torch.sum(label_one_hot * pred, dim=1), self.q)) / self.q
        return loss.mean() * self.scale


class NGCE(torch.nn.Module):
    def __init__(self, num_classes, scale=1.0, q=0.7):
        super(NGCE, self).__init__()
        self.num_classes = num_classes
        self.q = q
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(pred.device)
        numerators = 1. - torch.pow(torch.sum(label_one_hot * pred, dim=1), self.q)
        denominators = self.num_classes - pred.pow(self.q).sum(dim=1)
        ngce = numerators / denominators
        return self.scale * ngce.mean()
# class MAE(nn.Module):
#     def __init__(self, device, temp=0):
#         super(MAE, self).__init__()
#         self.device = device
#         self.temp = temp
#
#     def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#         if self.temp != 0:
#             norms = torch.norm(input, p=2, dim=-1, keepdim=True) + 1e-7
#             input = torch.div(input, norms) / self.temp
#         labels_one_hot = torch.zeros(target.shape[0], input.shape[-1]).to(self.device).scatter_(1,
#                                                                                                  target.unsqueeze(
#                                                                                                      1), 1)
#         softmax = torch.nn.Softmax(dim=-1)
#         loss_l1 = torch.nn.L1Loss()
#         loss = loss_l1(softmax(input), labels_one_hot)
#         return loss




class TaylorCE(nn.Module):
    def __init__(self, device, series=2):
        super(TaylorCE, self).__init__()
        self.series = series
        self.device = device

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n = int(self.series)
        k = input.shape[1]
        soft_max = nn.Softmax(dim=1)
        sm_outputs = soft_max(input)
        label_one_hot = nn.functional.one_hot(target, k).float().to(self.device)
        final_outputs = (sm_outputs * label_one_hot).sum(dim=1)

        total_loss = 0
        for i in range(n):  # 0 to n-1
            total_loss += torch.pow(torch.tensor([-1.0]).to(self.device), i + 1) * torch.pow(final_outputs - 1, i + 1) * 1.0 / (
                    i + 1)  # \sum_i=0^n(x-1)^(i+1)*(-1)^(i+1)/(i+1)
        average_loss = total_loss.mean()
        return average_loss


class NLNL(nn.Module):
    def __init__(self, device, train_loader, num_classes=10, ln_neg=1):
        super(NLNL, self).__init__()
        self.num_classes = num_classes
        self.ln_neg = ln_neg
        weight = torch.FloatTensor(num_classes).zero_() + 1.
        if not hasattr(train_loader.dataset, 'targets'):
            weight = [1] * num_classes
            weight = torch.FloatTensor(weight)
        else:
            for i in range(num_classes):
                weight[i] = (torch.from_numpy(np.array(train_loader.dataset.targets)) == i).sum()
            weight = 1 / (weight / weight.max())

        self.device = device
        self.weight = weight.to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss(weight=self.weight)
        self.criterion_nll = torch.nn.NLLLoss()


    def forward(self, pred, labels):

        labels_neg = (labels.unsqueeze(-1).repeat(1, self.ln_neg)
                      + torch.LongTensor(len(labels), self.ln_neg).to(self.device).random_(1, self.num_classes)) % self.num_classes
        labels_neg = torch.autograd.Variable(labels_neg)

        assert labels_neg.max() <= self.num_classes-1
        assert labels_neg.min() >= 0
        assert (labels_neg != labels.unsqueeze(-1).repeat(1, self.ln_neg)).sum() == len(labels)*self.ln_neg

        s_neg = torch.log(torch.clamp(1. - F.softmax(pred, 1), min=1e-5, max=1.))
        s_neg *= self.weight[labels].unsqueeze(-1).expand(s_neg.size()).to(self.device)
        labels = labels * 0 - 100
        loss = self.criterion(pred, labels) * float((labels >= 0).sum())
        loss_neg = self.criterion_nll(s_neg.repeat(self.ln_neg, 1), labels_neg.t().contiguous().view(-1)) * float((labels_neg >= 0).sum())
        loss = ((loss+loss_neg) / (float((labels >= 0).sum())+float((labels_neg[:, 0] >= 0).sum())))
        return loss

class NCELoss(nn.Module):
    def __init__(self, num_classes, scale=1.0, reduction='mean'):
        super(NCELoss, self).__init__()
        self.num_classes = num_classes
        self.scale = scale
        self.reduction = reduction

    def forward(self, pred, labels):
        pred = F.log_softmax(pred, dim=1)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        loss = -1 * torch.sum(label_one_hot * pred, dim=1) / (-pred.sum(dim=1))

        if self.reduction == 'mean':
            return self.scale * loss.mean()
        elif self.reduction == 'sum':
            return self.scale * loss.sum()
        elif self.reduction == 'none':
            return self.scale * loss
        else:
            raise ValueError("Invalid reduction option. Use 'mean', 'sum', or 'none'.")

class RCELoss(nn.Module):
    def __init__(self, num_classes=10, scale=1.0, reduction='mean'):
        super(RCELoss, self).__init__()
        self.num_classes = num_classes
        self.scale = scale
        self.reduction = reduction

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        loss = (-1 * torch.sum(pred * torch.log(label_one_hot), dim=1))

        if self.reduction == 'mean':
            return self.scale * loss.mean()
        elif self.reduction == 'sum':
            return self.scale * loss.sum()
        elif self.reduction == 'none':
            return self.scale * loss
        else:
            raise ValueError("Invalid reduction option. Use 'mean', 'sum', or 'none'.")
    
class NCEandRCE(nn.Module):
    def __init__(self, alpha=1., beta=1., num_classes=10):
        super(NCEandRCE, self).__init__()
        self.num_classes = num_classes
        self.nce = NCELoss(num_classes=num_classes, scale=alpha)
        self.rce = RCELoss(num_classes=num_classes, scale=beta)

    def forward(self, pred, labels):
        return self.nce(pred, labels) + self.rce(pred, labels)

class NCEandAGCE(torch.nn.Module):
    def __init__(self, alpha=1., beta = 1., num_classes=10, a=3, q=1.5):
        super(NCEandAGCE, self).__init__()
        self.num_classes = num_classes
        self.nce = NCELoss(num_classes=num_classes, scale=alpha)
        self.agce = AGCELoss(num_classes=num_classes, a=a, q=q, scale=beta)

    def forward(self, pred, labels):
        return self.nce(pred, labels) + self.agce(pred, labels)
 

class SMAPELoss(nn.Module):
    def __init__(self, num_classes=10, scale=2, reduction='mean'):
        super(SMAPELoss, self).__init__()
        self.num_classes = num_classes
        self.scale = scale
        self.reduction = reduction

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        pt = torch.sum(label_one_hot * pred, dim=1)
        pt = torch.clamp(pt, min=1e-3, max=1.0)
        loss = self.scale*(1. - pt)+ (1-pt)/pt
        
        if self.reduction == 'mean':
            return self.scale * loss.mean()
        elif self.reduction == 'sum':
            return self.scale * loss.sum()
        elif self.reduction == 'none':
            return self.scale * loss
        else:
            raise ValueError("Invalid reduction option. Use 'mean', 'sum', or 'none'.")




class MAELoss(nn.Module):
    def __init__(self, num_classes=10, scale=2.0, reduction='mean'):
        super(MAELoss, self).__init__()
        self.num_classes = num_classes
        self.scale = scale
        self.reduction = reduction

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        loss = self.scale*(1. - torch.sum(label_one_hot * pred, dim=1))

        if self.reduction == 'mean':
            return self.scale * loss.mean()
        elif self.reduction == 'sum':
            return self.scale * loss.sum()
        elif self.reduction == 'none':
            return self.scale * loss
        else:
            raise ValueError("Invalid reduction option. Use 'mean', 'sum', or 'none'.")




class NCEandMAE(nn.Module):
    def __init__(self, alpha=1., beta=1., num_classes=10):
        super(NCEandMAE, self).__init__()
        self.num_classes = num_classes
        self.nce = NCELoss(num_classes=num_classes, scale=alpha)
        self.mae = MAELoss(num_classes=num_classes, scale=beta)

    def forward(self, pred, labels):
        return self.nce(pred, labels) + self.mae(pred, labels)

class NGCEandMAE(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes, q=0.7):
        super(NGCEandMAE, self).__init__()
        self.num_classes = num_classes
        self.ngce = NGCE(num_classes=num_classes, scale=alpha, q=q)
        self.mae = MAELoss(scale=beta, num_classes=num_classes)

    def forward(self, pred, labels):
        return self.ngce(pred, labels) + self.mae(pred, labels)


class AUELoss(nn.Module):
    def __init__(self, num_classes=10, a=5.5, q=3, scale=1.0):
        super(AUELoss, self).__init__()
        self.num_classes = num_classes
        self.a = a
        self.q = q
        self.eps = 1e-7
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        loss = (torch.pow(self.a - torch.sum(label_one_hot * pred, dim=1), self.q) - (self.a-1)**self.q)/ self.q
        return loss.mean() * self.scale

class AExpLoss(torch.nn.Module):
    def __init__(self, num_classes=10, a=2.5, scale=1.0):
        super(AExpLoss, self).__init__()
        self.num_classes = num_classes
        self.a = a
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        loss = torch.exp(-torch.sum(label_one_hot * pred, dim=1) / self.a)
        return loss.mean() * self.scale



class CoresLoss(torch.nn.Module):
    def __init__(self, device):
        super(CoresLoss, self).__init__()
        self.device = device

    def forward(self, pred, labels, epoch):
        beta = f_beta(epoch)
        loss = F.cross_entropy(pred, labels, reduce=False)
        loss_numpy = loss.data.cpu().numpy()
        num_batch = len(loss_numpy)
        loss_v = np.zeros(num_batch)
        loss_ = -torch.log(F.softmax(pred) + 1e-8)
        # sel metric
        loss_sel = loss - torch.mean(loss_, 1)
        loss = loss - beta * torch.mean(loss_, 1)

        loss_div_numpy = loss_sel.data.cpu().numpy()

        for i in range(len(loss_numpy)):
            if epoch <= 60:
                loss_v[i] = 1.0
            elif loss_div_numpy[i] <= 0:
                loss_v[i] = 1.0
        loss_v = loss_v.astype(np.float32)
        loss_v_var = Variable(torch.from_numpy(loss_v)).to(self.device)
        loss_ = loss_v_var * loss
        if sum(loss_v) == 0.0:
            return torch.mean(loss_) / 100000000
        else:
            return torch.sum(loss_) / sum(loss_v)
# def loss_cores(epoch, y, t, class_list, ind, noise_or_not, loss_all, loss_div_all, noise_prior=None):

def f_beta(epoch):
    beta1 = np.linspace(0.0, 0.0, num=20)
    beta2 = np.linspace(0.0, 2, num=60)
    beta3 = np.linspace(2, 2, num=120)

    beta = np.concatenate((beta1, beta2, beta3), axis=0)
    return beta[epoch]