
from defenses.membership_inference.Normal import TrainTargetNormal
import torch
import numpy as np
import os
import time
from runx.logx import logx
import torch.nn.functional as F
import torch.nn as nn

class LogitNormLoss(nn.Module):

    def __init__(self, device, t=1.0):
        super(LogitNormLoss, self).__init__()
        self.device = device
        self.t = t

    def forward(self, x, target):
        norms = torch.norm(x, p=2, dim=-1, keepdim=True) + 1e-7
        logit_norm = torch.div(x, norms) / self.t
        return F.cross_entropy(logit_norm, target)

# ,device,num_class, epochs, learning_rate, momentum, weight_decay, smooth_eps, log_path

class TrainTargetLogitsNorm(TrainTargetNormal):
    def __init__(self,model, device="cuda:0", num_class=10, epochs=100, learning_rate=0.01, momentum=0.9, weight_decay=5e-4, smooth_eps=0.8, log_path="./", temperature=1):
        super(TrainTargetNormal,self).__init__(device,num_class, epochs, learning_rate, momentum, weight_decay, smooth_eps, log_path)
        self.device = device
        self.temperature = temperature
        self.criterion = LogitNormLoss(self.device,self.temperature)
        
        self.model = model
        self.device = device
        self.num_class = num_class
        self.epochs = epochs
        self.smooth_eps = smooth_eps

        self.model = self.model.to(self.device)

        self.optimizer = torch.optim.SGD(
            self.model.parameters(), learning_rate, momentum, weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.epochs)