
from defenses.membership_inference.Normal import TrainTargetNormal
import torch
import numpy as np
import os
import time
from runx.logx import logx
import torch.nn.functional as F
import torch.nn as nn

class LogitClipLoss(nn.Module):

    def __init__(self, device, tau=1.0 ,p =2):
        super(LogitClipLoss, self).__init__()
        self.tau =tau
        self.delta = tau
        self.device = device
        self.p = p
        
    def forward(self, x, target):
        # [N, C]
        norms = torch.norm(x, p=self.p, dim=-1, keepdim=True) + 1e-7 # [N,]      
        identify = (norms >= self.tau)
        logit_clip = self.delta*torch.div(x, norms)*identify + x*(~identify)
        
        #logit_clip  = [self.delta*torch.div(x, norm) for norm in norms if norm >= self.tau else x]
        #logit_clip = (norms >= self.tau)
        return F.cross_entropy(logit_clip, target)

# ,device,num_class, epochs, learning_rate, momentum, weight_decay, smooth_eps, log_path

class TrainTargetLogitClip(TrainTargetNormal):
    def __init__(self,model, device="cuda:0", num_class=10, epochs=100, learning_rate=0.01, momentum=0.9, weight_decay=5e-4, smooth_eps=0.8, log_path="./", tau =1.0, p=2):
        super(TrainTargetLogitClip,self).__init__(model, device, num_class, epochs, learning_rate, momentum, weight_decay, smooth_eps, log_path)
        self.device = device
        self.tau = tau
        self.p = p
        self.criterion = LogitClipLoss(device=self.device, tau=self.tau, p=self.p )
        

    def train_batch(self, index, inputs, targets, epoch):
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        logits = self.net(inputs)
        threshold = self.args.temp
        norms = torch.norm(logits, p=self.args.lp, dim=-1, keepdim=True) + 1e-7
        logits_norm = torch.div(logits, norms) * threshold
        clip = (norms > threshold).expand(-1, logits.shape[-1])
        logits_final = torch.where(clip, logits_norm, logits)
        if self.args.loss == "cores":
            loss = self.loss_function(logits_final, targets, epoch)
        else:
            loss = self.loss_function(logits_final, targets)
        if self.args.use_stat:
            clip_num, noisy_clip_num = self.clip_stat(index, clip[:, 0])
            return loss, clip_num, noisy_clip_num, clip.shape[0]
        else:
            return loss