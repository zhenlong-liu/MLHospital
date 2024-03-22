import torch
import numpy as np
import os
import time
import sys

from defenses.membership_inference.NormalLoss import TrainTargetNormal

sys.path.append("..")
sys.path.append("../..")
from utility.main_parse import save_dict_to_yaml
from runx.logx import logx
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from tqdm import tqdm

def mixup_data(x, y, device="cuda:0", alpha=1.0, use_cuda=True):
    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda().to(device)
        
    else:
        index = torch.randperm(batch_size)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]  
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class TrainTargetMixup(TrainTargetNormal):
    def __init__(self, model, args, mixup=1, **kwargs):

        super().__init__(model, args, **kwargs)
        self.mixup = mixup
        self.mixup_alpha = args.alpha
    def train(self, train_loader, test_loader):
        t_start = time.time()
        # check whether path exist
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        for e in range(1, self.epochs+1):
            batch_n = 0
            self.model.train()
            loss_num =0
            for img, label in tqdm(train_loader, desc="train mixup"):
                self.model.zero_grad()
                img, label = img.to(self.device), label.to(self.device)
                batch_n += 1
                if (self.mixup):
                    
                        inputs, targets_a, targets_b, lam = mixup_data(
                            img, label, self.device, self.mixup_alpha)
                        #inputs, targets_a, targets_b = inputs.to(self.device), targets_a.to(
                         #   self.device), targets_b.to(self.device)
                        outputs = self.model(inputs)
                        loss_func = mixup_criterion(targets_a, targets_b, lam)
                        loss = loss_func(self.criterion, outputs)
                        loss.backward()
                        loss_num = loss.item()
                        self.optimizer.step()
                    
                else:        
                    img, label = img.to(self.device), label.to(self.device)
                    # print("img", img.shape)
                    logits = self.model(img)
                    loss = self.criterion(logits, label)
                    loss.backward()
                    self.optimizer.step()
            """
            if self.args.dataset.lower() == 'imagenet' and e<self.epochs:
                self.scheduler.step()
                continue
            """
            train_acc = self.eval(train_loader)
            test_acc = self.eval(test_loader)
            logx.msg('Loss Type: %s, Train Epoch: %d, Total Sample: %d, Train Acc: %.3f, Test Acc: %.3f, Loss: %.3f, Total Time: %.3fs' % (
                self.args.loss_type, e, len(train_loader.dataset), train_acc, test_acc, loss_num, time.time() - t_start))
            self.scheduler.step()
            if e == self.epochs:
                log_dict = {'Loss Type' : self.args.loss_type,"Train Epoch" : e, "Total Sample": len(train_loader.dataset),
                            "Train Acc": train_acc, "Test Acc": test_acc, "Loss": loss_num, "Total Time" : time.time() - t_start}
                save_dict_to_yaml(log_dict, f'{self.log_path}/train_log.yaml')