import torch.nn.functional as F
import torch
import numpy as np
import os
import time
import sys
sys.path.append("..")
sys.path.append("../..")
import copy
# class LabelSmoothingLoss(torch.nn.Module):
from runx.logx import logx
import torch.nn.functional as F
from defenses.membership_inference.NormalLoss import TrainTargetNormalLoss 
import torch.nn as nn
from defenses.membership_inference.loss_function import get_loss, get_loss_adj
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from tqdm import tqdm
from utils import generate_save_path, get_optimizer, get_scheduler, get_init_args, dict_str
from utility.main_parse import save_namespace_to_yaml, save_dict_to_yaml
class TrainTargetEarlyStopping(TrainTargetNormalLoss):
    def __init__(self, model, args, 
                  **kwargs):
        """
        :param model: The student model
        :param teacher_model: The teacher model
        :param args: Arguments
        :param train_loader: Training data loader
        :param T: Temperature parameter
        :param alpha: Weighting between hard labels and soft labels
        """
        
        
        super().__init__(model, args, **kwargs)
        self.stop_eps = args.stop_eps
        self.args_copy = copy.deepcopy(args)
    def train(self, train_loader, test_loader):

        best_accuracy = 0
        t_start = time.time()
        # check whether path exist
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

        
        # torch.save(self.model.state_dict(), os.path.join(
        #     self.log_path, '%s_0.pth' % (self.model_save_name)))

        for e in range(1, min(self.epochs, max(self.stop_eps))):
            batch_n = 0
            self.model.train()
            for img, label in tqdm(train_loader):
                self.model.zero_grad()
                batch_n += 1

                img, label = img.to(self.device), label.to(self.device)
                # print("img", img.shape)
                logits = self.model(img)
                # 其形状是torch.Size([128, 10])
                
                loss = self.criterion(logits, label)
                
                loss.backward()
                loss_num = loss.item()
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
            
            
            
            
            if e in list(set(self.stop_eps + [self.epochs])):
                self.args_copy.tau = e
                log_path = generate_save_path(self.args_copy)
                if not os.path.exists(log_path):
                    os.makedirs(log_path)
                
                log_dict = {'Loss Type' : self.args.loss_type,"Train Epoch" : e, "Total Sample": len(train_loader.dataset),
                            "Train Acc": train_acc, "Test Acc": test_acc, "Loss": loss_num, "Total Time" : time.time() - t_start}
                save_namespace_to_yaml(self.args_copy, f'{log_path}/config.yaml')
                save_dict_to_yaml(log_dict, f'{log_path}/train_log.yaml')
                torch.save(self.model.state_dict(), os.path.join(log_path, f"{self.args.model}.pth"))