import torch.nn.functional as F
import torch
import numpy as np
import os
import time
import sys
sys.path.append("..")
sys.path.append("../..")

# class LabelSmoothingLoss(torch.nn.Module):
from runx.logx import logx
import torch.nn.functional as F
from defenses.membership_inference.NormalLoss import TrainTargetNormalLoss 
import torch.nn as nn
from defenses.membership_inference.loss_function import get_loss_adj
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from tqdm import tqdm
from utils import get_optimizer, get_scheduler, get_init_args, dict_str
from utility.main_parse import save_namespace_to_yaml, save_dict_to_yaml
class TrainTargetKnowledgeDistillation(TrainTargetNormalLoss):
    def __init__(self, model, teacher_model, args, T=2.0, alpha=0.5, 
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
        
        self.teacher_model = teacher_model
        self.T = args.gamma
        self.alpha = args.tau
        self.teacher_model.to(self.device)
        
    def compute_loss(self, student_logits, teacher_logits, labels):
        """
        Compute the knowledge distillation loss.
        :param student_logits: Logits from the student model
        :param teacher_logits: Logits from the teacher model
        :param labels: True labels
        :return: The loss value
        """
        # Calculate the KL Divergence between the teacher and student
        soft_loss = F.kl_div(F.log_softmax(student_logits/self.T, dim=1),
                             F.softmax(teacher_logits/self.T, dim=1),
                             reduction='batchmean') * (self.T**2)
        
        # Calculate the normal CE loss with true labels
        hard_loss = F.cross_entropy(student_logits, labels)
        
        # Combine the two loss terms
        return self.alpha * soft_loss + (1. - self.alpha) * hard_loss
    
    def train(self, train_loader, test_loader):
        # Overriding the train method to use the knowledge distillation loss
        best_accuracy = 0
        t_start = time.time()
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

        for e in range(1, self.epochs+1):
            batch_n = 0
            self.model.train()
            self.teacher_model.eval()  # Ensure the teacher is in eval mode
            loss_num = 0
            
            for img, label in tqdm(train_loader):
                self.model.zero_grad()
                batch_n += 1

                img, label = img.to(self.device), label.to(self.device)
                student_logits = self.model(img)
                
                # Get the teacher logits with no gradient
                with torch.no_grad():
                    teacher_logits = self.teacher_model(img)
                
                # Use the knowledge distillation loss
                loss = self.compute_loss(student_logits, teacher_logits, label)
                
                loss.backward()
                loss_num = loss.item()
                self.optimizer.step()

            train_acc = self.eval(train_loader)
            test_acc = self.eval(test_loader)
            logx.msg('Train Epoch: %d, Train Acc: %.3f, Test Acc: %.3f, Loss: %.3f, Total Time: %.3fs' % (
                e, train_acc, test_acc, loss_num, time.time() - t_start))
            self.scheduler.step()
            if e == self.epochs:
                log_dict = {"Train Epoch" : e, "Train Acc": train_acc, 
                            "Test Acc": test_acc, "Loss": loss_num, 
                            "Total Time" : time.time() - t_start}
                save_dict_to_yaml(log_dict, f'{self.log_path}/train_log.yaml')
