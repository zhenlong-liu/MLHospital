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
from defenses.membership_inference.trainer import Trainer
import torch.nn as nn
from defenses.membership_inference.loss_function import get_loss
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from tqdm import tqdm
from utils import get_optimizer, get_scheduler, get_init_args, dict_str
from utility.main_parse import save_namespace_to_yaml, save_dict_to_yaml
from utility.metrics import Metrics, StaMetrics
class TrainTargetNormal(Trainer):
    def __init__(self, model, args, momentum=0.9, weight_decay=5e-4, smooth_eps=0.8, log_path="./"):
        super().__init__(model, args)
        self.model = model.to(args.device)
        self.device = args.device
        self.weight_decay = args.weight_decay
        self.num_classes = args.num_class
        self.epochs = args.epochs
        self.smooth_eps = smooth_eps
        self.args = args
        self.loss_type = args.loss_type
        self.learning_rate = args.learning_rate
        self.optimizer = get_optimizer(args.optimizer, self.model.parameters(),self.learning_rate, momentum, self.weight_decay)
        self.sta_book = StaMetrics()
        self.scheduler = get_scheduler(scheduler_name = args.scheduler, optimizer =self.optimizer, t_max=self.epochs)
        self.criterion = self.initialize_criterion()
        self.log_path = log_path
        logx.initialize(logdir=self.log_path,coolname=False, tensorboard=False)
        
        logx.msg(f"optimizer:{args.optimizer}, learning rate:{args.learning_rate}, scheduler:{args.scheduler}, epoches:{self.epochs}")
        print(args.checkpoint)
        self.save_configs()
        if args.checkpoint:
            self.check_point()

        
    def initialize_criterion(self):
        """Initialize the loss criterion."""
        return get_loss(loss_type=self.loss_type, device=self.device, args=self.args, num_classes=self.num_classes)

    def save_configs(self):
        """Save configurations for better reproducibility and logging."""
        save_namespace_to_yaml(self.args, f'{self.log_path}/config.yaml')
        save_namespace_to_yaml(dict_str(get_init_args(self.criterion)), f'{self.log_path}/loss_config.yaml')
        
    
    def check_point(self):
        checkpoint = f"{self.log_path}/{self.args.model}.pth"
        if os.path.isfile(checkpoint):
            exit()

    def eval(self, data_loader, epoch, loader_type = "train_loader"):
        correct = 0
        total = 0
        self.model.eval()
        self.sta_book._metrics_list =[]
        with torch.no_grad():
            for data, label in data_loader:
                data, label = data.to(self.device), label.to(self.device)
                logits = self.model(data)
                metrics = Metrics(labels = label, logits = logits)
                self.sta_book.add_metrics(metrics.metrics)
                predicted = torch.argmax(logits, dim=1)
                total += label.size(0)
                correct += (predicted == label).sum().item()
            final_acc = 100 * correct / total
            self.sta_book.add_total_variance(epoch, loader_type)
        return final_acc

    def train(self, train_loader, test_loader):
        t_start = time.time()
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path) 
        for epoch in range(1, self.epochs+1):
            losses = []
            self.model.train()
            loss_num = 0
            for data, label in tqdm(train_loader):
                self.model.zero_grad()
                data, label = data.to(self.device), label.to(self.device)
                logits = self.model(data)
                loss = self.criterion(logits, label)
                loss.backward()
                losses.append(loss.item())
                self.optimizer.step()
            train_acc = self.eval(train_loader, epoch, loader_type = "train_loader")
            test_acc = self.eval(test_loader, epoch, loader_type="test_loader")
            
            logx.msg('Loss Type: %s, Train Epoch: %d, Total Sample: %d, Train Acc: %.3f, Test Acc: %.3f, Loss: %.3f, Total Time: %.3fs' % (
                self.args.loss_type, epoch, len(train_loader.dataset), train_acc, test_acc, np.mean(losses), time.time() - t_start))
        
            self.scheduler.step()
            if epoch == self.epochs:
                log_dict = {'Loss Type' : self.args.loss_type,"Train Epoch" : epoch, "Total Sample": len(train_loader.dataset),
                            "Train Acc": train_acc, "Test Acc": test_acc, "Loss": loss_num, "Total Time" : time.time() - t_start}
                save_dict_to_yaml(log_dict,  f'{self.log_path}/train_log.yaml')
            
            self.sta_book.sta_epochs.to_excel( f'{self.log_path}/epochs_data.xlsx', index=False)   
