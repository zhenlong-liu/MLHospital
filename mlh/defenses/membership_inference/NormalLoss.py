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
from defenses.membership_inference.loss_function import get_loss, get_loss_adj
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from tqdm import tqdm
from utils import get_optimizer, get_scheduler, get_init_args, dict_str
from utility.main_parse import save_namespace_to_yaml, save_dict_to_yaml
#     """
#     copy from:
#     https://github.com/pytorch/pytorch/issues/7455
#     """

#     def __init__(self, classes, smoothing=0.0, dim=-1):
#         super(LabelSmoothingLoss, self).__init__()
#         self.confidence = 1.0 - smoothing
#         self.smoothing = smoothing
#         self.cls = classes
#         self.dim = dim

#     def forward(self, pred, target):
#         pred = pred.log_softmax(dim=self.dim)
#         with torch.no_grad():
#             # true_dist = pred.data.clone()
#             true_dist = torch.zeros_like(pred)
#             true_dist.fill_(self.smoothing / (self.cls - 1))
#             true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
#         return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


class TrainTargetNormalLoss(Trainer):
    def __init__(self, model, args, momentum=0.9, weight_decay=5e-4, smooth_eps=0.8, log_path="./"):

        super().__init__()
        
        self.model = model.to(args.device)
        self.device = args.device
        self.num_classes = args.num_class
        self.epochs = args.epochs
        self.smooth_eps = smooth_eps
        self.args = args
        self.loss_type = args.loss_type
        self.learning_rate = args.learning_rate

        self.optimizer = get_optimizer(args.optimizer, self.model.parameters(),self.learning_rate, momentum, weight_decay)
        #self.optimizer = torch.optim.SGD( self.model.parameters(), self.learning_rate, momentum, weight_decay)
        
        self.scheduler = get_scheduler(scheduler_name = args.scheduler, optimizer =self.optimizer, t_max=self.epochs)
        #self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs)
        self.criterion = self.initialize_criterion()
        
        # self.log_path = "%smodel_%s_bs_%s_dataset_%s/%s/label_smoothing_%.1f" % (self.opt.model_save_path, self.opt.model,
        # #                                                                              self.opt.batch_size, self.opt.dataset, self.opt.mode, self.opt.smooth_eps)
        # self.model_save_name = 'model_%s_label_smoothing_%.1f' % (
        #     self.opt.mode, self.opt.smooth_eps)

        # logx.initialize(logdir=self.log_path,
        #                 coolname=False, tensorboard=False)

        self.log_path = log_path
        logx.initialize(logdir=self.log_path,
                        coolname=False, tensorboard=False)

        logx.msg(f"optimizer:{args.optimizer}, learning rate:{args.learning_rate}, scheduler:{args.scheduler}, epoches:{self.epochs}")
        print(args.checkpoint)
        self.save_configs()
        if args.checkpoint:
            self.check_point()
    
        
    def initialize_criterion(self):
        """Initialize the loss criterion."""
        if self.args.loss_adjust:
            return get_loss_adj(loss_type=self.loss_type, device=self.device, args=self.args, num_classes=self.num_classes)
        else:
            return get_loss(loss_type=self.loss_type, device=self.device, args=self.args, num_classes=self.num_classes)

    def save_configs(self):
        """Save configurations for better reproducibility and logging."""
        save_namespace_to_yaml(self.args, f'{self.log_path}/config.yaml')
        save_namespace_to_yaml(dict_str(get_init_args(self.criterion)), f'{self.log_path}/loss_config.yaml')

    
    def check_point(self):
        checkpoint = f"{self.log_path}/{self.args.model}.pth"
        if os.path.isfile(checkpoint):
            exit()
    
    
    
    @staticmethod
    def _sample_weight_decay():
        # We selected the l2 regularization parameter from a range of 45 logarithmically spaced values between 10−6 and 105
        weight_decay = np.logspace(-6, 5, num=45, base=10.0)
        weight_decay = np.random.choice(weight_decay)
        print("Sampled weight decay:", weight_decay)
        return weight_decay

    def eval(self, data_loader):

        correct = 0
        total = 0
        self.model.eval()
        with torch.no_grad():

            for img, label in data_loader:
                img, label = img.to(self.device), label.to(self.device)
                logits = self.model.eval().forward(img)

                predicted = torch.argmax(logits, dim=1)
                total += label.size(0)
                correct += (predicted == label).sum().item()

            final_acc = 100 * correct / total

        return final_acc

    def train(self, train_loader, test_loader):

        best_accuracy = 0
        t_start = time.time()
        # check whether path exist
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

        
        # torch.save(self.model.state_dict(), os.path.join(
        #     self.log_path, '%s_0.pth' % (self.model_save_name)))

        for e in range(1, self.epochs+1):
            batch_n = 0
            self.model.train()
            loss_num =0
            
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
            if e == self.epochs:
                log_dict = {'Loss Type' : self.args.loss_type,"Train Epoch" : e, "Total Sample": len(train_loader.dataset),
                            "Train Acc": train_acc, "Test Acc": test_acc, "Loss": loss_num, "Total Time" : time.time() - t_start}
                save_dict_to_yaml(log_dict, f'{self.log_path}/train_log.yaml')