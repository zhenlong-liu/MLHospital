# MIT License

# Copyright (c) 2022 The Machine Learning Hospital (MLH) Authors

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import numpy as np
import os
import time
from runx.logx import logx
import torch.nn as nn
from defenses.membership_inference.NormalLoss import TrainTargetNormalLoss
from defenses.membership_inference.trainer import Trainer

import sys
sys.path.append("..")
sys.path.append("../..")

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from defenses.membership_inference.loss_function import get_loss, get_loss_adj
from tqdm import tqdm
from utils import get_optimizer, get_scheduler, get_init_args, dict_str
from utility.main_parse import save_namespace_to_yaml, save_dict_to_yaml
class AttackAdvReg(nn.Module):
    def __init__(self, posterior_dim, class_dim):
        self.posterior_dim = posterior_dim
        self.class_dim = class_dim
        super(AttackAdvReg, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(self.posterior_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
        )
        self.labels = nn.Sequential(
            nn.Linear(self.class_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.combine = nn.Sequential(
            nn.Linear(64*2, 256),
            nn.ReLU(),
            nn.Linear(256, 128),

            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self.output = nn.Sigmoid()

    def forward(self, x, l):
        out_x = self.features(x)
        out_l = self.labels(l)
        is_member = self.combine(torch.cat((out_x, out_l), 1))
        return self.output(is_member)


class TrainTargetAdvReg(TrainTargetNormalLoss):
    def __init__(self, model, args, delta=1e-5,momentum=0.9, weight_decay=5e-4, **kwargs):

        super().__init__(model, args, **kwargs)
        
        self.attack_model = AttackAdvReg(self.num_classes, self.num_classes)
        self.attack_model.to(self.device)
        self.optimizer_adv = torch.optim.SGD(self.attack_model.parameters(
        ), self.learning_rate, momentum=momentum, weight_decay=weight_decay)
        self.scheduler_adv = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_adv, T_max=self.epochs)
    def train_attack_advreg(self, train_loader, inference_loader):
        """
        train the mia classifier to distinguish train and inference data
        """
        
        
        self.model.eval()
        self.attack_model.train()
        
        
        for batch_idx, ((train_data, train_target), (inference_data, inference_target)) in enumerate(zip(train_loader, inference_loader)):
            train_data, train_target = train_data.to(
                self.device), train_target.to(self.device)
            inference_data, inference_target = inference_data.to(
                self.device), inference_target.to(self.device)

            all_data = torch.cat([train_data, inference_data], dim=0)
            all_target = torch.cat([train_target, inference_target], dim=0)
            all_output = self.model(all_data)

            one_hot_tr = torch.from_numpy((np.zeros(
                (all_target.size(0), self.num_classes))-1)).type(torch.FloatTensor).to(self.device)
            infer_input_one_hot = one_hot_tr.scatter_(1, all_target.type(
                torch.LongTensor).view([-1, 1]).data.to(self.device), 1)
            attack_output = self.attack_model(all_output, infer_input_one_hot)
            # get something like [[1], [1], [1], [0], [0], [0]]
            att_labels = torch.cat([torch.ones(
    train_data.shape[0]), torch.zeros(train_data.shape[0])], dim=0).to(self.device).unsqueeze(1)
            # att_labels = torch.cat([torch.ones(train_data.shape[0]), torch.zeros(
            #     train_data.shape[0])], dim=0).type(torch.LongTensor).to(self.device)
            # print(att_labels, train_target)
            loss = torch.nn.functional.binary_cross_entropy(
                attack_output, att_labels)
            self.attack_model.zero_grad()
            loss.backward()
            self.optimizer_adv.step()
            self.scheduler_adv.step()
    def train_target_privately(self, train_loader):
        self.model.train()
        self.attack_model.eval()
        tau= self.args.tau

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            one_hot_tr = torch.from_numpy((np.zeros((output.size(
                0), self.num_classes)) - 1)).type(torch.FloatTensor).to(self.device)
            target_one_hot_tr = one_hot_tr.scatter_(1, target.type(
                torch.LongTensor).view([-1, 1]).data.to(self.device), 1)

            member_output = self.attack_model(output, target_one_hot_tr)
            loss = self.criterion(output, target) + (tau)*(torch.mean((member_output)) - 0.5)
            self.loss_num = loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
    def train(self, train_loader, inference_loader, test_loader):

        best_accuracy = 0
        t_start = time.time()
        # check whether path exist
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

        # torch.save(self.model.state_dict(), os.path.join(
        #     self.log_path, '%s_0.pth' % (self.model_save_name)))

        # first train target model for 5 epochs
        for e in range(1, self.epochs+1):

            if e < 100:
                self.train_target_privately(train_loader)
            else:
                self.train_attack_advreg(train_loader, inference_loader)
                self.train_target_privately(train_loader)

            train_acc = self.eval(train_loader)
            test_acc = self.eval(test_loader)

            logx.msg('Loss Type: %s, Train Epoch: %d, Total Sample: %d, Train Acc: %.3f, Test Acc: %.3f, Loss: %.3f, Total Time: %.3fs' % (
                self.args.loss_type, e, len(train_loader.dataset), train_acc, test_acc, self.loss_num, time.time() - t_start))
            
            

        #     if e % 10 == 0:
        #         torch.save(self.model.state_dict(), os.path.join(
        #             self.log_path, '%s_%d.pth' % (self.model_save_name, e)))

        # torch.save(self.model.state_dict(), os.path.join(
        #     self.log_path, "%s.pth" % self.model_save_name))
            if e == self.epochs:
                log_dict = {'Loss Type' : self.args.loss_type,"Train Epoch" : e, "Total Sample": len(train_loader.dataset),
                            "Train Acc": train_acc, "Test Acc": test_acc, "Loss": self.loss_num, "Total Time" : time.time() - t_start}
                save_dict_to_yaml(log_dict, f'{self.log_path}/train_log.yaml')