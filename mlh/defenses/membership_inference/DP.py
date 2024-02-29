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
import sys
sys.path.append("..")
sys.path.append("../..")
from utility.main_parse import save_namespace_to_yaml, save_dict_to_yaml
from runx.logx import logx
from defenses.membership_inference.trainer import Trainer
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from tqdm import tqdm
from utils import get_optimizer, get_scheduler, get_init_args, dict_str
from defenses.membership_inference.NormalLoss import TrainTargetNormalLoss


class TrainTargetDPSGD(TrainTargetNormalLoss):
    def __init__(self, model, args, delta=1e-5,momentum=0.9, weight_decay=5e-4, **kwargs):
        
        super().__init__(model, args, **kwargs)
        
        self.noise_scale = args.alpha
        self.grad_norm = args.temp
        self.delta = delta 
        self.model = ModuleValidator.fix(self.model)
        self.model = self.model.to(self.device)
        self.optimizer = get_optimizer(args.optimizer, self.model.parameters(),self.learning_rate, momentum, weight_decay)
        #self.optimizer = torch.optim.SGD(self.model.parameters(
        # ), args.learning_rate, momentum=momentum, weight_decay=weight_decay)
        
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #    self.optimizer, T_max=self.epochs)
        
    def train(self, train_loader, test_loader):
        
        # privacy_engine = PrivacyEngine(
        #     self.model,
        #     batch_size=self.args.batch_size,
        #     sample_size=len(train_loader.dataset),  # overall training set size
        #     alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
        #     noise_multiplier=self.noise_scale,  # sigma
        #     max_grad_norm=self.grad_norm,  # this is from dp-sgd paper
        # )
        # privacy_engine.attach(self.optimizer)
        
        
        """    
        self.model, self.optimizer, train_loader =privacy_engine.make_private(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=train_loader,
            # epochs=self.epochs,
            noise_multiplier=self.noise_scale,
            max_grad_norm=self.grad_norm,
            poisson_sampling= False
        )
        """
        privacy_engine = PrivacyEngine()
        self.model, self.optimizer, train_loader = privacy_engine.make_private(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=train_loader,
            noise_multiplier=self.noise_scale,
            max_grad_norm=self.grad_norm,
        )
       
        
        self.scheduler = get_scheduler(scheduler_name = self.args.scheduler, optimizer =self.optimizer, t_max=self.epochs)
        
        
        """
        
        self.model, self.optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=train_loader,
            epochs=self.epochs, 
            target_epsilon=self.noise_scale,
            target_delta=self.delta,
            max_grad_norm=self.grad_norm,
        )
        """
        """
        
        """
        t_start = time.time()

        # check whether path exist
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        losses = []
        for e in range(1, self.epochs+1):
            batch_n = 0
            self.model.train()
            for img, label in tqdm(train_loader):

                self.optimizer.zero_grad()
                batch_n += 1

                img, label = img.to(self.device), label.to(self.device)

                logits = self.model(img)
                loss = self.criterion(logits, label)
                loss.backward()
                losses.append(loss.item())
                self.optimizer.step()

            train_acc = self.eval(train_loader)
            test_acc = self.eval(test_loader)
            # print("epoch:%d\ttrain_acc:%.3f\ttest_acc:%.3f\ttotal_time:%.3fs" % (e, train_acc, test_acc, time.time() - t_start))
            logx.msg('Train Epoch: %d, Total Sample: %d, Train Acc: %.3f, Test Acc: %.3f, Loss: %.3f, Total Time: %.3fs' % (
                e, len(train_loader.dataset), train_acc, test_acc, np.mean(losses), time.time() - t_start))

            # print dp params
            # epsilon = privacy_engine.get_epsilon(self.delta)
            # logx.msg('eps: %.5f, delta:%.5f' % (epsilon, self.delta))

            self.scheduler.step()
            if e == self.epochs:
                log_dict = {'Loss Type' : self.args.loss_type,"Train Epoch" : e, "Total Sample": len(train_loader.dataset),
                            "Train Acc": train_acc, "Test Acc": test_acc, "Loss": float(np.mean(losses)), "Total Time" : time.time() - t_start}
                save_dict_to_yaml(log_dict, f'{self.log_path}/train_log.yaml')
                
        try:
            # model_path = os.path.join(self.log_path, f"{self.args.model}.pth")
            # torch.save(self.model.state_dict(),model_path)
            # print(f"Model successfully saved at {model_path}")
            model_path = os.path.join(self.log_path, f"{self.args.model}.pt")
            torch.save(self.model, os.path.join(self.log_path, f"{self.args.model}.pt"))
            print(f"Model successfully saved at {model_path}")
        except Exception as e:
            print(f"An error occurred while saving the model: {e}")
