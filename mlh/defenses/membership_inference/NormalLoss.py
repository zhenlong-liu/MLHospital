import torch
import numpy as np
import os
import time
from runx.logx import logx
import torch.nn.functional as F
from defenses.membership_inference.trainer import Trainer
import torch.nn as nn
from loss_function import*
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# class LabelSmoothingLoss(torch.nn.Module):
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
    def __init__(self, model, args, train_loader, loss_type ="ce", device="cuda:0", num_class=10, epochs=100, learning_rate=0.01, momentum=0.9, weight_decay=5e-4, smooth_eps=0.8, log_path="./"):

        super().__init__()

        self.model = model
        self.device = device
        self.num_class = num_class
        self.epochs = epochs
        self.smooth_eps = smooth_eps
        self.args = args
        self.loss_type = loss_type
        self.model = self.model.to(self.device)
        self.train_loader = train_loader
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), learning_rate, momentum, weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.epochs)

        self.criterion = self.get_loss(loss_type =self.loss_type, device=self.device, train_loader=self.train_loader, args = self.args)

        # self.log_path = "%smodel_%s_bs_%s_dataset_%s/%s/label_smoothing_%.1f" % (self.opt.model_save_path, self.opt.model,
        # #                                                                              self.opt.batch_size, self.opt.dataset, self.opt.mode, self.opt.smooth_eps)
        # self.model_save_name = 'model_%s_label_smoothing_%.1f' % (
        #     self.opt.mode, self.opt.smooth_eps)

        # logx.initialize(logdir=self.log_path,
        #                 coolname=False, tensorboard=False)

        self.log_path = log_path
        logx.initialize(logdir=self.log_path,
                        coolname=False, tensorboard=False)

    # 需要通过装饰器 @staticmethod 来进行修饰， 静态方法既不需要传递类对象也不需要传递实例对象（形参没有self/cls ） 。

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
            for img, label in train_loader:
                self.model.zero_grad()
                batch_n += 1

                img, label = img.to(self.device), label.to(self.device)
                # print("img", img.shape)
                logits = self.model(img)
                # 其形状是torch.Size([128, 10])
                loss = self.criterion(logits, label)
                
                loss.backward()
                self.optimizer.step()
    
            train_acc = self.eval(train_loader)
            test_acc = self.eval(test_loader)
            logx.msg('Train Epoch: %d, Total Sample: %d, Train Acc: %.3f, Test Acc: %.3f, Total Time: %.3fs' % (
                e, len(train_loader.dataset), train_acc, test_acc, time.time() - t_start))
            self.scheduler.step()
            
            
    def get_loss(self, loss_type, device, train_loader, args):
        CIFAR10_CONFIG = {
            "ce": nn.CrossEntropyLoss(),
            "focal": FocalLoss(gamma=0.5),
            "mae": MAELoss(num_classes=self.num_classes),
            "gce": GCE(self.device, k=self.num_classes),
            "sce": SCE(alpha=0.5, beta=1.0, num_classes=self.num_classes),
            "ldam": LDAMLoss(device=device),
            "logit_norm": LogitNormLoss(device, self.args.temp, p=self.args.lp),
            "normreg": NormRegLoss(device, self.args.temp, p=self.args.lp),
            "logneg": logNegLoss(device, t=self.args.temp),
            "logit_clip": LogitClipLoss(device, threshold=self.args.temp),
            "cnorm": CNormLoss(device, self.args.temp),
            "tlnorm": TLogitNormLoss(device, self.args.temp, m=10),
            "nlnl": NLNL(device, train_loader=train_loader, num_classes=self.num_classes),
            "nce": NCELoss(num_classes=self.num_classes),
            "ael": AExpLoss(num_classes=10, a=2.5),
            "aul": AUELoss(num_classes=10, a=5.5, q=3),
            "phuber": PHuberCE(tau=10),
            "taylor": TaylorCE(device=self.device, series=args.series),
            "cores": CoresLoss(device=self.device),
            "ncemae": NCEandMAE(alpha=1, beta=1, num_classes=10),
            "ngcemae": NGCEandMAE(alpha=1, beta=1, num_classes=10),
            "ncerce": NGCEandMAE(alpha=1, beta=1.0, num_classes=10),
            "nceagce": NCEandAGCE(alpha=1, beta=4, a=6, q=1.5, num_classes=10),
        }
        CIFAR100_CONFIG = {
            "ce": nn.CrossEntropyLoss(),
            "focal": FocalLoss(gamma=0.5),
            "mae": MAELoss(num_classes=self.num_classes),
            "gce": GCE(self.device, k=self.num_classes),
            "sce": SCE(alpha=0.5, beta=1.0, num_classes=self.num_classes),
            "ldam": LDAMLoss(device=device),
            "logit_clip": LogitClipLoss(device, threshold=self.args.temp),
            "logit_norm": LogitNormLoss(device, self.args.temp, p=self.args.lp),
            "normreg": NormRegLoss(device, self.args.temp, p=self.args.lp),
            "tlnorm": TLogitNormLoss(device, self.args.temp, m=100),
            "cnorm": CNormLoss(device, self.args.temp),
            "nlnl": NLNL(device, train_loader=train_loader, num_classes=self.num_classes),
            "nce": NCELoss(num_classes=self.num_classes),
            "ael": AExpLoss(num_classes=100, a=2.5),
            "aul": AUELoss(num_classes=100, a=5.5, q=3),
            "phuber": PHuberCE(tau=30),
            "taylor": TaylorCE(device=self.device, series=args.series),
            "cores": CoresLoss(device=self.device),
            "ncemae": NCEandMAE(alpha=50, beta=1, num_classes=100),
            "ngcemae": NGCEandMAE(alpha=50, beta=1, num_classes=100),
            "ncerce": NGCEandMAE(alpha=50, beta=1.0, num_classes=100),
            "nceagce": NCEandAGCE(alpha=50, beta=0.1, a=1.8, q=3.0, num_classes=100),
        }
        WEB_CONFIG = {
            "ce": nn.CrossEntropyLoss(),
            "focal": FocalLoss(gamma=0.5),
            "mae": MAELoss(num_classes=self.num_classes),
            "gce": GCE(self.device, k=self.num_classes),
            "sce": SCE(alpha=0.5, beta=1.0, num_classes=self.num_classes),
            "ldam": LDAMLoss(device=device),
            "logit_norm": LogitNormLoss(device, self.args.temp, p=self.args.lp),
            "logit_clip": LogitClipLoss(device, threshold=self.args.temp),
            "normreg": NormRegLoss(device, self.args.temp, p=self.args.lp),
            "cnorm": CNormLoss(device, self.args.temp),
            "tlnorm": TLogitNormLoss(device, self.args.temp, m=50),
            "nlnl": NLNL(device, train_loader=train_loader, num_classes=self.num_classes),
            "nce": NCELoss(num_classes=self.num_classes),
            "ael": AExpLoss(num_classes=50, a=2.5),
            "aul": AUELoss(num_classes=50, a=5.5, q=3),
            "phuber": PHuberCE(tau=30),
            "taylor": TaylorCE(device=self.device, series=args.series),
            "cores": CoresLoss(device=self.device),
            "ncemae": NCEandMAE(alpha=50, beta=0.1, num_classes=50),
            "ngcemae": NGCEandMAE(alpha=50, beta=0.1, num_classes=50),
            "ncerce": NGCEandMAE(alpha=50, beta=0.1, num_classes=50),
            "nceagce": NCEandAGCE(alpha=50, beta=0.1, a=2.5, q=3.0, num_classes=50),
        }
        if "CIFAR10" in args.dataset:
            return CIFAR10_CONFIG[loss_type]
        elif args.dataset == "cifar100":
            return CIFAR100_CONFIG[loss_type]
        elif args.dataset == "webvision":
            return WEB_CONFIG[loss_type]   