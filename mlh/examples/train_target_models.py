import os
import sys
sys.path.append('/home/liuzhenlong/MIA/MLHospital/mlh/')
sys.path.append('/home/liuzhenlong/MIA/MLHospital/mlh/defenses')

import torchvision
import utils
from utils import get_loss
from defenses.membership_inference.AdvReg import TrainTargetAdvReg
from defenses.membership_inference.DPSGD import TrainTargetDP
from defenses.membership_inference.LabelSmoothing import TrainTargetLabelSmoothing
from defenses.membership_inference.MixupMMD import TrainTargetMixupMMD
from defenses.membership_inference.PATE import TrainTargetPATE
from defenses.membership_inference.Normal import TrainTargetNormal

from defenses.membership_inference.logit_norm import TrainTargetLogitsNorm

from defenses.membership_inference.logit_norm import LogitNormLoss

from defenses.membership_inference.LogitClip import TrainTargetLogitClip

from defenses.membership_inference.NormalLoss import TrainTargetNormalLoss
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from data_preprocessing.data_loader import GetDataLoader
from torchvision import datasets
import torchvision.transforms as transforms
import argparse
import numpy as np
import torch.optim as optim
torch.manual_seed(0)
np.random.seed(0)
torch.set_num_threads(1)


def parse_args():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--batch-size', type=int, default=512,
                        help='batch_size')
    parser.add_argument('--num-workers', type=int, default=10,
                        help='num of workers to use')

    parser.add_argument('--training_type', type=str, default="Normal",
                        help='Normal, LabelSmoothing, AdvReg, DP, MixupMMD, PATE')
    parser.add_argument('--mode', type=str, default="shadow",
                        help='target, shadow')

    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')
    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu index used for training')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--load-pretrained', type=str, default='no')
    parser.add_argument('--task', type=str, default='mia',
                        help='specify the attack task, mia or ol')
    parser.add_argument('--dataset', type=str, default='CIFAR10',
                        help='dataset')
    parser.add_argument('--num-class', type=int, default=10,
                        help='number of classes')
    parser.add_argument('--inference-dataset', type=str, default='CIFAR10',
                        help='if yes, load pretrained the attack model to inference')
    parser.add_argument('--data-path', type=str, default='../datasets/',
                        help='data_path')
    parser.add_argument('--input-shape', type=str, default="32,32,3",
                        help='comma delimited input shape input')
    parser.add_argument('--log_path', type=str,
                        default='./save', help='data_path')
    
    parser.add_argument('--temp', type=float, default=1,
                        help='temperature')
    # 默认储存到save里
    parser.add_argument('--tau', type=float, default=1, help = "logitclip tau")
    
    parser.add_argument('--loss_type', type=str, default="ce", help = "Loss function")
    args = parser.parse_args()

    args.input_shape = [int(item) for item in args.input_shape.split(',')]
    args.device = 'cuda:%d' % args.gpu if torch.cuda.is_available() else 'cpu'

    return args


def get_target_model(name="resnet18", num_classes=10):
    if name == "resnet18":
        model = torchvision.models.resnet18()
        model.fc = nn.Sequential(nn.Linear(512, 10))
        # 代码修改了ResNet-18模型的最后一层全连接层，将其替换为一个新的全连接层nn.Linear(512, 10)，
        # 其中512是ResNet-18模型中最后一个卷积层的输出通道数，10是类别数量。这样做是为了将模型的输出调整为与任务中的类别数量相匹配。
    else:
        raise ValueError("Model not implemented yet :P")
    return model


def evaluate(args, model, dataloader):
    model.eval()
    correct = 0
    total = 0
    for data in dataloader:
        inputs, labels = data
        inputs, labels = inputs.to(args.device), labels.to(args.device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    model.train()
    return correct / total


if __name__ == "__main__":

    opt = parse_args()
    s = GetDataLoader(opt)
    target_train_loader, target_inference_loader, target_test_loader, shadow_train_loader, shadow_inference_loader, shadow_test_loader = s.get_data_supervised()


    # 选择是训练target model 还是训练shadow model
    if opt.mode == "target":
        train_loader, inference_loader, test_loader = target_train_loader, target_inference_loader, target_test_loader
        
        # train_loader is a dataloader, using next(), feature shape is [128,3,32,32], label shape [128]
    # 
    elif opt.mode == "shadow":
        train_loader, inference_loader, test_loader = shadow_train_loader, shadow_inference_loader, shadow_test_loader
    else:
        raise ValueError("opt.mode should be target or shadow")

    target_model = get_target_model(name="resnet18", num_classes=10)

    save_pth = f'{opt.log_path}/{opt.dataset}/{opt.training_type}/{opt.mode}/{opt.temp}'

    if opt.training_type == "Normal":
        
        total_evaluator = TrainTargetNormal(
            model=target_model, epochs=opt.epochs, log_path=save_pth)
        total_evaluator.train(train_loader, test_loader)
        
    elif opt.training_type == "TrainTargetLogitNorm":
        
        total_evaluator = TrainTargetNormal(
            model=target_model, epochs=opt.epochs, log_path=save_pth)
        total_evaluator.criterion = LogitNormLoss(opt.device,opt.temp)
        total_evaluator.train(train_loader, test_loader)
        
    elif opt.training_type == "LogitClip":
        
        total_evaluator = TrainTargetLogitClip(
            model=target_model, epochs=opt.epochs, log_path=save_pth, tau = opt.tau)
        total_evaluator.train(train_loader, test_loader)
        
    elif opt.training_type == "NormalLoss":
        
        total_evaluator = TrainTargetNormalLoss(
            model=target_model, args=opt, train_loader=train_loader, loss_type=opt.loss_type ,epochs=opt.epochs, log_path=save_pth)
        total_evaluator.train(train_loader, test_loader)

    elif opt.training_type == "LabelSmoothing":

        total_evaluator = TrainTargetLabelSmoothing(
            model=target_model, epochs=opt.epochs, log_path=save_pth)
        total_evaluator.train(train_loader, test_loader)

    elif opt.training_type == "AdvReg":

        total_evaluator = TrainTargetAdvReg(
            model=target_model, epochs=opt.epochs, log_path=save_pth)
        total_evaluator.train(train_loader, inference_loader, test_loader)
        model = total_evaluator.model

    elif opt.training_type == "DP":
        total_evaluator = TrainTargetDP(
            model=target_model, epochs=opt.epochs, log_path=save_pth)
        total_evaluator.train(train_loader, test_loader)

    elif opt.training_type == "MixupMMD":

        target_train_sorted_loader, target_inference_sorted_loader, shadow_train_sorted_loader, shadow_inference_sorted_loader, start_index_target_inference, start_index_shadow_inference, target_inference_sorted, shadow_inference_sorted = s.get_sorted_data_mixup_mmd()
        if opt.mode == "target":
            train_loader_ordered, inference_loader_ordered, starting_index, inference_sorted = target_train_sorted_loader, target_inference_sorted_loader, start_index_target_inference, target_inference_sorted

        elif opt.mode == "shadow":
            train_loader_ordered, inference_loader_ordered, starting_index, inference_sorted = shadow_train_sorted_loader, shadow_inference_sorted_loader, start_index_shadow_inference, shadow_inference_sorted

        total_evaluator = TrainTargetMixupMMD(
            model=target_model, epochs=opt.epochs, log_path=save_pth)
        total_evaluator.train(train_loader, train_loader_ordered,
                              inference_loader_ordered, test_loader, starting_index, inference_sorted)

    elif opt.training_type == "PATE":

        total_evaluator = TrainTargetPATE(
            model=target_model, epochs=opt.epochs, log_path=save_pth)
        total_evaluator.train(train_loader, inference_loader, test_loader)

    else:
        raise ValueError(
            "opt.training_type should be Normal, LabelSmoothing, AdvReg, DP, MixupMMD, PATE")

    torch.save(target_model.state_dict(),
               os.path.join(save_pth, f"{opt.model}.pth"))
    print("Finish Training")

