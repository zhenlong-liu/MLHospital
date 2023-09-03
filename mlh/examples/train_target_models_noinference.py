import os
import sys
sys.path.append("..")
sys.path.append("../..")
from utility.main_parse import add_argument_parameter

from defenses.membership_inference.NormalRelaxLoss import TrainTargetNormalRelaxLoss


import torchvision
import utils
from defenses.membership_inference.loss_function import get_loss
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

from models.resnet import resnet20
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from data_preprocessing.data_loader import GetDataLoader
from data_preprocessing.data_loader_target import GetDataLoaderTarget
from torchvision import datasets
import torchvision.transforms as transforms
import argparse
import numpy as np
import torch.optim as optim

# `torch.set_num_threads(1)` is setting the number of OpenMP threads used for parallelizing CPU
# operations to 1. This means that only one thread will be used for CPU operations, which can be
# useful in cases where parallelization may cause issues or when you want to limit the number of
# threads used for performance reasons.
torch.set_num_threads(1)
from utils import add_dropout_to_last_fc_layer, get_target_model, generate_save_path

def parse_args():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--batch_size', type=int, default=512,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=10,
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
    parser.add_argument('--num_class', type=int, default=10,
                        help='number of classes')
    parser.add_argument('--inference-dataset', type=str, default='CIFAR10',
                        help='if yes, load pretrained the attack model to inference')
    parser.add_argument('--data_path', type=str, default='../datasets/',
                        help='data_path')
    parser.add_argument('--input_shape', type=str, default="32,32,3",
                        help='comma delimited input shape input')
    
    
    add_argument_parameter(parser)
    
    
    
    
    args = parser.parse_args()

    args.input_shape = [int(item) for item in args.input_shape.split(',')]
    args.device = 'cuda:%d' % args.gpu if torch.cuda.is_available() else 'cpu'

    return args




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
    s = GetDataLoaderTarget(opt)
    seed = opt.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)#让显卡产生的随机数一致
    torch.cuda.manual_seed_all(seed)    
    
    #split_num = [0.25,0,0.25,0.25,0,0.25]
    target_train_loader, target_test_loader, shadow_train_loader, shadow_test_loader  = s.get_data_supervised_ni(batch_size =opt.batch_size, num_workers =opt.num_workers)


    
    # 选择是训练target model 还是训练shadow model
    if opt.mode == "target":
        train_loader, test_loader = target_train_loader, target_test_loader
        
        # train_loader is a dataloader, using next(), feature shape is [128,3,32,32], label shape [128]
    # 
    elif opt.mode == "shadow":
        train_loader, test_loader = shadow_train_loader, shadow_test_loader
    else:
        raise ValueError("opt.mode should be target or shadow")
    
    temp_save = str(opt.temp).rstrip('0').rstrip('.') if '.' in str(opt.temp) else str(opt.temp)
    target_model = get_target_model(name=opt.model, num_classes=opt.num_class)

    save_pth = generate_save_path(opt)
    #save_pth = f'{opt.log_path}/{opt.dataset}/{opt.model}/{opt.training_type}/{opt.mode}/{opt.loss_type}/epochs{opt.epochs}/seed{seed}/{temp_save}'

    if opt.training_type == "Normal":
        
        total_evaluator = TrainTargetNormal(
            model=target_model, epochs=opt.epochs, log_path=save_pth)
        total_evaluator.train(train_loader, test_loader)
        
    elif opt.training_type == "NormalRelaxLoss":
        
        total_evaluator = TrainTargetNormalRelaxLoss(
            model=target_model, args=opt, train_loader=train_loader, loss_type=opt.loss_type , device= opt.device, epochs=opt.epochs, log_path=save_pth)
        total_evaluator.train(train_loader, test_loader)    
        
    elif opt.training_type == "NormalLoss":
        
        total_evaluator = TrainTargetNormalLoss(
            model=target_model, args=opt, train_loader=train_loader, loss_type=opt.loss_type , device= opt.device, num_classes= opt.num_class, epochs=opt.epochs, log_path=save_pth)
        total_evaluator.train(train_loader, test_loader)

    elif opt.training_type == "Dropout":
        target_model  = add_dropout_to_last_fc_layer(target_model, opt.beta)
        total_evaluator = TrainTargetNormalLoss(
            model=target_model, args=opt, train_loader=train_loader, loss_type=opt.loss_type , device= opt.device, num_classes= opt.num_class, epochs=opt.epochs, log_path=save_pth)
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
            model=target_model, args=opt, log_path=save_pth)
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

