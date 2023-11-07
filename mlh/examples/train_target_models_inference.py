import os
import sys

sys.path.append("..")
sys.path.append("../..")
from utility.main_parse import add_argument_parameter

from defenses.membership_inference.NormalRelaxLoss import TrainTargetNormalRelaxLoss
from mlh.defenses.membership_inference.Mixup_no_inf import TrainTargetMixup
import copy
import torchvision
import utils
from defenses.membership_inference.loss_function import get_loss
from defenses.membership_inference.AdvReg import TrainTargetAdvReg
from defenses.membership_inference.DPSGD import TrainTargetDP # origin
from defenses.membership_inference.DP import TrainTargetDPSGD # new
from defenses.membership_inference.LabelSmoothing import TrainTargetLabelSmoothing
from defenses.membership_inference.MixupMMD import TrainTargetMixupMMD
from defenses.membership_inference.MixupMMDLoss import TrainTargetMixupMMDLoss

from defenses.membership_inference.PATE import TrainTargetPATE
from defenses.membership_inference.Normal import TrainTargetNormal
from defenses.membership_inference.KnowledgeDistillation import TrainTargetKnowledgeDistillation
from defenses.membership_inference.logit_norm import LogitNormLoss

from defenses.membership_inference.LogitClip import TrainTargetLogitClip

from defenses.membership_inference.NormalLoss import TrainTargetNormalLoss
from defenses.membership_inference.EarlyStopping import TrainTargetEarlyStopping
from defenses.membership_inference.RelaxLoss import TrainTargetRelaxLoss

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
from utils import add_new_last_layer, get_dropout_fc_layers, get_target_model, generate_save_path

def parse_args():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--batch_size', type=int, default=128,
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

def load_teacher_model(model,teacher_path ,device):
    """
    Load a teacher model from the given path.
    Returns:
        MyModel: The loaded model.
    """
    # Ensure the model path exists
    if not os.path.exists(teacher_path):
        raise FileNotFoundError(f"No model found at {teacher_path}")
    # Move model to appropriate device
    model_copy = copy.deepcopy(model)
    model_copy = model_copy.to(device)
    # Load the state dict into the model
    model_copy.load_state_dict(torch.load(teacher_path, map_location=device))
    
    # Set the model to evaluation mode
    model_copy.eval()

    return model_copy

def freeze_except_last_layer(model):
    """
    Freeze the parameters of all layers in the model except the last one.

    Args:
        model (nn.Module): The model to freeze the parameters of.
    """
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the last layer
    for param in list(model.children())[-1].parameters():
        param.requires_grad = True


if __name__ == "__main__":
    opt = parse_args()
    seed = opt.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)    
    os.environ['PYTHONHASHSEED'] = str(seed)
    s = GetDataLoaderTarget(opt)
    #split_num = [0.25,0,0.25,0.25,0,0.25]
    
    
    if opt.inference:  
        target_train_loader, target_test_loader, inference_loader,shadow_train_loader, shadow_test_loader  = s.get_data_supervised_inference(batch_size =opt.batch_size, num_workers =opt.num_workers)
        
    else:
        target_train_loader, target_test_loader, shadow_train_loader, shadow_test_loader  = s.get_data_supervised_ni(batch_size =opt.batch_size, num_workers =opt.num_workers)
    #  target model  shadow model
    if opt.mode == "target":
        train_loader, test_loader = target_train_loader, target_test_loader
        
    # train_loader is a dataloader, using next(), feature shape is [128,3,32,32], label shape [128]
    
    elif opt.mode == "shadow":
        train_loader, test_loader = shadow_train_loader, shadow_test_loader
    else:
        raise ValueError("opt.mode should be target or shadow")
    
    temp_save = str(opt.temp).rstrip('0').rstrip('.') if '.' in str(opt.temp) else str(opt.temp)
    
    if opt.training_type == "Dropout":
        target_model = get_target_model(name=opt.model, num_classes=opt.num_class, dropout = opt.tau,finetune= opt.fintune)
    else: 
        target_model = get_target_model(name=opt.model, num_classes=opt.num_class, fintune= opt.finetune)

    if opt.finetune:
        freeze_except_last_layer(target_model)
    
    
    
    
    
    save_pth = generate_save_path(opt)
    #save_pth = f'{opt.log_path}/{opt.dataset}/{opt.model}/{opt.training_type}/{opt.mode}/{opt.loss_type}/epochs{opt.epochs}/seed{seed}/{temp_save}'

    if opt.training_type == "Normal":
        
        total_evaluator = TrainTargetNormal(
            model=target_model, epochs=opt.epochs, log_path=save_pth)
        total_evaluator.train(train_loader, test_loader)
        
    elif opt.training_type == "RelaxLoss":
        
        total_evaluator = TrainTargetRelaxLoss(
            model=target_model, args=opt,log_path=save_pth)
        total_evaluator.train(train_loader, test_loader) 
        """
        total_evaluator = TrainTargetNormalRelaxLoss(
            model=target_model, args=opt, train_loader=train_loader, loss_type=opt.loss_type , device= opt.device, epochs=opt.epochs, log_path=save_pth)
        total_evaluator.train(train_loader, test_loader)    
        """
    elif opt.training_type == "NormalLoss":
        
        total_evaluator = TrainTargetNormalLoss(
            model=target_model, args=opt, log_path=save_pth)
        total_evaluator.train(train_loader, test_loader)

    elif opt.training_type == "Dropout":
        total_evaluator = TrainTargetNormalLoss(
            model=target_model, args=opt, log_path=save_pth)
        total_evaluator.train(train_loader, test_loader)
    
    elif opt.training_type == "KnowledgeDistillation":
        teacher_model = load_teacher_model(target_model, opt.teacher_path, opt.device)
        total_evaluator = TrainTargetKnowledgeDistillation(model= target_model,teacher_model =teacher_model ,args=opt,log_path=save_pth, T= opt.tau)
        
        total_evaluator.train(train_loader, test_loader)
    
    
    elif opt.training_type == "LabelSmoothing":

        total_evaluator = TrainTargetLabelSmoothing(
            model=target_model, epochs=opt.epochs, log_path=save_pth)
        total_evaluator.train(train_loader, test_loader)
    
    elif opt.training_type == "AdvReg":

        total_evaluator = TrainTargetAdvReg(
            model=target_model, args = opt,  log_path=save_pth)
        total_evaluator.train(train_loader, inference_loader, test_loader)
        #model = total_evaluator.model

    # elif opt.training_type == "DP":
    #     total_evaluator = TrainTargetDP(
    #         model=target_model, args=opt, log_path=save_pth)
    #     total_evaluator.train(train_loader, test_loader)
    
    elif opt.training_type == "DPSGD":
        total_evaluator = TrainTargetDPSGD(
            model=target_model, args=opt, log_path=save_pth)
        total_evaluator.train(train_loader, test_loader)

    elif opt.training_type == "Mixup":
        total_evaluator = TrainTargetMixup(
            model=target_model, args=opt, train_loader=train_loader, loss_type=opt.loss_type , device= opt.device, num_classes= opt.num_class, epochs=opt.epochs, log_path=save_pth)
        total_evaluator.train(train_loader, test_loader)
    
    
    
    elif opt.training_type == "MixupMMD":

        target_train_sorted_loader, target_inference_sorted_loader, shadow_train_sorted_loader, shadow_inference_sorted_loader, start_index_target_inference, start_index_shadow_inference, target_inference_sorted, shadow_inference_sorted = s.get_sorted_data_mixup_mmd_one_inference()
        if opt.mode == "target":
            train_loader_ordered, inference_loader_ordered, starting_index, inference_sorted = target_train_sorted_loader, target_inference_sorted_loader, start_index_target_inference, target_inference_sorted

        elif opt.mode == "shadow":
            train_loader_ordered, inference_loader_ordered, starting_index, inference_sorted = shadow_train_sorted_loader, shadow_inference_sorted_loader, start_index_shadow_inference, shadow_inference_sorted

        total_evaluator = TrainTargetMixupMMDLoss(
            model=target_model, args=opt, log_path=save_pth)
        total_evaluator.train(train_loader, train_loader_ordered,
                              inference_loader_ordered, test_loader, starting_index, inference_sorted)

    elif opt.training_type == "PATE":

        total_evaluator = TrainTargetPATE(
            model=target_model, args = opt, log_path=save_pth)
        total_evaluator.train(train_loader, inference_loader, test_loader)
        
    elif opt.training_type == "EarlyStopping":
        total_evaluator = TrainTargetEarlyStopping(
            model=target_model, args = opt, log_path=save_pth)
        total_evaluator.train(train_loader, test_loader)
        print("Finish Training")
        exit()
        
    else:
        raise ValueError(
            "opt.training_type should be Normal, LabelSmoothing, AdvReg, DP, MixupMMD, PATE")
    
    torch.save(target_model.state_dict(),
               os.path.join(save_pth, f"{opt.model}.pth"))
    print("Finish Training")

