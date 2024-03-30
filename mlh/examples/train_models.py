import os
import sys
import argparse
import numpy as np
sys.path.append("..")
sys.path.append("../..")
from utility.main_parse import add_argument_parameter
from utility.utilis_train import load_teacher_model, freeze_except_last_layer, set_random_seed
torch.set_num_threads(1)

import torch
import torch.nn as nn
from datasets.data_loader import BuildDataLoader
from models.model_utils import get_model

from utils import  generate_save_path
from defenses.membership_inference.NormalLoss import TrainTargetNormal


def parse_args():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--mode', type=str, default="shadow",help='target, shadow')
    parser.add_argument('--load-pretrained', type=str, default= None, help='load the pretrained model')
    parser.add_argument('--task', type=str, default='mia', help='specify the attack task, mia or ol')
    #parser.add_argument('--split_size', type=int, default=5, help='number of data used for training')
    add_argument_parameter(parser) 
    
    args = parser.parse_args()
    args.input_shape = [int(item) for item in args.input_shape.split(',')]
    
    args.device = 'cuda:%d' % args.gpu if torch.cuda.is_available() else 'cpu'
    return args

if __name__ == "__main__":
    opt = parse_args()
    seed = opt.seed
    set_random_seed(seed)
    
    data_loader = BuildDataLoader(opt)
    
    if opt.inference:  
        target_train_loader, target_test_loader, inference_loader, shadow_train_loader, shadow_test_loader  = data_loader.get_split_dataset(batch_size =opt.batch_size,  num_workers =opt.num_workers, split_size=5)
    else:
        target_train_loader, target_test_loader, shadow_train_loader, shadow_test_loader  = data_loader.get_split_dataset(batch_size =opt.batch_size, num_workers =opt.num_workers, split_size=4)
    
    if opt.mode == "target":
        train_loader, test_loader = target_train_loader, target_test_loader    
    elif opt.mode == "shadow":
        train_loader, test_loader = shadow_train_loader, shadow_test_loader
    else:
        raise ValueError("opt.mode should be target or shadow")
    
    temp_save = str(opt.temp).rstrip('0').rstrip('.') if '.' in str(opt.temp) else str(opt.temp)
    
    target_model = get_model(name=opt.model, num_classes=opt.num_class)
    
    save_pth = generate_save_path(opt)
    
    if opt.training_type == "Normal":
        total_evaluator = TrainTargetNormal(
            model=target_model, args=opt, log_path=save_pth)
        total_evaluator.train(train_loader, test_loader)
    else:
        raise ValueError(
            "opt.training_type has not been implemented yet")
    
    torch.save(target_model.state_dict(),
               os.path.join(save_pth, f"{opt.model}.pth"))
    print("Finish Training")

