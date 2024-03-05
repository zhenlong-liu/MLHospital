import os
import sys
sys.path.append("..")
sys.path.append("../..")
from utility.main_parse import add_argument_parameter
from utility.utilis_train import load_teacher_model, freeze_except_last_layer, set_random_seed
from defenses.membership_inference.Mixup_no_inf import TrainTargetMixup
from defenses.membership_inference.AdvReg import TrainTargetAdvReg
from defenses.membership_inference.DP import TrainTargetDPSGD
from defenses.membership_inference.MixupMMDLoss import TrainTargetMixupMMDLoss
from defenses.membership_inference.PATE import TrainTargetPATE
from defenses.membership_inference.Normal import TrainTargetNormal
from defenses.membership_inference.KnowledgeDistillation import TrainTargetKnowledgeDistillation
from defenses.membership_inference.NormalLoss import TrainTargetNormalLoss
from defenses.membership_inference.EarlyStopping import TrainTargetEarlyStopping
from defenses.membership_inference.RelaxLoss import TrainTargetRelaxLoss
import torch
import torch.nn as nn
from data_preprocessing.data_loader_target import BuildDataLoader
import argparse
import numpy as np
torch.set_num_threads(1)
from utils import get_target_model, generate_save_path
def parse_args():
    parser = argparse.ArgumentParser('argument for training')
    add_argument_parameter(parser)
    args = parser.parse_args()
    args.input_shape = [int(item) for item in args.input_shape.split(',')]
    args.device = 'cuda:%d' % args.gpu if torch.cuda.is_available() else 'cpu'
    return args

if __name__ == "__main__":
    opt = parse_args()
    seed = opt.seed
    set_random_seed(seed)
    s = BuildDataLoader(opt)
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
        target_model = get_target_model(name=opt.model, num_classes=opt.num_class, dropout = opt.tau)
    else: 
        target_model = get_target_model(name=opt.model, num_classes=opt.num_class)
    if opt.finetune:
        freeze_except_last_layer(target_model)
    save_pth = generate_save_path(opt)
    #save_pth = f'{opt.log_path}/{opt.dataset}/{opt.model}/{opt.training_type}/{opt.mode}/{opt.loss_type}/epochs{opt.epochs}/seed{seed}/{temp_save}'
    if opt.training_type == "NormalLoss":
        total_evaluator = TrainTargetNormalLoss(
            model=target_model, args=opt, log_path=save_pth)
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

    elif opt.training_type == "Dropout":
        total_evaluator = TrainTargetNormalLoss(
            model=target_model, args=opt, log_path=save_pth)
        total_evaluator.train(train_loader, test_loader)
    
    elif opt.training_type == "KnowledgeDistillation":
        teacher_model = load_teacher_model(target_model, opt.teacher_path, opt.device)
        total_evaluator = TrainTargetKnowledgeDistillation(model= target_model,teacher_model =teacher_model ,args=opt,log_path=save_pth, T= opt.tau)
        
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
        print("Finish Training")
        exit()

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
            "opt.training_type has not been implemented yet")
    
    torch.save(target_model.state_dict(),
               os.path.join(save_pth, f"{opt.model}.pth"))
    print("Finish Training")

