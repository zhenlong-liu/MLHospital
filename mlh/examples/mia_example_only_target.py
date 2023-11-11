import torchvision
import sys
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../..")
from mlh.utility.main_parse import add_argument_parameter

#sys.path.append('/home/liuzhenlong/MIA/MLHospital/')
#sys.path.append('/home/liuzhenlong/MIA/MLHospital/mlh/')
from defenses.membership_inference.loss_function import get_loss
from attacks.membership_inference.attacks import AttackDataset, BlackBoxMIA, MetricBasedMIA, LabelOnlyMIA
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from data_preprocessing.data_loader import GetDataLoader
from data_preprocessing.data_loader_target import GetDataLoaderTarget
from torchvision import datasets
#from opacus.validators import ModuleValidator
from utils import add_new_last_layer, get_dropout_fc_layers, get_target_model, plot_celoss_distribution_together, plot_entropy_distribution_together, generate_save_path_1, generate_save_path_2, generate_save_path
import torchvision.transforms as transforms
import argparse
import numpy as np
import torch.optim as optim
import os
torch.set_num_threads(1)
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import yaml
from ruamel.yaml import YAML
from ruamel.yaml.scalarfloat import ScalarFloat
"""
torch.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed(0)#让显卡产生的随机数一致
torch.cuda.manual_seed_all(0)#多卡模式下，让所有显卡生成的随机数一致？这个待验证
"""



def parse_args():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--batch_size', type=int, default=512,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=10,
                        help='num of workers to use')

    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')
    parser.add_argument('--gpu', type=int, default=5,
                        help='gpu index used for training')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--load-pretrained', type=str, default='no')
    parser.add_argument('--dataset', type=str, default='CIFAR10',
                        help='dataset')
    parser.add_argument('--num_class', type=int, default=10,
                        help='number of classes')
    parser.add_argument('--training_type', type=str, default="Normal",
                        help='Normal, LabelSmoothing, AdvReg, DP, MixupMMD, PATE')
    #--training type there is used for specifying path to load model
    parser.add_argument('--inference-dataset', type=str, default='CIFAR10',
                        help='if yes, load pretrained attack model to inference')
    parser.add_argument('--attack_type', type=str, default='black-box',
                        help='attack type: "black-box", "black-box-sorted", "black-box-top3", "metric-based", and "label-only"')
    parser.add_argument('--data_path', type=str, default='../datasets/',
                        help='data_path')
    parser.add_argument('--input-shape', type=str, default="32,32,3",
                        help='comma delimited input shape input')
    
    
    add_argument_parameter(parser)
    
    args = parser.parse_args()
    args.input_shape = [int(item) for item in args.input_shape.split(',')]
    args.device = 'cuda:%d' % args.gpu if torch.cuda.is_available() else 'cpu'

    return args


def check_loss_distr(save_path,file_name= "loss_distribution.yaml"):
    yaml_2 = YAML()
    loss_distr = f"{save_path}/{file_name}"
    if os.path.exists(loss_distr):
        with open(loss_distr, 'r') as f:
            #distribution = yaml.safe_load(f)
            distribution = yaml_2.load(f)
            return not isinstance(distribution["loss_train_mean"], ScalarFloat)
    else:
        return True
    
    
import torch

def get_image_shape(dataloader):
    try:
        # Get a batch of data
        data, labels = next(iter(dataloader))
        
        # Check the shape of a single image
        image_shape = data[0].shape
        
        return image_shape
    except StopIteration:
        print("Data loader is empty or exhausted.")
        return None






if __name__ == "__main__":

    args = parse_args()
    s = GetDataLoaderTarget(args)
    device = args.device
    
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)#让显卡产生的随机数一致
    torch.cuda.manual_seed_all(seed)
    
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)#让显卡产生的随机数一致
    torch.cuda.manual_seed_all(seed)    
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    
    if args.inference:
        target_train_loader, target_test_loader, _,shadow_train_loader, shadow_test_loader  = s.get_data_supervised_inference(batch_size =args.batch_size, num_workers =args.num_workers)
        
    else:
        target_train_loader, target_test_loader, shadow_train_loader, shadow_test_loader  = s.get_data_supervised_ni(batch_size =args.batch_size, num_workers =args.num_workers)
    #target_train_loader, target_test_loader, shadow_train_loader, shadow_test_loader = s.get_data_supervised_ni()

    target_model = get_target_model(name= args.model, num_classes=args.num_class)
    

    
    if args.training_type == "Dropout":
        target_model = get_target_model(name=args.model, num_classes=args.num_class, dropout = args.tau)
        shadow_model = get_target_model(name= args.model, num_classes=args.num_class, dropout = args.tau)
    else: 
        target_model = get_target_model(name=args.model, num_classes=args.num_class)
        shadow_model = get_target_model(name= args.model, num_classes=args.num_class)
    
    

    
    
    
    
    temp_save = str(args.temp).rstrip('0').rstrip('.') if '.' in str(args.temp) else str(args.temp)

    
    if args.specific_path:
        load_path_target = f"{args.load_model_path}/{args.model}.pth"
        load_path_shadow = load_path_target.replace("/target/", "/shadow/")
        save_path = args.load_model_path
    elif args.training_type == "DPSGD":
        # load_path_target = f'{generate_save_path(args, mode = "target")}/{args.model}.pth'
        # load_path_shadow = f'{generate_save_path(args, mode = "shadow")}/{args.model}.pth'
        
        load_path_target = f'{generate_save_path(args, mode = "target")}/{args.model}.pt'
        load_path_shadow = f'{generate_save_path(args, mode = "shadow")}/{args.model}.pt'
        save_path = generate_save_path(args, mode = "target")
        
    else:
        load_path_target = f'{generate_save_path(args, mode = "target")}/{args.model}.pth'
        load_path_shadow = f'{generate_save_path(args, mode = "shadow")}/{args.model}.pth'
        save_path = generate_save_path(args, mode = "target")
    # load target/shadow model to conduct the attacks
    
    if args.training_type == "DPSGD":
        
        #target_model = ModuleValidator.fix(target_model)
        #shadow_model = ModuleValidator.fix(shadow_model)
        
        # target_model.load_state_dict(torch.load(load_path_target, map_location=args.device))
        # shadow_model.load_state_dict(torch.load(load_path_shadow, map_location=args.device))
        target_model = torch.load(load_path_target)
        shadow_model = torch.load(load_path_shadow)
    else:
        target_model.load_state_dict(torch.load(load_path_target, map_location=args.device))
        shadow_model.load_state_dict(torch.load(load_path_shadow, map_location=args.device))
    

    target_model = target_model.to(args.device)
    target_model.eval()
    shadow_model = shadow_model.to(args.device)
    shadow_model.eval()
    # generate attack dataset
    # or "black-box, black-box-sorted", "black-box-top3", "metric-based", and "label-only"
    attack_type = args.attack_type

    # save_path = f'{args.log_path}/{args.dataset}/{args.model}/{args.training_type}/target/{args.loss_type}/epochs{args.epochs}/seed{seed}/{temp_save}'
    
    
    # check whether there exits loss distribtion
    if check_loss_distr:
        plot_entropy_distribution_together(target_train_loader, target_test_loader, target_model, save_path, device)

        plot_celoss_distribution_together(target_train_loader, target_test_loader, target_model, save_path, device)
    
    # attack_type = "metric-based"
    input_shape = get_image_shape(target_train_loader)
    if attack_type == "label-only":
        attack_model = LabelOnlyMIA(
            device=args.device,
            target_model=target_model.eval(), # 打开eval()模式
            shadow_model=shadow_model.eval(),
            save_path = save_path,
            target_loader=(target_train_loader, target_test_loader),
            shadow_loader=(shadow_train_loader, shadow_test_loader),
            input_shape=input_shape,
            nb_classes=args.num_class)
        auc = attack_model.Infer()
        print(auc)

    else:
        attack_dataset = AttackDataset(args, attack_type, target_model, shadow_model,
                                        target_train_loader, target_test_loader, shadow_train_loader, shadow_test_loader)


        # train attack model

        if "black-box" in attack_type:
            attack_model = BlackBoxMIA(
                num_class=args.num_class,
                device=args.device,
                attack_type=attack_type,
                attack_train_dataset=attack_dataset.attack_train_dataset,
                attack_test_dataset=attack_dataset.attack_test_dataset,
                save_path = save_path,
                batch_size=128)
        elif "metric-based" in attack_type:

            attack_model = MetricBasedMIA(
                args = args,
                num_class=args.num_class,
                device=args.device,
                attack_type=attack_type,
                attack_train_dataset=attack_dataset.attack_train_dataset,
                attack_test_dataset=attack_dataset.attack_test_dataset,
                train_loader = target_train_loader,
                save_path = save_path,
                batch_size=128)
        elif "white_box" in attack_type:
            attack_model = MetricBasedMIA(
                args = args,
                num_class=args.num_class,
                device=args.device,
                attack_type=attack_type,
                attack_train_dataset=attack_dataset.attack_train_dataset,
                attack_test_dataset=attack_dataset.attack_test_dataset,
                train_loader = target_train_loader,
                save_path = save_path,
                batch_size=128)