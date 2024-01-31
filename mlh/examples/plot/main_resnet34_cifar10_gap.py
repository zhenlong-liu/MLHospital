import torchvision
import sys
from attacks.membership_inference.model_loader import ModelLoader
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../..")
from mlh.utility.main_parse import add_argument_parameter
from attacks.membership_inference.data_augmentation_attack import AugemtaionAttackDataset, DataAugmentationMIA
from mlh.attacks.membership_inference.attack_dataset import AttackDataset
from mlh.attacks.membership_inference.black_box_attack import BlackBoxMIA
from mlh.attacks.membership_inference.label_only_attack import LabelOnlyMIA
from mlh.attacks.membership_inference.metric_based_attack import MetricBasedMIA
import torch
from data_preprocessing.data_loader_target import BuildDataLoader
from utils import get_target_model, generate_save_path, plot_celoss_distribution_together
import argparse
import numpy as np
import os
torch.set_num_threads(1)
from ruamel.yaml import YAML
from ruamel.yaml.scalarfloat import ScalarFloat

torch.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)



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
    device = args.device
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)#让显卡产生的随机数一致
    torch.cuda.manual_seed_all(seed)    
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    if args.save_attack_path:
        save_path = args.save_attack_path
    else:
        save_path = generate_save_path(args, mode="target")

    s = BuildDataLoader(args)
    target_model = ModelLoader(args, "target")()
    target_train_loader, target_test_loader, shadow_train_loader, shadow_test_loader  = s.get_data_supervised_ni(batch_size =args.batch_size, num_workers =args.num_workers)
    plot_celoss_distribution_together(target_train_loader,target_test_loader,target_train_loader,save_path=save_path,device= device) 
    
