import sys
sys.path.append('..')
sys.path.append('../..')
sys.path.append('../../..')
import argparse
import torch
#from mlh.examples.mia_example_only_target import parse_args
from mlh.attacks.membership_inference.attack_dataset_muti_shdow_models import AttackDatasetMutiShadowModels
from attacks.membership_inference.enhanced_attack import ReferenceMIA
from attacks.membership_inference.model_loader import ModelLoader, ShadowModelLoader
from mlh.utility.main_parse import add_argument_parameter
from attacks.membership_inference.data_augmentation_attack import AugemtaionAttackDataset, DataAugmentationMIA
from mlh.attacks.membership_inference.attack_dataset import AttackDataset
from mlh.attacks.membership_inference.black_box_attack import BlackBoxMIA
from mlh.attacks.membership_inference.metric_based_attack import MetricBasedMIA
import torch
from data_preprocessing.data_loader_target import BuildDataLoader
from utils import get_target_model, generate_save_path, call_function_from_module
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
from utils import compute_phi_stable
from matplotlib import pyplot as plt
from utils import compute_cross_entropy_losses, compute_phi_stable
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.lines as mlines
import matplotlib.ticker as ticker
plt.rcParams['font.family'] = 'Times New Roman'
from utility.main_parse import add_argument_parameter
def plot_phi_distribution_together(target_train_loader, target_test_loader, target_model, device):    
    train_loss = compute_phi_stable(target_train_loader, target_model, device)
    test_loss = compute_phi_stable(target_test_loader, target_model, device)
    train_mean = np.mean(train_loss)
    train_variance = np.var(train_loss)
    test_mean = np.mean(test_loss)
    test_variance = np.var(test_loss)
    
    
    print(f'Loss: train_mean:{train_mean: .8f} train_variance:{train_variance: .8f} test_mean:{test_mean: .8f} test_variance:{test_variance: .8f}')

    plt.figure(figsize=(8, 6))
    plt.hist(train_loss, bins=50, alpha=0.5, label=f'Train Loss\nMean: {train_mean:.2f}\nVariance: {train_variance:.2f}', color='blue')
    plt.hist(test_loss, bins=50, alpha=0.5, label=f'Test Loss\nMean: {test_mean:.2f}\nVariance: {test_variance:.2f}', color='red')
    plt.xlabel('Loss') 
    plt.ylabel('Frequency')
    plt.title('Loss Distribution for Target Data')
    plt.legend()
    plt.grid(True)
    plt.show()
    save_path = 'loss_distribution_comparison.png'
    # Save the plot to the specified path
    plt.savefig("cifar100_loss_compare.pdf")
    torch.cuda.empty_cache()

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
    args = parser.parse_args(args=[])
    args_dict = vars(args)
    args_dict['input_shape'] = [int(item) for item in args_dict['input_shape'].split(',')]
    args_dict['device'] = 'cuda:%d' % args_dict['gpu'] if torch.cuda.is_available() else 'cpu'
    return args_dict

if __name__ == "__main__":
    args_dict = {
        'python': "../train_models.py" ,
        #"../train_shadow_models.py", # "../train_target_models_noinference.py"
        "data_path": "../../datasets",
        "dataset": "CIFAR100", # purchase texas  
        "num_class": 100,
        'log_path': "../save_adj", #'../save_300_cosine', # '../save_p2' save_adj # ../save_adj/combine
        'training_type': "NoramlLoss", #'EarlyStopping', # 
        'loss_type': 'ce', # concave_log  concave_exp
        'learning_rate': 0.005,
        'epochs': 300, # 100 300
        "model": "densenet121",  # resnet18 # densenet121 # wide_resnet50 resnet34 # TexasClassifier # PurchaseClassifier
        'optimizer' : "sgd",
        'seed' : 0,
        "alpha" : 1,
        "tau" : 1,
        'scheduler' : 'multi_step',
        "temp" : 1,
        'batch_size' : 128,
        "num_workers" : 8,
        "loss_adjust" : None,
        #"inference" : None,
        "gamma" :1,
        "load_model_path" : "/data/home/liuzl/MIA/MLHospital/mlh/examples/save_adj/CIFAR100/densenet121/NormalLoss/target/ce/epochs300/seed0/1/1/1/1/densenet121.pth",
        #"shadow_split_num" : 16,
        #"threshold_function": "gaussian_threshold_func",
        #"fpr_tolerance_rate_list":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
        #"stop_eps": ["25 50 75 100 125 150 175 200 225 250 275"]
        #"teacher_path": "../save_adj/CIFAR100/densenet121/NormalLoss/target/concave_exp_one/epochs300/seed0/0.05/0.7/1/1/densenet121.pth"
        }
    aa = parse_args() 
    aa.update(args_dict)
    args = argparse.Namespace(**aa)
    args.device = 'cuda:0'
    s = BuildDataLoader(args,shuffle=False)
    target_train, target_test, *_ = s.get_data_supervised_inference(batch_size =args.batch_size, num_workers =args.num_workers, if_dataset=True)
    
    target_model = get_target_model(name=args.model, num_classes=args.num_class)
    load_path_target = f"{args.load_model_path}/{args.model}.pth"
    save_path = args.load_model_path
    
    load_path_target ="/data/home/liuzl/MIA/MLHospital/mlh/examples/save_adj/CIFAR100/densenet121/NormalLoss/target/ce/epochs300/seed0/1/1/1/1/densenet121.pth"
    target_model.load_state_dict(torch.load(load_path_target, map_location=args.device))
    plot_phi_distribution_together(target_train, target_test,target_model,save_path=save_path,device= args.device)
    
            
        # tmux kill-session -t 1
        # tmux new -s 3
        # conda activate mlh 
        # cd mlh/examples/where_plot/
        # python main_plot_compare_distribution.py
        # python3 -m pdb -c continue 
        