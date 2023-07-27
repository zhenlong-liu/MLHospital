import torchvision

import sys
sys.path.append('/home/liuzhenlong/MIA/MLHospital/')
sys.path.append('/home/liuzhenlong/MIA/MLHospital/mlh/')
from mlh.defenses.membership_inference.loss_function import get_loss
from mlh.attacks.membership_inference.attacks import AttackDataset, BlackBoxMIA, MetricBasedMIA, LabelOnlyMIA
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from mlh.data_preprocessing.data_loader import GetDataLoader
from mlh.data_preprocessing.data_loader_target import GetDataLoaderTarget
from torchvision import datasets

from utils import get_target_model, plot_celoss_distribution_together, plot_entropy_distribution_together
import torchvision.transforms as transforms
import argparse
import numpy as np
import torch.optim as optim
torch.manual_seed(0)
np.random.seed(0)
torch.set_num_threads(1)
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt




def parse_args():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--batch-size', type=int, default=512,
                        help='batch_size')
    parser.add_argument('--num-workers', type=int, default=10,
                        help='num of workers to use')

    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')
    parser.add_argument('--gpu', type=int, default=0,
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
    parser.add_argument('--data-path', type=str, default='../datasets/',
                        help='data_path')
    parser.add_argument('--input-shape', type=str, default="32,32,3",
                        help='comma delimited input shape input')
    parser.add_argument('--log_path', type=str,
                        default='./save', help='')

    parser.add_argument('--temp', type=float, default=1,
                        help='temperature')
    parser.add_argument('--loss_type', type=str, default="ce", help = "Loss function")
    
    parser.add_argument('--lp', type=int, default=2, help = "lp norm")
    parser.add_argument('--series', type=int, default=2, help = "taylor ce series")
    
    parser.add_argument('--optimizer', type=str, default="sgd", help = "sgd or adam")
    
    parser.add_argument('--schedular', type=str, default="cosine", help = "cosine or step")
    args = parser.parse_args()

    
    
    args.input_shape = [int(item) for item in args.input_shape.split(',')]
    args.device = 'cuda:%d' % args.gpu if torch.cuda.is_available() else 'cpu'

    return args




def evaluate(model, dataloader):
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

    args = parse_args()
    s = GetDataLoaderTarget(args)
    device = args.device
    
    target_train_loader, target_test_loader, shadow_train_loader, shadow_test_loader = s.get_data_supervised_ni()

    target_model = get_target_model(name= args.model, num_classes=args.num_class)
    shadow_model = get_target_model(name= args.model, num_classes=args.num_class)

    temp_save = str(args.temp).rstrip('0').rstrip('.') if '.' in str(args.temp) else str(args.temp)

    # load target/shadow model to conduct the attacks
    target_model.load_state_dict(torch.load(
        f'{args.log_path}/{args.dataset}/{args.model}/{args.training_type}/target/{args.loss_type}/{temp_save}/{args.model}.pth'))
    target_model = target_model.to(args.device)

    shadow_model.load_state_dict(torch.load(
        f'{args.log_path}/{args.dataset}/{args.model}/{args.training_type}/shadow/{args.loss_type}/{temp_save}/{args.model}.pth'))
    shadow_model = shadow_model.to(args.device)
    
    # generate attack dataset
    # or "black-box, black-box-sorted", "black-box-top3", "metric-based", and "label-only"
    attack_type = args.attack_type

    save_path = f'{args.log_path}/{args.dataset}/{args.model}/{args.training_type}/target/{args.loss_type}/{temp_save}'
    
    
    plot_entropy_distribution_together(target_train_loader, target_test_loader, target_model, save_path, device)
    
    plot_celoss_distribution_together(target_train_loader, target_test_loader, target_model, save_path, device)
    
    # attack_type = "metric-based"
    
    if attack_type == "label-only":
        attack_model = LabelOnlyMIA(
            device=args.device,
            target_model=target_model.eval(), # 打开eval()模式
            shadow_model=shadow_model.eval(),
            target_loader=(target_train_loader, target_test_loader),
            shadow_loader=(shadow_train_loader, shadow_test_loader),
            input_shape=(3, 32, 32),
            nb_classes=10)
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
                batch_size=128)
