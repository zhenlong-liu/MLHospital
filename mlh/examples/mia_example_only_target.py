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

from utils import add_new_last_layer, get_dropout_fc_layers, get_target_model, plot_celoss_distribution_together, plot_entropy_distribution_together, generate_save_path_1, generate_save_path_2, generate_save_path
import torchvision.transforms as transforms
import argparse
import numpy as np
import torch.optim as optim

torch.set_num_threads(1)
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

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
    parser.add_argument('--data_path', type=str, default='../datasets/',
                        help='data_path')
    parser.add_argument('--input-shape', type=str, default="32,32,3",
                        help='comma delimited input shape input')
    
    
    add_argument_parameter(parser)
    
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
    
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)#让显卡产生的随机数一致
    torch.cuda.manual_seed_all(seed)    
    
    
    target_train_loader, target_test_loader, shadow_train_loader, shadow_test_loader = s.get_data_supervised_ni()

    target_model = get_target_model(name= args.model, num_classes=args.num_class)
    shadow_model = get_target_model(name= args.model, num_classes=args.num_class)

    if args.training_type == "Dropout":
        new_last_layer = get_dropout_fc_layers(target_model, rate = args.alpha)
        add_new_last_layer(target_model, new_last_layer)
        add_new_last_layer(shadow_model, new_last_layer)
    
    
    temp_save = str(args.temp).rstrip('0').rstrip('.') if '.' in str(args.temp) else str(args.temp)

    # load target/shadow model to conduct the attacks
    target_model.load_state_dict(torch.load(
        f'{generate_save_path(args, mode = "target")}/{args.model}.pth'))
    target_model = target_model.to(args.device)
    target_model.eval()
    shadow_model.load_state_dict(torch.load(
        f'{generate_save_path(args, mode = "shadow")}/{args.model}.pth'))
    shadow_model = shadow_model.to(args.device)
    shadow_model.eval()
    # generate attack dataset
    # or "black-box, black-box-sorted", "black-box-top3", "metric-based", and "label-only"
    attack_type = args.attack_type

    # save_path = f'{args.log_path}/{args.dataset}/{args.model}/{args.training_type}/target/{args.loss_type}/epochs{args.epochs}/seed{seed}/{temp_save}'
    save_path = generate_save_path(args, mode = "target")
    
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
                save_path = save_path,
                batch_size=128)
