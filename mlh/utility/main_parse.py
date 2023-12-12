import argparse
import torch
import yaml
import os
from datetime import datetime
def add_argument_parameter(parser):
    parser.add_argument('--log_path', type=str,
                        default='./save', help='data_path')
    
    parser.add_argument('--temp', type=float, default=1,
                        help='temperature')
    # 默认储存到save里
    parser.add_argument('--tau', type=float, default=1, help = "logitclip tau")
    
    parser.add_argument('--loss_type', type=str, default="ce", help = "Loss function")
    
    parser.add_argument('--lp', type=int, default=2, help = "lp norm")
    parser.add_argument('--series', type=int, default=2, help = "taylor ce series")
    
    parser.add_argument('--learning_rate', type=float, default=0.01, help = "learning rate")
    
    parser.add_argument('--optimizer', type=str, default="sgd", help = "sgd or adam")
    
    parser.add_argument('--scheduler', type=str, default="cosine", help = "cosine or step")
    
    parser.add_argument('--seed', type=int, default=0, help='seed')
    
    parser.add_argument('--alpha', type=float, default=1, help='adjust parameter')
    
    parser.add_argument('--augment_kwarg_translation', type=int, default=1, help='translation parameter')
    parser.add_argument('--augment_kwarg_rotation', type=int, default=4, help='rotation parameter')
    
    parser.add_argument('--noise_scale', type=float, default=100, help='noise scale for DPSGD')
    parser.add_argument('--beta', type=float, default=0, help='Dropout rate')
    parser.add_argument('--gamma', type=float, default=1, help='adjust parameter')
    parser.add_argument('--loss_adjust', action='store_true', default=False, help='if invoke loss_adj function')
    parser.add_argument('--inference', action='store_true', default=False, help='if spilt inference dataset')
    parser.add_argument('--teacher_path', type=str, default="../save0/CIFAR100/densenet121/NormalLoss/target/ce/epochs150/seed0/1/1/1/1/densenet121.pth", help='teacher path for Knowledge Distillation')
    parser.add_argument('--stop_eps', type=int, nargs='+', help='a list of stop epoches')
    
    parser.add_argument('--checkpoint', action='store_true', default=False, help='if check point')
    
    parser.add_argument('--specific_path', action='store_true', default=False, help='whether load specific path')
    
    parser.add_argument('--load_model_path', type=str, default="../save0/CIFAR100/densenet121/NormalLoss/target/ce/epochs150/seed0/1/1/1/1/densenet121.pth", help='load model path')
    
    parser.add_argument('--finetune', action='store_true', default=False, help='whether finetune')
    
    parser.add_argument('--plot_distribution', action='store_true', default=False, help='whether finetune')

    parser.add_argument('--shadow_split_num', type=int, default=16, help='number of shadow models')

    parser.add_argument('--shadow_datapoint_num', type=int, default=15000, help='number of datapoints for shadow models')


def save_namespace_to_yaml(namespace, output_path):
    """
    Save a Namespace object to a YAML file.

    Args:
        namespace (argparse.Namespace): The Namespace object to be saved.
        output_path (str): The path to the output YAML file.
    """
    # Convert Namespace object to a dictionary
    config_data = vars(namespace)
    output_yaml = f'{output_path}/config.yaml'
    # Create the output directory if it doesn't exist
    output_folder = os.path.dirname(output_yaml)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save the configuration dictionary to a YAML file
    with open(output_yaml, 'w') as yaml_file:
        yaml.dump(config_data, yaml_file, default_flow_style=False)

    print(f"Configuration saved to {output_yaml}")


def save_namespace_to_yaml(namespace, output_path):
    """
    Save a Namespace object to a YAML file.

    Args:
        namespace (argparse.Namespace): The Namespace object to be saved.
        output_path (str): The path to the output YAML file.
    """
    # Convert Namespace object to a dictionary
    if isinstance(namespace, argparse.Namespace):
        config_data = vars(namespace)
    else: config_data = namespace
    current_time = datetime.now()
    config_data['current_time'] = current_time
    #output_yaml = f'{output_path}/config.yaml'
    output_yaml = output_path
    # Create the output directory if it doesn't exist
    output_folder = os.path.dirname(output_yaml)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save the configuration dictionary to a YAML file
    with open(output_yaml, 'w') as yaml_file:
        yaml.dump(config_data, yaml_file, default_flow_style=False)

    print(f"Configuration saved to {output_yaml}")
    
def save_dict_to_yaml(dict, output_path):
    # output_yaml = f'{output_path}/train_log.yaml'
    
    output_yaml = output_path
    # Create the output directory if it doesn't exist
    output_folder = os.path.dirname(output_yaml)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save the configuration dictionary to a YAML file
    with open(output_yaml, 'w') as yaml_file:
        yaml.dump(dict, yaml_file, default_flow_style=False)

    print(f"Training log saved to {output_yaml}")