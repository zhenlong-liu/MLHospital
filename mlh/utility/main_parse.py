import argparse
import torch
import yaml
import os

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
    
    parser.add_argument('--noise_scale', type=float, default=100, help='noise scale for DPSGD')
    parser.add_argument('--beta', type=float, default=0, help='Dropout rate')
    parser.add_argument('--gamma', type=float, default=1, help='adjust parameter')
    parser.add_argument('--loss_adjust', action='store_true', default=False, help='if invoke loss_adj function')
    
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