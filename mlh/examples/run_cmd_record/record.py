import copy
import os
import sys
import yaml
from datetime import datetime
sys.path.append("..")
sys.path.append("../..")
from run_cmd_record.parameter_space_cifar10 import get_cifar10_parameter_set
from run_cmd_record.parameter_space_cifar100 import get_cifar100_parameter_set
from run_cmd_record.parameter_space_imagenet import get_imagenet_parameter_set
# mlh.examples.run_cmd_record.

def save_merged_dicts_to_yaml(params, loss_set, root, dataset ="cifar10"):
    """
    Merge two dictionaries, add the current time, and save the result to a YAML file.
    
    Args:
    params (dict): The first dictionary.
    param_dict (dict): The second dictionary.
    root (str): The path where the YAML file will be saved.

    Returns:
    None
    """
    record = copy.deepcopy(params)
    if not os.path.exists(root):
        os.makedirs(root)

    # Get the current time and add it to the dictionary
    current_time = datetime.now()
    record['current_time'] = current_time.strftime('%Y-%m-%d %H:%M:%S')
    
    if isinstance(loss_set[0], tuple):
        for method, loss in loss_set:
            if dataset.lower() == "cifar10":
                record[loss+method] = get_cifar10_parameter_set(method)
                
                if record[loss+method] == None:
                    record[loss+method] = get_cifar10_parameter_set(loss)
            elif dataset.lower() == "cifar100":
                record[loss+method] = get_cifar100_parameter_set(method)
                
                if record[loss+method] == None:
                    record[loss+method] = get_cifar100_parameter_set(loss)
            if dataset.lower() == "imagenet":
                record[loss+method] = get_imagenet_parameter_set(method)
                
                if record[loss+method] == None:
                    record[loss+method] = get_imagenet_parameter_set(loss)
                
                #record[loss] = get_cifar100_parameter_set(loss)
    else:
        for loss in loss_set:
            if dataset.lower() == "cifar10":
                record[loss] = get_cifar10_parameter_set(loss)
            elif dataset.lower() == "cifar100":
                record[loss] = get_cifar100_parameter_set(loss)
    # Define the filename based on the current month and day
    filename = current_time.strftime('%m-%d_%H') + '.yaml'

    # Create the full path for the file
    full_path = root.rstrip('/') + '/' + filename

    # Save the merged dictionary to a YAML file
    with open(full_path, 'w') as file:
        yaml.dump(record, file, default_flow_style=False)