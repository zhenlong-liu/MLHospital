

import copy
import os
import yaml
from datetime import datetime

def get_cifar10_parameter_set(loss_type, dataset = "cifar10", model ="resnet34"):
    gce_param = {
        "alpha": [0.01, 0.1, 1, 10],
        "temp": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        "tau": [1],
        "gamma": [1],
    }

    concave_exp_param = {
        "alpha": [0.01, 0.1, 0.5, 1, 5],
        "temp": [0.01, 0.1, 0.5, 1, 5, 10],
        "tau": [1],
        "gamma": [0.2, 0.4, 0.8, 1, 2, 3, 4],
    }

    concave_log_param = {
        "alpha": [0.01, 0.1, 0.5, 1, 5],
        "temp": [0.01, 0.05, 0.1, 0.5, 1],
        "tau": [1],
        "gamma": [0, 0.01, 0.1, 1, 2, 4, 8],
    }

    sce_param = {
        "alpha": [0.01, 0.1, 0.5, 1, 5, 10],
        "temp": [0.01, 0.1, 0.5, 1, 5, 10],
        "tau": [1],
        "gamma": [1],
    }

    flood_param = {
        "alpha": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.1, 0.11, 0.12, 0.13, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2],
        "temp": [1],
        "tau": [1],
        "gamma": [1],
    }

    taylor_param = {
        "alpha": [1, 2, 3, 4, 5, 6, 7, 8],
        "temp": [1],
        "tau": [1],
        "gamma": [1],
    }

    loss_type_param_space = {
        "gce": gce_param,
        "sce": sce_param,
        "flood": flood_param,
        "taylor": taylor_param,
        "concave_exp": concave_exp_param,
        "concave_log": concave_log_param,
    }

    return loss_type_param_space.get(loss_type)


def save_merged_dicts_to_yaml(params, loss_set, root):
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

    for loss in loss_set:
        record[loss] = get_cifar10_parameter_set(loss)
    # Define the filename based on the current month and day
    filename = current_time.strftime('%m-%d_%H') + '.yaml'

    # Create the full path for the file
    full_path = root.rstrip('/') + '/' + filename

    # Save the merged dictionary to a YAML file
    with open(full_path, 'w') as file:
        yaml.dump(record, file, default_flow_style=False)