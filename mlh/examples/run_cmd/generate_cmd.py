import os
import numpy as np
import sys
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../..")
from mlh.utils import generate_save_path


def generate_train_command(params, mode, gpu):
    args_command = "nohup "
    for key, value in params.items():
        args_command += f"{key} {value} " if key == "python" else f"--{key} {value} "
    args_command += f"--mode {mode} "
    args_command += f"--gpu {gpu} "
    args_command += "2>&1 &"
    return args_command.strip()
def generate_train_command_2(params, mode, gpu):
    args_command = ""
    for key, value in params.items():
        if key == "python":
            args_command += f"{key} {value} "
        
        elif value == None:
            args_command += f"--{key} "
        elif isinstance(value, list):
            value = ' '.join(map(str, value))
            args_command +=f"--{key} {value} "
        else: args_command +=f"--{key} {value} "
    args_command += f"--mode {mode} "
    args_command += f"--gpu {gpu} "
    return args_command.strip()

def generate_mia_command(params, mia ="mia_example_only_target.py", attack_type ='metric-based', inference_type = "reference_attack",nohup = True, gpu =0, store =True):
    if nohup:
        args_command = "nohup"
    else: args_command = ""
    for key, value in params.items():
        if key == "python":
            args_command += f"{key} {mia} "
        
        elif value == None:
            args_command += f"--{key} "
        elif value is True:
            args_command += f"--{key} "
        elif value is False:
            continue
        elif isinstance(value, list):
            value = ' '.join(map(str, value))
            args_command +=f"--{key} {value} "
        else: args_command +=f"--{key} {value} "
        
    args_command += f"--attack_type {attack_type} "
    args_command += f"--gpu {gpu} "
    args_command += f"--inference_type {inference_type} "
    
    save_pth = generate_save_path(params, 'target')
    if store:
        if "specific_path" in params.keys():
            save_pth = params["load_model_path"]
        args_command += f"> {save_pth}/mia_{attack_type}.log"
    else: 
        return args_command.strip()
    return args_command.strip()
def generate_cmd(params, gpu0 =0, gpu1 =1):
    cmd1 = generate_train_command(params, mode = "target", gpu=gpu0)
    cmd2 = generate_train_command(params, mode = "shadow", gpu=gpu1)
    
    return cmd1,cmd2

def generate_cmd_hup(params, gpu0 =0, gpu1 =1):
    cmd1 = generate_train_command_2(params, mode = "target", gpu=gpu0)
    cmd2 = generate_train_command_2(params, mode = "shadow", gpu=gpu1)
    
    return cmd1,cmd2