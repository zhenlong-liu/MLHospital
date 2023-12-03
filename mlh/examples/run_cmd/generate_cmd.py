import os
import numpy as np
import sys
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../..")
from mlh.utils import generate_save_path


def generate_train_command(params, mode, gpu):
    # 初始化一个空字符串，用于存储生成的args命令
    args_command = "nohup "

    # 遍历字典中的每个键值对
    for key, value in params.items():
        args_command += f"{key} {value} " if key == "python" else f"--{key} {value} "
    args_command += f"--mode {mode} "
    args_command += f"--gpu {gpu} "
    args_command += "2>&1 &"
    # 返回生成的args命令，去除末尾的多余空格
    return args_command.strip()




def generate_train_command_2(params, mode, gpu):
    # 初始化一个空字符串，用于存储生成的args命令
    args_command = ""

    # 遍历字典中的每个键值对
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
    # 返回生成的args命令，去除末尾的多余空格
    return args_command.strip()

def generate_mia_command(params, mia ="mia_example_only_target.py", attack_type ='metric-based', nohup = True, gpu =0, store =True):
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
    
    save_pth = generate_save_path(params, 'target')
    if store:
        if "specific_path" in params.keys():
            save_pth = params["load_model_path"]
        args_command += f"> {save_pth}/mia_{attack_type}.log"
    else: 
        return args_command.strip()
    return args_command.strip()




"""
def generate_mia_command(params, mia ="mia_example_only_target.py", attack_type ='metric-based', nohup = True, gpu =0):
    # 初始化一个空字符串，用于存储生成的args命令
    log_path = params.get('log_path', './save')
    
    loss_type = params.get('loss_type', 'gce')
    training_type = params.get('training_type', 'NormalLoss')
    
    dataset = params.get("dataset", "CIFAR10")
    temp = params.get('temp', 1)
    model = params.get('model', 'resnet20')
    epochs = params.get("epochs", 100)
    seed = params.get('seed',0)
    num_class = params.get('num_class',10)
    alpha = params.get('alpha',1)
    data_path = params.get('data_path','../datasets/')
    batch_size = params.get('batch_size',128)
    num_workers = params.get('num_workers',2)
    
    # 遍历字典中的每个键值对
    # save_pth = f'{log_path}/{dataset}/{model}/{training_type}/target/{loss_type}/epochs{epochs}/seed{seed}/{temp_save}'
    save_pth = generate_save_path(params, 'target')
    
    
    
    if nohup:
        args_command =f'nohup python {mia} --training_type {training_type} --loss_type {loss_type} --model {model} --log_path {log_path} --seed {seed} \
--epochs {epochs} --temp {temp} --alpha {alpha} --attack_type {attack_type} --dataset {dataset} --num_class {num_class} --gpu {gpu} \
    --data_path {data_path} --num_workers {num_workers} --batch_size {batch_size} > {save_pth}/mia_{attack_type}.log 2>&1 &'
    else: 
        args_command =f'python {mia} --training_type {training_type} --loss_type {loss_type} --model {model} --log_path {log_path} --seed {seed} \
--epochs {epochs} --temp {temp} --alpha {alpha} --attack_type {attack_type} --dataset {dataset} --num_class {num_class} --gpu {gpu} \
    --data_path {data_path} --num_workers {num_workers} --batch_size {batch_size} > {save_pth}/mia_{attack_type}.log'

    # 返回生成的args命令，去除末尾的多余空格
    return args_command.strip()
"""


def generate_cmd(params, gpu0 =0, gpu1 =1):
    cmd1 = generate_train_command(params, mode = "target", gpu=gpu0)
    cmd2 = generate_train_command(params, mode = "shadow", gpu=gpu1)
    
    return cmd1,cmd2

def generate_cmd_hup(params, gpu0 =0, gpu1 =1):
    cmd1 = generate_train_command_2(params, mode = "target", gpu=gpu0)
    cmd2 = generate_train_command_2(params, mode = "shadow", gpu=gpu1)
    
    return cmd1,cmd2