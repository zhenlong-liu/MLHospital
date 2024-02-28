import concurrent.futures
import sys
sys.path.append('..')
sys.path.append('../..')
sys.path.append('../../..')
import os
from run_cmd.generate_cmd import generate_cmd, generate_cmd_hup, generate_mia_command
from utility.plot import plot_acc_auc, plot_distribution_subplots, plot_scatter_with_lines

from utility.process import df_files, process_files, process_files_yaml
import itertools
import pandas as pd
import yaml
from ruamel.yaml import YAML
from ruamel.yaml.scalarfloat import ScalarFloat
def run_command(cmd):
    os.system(cmd)

def update_dict1_from_dict2(dict1, dict2):
    for key in dict1:
        if key in dict2:
            dict1[key] = dict2[key]

import GPUtil
import time
def check_gpu_memory():
        """
        Check the GPUs and return the IDs of the first two GPUs with less than 4MB memory used.
        """
        GPUs = GPUtil.getGPUs()
        valid_gpus = [gpu.id for gpu in GPUs if gpu.memoryUsed < 200]
        if len(valid_gpus) >= 2:
            return valid_gpus[:2]
        return None
    
    
if __name__ == "__main__":
    
    
    params = {
    'python': "../train_models.py", # "../train_target_models_noinference.py"
    "dataset": "CIFAR10",
    "num_class": 10,
    'log_path': "../save_adj", #'../save_300_cosine', # '../save_p2' save_adj
    'training_type': "NoramlLoss", #'EarlyStopping', # 
    'loss_type': 'ce', # concave_log  concave_exp
    'learning_rate': 0.1,
    'epochs': 300, # 100 300
    "model": "resnet34",  # resnet18 # densenet121 # wide_resnet50
    'optimizer' : "sgd",
    'seed' : 0,
    "alpha" : 1,
    "tau" : 1,
    #'scheduler' : 'multi_step',
    "temp" : 1,
    'batch_size' : 128,
    "num_workers" : 8,
    "loss_adjust" : None,
    #"inference" : None,
    "gamma" :1,
    #"stop_eps": "25 50 75 100 125 150 175 200 225 250 275"
    }


    toggle_executor = True
    gpu_list = [0,1]
    gpu_tuple = zip(range(len(gpu_list)),gpu_list)
    gpu_iter = itertools.cycle(gpu_tuple)
    root_dir = '../save_adj/CIFAR10/resnet34'
    #'../save_adj/CIFAR100/densenet121'
    #'../save_adj/CIFAR100/densenet121/MixupMMD/target/ce/epochs300/'
    
    
    executors = [concurrent.futures.ThreadPoolExecutor(max_workers=4) for _ in gpu_list]
    
    futures = []
    for subdir, dirs, files in os.walk(root_dir):
        if any(file.endswith('.pth') for file in files):
            for file in files:
                if file == 'config.yaml':
                    log_file_path = os.path.join(subdir, file)
                    if "shadow" in log_file_path:
                        continue
                    with open(log_file_path, 'r') as f:
                        data_config = yaml.safe_load(f)
                        #print(type(data_config))
                    if "gamma" not in data_config.keys():
                        data_config["gamma"] =1
                    
                    
                    
                    if data_config["epochs"] != 300:
                        continue
                    update_dict1_from_dict2(params,data_config)
                    
                    params["specific_path"] = None
                    params["loss_adjust"] = None
                    params["load_model_path"] = subdir
                    
                    
                    
                    # cmd32 = generate_mia_command(params, gpu = gpu1,  nohup = False, mia = "../mia.py")
                    
                    
                    # cmd4 = generate_mia_command(params, attack_type= "black-box", gpu = gpu1,  nohup = False, mia = "../mia.py")
                    # cmd5 = generate_mia_command(params, attack_type= "white_box", gpu = gpu2,  nohup = False, mia = "../mia.py")
                    
                    
                    index, gpu =next(gpu_iter)
                    cmd3 =generate_mia_command(params, gpu = gpu,  attack_type= "augmentation", nohup = False, mia = "../mia.py")
                    #cmd6 = generate_mia_command(params, attack_type= "augmentation", gpu = gpu,  nohup = False, mia = "../mia.py")
                    
                    print(cmd3)
                    # if not data_config["inference"]:
                    #     continue
                    # # else:
                    # #     print(cmd6)
                    # #     continue
                                        
                    
                    
                    #print(data_config)
                    #isinstance(x, ScalarFloat)
                    log_file_path_train_log = os.path.join(subdir, "train_log.yaml")
                    if os.path.exists(log_file_path_train_log):
                        with open(log_file_path_train_log, 'r') as f:
                            data_train_log = yaml.safe_load(f)
                    else: continue

                    
                    if data_train_log["Train Acc"] < 100/data_config["num_class"]*1.5:
                        continue
                    
                    
                    
                    futures.append(executors[index].submit(run_command, cmd3))
                    
                    
    concurrent.futures.wait(futures)                    
    
        # tmux kill-session -t 1
        # tmux new -s 1
        # conda activate mlh
        # cd mlh/examples/mia_lira.py
        # 
        # CUDA_VISIBLE_DEVICES=1,3,4 python mia_lira.py.py