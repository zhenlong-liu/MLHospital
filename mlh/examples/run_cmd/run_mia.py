import concurrent.futures
import sys
sys.path.append('..')
sys.path.append('../..')
sys.path.append('../../..')
import os
from generate_cmd import generate_cmd, generate_cmd_hup, generate_mia_command
from utility.plot import plot_acc_auc, plot_distribution_subplots, plot_scatter_with_lines

from utility.process import df_files, process_files, process_files_yaml

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

if __name__ == "__main__":
    
    params = {
    'python': "../train_target_models_inference.py", # "../train_target_models_noinference.py"
    "dataset": "CIFAR100",
    "num_class": 100,
    'log_path': "../save_adj", #'../save_300_cosine', # '../save_p2' save_adj
    'training_type': "NoramlLoss", #'EarlyStopping', # 
    'loss_type': 'ce', # concave_log  concave_exp
    'learning_rate': 0.1,
    'epochs': 150, # 100 300
    "model": "densenet121",  # resnet18 # densenet121 # wide_resnet50
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
    
    
    os.environ['MKL_THREADING_LAYER'] = 'GNU' 
    gpu0 = 0
    gpu1 = 1
    gpu2 = 2
    root_dir = '../save_adj/CIFAR100/densenet121'
    #'../save_adj/CIFAR100/densenet121'
    #'../save_adj/CIFAR100/densenet121/MixupMMD/target/ce/epochs300/'
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor1, concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor2:
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
                        
                        if data_config["loss_type"] == "concave_exp_one":
                            continue
                        update_dict1_from_dict2(params,data_config)
                        
                        params["specific_path"] = None
                        params["loss_adjust"] = None
                        params["load_model_path"] = subdir
                        
                        
                        cmd3 =generate_mia_command(params, gpu = gpu0,  nohup = False, mia = "../mia_example_only_target.py")
                        cmd4 = generate_mia_command(params, attack_type= "black-box", gpu = gpu1,  nohup = False, mia = "../mia_example_only_target.py")
                        cmd5 = generate_mia_command(params, attack_type= "white_box", gpu = gpu2,  nohup = False, mia = "../mia_example_only_target.py")
                        
                        #print(data_config)
                        #isinstance(x, ScalarFloat)
                        log_file_path_train_log = os.path.join(subdir, "train_log.yaml")
                        if os.path.exists(log_file_path_train_log):
                            with open(log_file_path_train_log, 'r') as f:
                                data_train_log = yaml.safe_load(f)
                        else: continue
                        
                        if data_train_log["Train Acc"] < 100/data_config["num_class"]*1.5:
                            continue
                        mia_yaml = os.path.join(subdir, 'mia_metric_based.yaml')
                        mia_bb_yaml = os.path.join(subdir, 'mia_black_box.yaml')
                        mia_wb_yaml = os.path.join(subdir, 'white_box_grid_attacks.yaml')
                        #print(mia_wb_yaml)
                        mia_loss = os.path.join(subdir, 'loss_distribution.yaml')
                        if os.path.exists(mia_loss):
                            with open(mia_loss, 'r') as f:
                                data_loss_distribution = yaml.safe_load(f)
                        #print(data_loss_distribution["loss_train_mean"])
                        #print(type(data_loss_distribution["loss_train_mean"]))
                        #continue
                        """
                        if not (isinstance(data_loss_distribution["loss_train_mean"], float)):
                            print(isinstance(data_loss_distribution["loss_train_mean"], ScalarFloat))
                            print(isinstance(data_loss_distribution["loss_train_mean"], float))
                            print(type(data_loss_distribution["loss_train_mean"]))
                            print(cmd3)
                            exit()
                        
                        """
                        
                        """
                        if not os.path.exists(mia_yaml):
                            #print(data_train_log["Train Acc"])
                            #print(cmd3)
                            #exit()
                            futures.append(executor1.submit(run_command, cmd3))
                        if not os.path.exists(mia_bb_yaml):
                            #print(cmd4)
                            #exit()
                            futures.append(executor2.submit(run_command, cmd4))
                            
                        """
                        if not os.path.exists(mia_wb_yaml):
                            
                            #print(cmd5)
                            #exit()
                            futures.append(executor1.submit(run_command, cmd5))
                        else:
                            with open(mia_wb_yaml, 'r') as f:
                                mia_wb_log = yaml.safe_load(f)
                                #print(mia_wb_log)
                            if "grid_w_l2_test_acc" not in mia_wb_log.keys():
                                #print(cmd5)
                                
                                futures.append(executor1.submit(run_command, cmd5))
                                
        concurrent.futures.wait(futures)                
        # tmux kill-session -t 1
        # tmux new -s 1
        # conda activate mlh
        # cd mlh/examples/run_cmd/
        # CUDA_VISIBLE_DEVICES=1,3,4 python run_mia.py