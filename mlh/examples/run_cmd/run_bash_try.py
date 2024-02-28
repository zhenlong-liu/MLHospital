import os
import concurrent.futures
import sys
sys.path.append("..")
sys.path.append("../..")
from generate_cmd import generate_cmd, generate_cmd_hup, generate_mia_command

def run_command(cmd):
    os.system(cmd)

if __name__ == "__main__":
    #lossfunction =["nceagce","ncemae","ngcemae","ncerce"]

    lossfunction =["ce","ereg","flood","focal","gce","mae","sce","nceagce","ncemae","ngcemae","ncerce"]
    params_temp = {
    'python': "../train_target_models_noinference.py",
    "dataset": "FashionMNIST", # CIFAR10 CIFAR100  FashionMNIST
    "num_class": 10,
    'log_path': '../save0',
    'training_type': 'NormalLoss',
    'loss_type': 'nceagce',
    'learning_rate': 0.01,
    'epochs': 100,
    "model": "resnet18",
    'optimizer' : "sgd",
    'seed' : 0,
    "alpha" : 1,
    "temp" :1,
    #'scheduler' : 'multi_step2',
    }
    os.environ['MKL_THREADING_LAYER'] = 'GNU' 
    
    ss =  [1]
    # aa = [0.02, 0.1, 0.5, 1, 3, 6, 10]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        """
        for loss in lossfunction:
            params_temp["loss_type"] = loss
            cmd1, cmd2 = generate_cmd_hup(params_temp,6,7)
            futures.append(executor.submit(run_command, cmd1))
            futures.append(executor.submit(run_command, cmd2))
        concurrent.futures.wait(futures)
        """
        for loss in lossfunction:
            params_temp["loss_type"] = loss
            cmd3 =generate_mia_command(params_temp, mia ="../mia_example_only_target.py",nohup = False)
            futures.append(executor.submit(run_command, cmd3))
        
        concurrent.futures.wait(futures)
