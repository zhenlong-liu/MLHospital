import os
import concurrent.futures
import sys
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../..")
from generate_cmd import generate_cmd, generate_cmd_hup, generate_mia_command

def run_command(cmd):
    os.system(cmd)

if __name__ == "__main__":
    # lossfunction =["ce","flood","focal","gce","sce","ereg","mae","nceagce","ncemae","ncerce","ngcemae"]
    lossfunction =["ce_ls"]
    params_loss = {
    'python': "../train_target_models_noinference.py",
    "dataset": "FashionMNIST", # CIFAR10 CIFAR100  FashionMNIST
    "num_class": 10,
    'log_path': '../save0',
    'training_type': 'NormalLoss',
    'loss_type': 'ce_ls',
    'learning_rate': 0.01,
    'epochs': 100,
    "model": "resnet18", # re
    'optimizer' : "sgd",
    #'scheduler' : 'multi_step',
    'seed' : 0,
    'temp' : 1,
    'alpha' : 1,
}

    os.environ['MKL_THREADING_LAYER'] = 'GNU' 
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        
        
        for loss in lossfunction:
            params_loss['loss_type'] = loss
            cmd1, cmd2 = generate_cmd_hup(params_loss, 0 , 1)
            futures.append(executor.submit(run_command, cmd1))
            futures.append(executor.submit(run_command, cmd2))
        concurrent.futures.wait(futures)
        
        
        for loss in lossfunction:
            params_loss['loss_type'] = loss
            cmd3 =generate_mia_command(params_loss, nohup = False, mia = "../mia_example_only_target.py")
            print(cmd3)
            futures.append(executor.submit(run_command, cmd3))
        
        concurrent.futures.wait(futures)
        
        
        # python run_bash.py