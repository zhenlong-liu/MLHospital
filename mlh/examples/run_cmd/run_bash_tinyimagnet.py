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
    # lossfunction =["ce","sce","ereg"]
    
    # lossfunction =["ce_ls", "gce"]
    lossfunction =["sce","ereg","ce_ls", "gce"]
    params_loss = {
    'python': "../train_target_models_noinference.py",
    'log_path': '../save0',
    'data_path' :'../../datasets/',
    "dataset": "TinyImagenet",
    "num_class": 200,
    'training_type': 'NormalLoss',
    'loss_type': 'ce',
    'learning_rate': 0.1,
    'epochs': 90,
    "model": "densenet121",
    'optimizer' : "sgd",
    'scheduler' : 'multi_step_imagenet',
    'seed' : 0,
    'temp' : 1,
    'alpha' : 1,
    'batch_size' : 128,
    "num_workers" : 10
}

    os.environ['MKL_THREADING_LAYER'] = 'GNU' 
    
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = []
        
        
        for loss in lossfunction:
            params_loss['loss_type'] = loss
            cmd1, cmd2 = generate_cmd_hup(params_loss, 6, 7)  # 请确保已定义generate_cmd函数和相关参数
            if loss == "gce":
                futures.append(executor.submit(run_command, cmd2))
                continue
            elif loss == "ce_ls":
                futures.append(executor.submit(run_command, cmd1))
                continue
            else:
                futures.append(executor.submit(run_command, cmd1))
                futures.append(executor.submit(run_command, cmd2))
        # 等待所有任务完成
        concurrent.futures.wait(futures)
        
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        futures = []
        params_loss['num_workers'] = 2
        params_loss['batch_size'] = 64
        for loss in lossfunction:
            params_loss['loss_type'] = loss
            
            cmd3 =generate_mia_command(params_loss,gpu = 7,  nohup = False, mia = "../mia_example_only_target.py")
            print(cmd3)
            futures.append(executor.submit(run_command, cmd3))
        
        concurrent.futures.wait(futures)
        
        
        # python run_bash_tinyimagnet.py