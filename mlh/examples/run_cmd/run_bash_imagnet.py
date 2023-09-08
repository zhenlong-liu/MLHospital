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
    lossfunction =["gce", "sce","ce_ls"]
    params_loss = {
    'python': "../train_target_models_noinference.py",
    'log_path': '../save0',
    "dataset": "Imagenet",
    "num_class": 1000,
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
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor1, concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor2:
        futures = []
        
        
        for loss in lossfunction:
            params_loss['loss_type'] = loss
            cmd1, cmd2 = generate_cmd_hup(params_loss, 2, 1)  # 请确保已定义generate_cmd函数和相关参数
            futures.append(executor1.submit(run_command, cmd1))
            futures.append(executor2.submit(run_command, cmd2))
        # 等待所有任务完成
        concurrent.futures.wait(futures)
        
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        futures = []     
        for loss in lossfunction:
            params_loss['loss_type'] = loss
            cmd3 =generate_mia_command(params_loss, nohup = False, mia = "../mia_example_only_target.py", gpu=1)
            futures.append(executor.submit(run_command, cmd3))
        
        concurrent.futures.wait(futures)
        
        
        # python run_bash_imagnet.py