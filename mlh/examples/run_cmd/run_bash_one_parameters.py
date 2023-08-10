import os
import concurrent.futures
import sys
sys.path.append("..")
sys.path.append("../..")
from generate_cmd import generate_cmd, generate_cmd_hup, generate_mia_command

def run_command(cmd):
    os.system(cmd)

if __name__ == "__main__":
    lossfunction =["ce","flood","focal","gce","sce","ereg"]

    
    params_temp = {
    'python': "../train_target_models_noinference.py",
    "dataset": "CIFAR100",
    "num_class": 100,
    'log_path': '../save_adj',
    'training_type': 'NormalLoss',
    'loss_type': 'flood',
    'learning_rate': 0.1,
    'epochs': 150,
    "model": "densenet121",
    'optimizer' : "sgd",
    'seed' : 0,
    "alpha" : 1
    #'scheduler' : 'multi_step2',
    }
    os.environ['MKL_THREADING_LAYER'] = 'GNU' 
    
    ss =  [0, 0.1, 0.2, 0.5, 1,2,5]
    # aa = [0.02, 0.1, 0.5, 1, 3, 6, 10]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for temp in ss:
            params_temp["temp"] = temp
            cmd1, cmd2 = generate_cmd_hup(params_temp,4,4)
            futures.append(executor.submit(run_command, cmd1))
            futures.append(executor.submit(run_command, cmd2))

        # 等待所有任务完成
        concurrent.futures.wait(futures)

        for temp in ss:
            params_temp["temp"] = temp
            cmd3 =generate_mia_command(params_temp, nohup = False)
            futures.append(executor.submit(run_command, cmd3))
        
        concurrent.futures.wait(futures)
        
        # python run_bash_one_parameters.py