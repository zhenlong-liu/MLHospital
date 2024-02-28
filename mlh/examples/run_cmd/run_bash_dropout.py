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
    # lossfunction =["ce"]
    
    # ss = [0.01, 0.05, 1,  10, 50] # sce
    # ss =  [1] # gce
    # aa = [0.1, 0.5, 1,  5, 10] # sce
    
    # aa =  [10, 50, 100, 150] # gce
    # aa = [50]
    
    aa = [0.1,0.3,0.7,0.9] # drop out rate
    params = {
    'python': "../train_target_models_noinference.py",
    'log_path': '../save_adj',
    # 'data_path' :'../../datasets/',
    "dataset": "CIFAR10", # TinyImagenet # CIFAR10
    "num_class": 10, 
    'training_type': 'Dropout', # NormalLoss
    'loss_type': 'ce',
    'learning_rate': 0.01,
    'epochs': 100,
    "model": "resnet18", # densenet121 # resnet18 # wide_resnet50
    'optimizer' : "sgd",
    # 'scheduler' : 'multi_step2',
    'seed' : 0,
    'temp' : 1,
    'alpha' : 1,
    'batch_size' : 128,
    "num_workers" : 10,
    "beta" : 0
    }

    os.environ['MKL_THREADING_LAYER'] = 'GNU' 
    
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor1, concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor2:
        futures = []
        for alpha in aa:
            params["alpha"] = alpha
            cmd1, cmd2 = generate_cmd_hup(params, 5, 6)
            futures.append(executor1.submit(run_command, cmd1))
            futures.append(executor2.submit(run_command, cmd2))
        concurrent.futures.wait(futures)
    
    
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        futures = [] 
        params['num_workers'] = 2
        params['batch_size'] = 64
        for alpha in aa:
            params["alpha"] = alpha
            cmd3 =generate_mia_command(params, gpu = 5,  nohup = False, mia = "../mia.py")
            cmd4 = generate_mia_command(params, attack_type= "black-box",gpu = 5,  nohup = False, mia = "../mia.py")
            # print(cmd3)
            futures.append(executor.submit(run_command, cmd3))
        
        concurrent.futures.wait(futures)
        # conda activate mlh
        # cd mlh/examples/run_cmd
        # python run_bash_dropout.py
        
        