import os
import concurrent.futures
import sys
sys.path.append("..")
sys.path.append("../..")
from generate_cmd import generate_cmd, generate_cmd_hup, generate_mia_command

def run_command(cmd):
    os.system(cmd)

if __name__ == "__main__":
    # lossfunction =["ce","flood","focal","gce","sce","ereg"]

    
    params = {
    'python': "../train_target_models_noinference.py",
    "dataset": "CIFAR100",
    "num_class": 100,
    'log_path': '../save_p4', # '../save_p2'
    'training_type': 'NormalLoss',
    'loss_type': 'mixup_py', # concave_log  concave_exp
    'learning_rate': 0.1,
    'epochs': 150, # 100
    "model": "densenet121",  # resnet18 # densenet121 # wide_resnet50
    'optimizer' : "sgd",
    'seed' : 0,
    "alpha" : 1,
    "tau" : 1,
    #'scheduler' : 'multi_step2',
    "temp" : 1,
    "gamma" :1,
    'batch_size' : 128,
    "num_workers" : 10,
    "loss_adjust" : None,
    }
    os.environ['MKL_THREADING_LAYER'] = 'GNU' 
    # lossfunction =["ce","flood","fo cal","gce","sce","ereg","mae","nceagce","ncemae","ncerce","ngcemae","taylor"]
    # ss = [0.1,1,10,100] # beta nce
    # ss = [0.02,0.05,0.1,0.2,0.4] # sce
    # ss =  [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8] # gce
    #aa = [0.02, 0.1, 0.5, 1, 3, 6, 10] # sce
    # aa =  [0.002, 0.02, 0.2, 1] # nce mae
    # aa = [0.01,0.05,0.1,0.5,1] # gce
    # aa = [0.1 , 1, 5, 10]  # concave
    #tt = [0.1 , 1, 3, 5,  10, 50, 100] # concave
    
    #aa = [0.05, 0.1, 0.5 , 1] #concave
    #tt = [0.05, 0.1, 0.2, 0.5 ,1] #concave
    # gg = [0.1, 0.5, 1, 3, 5, 10, 50, 100] # concave
    #aa = [1]  # concave
    #tt = [6, 8 , 9, 20 , 50] # concave
    #aa = [0.01, 0.02, 0.05, 0.1, 0.5] #concave_exp 
    #tt = [0.01, 0.02, 0.05, 0.1, 0.5] #concave_exp
    #aa = [0.05, 0.1] #concave_exp 
    #tt = [0.005, 0.01,0.02 ,0.05]
    #aa = [0.01, 0.05, 0.1, 1] #concave_log
    
    # tt = [0.01, 0.05, 0.1, 1, 10] #concave_log
    
    
    #aa = [0.05, 0.1]  # concave_loss
    #tt = [0.05, 0.1] # concave_loss
    #gg = [0.05, 0.1, 0.2,0.5,0.9]
    #uu = [0, 1e-7,1e-6,1e-5,1e-4,1e-3,1e-2]
    #gg = [0.5,1,3] # concave_loss
    
    aa = [0.01, 0.05, 0.1, 0.5 , 1]
    tt = [0.05]
    #gg = [0.05]
    #uu = [0.05]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor1, concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor2:
        futures = []
        for temp in tt:
            for alpha in aa:
                for gamma in gg:
                    for tau in uu:
                        # params['loss_type'] = loss
                        params["alpha"] = alpha
                        params["temp"] = temp
                        params["tau"] = tau
                        params["gamma"] = gamma
                        cmd1, cmd2 = generate_cmd_hup(params,5,6)
                        
                        
                        futures.append(executor1.submit(run_command, cmd1))
                        futures.append(executor2.submit(run_command, cmd2))
            

        concurrent.futures.wait(futures)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor1, concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor2:
        
        futures = []
        for temp in tt:
            for alpha in aa:
                for tau in uu:
                    for gamma in gg:
                        params["alpha"] = alpha
                        params["temp"] = temp
                        params["tau"] = tau
                        params["gamma"] = gamma
                        cmd3 =generate_mia_command(params, gpu = 5,  nohup = False, mia = "../mia.py")
                        cmd4 = generate_mia_command(params, attack_type= "black-box", gpu = 6,  nohup = False, mia = "../mia.py")
                        
                        futures.append(executor1.submit(run_command, cmd3))
                        futures.append(executor2.submit(run_command, cmd4))

        concurrent.futures.wait(futures)
        # tmux kill-session -t 0
        # tmux new -s <session-name>
        # conda activate mlh
        # cd mlh/examples/run_cmd
        # python run_bash_parameters0915.py
