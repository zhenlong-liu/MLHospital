import os
import concurrent.futures
import sys
sys.path.append("..")
sys.path.append("../..")
from run_cmd.generate_cmd import generate_cmd, generate_cmd_hup, generate_mia_command

def run_command(cmd):
    os.system(cmd)

if __name__ == "__main__":
    # lossfunction =["ce","gce","sce"]

    
    params = {
    'python': "../train_target_models_noinference.py",
    "dataset": "CIFAR10",
    "num_class": 10,
    'log_path': '../save_adj', # '../save_p2'
    'training_type': 'NormalLoss',
    'loss_type': 'ce', # concave_log  concave_exp
    'learning_rate': 0.01,
    'epochs': 120, # 100
    "model": "resnet34",  # resnet18 # densenet121 # wide_resnet50 resnet34
    'optimizer' : "sgd",
    'seed' : 0,
    "alpha" : 1,
    "tau" : 1,
    #'scheduler' : 'multi_step_wide_resnet',
    "temp" : 1,
    'batch_size' : 128,
    "num_workers" : 8,
    "loss_adjust" : None,
    "gamma" :1.0,
    }
    os.environ['MKL_THREADING_LAYER'] = 'GNU' 
    
    
    
    gg = [1]
    aa = [1]
    tt = [1]
    uu = [1]
    # ss = [0.1,1,10,100] # beta nce
    # ss = [0.02,0.05,0.1,0.2,0.4] # sce
    # ss =  [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8] # gce
    aa = [0.01, 0.1, 0.5, 1, 5, 10] # sce cifar10
    tt = [0.01, 0.1, 0.5, 1, 5, 10] # sce cifar10
    
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
    
    
    # aa = [0.1]  # concave
    # tt = [0.1] # concave
    #gg = [0.5,1,3] # concave
    lossfunction =["sce"]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor1, concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor2:
        futures = []
        for temp in tt:
            for alpha in aa:
                for gamma in gg:
                    for tau in uu:
                        for loss in lossfunction:
                            params['loss_type'] = loss
                            params["alpha"] = alpha
                            params["temp"] = temp
                            params["gamma"] = gamma
                            params["tau"] = tau
                            cmd1, cmd2 = generate_cmd_hup(params,0,1)
                            
                            
                            futures.append(executor1.submit(run_command, cmd1))
                            futures.append(executor2.submit(run_command, cmd2))
            
        # 等待所有任务完成
        concurrent.futures.wait(futures)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor1, concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor2:
        
        futures = []
        for temp in tt:
            for alpha in aa:
                for gamma in gg:
                    for tau in uu:
                        for loss in lossfunction:
                            params['loss_type'] = loss
                            params["alpha"] = alpha
                            params["temp"] = temp
                            params["gamma"] = gamma
                            params["tau"] = tau
                            cmd3 =generate_mia_command(params, gpu = 0,  nohup = False, mia = "../mia_example_only_target.py")
                            cmd4 = generate_mia_command(params, attack_type= "black-box", gpu = 1,  nohup = False, mia = "../mia_example_only_target.py")
                            
                            futures.append(executor1.submit(run_command, cmd3))
                            futures.append(executor2.submit(run_command, cmd4))

        concurrent.futures.wait(futures)
        # tmux kill-session -t 0
        # tmux new -s <session-name>
        # conda activate mlh
        # cd run_cmd_A100/
        # python run_bash_parameters0917_cifar10_resnet34.py
      