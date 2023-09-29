import os
import concurrent.futures
import sys
sys.path.append("..")
sys.path.append("../..")
from generate_cmd import generate_cmd, generate_cmd_hup, generate_mia_command
from mlh.examples.run_cmd_record.parameter_space_cifar100 import get_cifar100_parameter_set
from run_cmd_record.parameter_space_cifar10 import get_cifar10_parameter_set

from run_cmd_record.record import save_merged_dicts_to_yaml

def run_command(cmd):
    os.system(cmd)

if __name__ == "__main__":
    
    
    params = {
    'python': "../train_target_models_noinference.py",
    "dataset": "CIFAR100",
    "num_class": 100,
    'log_path': '../save_adj', # '../save_p2'
    'training_type': 'NormalLoss',
    'loss_type': 'ce', # concave_log  concave_exp
    'learning_rate': 0.1,
    'epochs': 200, # 100
    "model": "wide_resnet50",  # resnet18 # densenet121 # wide_resnet50
    'optimizer' : "sgd",
    'seed' : 0,
    "alpha" : 1,
    "tau" : 1,
    'scheduler' : 'multi_step_wide_resnet',
    "temp" : 1,
    'batch_size' : 128,
    "num_workers" : 8,
    "loss_adjust" : None,
    "gamma" :1.
    }
    os.environ['MKL_THREADING_LAYER'] = 'GNU' 
    
    #["concave_log","mixup_py"]
    loss_function =["concave_exp"]
    save_merged_dicts_to_yaml(params, loss_function, "./4090_record", dataset= params.get("dataset"))
    
    
    gpu0 = 7
    gpu1 = 6
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor1, concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor2:
        futures = []
        for loss in loss_function:
            
            param_dict = get_cifar100_parameter_set(loss)
            for temp in param_dict["temp"]:
                for alpha in param_dict["alpha"]:
                    for gamma in param_dict["gamma"]:
                        for tau in param_dict["tau"]:
                            
                            params['loss_type'] = loss
                            params["alpha"] = alpha
                            params["temp"] = temp
                            params["gamma"] = gamma
                            params["tau"] = tau
                            
                            cmd1, cmd2 = generate_cmd_hup(params,gpu0,gpu1)
                            
                            
                            futures.append(executor1.submit(run_command, cmd1))
                            futures.append(executor2.submit(run_command, cmd2))
             
        # 等待所有任务完成
        concurrent.futures.wait(futures)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor1, concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor2:
        
        futures = []
        for loss in loss_function:   
            param_dict = get_cifar100_parameter_set(loss)
            for temp in param_dict["temp"]:
                for alpha in param_dict["alpha"]:
                    for gamma in param_dict["gamma"]:
                        for tau in param_dict["tau"]:
                            params['loss_type'] = loss
                            params["alpha"] = alpha
                            params["temp"] = temp
                            params["gamma"] = gamma
                            params["tau"] = tau
                            
                            cmd3 =generate_mia_command(params, gpu = gpu0,  nohup = False, mia = "../mia_example_only_target.py")
                            cmd4 = generate_mia_command(params, attack_type= "black-box", gpu = gpu1,  nohup = False, mia = "../mia_example_only_target.py")
                            
                            futures.append(executor1.submit(run_command, cmd3))
                            futures.append(executor2.submit(run_command, cmd4))

        concurrent.futures.wait(futures)
        # tmux kill-session -t 0
        # tmux new -s 1
        # conda activate mlh
        # cd mlh/examples/run_cmd/
        # python run_bash_parameters0919_cifar100_wideresnet_multi.py
      