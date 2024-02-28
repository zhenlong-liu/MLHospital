import os
import concurrent.futures
import sys
sys.path.append("..")
sys.path.append("../..")
from run_cmd.generate_cmd import generate_cmd_hup, generate_mia_command
from mlh.examples.run_cmd_record.parameter_space_cifar100 import get_cifar100_parameter_set
from run_cmd_record.parameter_space_cifar10 import get_cifar10_parameter_set

from run_cmd_record.record import save_merged_dicts_to_yaml

def run_command(cmd):
    os.system(cmd)

if __name__ == "__main__":
    
    
    params = {
    'python': "../train_models.py", # "../train_target_models_noinference.py"
    "dataset": "CIFAR10",
    "num_class": 10,
    'log_path': '../save_adj', # '../save_p2'
    'training_type': 'AdvReg', # 
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
    "num_workers" : 2,
    "loss_adjust" : None,
    "inference" : None,
    "gamma" :1.
    }
    os.environ['MKL_THREADING_LAYER'] = 'GNU' 
    
    #["concave_log","mixup_py","concave_exp","focal","ereg","ce_ls","flood","phuber"]
    denfense_method =["AdvReg", "MixupMMD"]
    save_merged_dicts_to_yaml(params, denfense_method, "./A100_record", dataset= params.get("dataset"))
    
    
    gpu0 = 0
    gpu1 = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor1, concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor2:
        futures = []
        for method in denfense_method:
            
            param_dict = get_cifar10_parameter_set(method)
            for temp in param_dict["temp"]:
                for alpha in param_dict["alpha"]:
                    for gamma in param_dict["gamma"]:
                        for tau in param_dict["tau"]:
                            params['training_type'] = method
                            params["alpha"] = alpha
                            params["temp"] = temp
                            params["gamma"] = gamma
                            params["tau"] = tau
                            
                            cmd1, cmd2 = generate_cmd_hup(params,gpu0,gpu1)
                            futures.append(executor1.submit(run_command, cmd1))
                            futures.append(executor2.submit(run_command, cmd2))
            
     
        concurrent.futures.wait(futures)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor1, concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor2:

        futures = []
        for method in denfense_method:   
            param_dict = get_cifar10_parameter_set(method)
            for temp in param_dict["temp"]:
                for alpha in param_dict["alpha"]:
                    for gamma in param_dict["gamma"]:
                        for tau in param_dict["tau"]:
                            params['training_type'] = method
                            params["alpha"] = alpha
                            params["temp"] = temp
                            params["gamma"] = gamma
                            params["tau"] = tau
                            
                            cmd3 = generate_mia_command(params, gpu = gpu0,  nohup = False, mia = "../mia.py")
                            cmd4 = generate_mia_command(params, attack_type= "black-box", gpu = gpu1,  nohup = False, mia = "../mia.py")
                            
                            cmd5 = generate_mia_command(params, attack_type= "white_box", gpu = gpu0,  nohup = False, mia = "../mia.py")
                            futures.append(executor1.submit(run_command, cmd3))
                            futures.append(executor1.submit(run_command, cmd4))
                            futures.append(executor1.submit(run_command, cmd5))

        concurrent.futures.wait(futures)
        # tmux attach -t 0
        # tmux kill-session -t 0
        # tmux new -s 0
        # conda activate ml-hospital
        # cd mlh/examples/run_cmd_A100/
        # python run_bash_parameters1013_cifar10_advreg.py
        