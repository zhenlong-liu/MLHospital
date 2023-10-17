import os
import concurrent.futures
import sys
sys.path.append("..")
sys.path.append("../..")
from generate_cmd import generate_cmd, generate_cmd_hup, generate_mia_command
from mlh.examples.run_cmd_record.parameter_space_cifar100 import get_cifar100_parameter_set
from run_cmd_record.parameter_space_cifar10 import get_cifar10_parameter_set

from run_cmd_record.record import save_merged_dicts_to_yaml
import GPUtil
import time
def check_gpu_memory():
        """
        Check the GPUs and return the IDs of the first two GPUs with less than 4MB memory used.
        """
        GPUs = GPUtil.getGPUs()
        valid_gpus = [gpu.id for gpu in GPUs if gpu.memoryUsed < 100]
        if len(valid_gpus) >= 2:
            return valid_gpus[:2]
        return None


def run_command(cmd):
    os.system(cmd)

if __name__ == "__main__":
    
    
    params = {
    'python': "../train_target_models_inference.py", # "../train_target_models_noinference.py"
    "dataset": "CIFAR100",
    "num_class": 100,
    'log_path': '../save_adj', # '../save_p2'
    'training_type': 'Dropout', # 
    'loss_type': 'ce', # concave_log  concave_exp
    'learning_rate': 0.1,
    'epochs': 300, # 100
    "model": "densenet121",  # resnet18 # densenet121 # wide_resnet50
    'optimizer' : "sgd",
    'seed' : 0,
    "alpha" : 1,
    "tau" : 1,
    'scheduler' : 'multi_step',
    "temp" : 1,
    'batch_size' : 128,
    "num_workers" : 8,
    "loss_adjust" : None,
    "inference" : None,
    "gamma" :1.
    }
    os.environ['MKL_THREADING_LAYER'] = 'GNU' 
    #"RelaxLoss"
    #["concave_log","mixup_py","concave_exp","focal","ereg","ce_ls","flood","phuber"]
    methods =["MixupMMD", "AdvReg"]
    # ["Dropout", "MixupMMD", "AdvReg", "DPSGD", "RelaxLoss"]
    gpu0 = 3
    gpu1 = 4
    
    
    
    """
    end_time = time.time() + 24*60*60  # 24 hours from now
    found_gpus = False
    while time.time() < end_time:
        gpu_ids = check_gpu_memory()
        if gpu_ids:
            print(f"Found suitable GPUs with IDs: {gpu_ids[0]} and {gpu_ids[1]}")
            found_gpus = True
            break
        time.sleep(10*60)  # Wait for 10 minutes

    if not found_gpus:
        print("Did not find suitable GPUs within 24 hours. Exiting the program.")
        exit()
    #gpu0 = gpu_ids[0]
    #gpu1 = gpu_ids[1]
    """
    
    
    save_merged_dicts_to_yaml(params, methods, "./4090_record", dataset= params.get("dataset"))
    
    
    params_backup = params.copy()
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor1, concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor2:
        futures = []
        for method in methods:
            
            param_dict = get_cifar100_parameter_set(method)
            params["num_workers"] = param_dict.get("num_workers", params_backup["num_workers"])
       
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
            
        # 等待所有任务完成
        concurrent.futures.wait(futures)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor1:
        
        futures = []
        for method in methods:   
            param_dict = get_cifar100_parameter_set(method)
            params["num_workers"] = param_dict.get("num_workers", params_backup["num_workers"])
            for temp in param_dict["temp"]:
                for alpha in param_dict["alpha"]:
                    for gamma in param_dict["gamma"]:
                        for tau in param_dict["tau"]:
                            #params['loss_type'] = loss
                            params['training_type'] = method
                            params["alpha"] = alpha
                            params["temp"] = temp
                            params["gamma"] = gamma
                            params["tau"] = tau
                            
                            cmd3 =generate_mia_command(params, gpu = gpu0,  nohup = False, mia = "../mia_example_only_target.py")
                            cmd4 = generate_mia_command(params, attack_type= "black-box", gpu = gpu1,  nohup = False, mia = "../mia_example_only_target.py")
                            cmd5 = generate_mia_command(params, attack_type= "white_box", gpu = gpu0,  nohup = False, mia = "../mia_example_only_target.py")
                            futures.append(executor1.submit(run_command, cmd3))
                            futures.append(executor1.submit(run_command, cmd4))
                            futures.append(executor1.submit(run_command, cmd5))

        concurrent.futures.wait(futures)
        # tmux kill-session -t 1
        # tmux new -s 1
        # conda activate mlh
        # cd mlh/examples/run_cmd/
        # python run_bash_parameters1017_cifar100_multi_scheduler.py
        