import os
import concurrent.futures
import sys
sys.path.append("..")
sys.path.append("../..")
from run_cmd.generate_cmd import generate_cmd, generate_cmd_hup, generate_mia_command
from mlh.examples.run_cmd_record.parameter_space_cifar10 import get_cifar10_parameter_set
from run_cmd_record.record import save_merged_dicts_to_yaml
import GPUtil
import time
def run_command(cmd):
    os.system(cmd)

def check_gpu_memory():
        """
        Check the GPUs and return the IDs of the first two GPUs with less than 4MB memory used.
        """
        GPUs = GPUtil.getGPUs()
        valid_gpus = [gpu.id for gpu in GPUs if gpu.memoryUsed < 100]
        if len(valid_gpus) >= 2:
            return valid_gpus[:2]
        return None


if __name__ == "__main__":
    
    
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
    
    
    #loss_function =["concave_exp","concave_log","gce","flood","taylor","ce_ls","ereg","focal","ncemae"]
    # "mixup_py","phuber"
    loss_function =["sce","ce_ls","ereg","flood","focal"]
    
    
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
    
    save_merged_dicts_to_yaml(params, loss_function, "./A100_record")
    
    
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor1, concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor2:
        futures = []
        for loss in loss_function:
            
            param_dict = get_cifar10_parameter_set(loss)
            print(param_dict)
            for temp in param_dict["temp"]:
                for alpha in param_dict["alpha"]:
                    for gamma in param_dict["gamma"]:
                        for tau in param_dict["tau"]:
                            
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
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor1, concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor2:
        
        futures = []
        for loss in loss_function:   
            param_dict = get_cifar10_parameter_set(loss)
            for temp in param_dict["temp"]:
                for alpha in param_dict["alpha"]:
                    for gamma in param_dict["gamma"]:
                        for tau in param_dict["tau"]:
                            params['loss_type'] = loss
                            params["alpha"] = alpha
                            params["temp"] = temp
                            params["gamma"] = gamma
                            params["tau"] = tau
                            
                            cmd3 =generate_mia_command(params, gpu = 0,  nohup = False, mia = "../mia_example_only_target.py")
                            #cmd4 = generate_mia_command(params, attack_type= "black-box", gpu = 1,  nohup = False, mia = "../mia_example_only_target.py")
                            
                            futures.append(executor1.submit(run_command, cmd3))
                            #futures.append(executor2.submit(run_command, cmd4))

        concurrent.futures.wait(futures)
        # tmux kill-session -t 0
        # tmux new -s <session-name>
        # conda activate ml-hospital
        # cd mlh/examples/run_cmd_A100/
        # python search_0929_cifar10_resnet34_multi.py
