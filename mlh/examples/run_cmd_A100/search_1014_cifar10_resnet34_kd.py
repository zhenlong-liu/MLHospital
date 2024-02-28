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
        if len(valid_gpus) ==1:
            return [valid_gpus[0], valid_gpus[0]]
        return None


if __name__ == "__main__":
    
    need = 1
    params = {
    'python': "../train_models.py",
    "dataset": "CIFAR10",
    "num_class": 10,
    'log_path': '../save_adj', # '../save_p2'
    'training_type': 'KnowledgeDistillation',
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
    "inference": None,
    "gamma" :1.0,
    "teacher_path": "../save_adj/CIFAR10/resnet34/NormalLoss/target/ce/epochs120/seed0/1/1/1/1/resnet34.pth"
    }
    os.environ['MKL_THREADING_LAYER'] = 'GNU' 
    
    
    #loss_function =["concave_exp","concave_log","gce","flood","taylor","ce_ls","ereg","focal","ncemae", "mixup_py","phuber"]
    # 
    denfense_method =["KnowledgeDistillation"]
    
    end_time = time.time() + 24*60*60  # 24 hours from now
    found_gpus = False
    while time.time() < end_time:
        gpu_ids = check_gpu_memory()
        if gpu_ids:
            print(f"Found suitable GPUs with IDs: {gpu_ids[0]} and {gpu_ids[need-1]}")
            found_gpus = True
            break
        time.sleep(10*60)  # Wait for 10 minutes

    if not found_gpus:
        print("Did not find suitable GPUs within 24 hours. Exiting the program.")
        exit()
    
    save_merged_dicts_to_yaml(params, denfense_method, "./A100_record")
    
    gpu0 = gpu_ids[0]
    gpu1 = gpu_ids[1]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor1, concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor2:
        futures = []
        for method in denfense_method:
            
            param_dict = get_cifar10_parameter_set(method)
            print(param_dict)
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
                            cmd3 =generate_mia_command(params, gpu = gpu0,  nohup = False, mia = "../mia.py")
                            cmd4 = generate_mia_command(params, attack_type= "black-box", gpu = gpu1,  nohup = False, mia = "../mia.py")
                            
                            futures.append(executor1.submit(run_command, cmd3))
                            futures.append(executor2.submit(run_command, cmd4))

        concurrent.futures.wait(futures)
        # tmux kill-session -t 0
        # tmux new -s <session-name>
        # conda activate ml-hospital
        # cd mlh/examples/run_cmd_A100/
        # python search_1014_cifar10_resnet34_kd.py
