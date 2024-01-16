import copy
import itertools
import os
import concurrent.futures
import sys
sys.path.append("..")
sys.path.append("../..")
from run_cmd.generate_cmd import generate_cmd, generate_cmd_hup, generate_mia_command
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
        valid_gpus = [gpu.id for gpu in GPUs if gpu.memoryUsed < 200]
        if len(valid_gpus) >= 2:
            return valid_gpus[:2]
        return None


def run_command(cmd):
    os.system(cmd)

if __name__ == "__main__":
    
    
    params = {
    'python': "../train_target_models_inference.py", # "../train_target_models_noinference.py"
    "dataset": "CIFAR10",
    "num_class": 10,
    'log_path': "../save_adj", #'../save_300_cosine', # '../save_p2' save_adj
    'training_type': "NoramlLoss", #'EarlyStopping', # 
    'loss_type': 'ce', # concave_log  concave_exp
    'learning_rate': 0.1,
    'epochs': 300, # 100 300
    "model": "resnet34",  # resnet18 # densenet121 # wide_resnet50
    'optimizer' : "sgd",
    'seed' : 0,
    "alpha" : 1,
    "tau" : 1,
    'scheduler' : 'multi_step',
    "temp" : 1,
    'batch_size' : 128,
    "num_workers" : 8,
    "loss_adjust" : None,
    #"inference" : None,
    #"checkpoint":None,
    "gamma" :1,
    #"stop_eps": [25,50, 75, 100, 125, 150, 175, 200, 225, 250, 275]
    "teacher_path": "../save_adj/CIFAR10/resnet34/NormalLoss/target/ce/epochs300/seed0/1/1/1/1/resnet34.pth"
    }
    
    os.environ['MKL_THREADING_LAYER'] = 'GNU' 
    
    
    
    
    params_copy =copy.deepcopy(params)
    methods = [("NormalLoss","ereg"),("NormalLoss","ce_ls")]
    #[("NormalLoss", "concave_exp_one")]
    #[("KnowledgeDistillation","ce")]
    #[("EarlyStopping", "ce")]
    # ("MixupMMD","ce")
    #[("NormalLoss", "concave_exp_one")]
    #methods  = [("AdvReg","concave_exp"),("MixupMMD","concave_exp")]
    #[("NormalLoss", "concave_exp")]
    # ("RelaxLoss","ce")
    #methods =[("RelaxLoss","ce"),("NormalLoss", "concave_exp")]
    #[("EarlyStopping", "ce")] ("RelaxLoss","ce") ()
    #loss_funtion = ["concave_exp"]
    # 
    gpu_list = [0,1]
    gpu_tuple = zip(range(len(gpu_list)),gpu_list)
    gpu_iter = itertools.cycle(gpu_tuple)
    
    
    #print(next(gpu_iter))

    
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
    gpu0 = gpu_ids[0]
    gpu1 = gpu_ids[1]
    """
    
    save_merged_dicts_to_yaml(params, methods, "./A100_record", dataset= params.get("dataset"))
    
    max_workers = 3  # Number of commands per GPU

    ##Create a list of ThreadPoolExecutors, one for each GPU
    executors = [concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) for _ in gpu_list]

    # futures = []


    # for method, loss in methods:
    #     param_dict = get_cifar10_parameter_set(method) or get_cifar10_parameter_set(loss)

    #     all_combinations = itertools.product(
    #         param_dict["temp"], 
    #         param_dict["alpha"], 
    #         param_dict["gamma"], 
    #         param_dict["tau"]
    #     )

    #     for temp, alpha, gamma, tau in all_combinations:
            

    #         params.update({
    #             'training_type': method,
    #             'loss_type': loss,
    #             'alpha': alpha,
    #             'temp': temp,
    #             'gamma': gamma,
    #             'tau': tau,
    #         })
    #         gpu_index0, gpu0 = next(gpu_iter)
    #         gpu_index1, gpu1 = next(gpu_iter)
    #         cmd0, cmd1 = generate_cmd_hup(params,gpu0,gpu1)
    #         #print(type(cmd))
    #         print(cmd0)
    #         futures.append(executors[gpu_index0].submit(run_command, cmd0))
    #         futures.append(executors[gpu_index1].submit(run_command, cmd1))

    # concurrent.futures.wait(futures)
    
    attack_py = "../mia_example_only_target.py" #"../mia_enhance.py"#"../mia_example_only_target.py"
    
    #with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor1:
        
    futures = []
    for method, loss in methods:
        param_dict = get_cifar10_parameter_set(method) or get_cifar10_parameter_set(loss)
        
        all_combinations = itertools.product(
            param_dict["temp"], 
            param_dict["alpha"], 
            param_dict["gamma"], 
            param_dict["tau"], 
        )
        
        for temp, alpha, gamma, tau in all_combinations:
            # gpu_index0, gpu0 = next(gpu_iter)
            # gpu_index1, gpu1 = next(gpu_iter)
            # gpu_index2, gpu2 = next(gpu_iter)
            gpu_index3, gpu3 = next(gpu_iter)
            params.update({
                'training_type': method,
                'loss_type': loss,
                'alpha': alpha,
                'temp': temp,
                'gamma': gamma,
                'tau': tau,
            })
        
        
            if param_dict.get("stop_eps") is not None:
                for epoch in param_dict["stop_eps"]:
                    params["tau"] = epoch

                    # cmd3 =generate_mia_command(params, gpu = gpu0,  nohup = False, mia = attack_py, store=False)
                    # cmd4 = generate_mia_command(params, attack_type= "black-box", gpu = gpu1,  nohup = False, mia = attack_py)
                    # cmd5 = generate_mia_command(params, attack_type= "white_box", gpu = gpu2,  nohup = False, mia = attack_py)
                    cmd6 = generate_mia_command(params, attack_type= "augmentation", gpu = gpu3,  nohup = False, mia = attack_py)
                    # futures.append(executors[gpu_index0].submit(run_command, cmd3))
                    # futures.append(executors[gpu_index1].submit(run_command, cmd4))
                    # futures.append(executors[gpu_index2].submit(run_command, cmd5))
                    futures.append(executors[gpu_index3].submit(run_command, cmd6))
            else:
                params["epochs"] = params_copy["epochs"]
                # cmd3 =generate_mia_command(params, gpu = gpu0,  nohup = False, mia = attack_py, store=False)
                # cmd4 = generate_mia_command(params, attack_type= "black-box", gpu = gpu1,  nohup = False, mia = attack_py)
                # cmd5 = generate_mia_command(params, attack_type= "white_box", gpu = gpu2,  nohup = False, mia = attack_py)
                cmd6 = generate_mia_command(params, attack_type= "augmentation", gpu = gpu3,  nohup = False, mia = attack_py)
                # futures.append(executors[gpu_index0].submit(run_command, cmd3))
                # futures.append(executors[gpu_index1].submit(run_command, cmd4))
                # futures.append(executors[gpu_index2].submit(run_command, cmd5))
                futures.append(executors[gpu_index3].submit(run_command, cmd6))

    concurrent.futures.wait(futures)
    
    # python cmd_cifar10.py