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
    # lossfunction =["ce","flood","focal","gce","sce","ereg","mae","ncemae" ] #"nceagce",,"ncerce","ngcemae"]
    lossfunction =["ce","sce","ereg"]
    # lossfunction =["ce"]
    
    # ss = [0.01, 0.05, 1,  10, 50] # sce
    # ss =  [1] # gce
    # aa = [0.1, 0.5, 1,  5, 10] # sce
    
    # aa =  [1] # gce
    params = {
    'python': "../train_target_models_noinference.py",
    'log_path': '../save_adj',
    'data_path' :'../../datasets/',
    "dataset": "TinyImagenet",
    "num_class": 200,
    'training_type': 'NormalLoss',
    'loss_type': 'gce',
    'learning_rate': 0.1,
    'epochs': 100,
    "model":  "resnet50"  ,#"densenet121",
    'optimizer' : "sgd",
    'scheduler' : 'multi_step_imagenet',
    'seed' : 0,
    'temp' : 1,
    'alpha' : 1,
    'batch_size' : 128,
    "num_workers" : 10
}

    os.environ['MKL_THREADING_LAYER'] = 'GNU' 
    
    # """
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = []
        
        
        
        #for temp in ss:
        #   for alpha in aa:
        for loss in lossfunction:
            params['loss_type'] = loss
            
        
            #params["temp"] = temp
            #params["alpha"] = alpha
            cmd1, cmd2 = generate_cmd_hup(params, 6, 7)  # 请确保已定义generate_cmd函数和相关参数
            futures.append(executor.submit(run_command, cmd1))
            futures.append(executor.submit(run_command, cmd2))        
        # 等待所有任务完成
        concurrent.futures.wait(futures)
        
    # """
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        futures = [] 
        params['num_workers'] = 2
        params['batch_size'] = 64
        #for temp in ss:
        #    for alpha in aa:
        for loss in lossfunction:
            params['loss_type'] = loss
            cmd3 =generate_mia_command(params, gpu = 1,  nohup = False, mia = "../mia_example_only_target.py")
            print(cmd3)
            futures.append(executor.submit(run_command, cmd3))
        
        concurrent.futures.wait(futures)
        # conda activate mlh
        # cd mlh/examples/run_cmd
        # python run_bash_tinyimagnet_parameters.py
        
        # python run_bash_tinyimagnet_parameters.py