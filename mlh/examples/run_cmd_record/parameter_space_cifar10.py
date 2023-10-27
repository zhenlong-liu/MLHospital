

def get_cifar10_parameter_set(loss_type, method ="NormalLoss", dataset = "cifar10", model ="resnet34"):
    
    ce_param = {
        "alpha": [1],
        "temp": [1],
        "tau": [1],
        "gamma": [1],
    }
    
    
    ncemae_param ={
        "alpha": [0.1, 0.5,1, 5, 10],
        "temp": [0.1, 0.5,1, 5,10],
        "tau": [1],
        "gamma": [1],
    }

    gce_param = {
        "alpha": [0.01, 0.1, 1, 10],
        "temp": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        "tau": [1],
        "gamma": [1],
    }

    concave_exp_param = {
        "alpha":[0.01, 0.1, 0.5, 1, 5],
            #[0.01, 0.1, 0.5, 1, 5],
        "temp":[0.01, 0.1, 0.5, 1, 5, 10],
            #[0.01, 0.02, 0.05, 0.08, 0.1, 0.2, 0.5], 
            #[0.01, 0.1, 0.5, 1, 5, 10],
        "tau": [1],
        "gamma": [1],
        # "gamma": [0.2, 0.4, 0.8, 1, 2, 3, 4],
    }

    concave_log_param = {
        #"alpha": [0.01, 0.1, 0.5, 1, 5],
        "alpha": [2,4,6, 10],
        "temp": [0.01, 0.05, 0.1, 0.5, 1],
        "tau": [1],
        "gamma": [0, 0.01, 0.1, 1, 2, 4, 8],
    }

    sce_param = {
        "alpha": [0.5],
        "temp": [12,16,32,50,64,128],
        "tau": [1],
        "gamma": [1],
    }

    flood_param = {
        "alpha": [1],
        "temp": [0.4, 0.8, 1, 1.6, 2, 3.2],
        # [0.01, 0.02, 0.04, 0.06, 0.08, 0.1,0.12, 0.14, 0.16, 0.18, 0.2],
        "tau": [1],
        "gamma": [1],
    }

    taylor_param = {
        "alpha": [1, 2, 3, 4, 5, 6, 7, 8],
        "temp": [1],
        "tau": [1],
        "gamma": [1],
    }

    mixup_py_param = {
        "alpha": [0.2, 0.5, 0.6, 0.8 ],
        "temp": [0.005, 0.01, 0.05, 0.1, 1, 5, 10, 20, 64],
        "tau": [1],
        "gamma": [1],
    }

    ce_ls_param ={
        "alpha": [1],
        "temp": [0.01, 0.05,0.95,0.98, 0.99,0.999],
        #[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
        "tau": [1],
        "gamma": [1],
    }

    ereg_param ={
        "alpha": [16,32,50,64,80,128],
        # [0.05,0.1,0.2,0.4,0.8,1,2,4,8]
        "temp": [1],
        "tau": [1],
        "gamma": [1],
    }

    focal_param ={
        "alpha": [1],
        "temp": [1],
        "tau": [1],
        "gamma": [16,32,64,128,256,512,1024],
    }

    phuber_param ={
        "alpha": [1],
        "temp": [1],
        "tau": [0.5,1,2,4,8,16,32,64],
        "gamma": [1],
    }
    advreg_param = {
        "alpha": [1],
        "temp": [1],
        "tau": [0.01,0.1,0.8,1,1.2,1.4,1.6,1.8],
        "gamma": [1]}
    
    kd_param ={
        "alpha": [1],
        "temp": [1],
        "tau": [0.1,0.3,0.5,0.7,0.9],
        "gamma": [1]
    }
    
    mixupmmd_param ={
        "alpha": [1],
        "temp": [1],
        #"tau": [3],
        "tau": [0.01, 0.02,0.05 ,0.1, 0.2, 0.5,1,2,4,8],
        "gamma": [1]
    }
    relaxloss_param = {
        "alpha": [0.001,0.005, 0.01, 0.04,0.1,0.16, 0.2, 0.4, 0.8,1.6,3.2],
        "temp": [1],
        #"tau": [3],
        "tau": [1],
        "gamma": [1]
    }
    concave_exp_one_param = {
        "alpha": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        "temp": [0.01,0.03, 0.05, 0.1],
        #"tau": [3],
        "tau": [1],
        "gamma": [1]
    }
    dpsgd_param = {
        "alpha": [0.1, 0.2,0.4,0.8,1.6,3.2,6.4,12.8],#[0.1, 0.2,0.4,0.8,1.6,3.2,6.4,12.8],
        "temp": [1],
        #"tau": [3],
        "tau": [1],
        "gamma": [1]
    }
    
    dropout_param = {
        "alpha": [1],
        "temp": [1],
        #"tau": [3],
        "tau": [0.01, 0.02,0.05 ,0.1, 0.3, 0.5, 0.7, 0.9],
        "gamma": [1]
    } 
    
    early_param = {
        "alpha": [1],
        "temp": [1],
        "tau": [1],
        #"tau":[25, 50, 75 ,100, 125 ,150 ,175, 200 ,225, 250, 275],
        "gamma": [1],
        "stop_eps":[25, 50, 75 ,100, 125 ,150 ,175, 200 ,225, 250, 275] 
            #["25 50 75 100 125 150 175 200 225 250 275"]
    }
    
    loss_type_param_space = {
        "ce" : ce_param,
        "focal": focal_param,
        "gce": gce_param,
        "sce": sce_param,
        "flood": flood_param,
        "taylor": taylor_param,
        "concave_exp": concave_exp_param,
        "concave_log": concave_log_param,
        "ncemae": ncemae_param,
        "mixup_py": mixup_py_param,
        "ce_ls": ce_ls_param,
        "ereg": ereg_param,
        "phuber":phuber_param,
        "AdvReg":advreg_param,
        "concave_exp_one": concave_exp_one_param,
        "KnowledgeDistillation": kd_param,
        "MixupMMD": mixupmmd_param,
        "RelaxLoss" :relaxloss_param,
        "EarlyStopping": early_param,
        "DPSGD":dpsgd_param,
        "Dropout" : dropout_param,
    }

    
    
    
    
    method_type_param_space = {
        "RelaxLoss" :relaxloss_param,
        "AdvReg":advreg_param,
        "KnowledgeDistillation": kd_param,
        "MixupMMD": mixupmmd_param,
    }
    
    
    
    return loss_type_param_space.get(loss_type) 