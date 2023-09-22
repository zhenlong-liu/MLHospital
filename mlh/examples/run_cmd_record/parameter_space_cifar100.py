



def get_cifar100_parameter_set(loss_type, dataset = "cifar100", model ="wideresnet"):
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
        "alpha": [0.01, 0.1, 0.5, 1, 5],
        "temp": [0.01, 0.1, 0.5, 1, 5, 10],
        "tau": [1],
        "gamma": [0.2, 0.4, 0.8, 1, 2, 3, 4],
    }

    concave_log_param = {
        "alpha": [0.01, 0.04, 0.07, 0.1],
        "temp": [0.01, 0.04 , 0.07, 0.1],
        "tau": [1],
        "gamma": [0.6, 0.8, 1, 1.2 , 1.4],
    }

    sce_param = {
        "alpha": [0.01, 0.1, 0.5, 1, 5, 10],
        "temp": [0.01, 0.1, 0.5, 1, 5, 10],
        "tau": [1],
        "gamma": [1],
    }

    flood_param = {
        "alpha": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.1, 0.11, 0.12, 0.13, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2],
        "temp": [1],
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
        "alpha": [0.002, 0.004, 0.006, 0.008,  0.01, 0.02],
        "temp": [0.001, 0.01, 0.05, 0.1, 1, 5, 10],
        "tau": [1],
        "gamma": [1],
    }
    
    concave_loss_param ={
        "alpha": [0.05, 0.01],
        "temp": [0.05, 0.01],
        "tau": [0, 1e-6,1e-4,1e-2],
        "gamma": [0.05, 0.1, 0.5, 0.9, 1, 3],
    }

    loss_type_param_space = { 
        "gce": gce_param,
        "sce": sce_param,
        "flood": flood_param,
        "taylor": taylor_param,
        "concave_exp": concave_exp_param,
        "concave_log": concave_log_param,
        "ncemae" : ncemae_param,
        "mixup_py":mixup_py_param,
        "concave_loss": concave_loss_param,
    }
    
    

    return loss_type_param_space.get(loss_type)