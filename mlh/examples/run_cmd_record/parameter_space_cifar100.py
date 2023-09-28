



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
        "alpha": [0.02],
        "temp": [0.05],
        "tau": [1],
        "gamma": [0.1, 0.2, 0.4, 0.8, 1, 2, 3, 4, 8,16,32],
    }

    concave_log_param = {
        "alpha": [0.01, 0.02],
        "temp": [0.01,0.02],
        "tau": [1],
        "gamma": [1,2, 4, 8, 16, 32],
    }

    sce_param = {
        "alpha": [0.01, 0.1],
        "temp": [0.001, 0.005,  0.05, 0.1],
        "tau": [1],
        "gamma": [1],
    }

    flood_param = {
        "alpha": [0.4,0.8,1.6,3.2, 4, 6.4],
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
        "alpha": [0.04],
        "temp": [ 2, 4, 8, 16 ,20,30,32,40,50,60,64,80,100,128],
        "tau": [1],
        "gamma": [1],
    }
    

    ce_ls_param = {
        "alpha": [1],
         #[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],s
        "temp": [0.001,0.01,0.95,0.99,0.999,0.9999],
        "tau": [1],
        "gamma": [1],
    }
    
    ereg_param ={
        "alpha": [8,16,20,32,40,50,64,128], # [0.1,0.3,0.5,1,2,4,8]
        "temp": [1],
        "tau": [1],
        "gamma": [1],
    }
    focal_param ={
        "alpha": [1],
        "temp": [1],
        "tau": [1],
        "gamma": [8,16,20,32,40,50,64,128],
    }
    concave_loss_param ={
        "alpha": [0.01,0.02],
        "temp": [0.01,0.02],
        "tau": [0, 0.001, 0.01, 0.1, 1],
        "gamma": [0.5]
    }
    
    loss_type_param_space = {
        "ce_ls":ce_ls_param,
        "ereg":ereg_param,
        "focal":focal_param,
        "gce": gce_param,
        "sce": sce_param,
        "flood": flood_param,
        "taylor": taylor_param,
        "concave_exp": concave_exp_param,
        "concave_log": concave_log_param,
        "ncemae" : ncemae_param,
        "mixup_py":mixup_py_param,
        "ce_ls": ce_ls_param,
        "ereg": ereg_param,
        "concave_loss":concave_loss_param,
    }
    
    

    return loss_type_param_space.get(loss_type)