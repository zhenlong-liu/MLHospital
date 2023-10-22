



def get_cifar100_parameter_set(method, loss_type = "ce", dataset = "cifar100", model ="wideresnet"):
    
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
        "alpha": [0.01],
            #[0.01, 0.1, 1],
        "temp": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        "tau": [1],   
        "gamma": [1],
    }
    gce_mixup_param = {
        "alpha": [0.01, 0.1, 1],
        "temp": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        "tau": [1],
        "gamma": [1],
    }
    concave_exp_param = {
        "alpha": [0.02, 0.05, 0.08, 0.1],
            #[0.01,0.02, 0.05, 0.08, 0.1 ],
            #[0.01, 0.05, 0.08,0.1],
        "temp":[0.01, 0.02, 0.05, 0.08, 0.1,],
            #[0.01, 0.02, 0.05, 0.08, 0.1],
            #[0.01, 0.02, 0.05, 0.08, 0.1, 0.12, 0.16, 0.2],
            #[0.01,0.02,0.05,0.08, 0.1],  
        "tau": [1],
        "gamma": [1]
            
            #[0.05,0.1,0.2,0.4,0.8,1.6, 2.5,2.8]
            #[2, 3 ,3.2, 4, 5, 6, 6.4]
        #[1.8, 2, 2.2,2.4,2.6, 2.8, 3 ]
            #[0.05,0.1,0.2,0.4,0.8,1.6,3.2],
    }

    concave_log_param = {
        "alpha": [0.01, 0.02,0.05,0.1],
        "temp": [0.01, 0.02,0.05,0.1],
        "tau": [1],
        "gamma": [0.1,0.5, 1,2, 4, 8, 16],
    }

    sce_param = {
        "alpha": [0.1],
        "temp": [1,2,4,8,16,32],
        # 0.01, 0.02,  0.05, 0.1, 0.2, 0.4, 0.8
        "tau": [1],
        "gamma": [1],
    }

    phuber_param ={
        "alpha": [1],
        "temp": [1],
        "tau": [0.5,1,2,4,8,16,32,64],
        "gamma": [1],
    }
    
    flood_param = {
        "alpha": [1],
        #0.01,0.02,0.04,0.08,0.1,0.16, 0.2
        
        "temp": [0.01,0.02,0.04,0.08,0.1,0.16, 0.2,0.25, 0.3, 0.4,0.8,1.6,3.2, 4, 6.4],
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
        "alpha": [0.01, 0.04, 0.1, 1],
        "temp": [0.01, 0.05, 0.1, 0.5,1, 2, 4, 8, 16 , 32],
        # [ 2, 4, 8, 16 ,20,30,32,40,50,60,64,80,100,128,256]
        "tau": [1],
        "gamma": [1],
    }
    

    ce_ls_param = {
        "alpha": [1],
        #[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],s
        # [0.001,0.01,0.95,0.99,0.999,0.9999]
        "temp": [0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.96,0.97, 0.98, 0.985],
        "tau": [1],
        "gamma": [1],
    }
    
    ereg_param ={
        "alpha": [0.1,0.3,0.5,1,2,4,8,16,32,64,128,256, 512,800, 900, 1024],
        # [0.1,0.3,0.5,1,2,4,8,8,16,20,32,40,50,64,128]
        "temp": [1],
        "tau": [1],
        "gamma": [1],
    }
    focal_param ={
        "alpha": [1],
        "temp": [1],
        "tau": [1],
        "gamma":[2,4,8,16,20,32,40,50,64,128, 256,400, 512,700, 800, 900, 1024],
        # [8,16,20,32,40,50,64,128]
    }
    concave_loss_param ={
        "alpha": [0.01,0.02],
        "temp": [0.01,0.02],
        "tau": [0, 0.001, 0.01, 0.1, 1],
        "gamma": [0.5]
    }
    
    relaxloss_param = {
        "alpha": [0.001,0.005, 0.01, 0.04, 0.1, 0.2, 0.4, 0.8, 1.6, 1.8, 2, 2.4, 3.2],
            #[0.001,0.005, 0.01, 0.04, 0.1, 0.16, 0.2, 0.4, 0.8,1.6, 3.2],
        # [0.01,0.02,0.04,0.08,0.1,0.16, 0.2,0.25, 0.3, 0.4,0.5, 0.8,1.6,3.2],
        "temp": [1],
        "tau": [1],
        "gamma": [1]
    }
    advreg_param = {
        "alpha": [1],
        "temp": [1],
        "tau": [0.01, 0.05, 0.1, 0.8, 1, 1.6],
            #[0.01,0.02,0.05,0.1,0.2,0.4,0.6],
        # "tau": [0.8,1,1.2,1.4,1.6,1.8],
        "gamma": [1],
        "num_workers":1,
    }
    
    kd_param ={
        "alpha": [1],
        "temp": [1],
        "tau": [1,2,5,10,20,50,100],
        "gamma": [1]
    }
    mixupmmd_param ={
        "alpha": [1],
        "temp": [1],
        #"tau": [3],
        "tau": [1e-3,5e-3,16],
            #[1e-5,1e-4,1e-3,5e-3] ,
            #[0.01, 0.02,0.05 ,0.1, 0.2, 0.5,1,2,4,8],
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
        #"tau": [3],
        "tau": [1],
        "gamma": [1]
    }
    
    dpsgd_param= {  
        "alpha": [0.1],
        "temp": [0.5,1,5,10,20,40,80,200, 400, 800, 1600],
        #"tau": [3],
        "tau": [1],
        "gamma": [1]
    }
    
    concave_exp_one_param = {
        "alpha": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        "temp": [0.01,0.02, 0.05, 0.1],
        #"tau": [3],
        "tau": [1],
        "gamma": [1]
    }
    
    
    loss_type_param_space = {
        "ce":ce_param,
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
        "gce_mixup":gce_mixup_param,
        "phuber": phuber_param,
        "RelaxLoss" :relaxloss_param,
        "AdvReg":advreg_param,
        "KnowledgeDistillation": kd_param,
        "MixupMMD": mixupmmd_param,
        "Dropout" : dropout_param,
        "EarlyStopping": early_param,
        "concave_exp_one": concave_exp_one_param,
        "DPSGD" : dpsgd_param,
    }
    
    

    return loss_type_param_space.get(method)