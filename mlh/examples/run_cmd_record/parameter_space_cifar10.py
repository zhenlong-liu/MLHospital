

def get_cifar10_parameter_set(loss_type, dataset = "cifar10", model ="resnet34"):
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
        "alpha": [0.01, 0.1, 0.5, 1, 5],
        "temp": [0.01, 0.05, 0.1, 0.5, 1],
        "tau": [1],
        "gamma": [0, 0.01, 0.1, 1, 2, 4, 8],
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

    loss_type_param_space = {
        "gce": gce_param,
        "sce": sce_param,
        "flood": flood_param,
        "taylor": taylor_param,
        "concave_exp": concave_exp_param,
        "concave_log": concave_log_param,
    }

    return loss_type_param_space.get(loss_type)