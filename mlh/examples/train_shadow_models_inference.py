import os
import sys

sys.path.append("..")
sys.path.append("../..")
from utility.main_parse import add_argument_parameter
from mlh.defenses.membership_inference.Mixup import TrainTargetMixup
import copy
from defenses.membership_inference.AdvReg import TrainTargetAdvReg
from defenses.membership_inference.DP import TrainTargetDPSGD # new
from defenses.membership_inference.MixupMMDLoss import TrainTargetMixupMMDLoss
from defenses.membership_inference.PATE import TrainTargetPATE
from defenses.membership_inference.KnowledgeDistillation import TrainTargetKnowledgeDistillation
from defenses.membership_inference.NormalLoss import TrainTargetNormalLoss
from defenses.membership_inference.EarlyStopping import TrainTargetEarlyStopping
from defenses.membership_inference.RelaxLoss import TrainTargetRelaxLoss
import torch
import torch.nn as nn
from data_preprocessing.data_loader_target import BuildDataLoader
import argparse
import numpy as np

# `torch.set_num_threads(1)` is setting the number of OpenMP threads used for parallelizing CPU
# operations to 1. This means that only one thread will be used for CPU operations, which can be
# useful in cases where parallelization may cause issues or when you want to limit the number of
# threads used for performance reasons.
torch.set_num_threads(1)
from utils import get_target_model, generate_save_path

def parse_args():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=10,
                        help='num of workers to use')

    parser.add_argument('--training_type', type=str, default="Normal",
                        help='Normal, LabelSmoothing, AdvReg, DP, MixupMMD, PATE')
    parser.add_argument('--mode', type=str, default="shadow",
                        help='target, shadow')

    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')
    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu index used for training')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--load-pretrained', type=str, default='no')
    parser.add_argument('--task', type=str, default='mia',
                        help='specify the attack task, mia or ol')
    parser.add_argument('--dataset', type=str, default='CIFAR10',
                        help='dataset')
    parser.add_argument('--num_class', type=int, default=10,
                        help='number of classes')
    parser.add_argument('--inference-dataset', type=str, default='CIFAR10',
                        help='if yes, load pretrained the attack model to inference')
    parser.add_argument('--data_path', type=str, default='../datasets/',
                        help='data_path')
    parser.add_argument('--input_shape', type=str, default="32,32,3",
                        help='comma delimited input shape input')
    add_argument_parameter(parser)
    args = parser.parse_args()
    args.input_shape = [int(item) for item in args.input_shape.split(',')]
    args.device = 'cuda:%d' % args.gpu if torch.cuda.is_available() else 'cpu'

    return args

def load_teacher_model(model,teacher_path ,device):
    """
    Load a teacher model from the given path.
    Returns:
        MyModel: The loaded model.
    """
    # Ensure the model path exists
    if not os.path.exists(teacher_path):
        raise FileNotFoundError(f"No model found at {teacher_path}")
    # Move model to appropriate device
    model_copy = copy.deepcopy(model)
    model_copy = model_copy.to(device)
    # Load the state dict into the model
    model_copy.load_state_dict(torch.load(teacher_path, map_location=device))
    # Set the model to evaluation mode
    model_copy.eval()
    return model_copy

def freeze_except_last_layer(model):
    """
    Freeze the parameters of all layers in the model except the last one.

    Args:
        model (nn.Module): The model to freeze the parameters of.
    """
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False
    # Unfreeze the last layer
    for param in list(model.children())[-1].parameters():
        param.requires_grad = True

if __name__ == "__main__":
    opt = parse_args()
    seed = opt.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    s = BuildDataLoader(opt)
    #split_num = [0.25,0,0.25,0.25,0,0.25]
    if opt.inference:
        s.get_split_shadow_dataset_inference(num_splits=opt.shadow_split_num)
        # generate self.shadow_dataset_list
    else:
        s.get_split_shadow_dataset_ni(num_splits=opt.shadow_split_num)

        # train_loader is a dataloader, using next(), feature shape is [128,3,32,32], label shape [128]
    training_type_to_class = {
        "NormalLoss": TrainTargetNormalLoss,
        "RelaxLoss": TrainTargetRelaxLoss,
        "Dropout": TrainTargetNormalLoss,
        "Mixup": TrainTargetMixup,
        "KnowledgeDistillation": TrainTargetKnowledgeDistillation,
        "AdvReg": TrainTargetAdvReg,
        "DPSGD": TrainTargetDPSGD,
        "MixupMMD": TrainTargetMixupMMDLoss,
        "PATE": TrainTargetPATE,
        "EarlyStopping": TrainTargetEarlyStopping
    }
    for index in range(opt.num_shadow_models):

        #
        if opt.inference:
            inference_loader, train_loader, test_loader = s.get_split_shadow_dataloader_inference(
                batch_size=opt.batch_size, num_workers=opt.num_workers, index=index)
        else:
            train_loader, test_loader = s.get_split_shadow_dataloader_inference(batch_size=opt.batch_size,
                                                                                num_workers=opt.num_workers,
                                                                                 index=index)
        if opt.training_type == "Dropout":
            shadow_model = get_target_model(name=opt.model, num_classes=opt.num_class, dropout=opt.tau)
        else:
            shadow_model = get_target_model(name=opt.model, num_classes=opt.num_class)
        if opt.finetune:
            freeze_except_last_layer(shadow_model)
        save_pth = generate_save_path(opt, f"shadow_{index}")
        evaluator_class = training_type_to_class.get(opt.training_type)

        if evaluator_class:
            if opt.training_type == "KnowledgeDistillation":
                teacher_model = load_teacher_model(shadow_model, opt.teacher_path, opt.device)
                total_evaluator = evaluator_class(model=shadow_model, teacher_model=teacher_model, args=opt, log_path=save_pth, T=opt.tau)
            elif opt.training_type == "MixupMMD":
                sorted_loaders = s.get_sorted_data_mixup_mmd_one_inference()
                mode_loaders = sorted_loaders[0:4] if opt.mode == "target" else sorted_loaders[4:8]
                train_loader_ordered, inference_loader_ordered, starting_index, inference_sorted = mode_loaders
                total_evaluator = evaluator_class(model=shadow_model, args=opt, log_path=save_pth)
                total_evaluator.train(train_loader, train_loader_ordered, inference_loader_ordered, test_loader, starting_index, inference_sorted)
            else:
                total_evaluator = evaluator_class(model=shadow_model, args=opt, log_path=save_pth)
                total_evaluator.train(train_loader, test_loader)
        else:
            raise ValueError("opt.training_type has not been implemented yet")
        if opt.training_type in ["DPSGD", "EarlyStopping"]:
            exit()
        torch.save(shadow_model.state_dict(), os.path.join(save_pth, f"{opt.model}.pth"))
        print("Finish Training")




