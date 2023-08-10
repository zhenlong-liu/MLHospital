# MIT License

# Copyright (c) 2022 The Machine Learning Hospital (MLH) Authors

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import sys
import numpy as np
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple, Union
from models.resnet import resnet20
import torch.nn as nn
import torchvision

import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import argparse



def dict_str(input_dict):
    for key, value in input_dict.items():
        input_dict[key] = str(value)
    return input_dict
def get_init_args(obj):
    init_args = {}
    for attr_name in dir(obj):
        if not callable(getattr(obj, attr_name)) and not attr_name.startswith("__"):
            init_args[attr_name] = getattr(obj, attr_name)
    return init_args


def generate_save_path_1(opt):
    if isinstance(opt, dict):
        opt = argparse.Namespace(**opt)
    save_path1 = f'{opt.log_path}/{opt.dataset}/{opt.model}/{opt.training_type}'
    return save_path1    

def generate_save_path_2(opt):
    if isinstance(opt, dict):
        opt = argparse.Namespace(**opt)
    #temp_save = str(opt.temp).rstrip('0').rstrip('.') if '.' in str(opt.temp) else str(opt.temp)
    temp_save= standard_float(opt.temp)
    alpha_save = standard_float(opt.alpha)
    #alpha_save = str(opt.alpha).rstrip('0').rstrip('.') if '.' in str(opt.alpha) else str(opt.alpha)
    save_path2 =  f"{opt.loss_type}/epochs{opt.epochs}/seed{opt.seed}/{temp_save}/{alpha_save}"
    return save_path2
        
def standard_float(hyper_parameter):
    return str(hyper_parameter).rstrip('0').rstrip('.') if '.' in str(hyper_parameter) else str(hyper_parameter)
    


def generate_save_path(opt, mode = None):
    if isinstance(opt, dict):
        opt = argparse.Namespace(**opt)
    if mode == None:
        save_pth = f'{generate_save_path_1(opt)}/{opt.mode}/{generate_save_path_2(opt)}'
    else:
        save_pth = f'{generate_save_path_1(opt)}/{mode}/{generate_save_path_2(opt)}'
    return save_pth



def get_optimizer(optimizer_name, model_parameters, learning_rate=0.1, momentum=0.9, weight_decay=1e-4):
    """
    获取指定优化器的实例

    参数：
        optimizer_name (str): 优化器的名称，可以是 'sgd' 或 'adam'.
        model_parameters (iterable): 模型的参数，通常通过 model.parameters() 获得.
        learning_rate (float): 初始学习率 (默认为 0.1).
        momentum (float): SGD优化器的动量参数 (默认为 0.9).
        weight_decay (float): L2正则化项的权重衰减参数 (默认为 1e-4).

    返回：
        torch.optim.Optimizer: 返回所选优化器的实例.
    """
    if optimizer_name.lower() == 'sgd':
        optimizer = optim.SGD(model_parameters, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'adam':
        optimizer = optim.Adam(model_parameters, lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError("'sgd' or 'adam'.")

    return optimizer

import torch.optim.lr_scheduler as lr_scheduler

def get_scheduler(scheduler_name, optimizer, decay_epochs=1, decay_factor=0.1, t_max=50):
    """
    Get the specified learning rate scheduler instance.

    Parameters:
        scheduler_name (str): The name of the scheduler, can be 'step' or 'cosine'.
        optimizer (torch.optim.Optimizer): The optimizer instance.
        decay_epochs (int): Number of epochs for each decay period, used for StepLR scheduler (default is 1).
        decay_factor (float): The factor by which the learning rate will be reduced after each decay period,
                             used for StepLR scheduler (default is 0.1).
        t_max (int): The number of epochs for the cosine annealing scheduler (default is 50).

    Returns:
        torch.optim.lr_scheduler._LRScheduler: The instance of the selected scheduler.
    
    """
    if isinstance(optimizer, torch.optim.Adam):
        return DummyScheduler()
    if scheduler_name.lower() == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=decay_factor)
    elif scheduler_name.lower() == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
    elif scheduler_name.lower() == "multi_step":
        decay_epochs = [150, 225]
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=decay_epochs, gamma=0.1)
    elif scheduler_name.lower() == "multi_step2":
        decay_epochs = [40, 80]
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=decay_epochs, gamma=0.1)
    else:
        raise ValueError("Unsupported scheduler name. Please choose 'step' or 'cosine'.")

    return scheduler

class DummyScheduler:
    def step(self):
        pass
def compute_losses(loader, net,device):
    """Auxiliary function to compute per-sample losses"""

    criterion = nn.CrossEntropyLoss(reduction="none")
    all_losses = []

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        logits = net(inputs)
        losses = criterion(logits, targets).numpy(force=True)
        all_losses.extend(iter(losses))
    return np.array(all_losses)

def cross_entropy(prob, label):
    # 避免概率值为0，加上一个很小的值进行平滑处理
    epsilon = 1e-12

    # 使用np.clip确保概率值不为0或1，以避免log(0)或log(1)出现无效值
    prob = np.clip(prob, epsilon, 1.0 - epsilon)

    # 将label转换为one-hot编码
    one_hot_label = np.zeros_like(prob)
    one_hot_label[np.arange(len(label)), label] = 1

    return -np.sum(one_hot_label * np.log(prob), axis=1)



def calculate_entropy(data_loader, model):
    entropies = []
    model.eval()
    with torch.no_grad():
        for inputs, _ in data_loader:
            outputs = model(inputs)
            
            
            probabilities = F.softmax(outputs, dim=1)
            
            entropy = -(probabilities * torch.log(probabilities + 1e-9)).sum(dim=1)    
            entropies.extend(entropy.cpu().numpy())
            #entropies.append(entropy)
            
    
    return entropies

def plot_entropy_distribution_together(target_train_loader, target_test_loader, target_model, save_path, device):
    # Calculate entropies for target_train_loader and target_test_loader
    target_train_loader = [(data.to(device), target.to(device)) for data, target in target_train_loader]
    target_test_loader = [(data.to(device), target.to(device)) for data, target in target_test_loader]
    
    
    train_entropies = calculate_entropy(target_train_loader, target_model)
    test_entropies = calculate_entropy(target_test_loader, target_model)
    train_mean = np.mean(train_entropies)
    train_variance = np.var(train_entropies)
    test_mean = np.mean(test_entropies)
    test_variance = np.var(test_entropies)
    print(f'Entropies: train_mean:{train_mean: .3f} train_variance:{train_variance: .3f} test_mean:{test_mean: .3f} test_variance:{test_variance: .3f}')
    # Plot the distribution of entropies
    plt.figure(figsize=(8, 6))
    plt.hist(train_entropies, bins=50, alpha=0.5, label=f'Train Entropy\nMean: {train_mean:.2f}\nVariance: {train_variance:.2f}', color='blue')
    plt.hist(test_entropies, bins=50, alpha=0.5, label=f'Test Entropy\nMean: {test_mean:.2f}\nVariance: {test_variance:.2f}', color='red')
    plt.xlabel('Entropy') 
    plt.ylabel('Frequency')
    plt.title('Entropy Distribution for Target Data')
    plt.legend()
    plt.grid(True)
    save_path = f'{save_path}/entropy_distribution_comparison.png'
    # Save the plot to the specified path
    plt.savefig(save_path)
    plt.close()

def plot_celoss_distribution_together(target_train_loader, target_test_loader, target_model, save_path, device):
    # Calculate loss for target_train_loader and target_test_loader
    target_train_loader = [(data.to(device), target.to(device)) for data, target in target_train_loader]
    target_test_loader = [(data.to(device), target.to(device)) for data, target in target_test_loader]
    
    
    train_loss = compute_losses(target_train_loader, target_model,device)
    test_loss = compute_losses(target_test_loader, target_model,device)
    train_mean = np.mean(train_loss)
    train_variance = np.var(train_loss)
    test_mean = np.mean(test_loss)
    test_variance = np.var(test_loss)
    print(f'Loss: train_mean:{train_mean: .3f} train_variance:{train_variance: .3f} test_mean:{test_mean: .3f} test_variance:{test_variance: .3f}')
    # Plot the distribution of entropies

    plt.figure(figsize=(8, 6))
    plt.hist(train_loss, bins=50, range= (0,5),alpha=0.5, label=f'Train Loss\nMean: {train_mean:.2f}\nVariance: {train_variance:.2f}', color='blue')
    plt.hist(test_loss, bins=50,range= (0,5), alpha=0.5, label=f'Test Loss\nMean: {test_mean:.2f}\nVariance: {test_variance:.2f}', color='red')
    plt.xlabel('Loss') 
    plt.ylabel('Frequency')
    plt.title('Loss Distribution for Target Data')
    plt.legend()
    plt.grid(True)
    save_path = f'{save_path}/loss_distribution_comparison.png'
    # Save the plot to the specified path
    plt.savefig(save_path)
    plt.close()
    
    


def get_target_model(name="resnet18", num_classes=10):
    if name == "resnet18":
        model = torchvision.models.resnet18()
        model.fc = nn.Sequential(nn.Linear(512, num_classes))
        # 代码修改了ResNet-18模型的最后一层全连接层，将其替换为一个新的全连接层nn.Linear(512, 10)，
        # 其中512是ResNet-18模型中最后一个卷积层的输出通道数，10是类别数量。这样做是为了将模型的输出调整为与任务中的类别数量相匹配。
    elif name == "resnet20":
        model = resnet20(num_classes =num_classes)
    elif name == "resnet34":
        model = torchvision.models.resnet34()
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif name == "vgg11":
        model = torchvision.models.vgg11()
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_ftrs, num_classes)
    elif name == "wide_resnet50":
        model = torch.hub.load('pytorch/vision:v0.10.0', 'wide_resnet50_2', weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif name == "densenet121":
        model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121',weights=None)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
    else:
        raise ValueError("Model not implemented yet :P")
    return model

def one_hot_embedding(y, num_classes=10, dtype=torch.cuda.FloatTensor):
    '''
    apply one hot encoding on labels
    :param y: class label
    :param num_classes: number of classes
    :param dtype: data type
    :return:
    '''
    scatter_dim = len(y.size())
    # y_tensor = y.type(torch.cuda.LongTensor).view(*y.size(), -1)
    y_tensor = y.view(*y.size(), -1)
    zeros = torch.zeros(*y.size(), num_classes).type(dtype)
    return zeros.scatter(scatter_dim, y_tensor, 1)


def CrossEntropy_soft(input, target, reduction='mean'):
    '''
    cross entropy loss on soft labels
    :param input:
    :param target:
    :param reduction:
    :return:
    '''
    logprobs = F.log_softmax(input, dim=1)
    losses = -(target * logprobs)
    if reduction == 'mean':
        return losses.sum() / input.shape[0]
    elif reduction == 'sum':
        return losses.sum()
    elif reduction == 'none':
        return losses.sum(-1)

def to_categorical(labels: Union[np.ndarray, List[float]], nb_classes: Optional[int] = None) -> np.ndarray:
    """
    Convert an array of labels to binary class matrix.

    :param labels: An array of integer labels of shape `(nb_samples,)`.
    :param nb_classes: The number of classes (possible labels).
    :return: A binary matrix representation of `y` in the shape `(nb_samples, nb_classes)`.
    """
    labels = np.array(labels, dtype=int)
    if nb_classes is None:
        nb_classes = np.max(labels) + 1
    categorical = np.zeros((labels.shape[0], nb_classes), dtype=np.float32)
    categorical[np.arange(labels.shape[0]), np.squeeze(labels)] = 1
    return categorical


def check_and_transform_label_format(
    labels: np.ndarray, nb_classes: Optional[int] = None, return_one_hot: bool = True
) -> np.ndarray:
    """
    Check label format and transform to one-hot-encoded labels if necessary

    :param labels: An array of integer labels of shape `(nb_samples,)`, `(nb_samples, 1)` or `(nb_samples, nb_classes)`.
    :param nb_classes: The number of classes.
    :param return_one_hot: True if returning one-hot encoded labels, False if returning index labels.
    :return: Labels with shape `(nb_samples, nb_classes)` (one-hot) or `(nb_samples,)` (index).
    """
    if labels is not None:
        # multi-class, one-hot encoded
        if len(labels.shape) == 2 and labels.shape[1] > 1:
            if not return_one_hot:
                labels = np.argmax(labels, axis=1)
                labels = np.expand_dims(labels, axis=1)
        elif (
            len(
                labels.shape) == 2 and labels.shape[1] == 1 and nb_classes is not None and nb_classes > 2
        ):  # multi-class, index labels
            if return_one_hot:
                labels = to_categorical(labels, nb_classes)
            else:
                labels = np.expand_dims(labels, axis=1)
        elif (
            len(
                labels.shape) == 2 and labels.shape[1] == 1 and nb_classes is not None and nb_classes == 2
        ):  # binary, index labels
            if return_one_hot:
                labels = to_categorical(labels, nb_classes)
        elif len(labels.shape) == 1:  # index labels
            if return_one_hot:
                labels = to_categorical(labels, nb_classes)
            else:
                labels = np.expand_dims(labels, axis=1)
        else:
            raise ValueError(
                "Shape of labels not recognised."
                "Please provide labels in shape (nb_samples,) or (nb_samples, nb_classes)"
            )

    return labels



