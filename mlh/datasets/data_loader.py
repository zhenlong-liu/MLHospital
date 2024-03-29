

import os
import torchvision.transforms as transforms
import torch
import numpy as np

from .data_no_image import prepare_purchase, prepare_texas
from .data_utils import prepare_dataset_target, prepare_dataset_shadow_splits, generate_dataset, SUPPORTED_IMAGE_DATASETS
from torchvision import datasets
from PIL import Image
import torchvision


class BuildDataLoader(object):
    def __init__(self, args,shuffle= True):
        self.args = args
        self.data_path = args.data_path
        self.input_shape = args.input_shape
        self.batch_size = args.batch_size
        self.num_splits = args.shadow_split_num
        self.shuffle = shuffle
        
    def parse_dataset(self, dataset, train_transform, test_transform):

        if dataset.lower() == "imagenet":
            self.data_path = f'{self.data_path}/images/'
            train_dataset = torchvision.datasets.ImageFolder(root=self.data_path + 'train', transform=train_transform)
            test_dataset = torchvision.datasets.ImageFolder(root=self.data_path + 'val', transform=test_transform)
            dataset = train_dataset + test_dataset        
        elif dataset.lower() == "tinyimagenet":
            self.data_path = f'{self.data_path}/tiny-imagenet-200/'
            image_datasets = {x: torchvision.datasets.ImageFolder(os.path.join(self.data_path, x), transform=train_transform)
                  for x in ['train', 'val','test']}
            dataset =  image_datasets['train'] + image_datasets['val'] +image_datasets['test']
        elif dataset.lower() == "imagenet_r":
            self.data_path = f'{self.data_path}/imagenet-rendition/imagenet-r/'
            dataset = torchvision.datasets.ImageFolder(root=self.data_path, transform=train_transform)
        elif dataset.lower() == "purchase":
            dataset = prepare_purchase(self.data_path)
        elif dataset.lower() == "texas":
            dataset = prepare_texas(self.data_path)
        elif dataset in SUPPORTED_IMAGE_DATASETS:
            _loader = getattr(datasets, dataset)
            if dataset != "EMNIST":
                train_dataset = _loader(root=self.data_path,
                                        train=True,
                                        transform=train_transform,
                                        download=True)
                test_dataset = _loader(root=self.data_path,
                                       train=False,
                                       transform=test_transform,
                                       download=True)
            else:
                train_dataset = _loader(root=self.data_path,
                                        train=True,
                                        split="byclass",
                                        transform=train_transform,
                                        download=True)
                test_dataset = _loader(root=self.data_path,
                                       train=False,
                                       split="byclass",
                                       transform=test_transform,
                                       download=True)
            dataset = train_dataset + test_dataset

        else:
            raise ValueError("Dataset Not Supported: ", dataset)
        return dataset

    def get_data_transform(self, dataset, use_transform="simple"):
        
        if dataset.lower() in ["imagenet", "imagenet_r"]:
            transform_list = [transforms.Resize(256),transforms.CenterCrop(224)]
            if use_transform == "simple":
                transform_list += [transforms.RandomHorizontalFlip()]
            transform_list+= [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])]
            transform_ = transforms.Compose(transform_list)
            return transform_
        
        if dataset.lower() in ["tinyimagenet"]:
            if self.args.finetune:
                transform_list = [transforms.RandomResizedCrop(224)]
            else: transform_list = [transforms.RandomResizedCrop(64)]
            if use_transform == "simple":
                transform_list += [transforms.RandomHorizontalFlip()]
            transform_list+= [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])]
            transform_ = transforms.Compose(transform_list)
            return transform_
        
        transform_list = [transforms.Resize(
            (self.input_shape[0], self.input_shape[0])), ]
        if use_transform == "simple":
            transform_list += [transforms.RandomCrop(
                32, padding=4), transforms.RandomHorizontalFlip(), ]
        transform_list.append(transforms.ToTensor())
        
        if dataset in ["MNIST", "FashionMNIST", "EMNIST"]:
            transform_list = [
                transforms.Grayscale(3), ] + transform_list
            
        transform_ = transforms.Compose(transform_list)
        
        return transform_


    def get_dataset(self, train_transform, test_transform):
        """
        The function "get_dataset" returns a parsed dataset using the specified train and test
        transformations.
        
        :param train_transform: train_transform is a transformation function that is applied to the
        training dataset. It can include operations such as data augmentation, normalization, resizing,
        etc. This function is used to preprocess the training data before it is fed into the model for
        training
        :param test_transform: The `test_transform` parameter is a transformation that is applied to the
        test dataset. It is used to preprocess or augment the test data before it is fed into the model
        for evaluation. This can include operations such as resizing, normalization, or data
        augmentation techniques like random cropping or flipping
        :return: The dataset is being returned.
        """
        dataset = self.parse_dataset(
            self.args.dataset, train_transform, test_transform)
        return dataset


    def get_split_dataset(self, batch_size, num_workers=2, split_size=4):
        train_transform = self.get_data_transform(self.args.dataset)
        test_transform = self.get_data_transform(self.args.dataset)

        dataset = self.get_dataset(train_transform, test_transform)

        dataset_all = generate_dataset(dataset, split_size=split_size)
        
        print("Preparing dataloader!")
        print("dataset: ", len(dataset))
        
        for i in range(len(dataset_all)):
            dataset_all[i] = torch.utils.data.DataLoader(
                dataset_all[i], batch_size=batch_size, shuffle=self.shuffle, num_workers=num_workers, pin_memory=True)

        
        return dataset_all
        
    def get_data_supervised(self, num_workers=2, select_num=None):
        batch_size = self.batch_size
        
        train_transform = self.get_data_transform(self.args.dataset)
        test_transform = self.get_data_transform(self.args.dataset)
        
        dataset = self.get_dataset(train_transform, test_transform)

        target_train, target_test = generate_dataset(
            dataset, split_size=2)

        print("Preparing dataloader!")
        print("dataset: ", len(dataset))
        print("target_train: %d  \t target_test: %s" %
            (len(target_train), len(target_test)))

        target_train_loader = torch.utils.data.DataLoader(
            target_train, batch_size=batch_size, shuffle=self.shuffle, num_workers=num_workers, pin_memory=True)
        
        target_test_loader = torch.utils.data.DataLoader(
            target_test, batch_size=batch_size, shuffle=self.shuffle, num_workers=num_workers, pin_memory=True)
        

        return target_train_loader, target_test_loader
    
    
    def get_split_shadow_dataset_ni(self, select_num=None, if_dataset =False, num_splits =16,shadow_datapoint_num =None):
        # inference 1/5
        # self.args.dataset default CIFAR10
        train_transform = self.get_data_transform(self.args.dataset)
        test_transform = self.get_data_transform(self.args.dataset)

        dataset = self.get_dataset(train_transform, test_transform)

        _, _,  shadow_train, shadow_test = generate_dataset(
            dataset, split_size=4)
        if shadow_datapoint_num is not None:
            split_size = shadow_datapoint_num
        else:
            split_size = len(shadow_train)
        shadow_list = prepare_dataset_shadow_splits(dataset = shadow_train+ shadow_test, num_splits= num_splits, split_size= split_size)# list[tuple]
        print(f"Prepare shadow dataset list, total num of the list: {num_splits}")
        self.shadow_dataset_list = shadow_list
        if if_dataset:
            return shadow_list

    def get_split_shadow_dataloader_ni(self, batch_size=128, num_workers=8,index= 0):
        train_dataset, test_dataset= self.shadow_dataset_list[index]
        print("Preparing dataloader!")
        print(f"shadow dataset index: {index}")
        print(f"train: {len(train_dataset)} \t target_test: {len(test_dataset)}")

        shadow_train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=self.shuffle, num_workers=num_workers, pin_memory=True)

        shadow_test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=self.shuffle, num_workers=num_workers, pin_memory=True)

        return shadow_train_loader, shadow_test_loader

    def get_split_shadow_dataset_inference(self, select_num=None, if_dataset =False, num_splits =16,shadow_datapoint_num =None):
        # inference 1/5
        # self.args.dataset default CIFAR10
        train_transform = self.get_data_transform(self.args.dataset)
        test_transform = self.get_data_transform(self.args.dataset)

        dataset = self.get_dataset(train_transform, test_transform)

        _, _, inference, shadow_train, shadow_test = generate_dataset(
            dataset, split_size=5)
        if shadow_datapoint_num is not None:
            split_size = shadow_datapoint_num
        else:
            split_size = len(shadow_train)+len(shadow_test)
        shadow_list = prepare_dataset_shadow_splits(dataset = shadow_train+ shadow_test, num_splits= num_splits, split_size= split_size)
        print(f"Prepare shadow dataset list, total num of the list: {num_splits}")
        self.inference_dataset = inference
        self.shadow_dataset_list = shadow_list
        if if_dataset:
            return inference, shadow_list
        
    def get_split_shadow_dataloader_inference(self, batch_size=128, num_workers=8,index= 0):
        
        train_dataset, test_dataset= prepare_dataset_target(self.shadow_dataset_list[index])

        print("Preparing dataloader!")
        print(f"shadow dataset index: {index}")
        print("train: %d \t target_test: %s inference_dataset: %s" %
              (len(train_dataset), len(test_dataset), len(self.inference_dataset)))


        inference_data_loader = torch.utils.data.DataLoader(
            self.inference_dataset, batch_size=batch_size, shuffle=self.shuffle, num_workers=num_workers, pin_memory=True)

        shadow_train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=self.shuffle, num_workers=num_workers, pin_memory=True)

        shadow_test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=self.shuffle, num_workers=num_workers, pin_memory=True)

        return inference_data_loader, shadow_train_loader, shadow_test_loader
    
    
  
    def get_ordered_dataset(self, target_dataset):
        """
    
        Sorts and returns a dataset based on the labels of the data points.

        Parameters:
        - target_dataset (Dataset): The dataset to be sorted.

        Returns:
        - Subset: The sorted dataset.

        Inspired by https://stackoverflow.com/questions/66695251/define-manually-sorted-mnist-dataset-with-batch-size-1-in-pytorch
        """
        label = np.array([row[1] for row in target_dataset])
        sorted_index = np.argsort(label)
        sorted_dataset = torch.utils.data.Subset(target_dataset, sorted_index)
        return sorted_dataset

    def get_label_index(self, target_dataset):
        """
        return starting index for different labels in the sorted dataset
        """
        label_index = []
        start_label = 0
        label = np.array([row[1] for row in target_dataset])
        for i in range(len(label)):
            if label[i] == start_label:
                label_index.append(i)
                start_label += 1
        return label_index
    
    def get_sorted_data_mixup_mmd_one_inference(self):

        train_transform = self.get_data_transform(self.args.dataset)
        test_transform = self.get_data_transform(self.args.dataset)
        dataset = self.get_dataset(train_transform, test_transform)

        target_train, _,  inference, shadow_train, _ = prepare_dataset_inference(
            dataset, select_num=None)
            # sort by label
        target_train_sorted = self.get_ordered_dataset(target_train)
        target_inference_sorted = self.get_ordered_dataset(inference) # dataset
        shadow_train_sorted = self.get_ordered_dataset(shadow_train)
        shadow_inference_sorted = self.get_ordered_dataset(inference)
 

        start_index_target_inference = self.get_label_index(
            target_inference_sorted)
        start_index_shadow_inference = self.get_label_index(
            shadow_inference_sorted)

        # note that we set the inference loader's batch size to 1
        target_train_sorted_loader = torch.utils.data.DataLoader(
            target_train_sorted, batch_size=self.args.batch_size, shuffle=self.shuffle, num_workers=self.args.num_workers, pin_memory=True)
        target_inference_sorted_loader = torch.utils.data.DataLoader(
            target_inference_sorted, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers, pin_memory=True)
        shadow_train_sorted_loader = torch.utils.data.DataLoader(
            shadow_train_sorted, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers, pin_memory=True)
        shadow_inference_sorted_loader = torch.utils.data.DataLoader(
            shadow_inference_sorted, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers, pin_memory=True)

        return target_train_sorted_loader, target_inference_sorted_loader, shadow_train_sorted_loader, shadow_inference_sorted_loader, start_index_target_inference, start_index_shadow_inference, target_inference_sorted, shadow_inference_sorted
    
    
    def get_sorted_data_mixup_mmd(self):

        train_transform = self.get_data_transform(self.args.dataset)
        test_transform = self.get_data_transform(self.args.dataset)
        dataset = self.get_dataset(train_transform, test_transform)

        target_train, target_inference, target_test, shadow_train, shadow_inference, shadow_test = generate_dataset(
            dataset, split_size=6)
            # sort by label
        target_train_sorted = self.get_ordered_dataset(target_train)
        target_inference_sorted = self.get_ordered_dataset(target_inference) # dataset
        shadow_train_sorted = self.get_ordered_dataset(shadow_train)
        shadow_inference_sorted = self.get_ordered_dataset(shadow_inference)
 

        start_index_target_inference = self.get_label_index(
            target_inference_sorted)
        start_index_shadow_inference = self.get_label_index(
            shadow_inference_sorted)

        # note that we set the inference loader's batch size to 1
        target_train_sorted_loader = torch.utils.data.DataLoader(
            target_train_sorted, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers, pin_memory=True)
        target_inference_sorted_loader = torch.utils.data.DataLoader(
            target_inference_sorted, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers, pin_memory=True)
        shadow_train_sorted_loader = torch.utils.data.DataLoader(
            shadow_train_sorted, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers, pin_memory=True)
        shadow_inference_sorted_loader = torch.utils.data.DataLoader(
            shadow_inference_sorted, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers, pin_memory=True)

        return target_train_sorted_loader, target_inference_sorted_loader, shadow_train_sorted_loader, shadow_inference_sorted_loader, start_index_target_inference, start_index_shadow_inference, target_inference_sorted, shadow_inference_sorted


