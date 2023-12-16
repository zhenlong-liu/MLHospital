import numpy as np
import torch

from attacks.membership_inference.attack_dataset import ModelParser
from attacks.membership_inference.attack_utils import ensure_list


class AttackDatasetMutiShadowModels:
    """
    Generate reference attack dataset
    """
    def __init__(self, args, attack_type, target_model, shadow_model_list, target_train_dataloader, target_test_dataloader,
                 shadow_train_loader_list, shadow_test_loader_list):
        self.args = args
        self.attack_type = attack_type
        self.target_model_parser = ModelParser(args, target_model)
        self.shadow_model_parser_list = [ModelParser(args, model) for model in shadow_model_list]

        if attack_type == "white_box":
            self.target_train_info = self.target_model_parser.combined_gradient_attack(target_train_dataloader)
            self.target_test_info = self.target_model_parser.combined_gradient_attack(target_test_dataloader)
             # target dataset into shadow models
            self.reference_mem_info_list = [shadow_model_parser.combined_gradient_attack(target_train_dataloader) for shadow_model_parser in self.shadow_model_parser_list]

            self.reference_no_mem_info_list =[shadow_model_parser.combined_gradient_attack(target_test_dataloader) for shadow_model_parser in self.shadow_model_parser_list]


        else:
            # if attack_type == "black-box":
            self.target_train_info = self.target_model_parser.get_posteriors(
                target_train_dataloader)
            self.target_test_info = self.target_model_parser.get_posteriors(
                target_test_dataloader)

            self.reference_mem_info_list = [shadow_model_parser.get_posteriors(target_train_dataloader) for shadow_model_parser in self.shadow_model_parser_list]

            self.reference_no_mem_info_list = [shadow_model_parser.get_posteriors(target_test_dataloader) for shadow_model_parser in self.shadow_model_parser_list]

        self.attack_train_dataset, self.attack_test_dataset = self.generate_attack_dataset()

    def parse_info(self, info, label=0, return_dict = False, **kwargs):
        mem_label = [label] * len(info["targets"])
        original_label = info["targets"]
        parse_type = self.attack_type
        if parse_type in ["black-box","black_box"]:
            mem_data = info["posteriors"]
        elif parse_type == "black-box-sorted":
            mem_data = [sorted(row, reverse=True)
                        for row in info["posteriors"]]
        elif parse_type == "black-box-top3":
            mem_data = [sorted(row, reverse=True)[:3]
                        for row in info["posteriors"]]
        elif parse_type in ["metric-based", "enhanced_attack"]:
            mem_data = info["posteriors"]

        elif parse_type == "white_box":
            mem_data = info["gird_x_w"]

        else:
            raise ValueError("More implementation is needed :P")

        if return_dict:
            return {"mem_data":mem_data, "mem_label":mem_label, "original_label":original_label, "parse_type":parse_type}

        return mem_data, mem_label, original_label

        ## mem_data posteriors prob; mem_label 0 or 1 ; original_label : class label 1-10

    def generate_attack_dataset(self):

        data_target_train = self.parse_info(
            self.target_train_info, label=1,return_dict= True)
        data_target_test = self.parse_info(
            self.target_test_info, label=0,return_dict= True)

        data_mem_reference_list = [self.parse_info(
            info, label=1) for info in self.reference_mem_info_list]
        data_no_mem_reference_list = [self.parse_info(
            info, label=0) for info in self.reference_no_mem_info_list]


        ## shadow_train_info and shadow_test_info construct the attack_train_dataset, means that using shadow model to train classification model
        attack_train_dataset = [
            {"shadow_mem_data": data_mem_reference_list, "shadow_no_mem_data": data_no_mem_reference_list}]
        # [dicts] dict : {"mem_data":mem_data, "mem_label":mem_label, "original_label":original_label, "parse_type":parse_type}
        # mem_data : posteriors
        # (tensor([2.4628e-04, 3.9297e-01, 3.7773e-04, 9.3001e-05, 2.3009e-04, 3.6489e-04, 1.2535e-04, 6.5634e-05, 1.3375e-03, 6.0419e-01]), tensor(1), tensor(9))
        attack_test_dataset = [{"target_mem_data": data_target_train, "target_no_mem_data": data_target_test}]


        return attack_train_dataset, attack_test_dataset

    def get_reference_info(self,target_train_dataloader, target_test_dataloader,keys = None ):

        if keys:
            metrics = ensure_list(keys)
        else:
            metrics = ["losses", "entropies", "confidences", "correctness", "phi_stable", "modified_entropies"]

        reference_mem_info_list = [shadow_model_parser.get_metrics(target_train_dataloader) for
                                        shadow_model_parser in self.shadow_model_parser_list]

        reference_no_mem_info_list = [shadow_model_parser.get_metrics(target_test_dataloader) for shadow_model_parser in self.shadow_model_parser_list]

        reference_mem_info_dict = {key: np.stack([d[key] for d in reference_mem_info_list]) for key in metrics}

        reference_no_mem_info_dict = {key: np.stack([d[key] for d in reference_no_mem_info_list]) for key in metrics}

        # metric =   "losses"  "entropies" "confidences"  "correctness" "phi_stable" modified_entropies
        return reference_mem_info_dict, reference_no_mem_info_dict

    def get_target_info(self,target_train_dataloader, target_test_dataloader):

        target_mem_info_dict = self.target_model_parser.get_metrics(
            target_train_dataloader)
        target_no_mem_info_dict = self.target_model_parser.get_metrics(
            target_test_dataloader)
        return target_mem_info_dict, target_no_mem_info_dict
