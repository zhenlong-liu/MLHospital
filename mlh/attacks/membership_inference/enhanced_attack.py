import numpy as np
from privacy_meter.audit import Audit
from privacy_meter.dataset import Dataset
from privacy_meter.hypothesis_test import threshold_func
from privacy_meter.information_source import InformationSource
from privacy_meter.information_source_signal import Signal
from privacy_meter.model import Model
from privacy_meter.metric import PopulationMetric
from privacy_meter.constants import InferenceGame
from privacy_meter.audit_report import ROCCurveReport, SignalHistogramReport
from sklearn.metrics import roc_auc_score

import model_loader
from attacks.membership_inference.attack_utils import ensure_list, default_quantile
from attacks.membership_inference.membership_Inference_attack import MembershipInferenceAttack
from defenses.membership_inference.loss_function import get_loss


class ReferenceMIA(MembershipInferenceAttack):
    def __init__(
            self,
            args,
            num_class,
            device,
            attack_type,
            attack_train_dataset,
            attack_test_dataset,
            save_path,
            batch_size=128):
        # traget train load
        super().__init__()
        self.args = args
        self.num_class = num_class
        self.device = device
        self.attack_type = attack_type
        self.attack_train_dataset = attack_train_dataset
        self.attack_test_dataset = attack_test_dataset

        self.loss_type = args.loss_type
        self.save_path = save_path
        self.threshold_func = threshold_func

    def get_threshould(self,threshould_function, alphas,metrics = None):

        reference_mem_info_dict, reference_no_mem_info_dict = self.attack_train_dataset
        # reference_mem_threshould ={f"{key}_threshould": threshould_function(reference_mem_info_dict[key],alphas) for key in reference_mem_info_dict.keys()}
        #
        # reference_no_mem_threshould = {f"{key}_threshould": threshould_function(reference_no_mem_info_dict[key],alphas) for key in reference_no_mem_info_dict.keys() }


        if metrics:
            metrics = ensure_list(metrics)
        else:
            metrics = reference_mem_info_dict.keys()
        reference_mem_threshould = {metric: threshould_function(reference_mem_info_dict[metric],alphas) for metric in metrics}

        reference_no_mem_threshould = {metric: threshould_function(reference_no_mem_info_dict[metric], alphas) for metric in metrics}


        return reference_mem_threshould, reference_no_mem_threshould


    def run_attack(self,threshould_function, fpr_tolerance_rate_list, metrics = None):




            if fpr_tolerance_rate_list is None:
                fpr_tolerance_rate_list = default_quantile()
            else:
                fpr_tolerance_rate_list = np.array(fpr_tolerance_rate_list)


            reference_member_threshold, reference_non_member_threshold = self.get_threshould(
                threshould_function,fpr_tolerance_rate_list,metrics =metrics
            )

            target_mem_info_dict, target_no_mem_info_dict = self.attack_test_dataset


            if metrics is None:
                metrics = target_mem_info_dict.keys()

            num_threshold = len(fpr_tolerance_rate_list)
            target_mem_info_dict = {key: value.repeat(num_threshold, 1).T for key,value in target_mem_info_dict.items() if key in metrics}

            target_no_mem_info_dict = {key: value.reshape(-1, 1).repeat(num_threshold,1).T for key, value in target_no_mem_info_dict.items() if key in metrics}


            member_preds_dict = {key: np.less(value, reference_member_threshold) for key,value in target_mem_info_dict.items()}

            non_member_preds_dict = {key: np.less(value, reference_non_member_threshold)
                                 for key,value in target_no_mem_info_dict.items()}


            predictions_dict = {key: np.concatenate([member_preds_dict[key], non_member_preds_dict[key]], axis=1) for key in metrics}

            true_labels_dict = {key: np.concatenate(
                [np.ones(len(member_preds_dict[key])), np.zeros(len(non_member_preds_dict[key]))]
            ) for key in metrics}



            result_metrics = {}
            for metric in metrics:
                predictions = predictions_dict[metric]
                true_labels = true_labels_dict[metric]

                # expand true_labels to match shape of predictions
                true_labels_expanded = np.expand_dims(true_labels, axis=0)

                # calculate TP, FP, TN, FN
                tp = np.sum((predictions == 1) & (true_labels_expanded == 1), axis=1)
                fp = np.sum((predictions == 1) & (true_labels_expanded == 0), axis=1)
                tn = np.sum((predictions == 0) & (true_labels_expanded == 0), axis=1)
                fn = np.sum((predictions == 0) & (true_labels_expanded == 1), axis=1)

                # calculate accuracy
                accuracy = (tp + tn) / (tp + fp + tn + fn)

                # calculate AUC fpr_tolerance_rate
                auc_scores = [roc_auc_score(true_labels, predictions[i, :]) for i in range(predictions.shape[0])]

                result_metrics[metric] = {
                    "TP": tp,
                    "FP": fp,
                    "TN": tn,
                    "FN": fn,
                    "Accuracy": accuracy,
                    "AUC": auc_scores
                }

            return result_metrics

