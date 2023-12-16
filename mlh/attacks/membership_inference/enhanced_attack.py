from privacy_meter.audit import Audit
from privacy_meter.dataset import Dataset
from privacy_meter.hypothesis_test import threshold_func
from privacy_meter.information_source import InformationSource
from privacy_meter.information_source_signal import Signal
from privacy_meter.model import Model
from privacy_meter.metric import PopulationMetric
from privacy_meter.constants import InferenceGame
from privacy_meter.audit_report import ROCCurveReport, SignalHistogramReport

import model_loader
from attacks.membership_inference.MembershipInferenceAttack import MembershipInferenceAttack
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
        """
        self.attack_train_loader = torch.utils.data.DataLoader(
            attack_train_dataset, batch_size=batch_size, shuffle=True)
        self.attack_test_loader = torch.utils.data.DataLoader(
            attack_test_dataset, batch_size=batch_size, shuffle=False)
        """
        self.loss_type = args.loss_type
        self.save_path = save_path
        self.criterion = get_loss(loss_type="ce", device=self.device, args=self.args)
        if self.attack_type == "metric-based":
            self.metric_based_attacks()
        elif self.attack_type == "white_box":
            self.white_box_grid_attacks()
        else:
            raise ValueError("Not implemented yet")


