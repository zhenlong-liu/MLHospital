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

import numpy as np
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple, Union


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



def get_loss(self, loss_type, device, train_loader, args):
    CIFAR10_CONFIG = {
        "ce": nn.CrossEntropyLoss(),
        "focal": FocalLoss(gamma=0.5),
        "mae": MAELoss(num_classes=self.num_classes),
        "gce": GCE(self.device, k=self.num_classes),
        "sce": SCE(alpha=0.5, beta=1.0, num_classes=self.num_classes),
        "ldam": LDAMLoss(device=device),
        "logit_norm": LogitNormLoss(device, self.args.temp, p=self.args.lp),
        "normreg": NormRegLoss(device, self.args.temp, p=self.args.lp),
        "logneg": logNegLoss(device, t=self.args.temp),
        "logit_clip": LogitClipLoss(device, threshold=self.args.temp),
        "cnorm": CNormLoss(device, self.args.temp),
        "tlnorm": TLogitNormLoss(device, self.args.temp, m=10),
        "nlnl": NLNL(device, train_loader=train_loader, num_classes=self.num_classes),
        "nce": NCELoss(num_classes=self.num_classes),
        "ael": AExpLoss(num_classes=10, a=2.5),
        "aul": AUELoss(num_classes=10, a=5.5, q=3),
        "phuber": PHuberCE(tau=10),
        "taylor": TaylorCE(device=self.device, series=args.series),
        "cores": CoresLoss(device=self.device),
        "ncemae": NCEandMAE(alpha=1, beta=1, num_classes=10),
        "ngcemae": NGCEandMAE(alpha=1, beta=1, num_classes=10),
        "ncerce": NGCEandMAE(alpha=1, beta=1.0, num_classes=10),
        "nceagce": NCEandAGCE(alpha=1, beta=4, a=6, q=1.5, num_classes=10),
    }
    CIFAR100_CONFIG = {
        "ce": nn.CrossEntropyLoss(),
        "focal": FocalLoss(gamma=0.5),
        "mae": MAELoss(num_classes=self.num_classes),
        "gce": GCE(self.device, k=self.num_classes),
        "sce": SCE(alpha=0.5, beta=1.0, num_classes=self.num_classes),
        "ldam": LDAMLoss(device=device),
        "logit_clip": LogitClipLoss(device, threshold=self.args.temp),
        "logit_norm": LogitNormLoss(device, self.args.temp, p=self.args.lp),
        "normreg": NormRegLoss(device, self.args.temp, p=self.args.lp),
        "tlnorm": TLogitNormLoss(device, self.args.temp, m=100),
        "cnorm": CNormLoss(device, self.args.temp),
        "nlnl": NLNL(device, train_loader=train_loader, num_classes=self.num_classes),
        "nce": NCELoss(num_classes=self.num_classes),
        "ael": AExpLoss(num_classes=100, a=2.5),
        "aul": AUELoss(num_classes=100, a=5.5, q=3),
        "phuber": PHuberCE(tau=30),
        "taylor": TaylorCE(device=self.device, series=args.series),
        "cores": CoresLoss(device=self.device),
        "ncemae": NCEandMAE(alpha=50, beta=1, num_classes=100),
        "ngcemae": NGCEandMAE(alpha=50, beta=1, num_classes=100),
        "ncerce": NGCEandMAE(alpha=50, beta=1.0, num_classes=100),
        "nceagce": NCEandAGCE(alpha=50, beta=0.1, a=1.8, q=3.0, num_classes=100),
    }
    WEB_CONFIG = {
        "ce": nn.CrossEntropyLoss(),
        "focal": FocalLoss(gamma=0.5),
        "mae": MAELoss(num_classes=self.num_classes),
        "gce": GCE(self.device, k=self.num_classes),
        "sce": SCE(alpha=0.5, beta=1.0, num_classes=self.num_classes),
        "ldam": LDAMLoss(device=device),
        "logit_norm": LogitNormLoss(device, self.args.temp, p=self.args.lp),
        "logit_clip": LogitClipLoss(device, threshold=self.args.temp),
        "normreg": NormRegLoss(device, self.args.temp, p=self.args.lp),
        "cnorm": CNormLoss(device, self.args.temp),
        "tlnorm": TLogitNormLoss(device, self.args.temp, m=50),
        "nlnl": NLNL(device, train_loader=train_loader, num_classes=self.num_classes),
        "nce": NCELoss(num_classes=self.num_classes),
        "ael": AExpLoss(num_classes=50, a=2.5),
        "aul": AUELoss(num_classes=50, a=5.5, q=3),
        "phuber": PHuberCE(tau=30),
        "taylor": TaylorCE(device=self.device, series=args.series),
        "cores": CoresLoss(device=self.device),
        "ncemae": NCEandMAE(alpha=50, beta=0.1, num_classes=50),
        "ngcemae": NGCEandMAE(alpha=50, beta=0.1, num_classes=50),
        "ncerce": NGCEandMAE(alpha=50, beta=0.1, num_classes=50),
        "nceagce": NCEandAGCE(alpha=50, beta=0.1, a=2.5, q=3.0, num_classes=50),
    }
    if "CIFAR10" in args.dataset:
        return CIFAR10_CONFIG[loss_type]
    elif args.dataset == "cifar100":
        return CIFAR100_CONFIG[loss_type]
    elif args.dataset == "webvision":
        return WEB_CONFIG[loss_type]