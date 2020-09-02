# Copyright (C) 2020 Tong Jia cecilio.jia@gmail.com. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
"""PyTorch learning rate schedulers."""
import warnings
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LambdaLR  # For customly define your own learning rate scheduler.


def poly_lr_scheduler(optimizer: Optimizer,
                      num_epochs: int,
                      power: float,
                      last_epoch: int = -1) -> LambdaLR:
    """Create a polynomial learning rate scheduler as following:
        lr = init_lr * (1 - epoch / num_epochs) ** power

    :param optimizer: (torch.optim.optimizer.Optimizer).
        An wrapped :obj:`torch.optim.optimizer.Optimizer` used for training.
    :param num_epochs: (int).
        Number of training epochs will be executed continuously soon.
    :param power: (float in range of (0, +inf)).
        Power of the polynomial term.
        NOTE: for the smoothing annealing of learning rate, we recommend `power` to be greater than 1.
    :param last_epoch: (int, optional, defaults to -1)
        The index of last epoch.
    :return:
    """
    def lr_lambda(epoch: int):
        """
        :param epoch: (int)
            Current epoch index from [0, <Number of training epochs will be executed continuously soon> - 1].
            NOTE: Not from the very begining.
        :return: (float).
            Learning rate annealing factor.
        """
        if epoch > num_epochs:
            raise ValueError("Current epoch: {} MUST NOT be greater than number of "
                             "epochs {}".format(epoch, num_epochs))
        return (1 - epoch / num_epochs) ** power

    if power <= 1.0:
        if power <= 0.0:
            raise ValueError("Power ({}) should to be greater than 1, for the validity (> 0.0) and of learning rate "
                             "annealing and smoothing annealing of learning rate (> 1.0).".format(power))
        else:
            warnings.warn(message="Power ({}) is suggested to be greater than 1, for the smoothing annealing of "
                                  "the learning rate.".format(power))

    return LambdaLR(optimizer=optimizer, lr_lambda=lr_lambda, last_epoch=last_epoch)


def cosine_with_warmup_lr_scheduler():
    """Create a cosine learning rate scheduler with warmup."""
    def lambda_lr(epoch: int):
