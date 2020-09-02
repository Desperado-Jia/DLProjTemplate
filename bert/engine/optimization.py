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
"""PyTorch optimizers, each definiition should include both ``OptimKeys` enum-like class and `Optimizer` class."""
import math
from typing import Callable, Dict, Iterable, Tuple, Optional
import torch
from torch.nn import Parameter
from torch.optim.optimizer import Optimizer


class AdamWOptimKeys:
    """An enum-like class for keywords in AdamW optimizer of BERT model."""
    LR = "lr"  # Learning rate.
    BETAS = "betas"
    EPS = "eps"
    WEIGHT_DECAY = "weight_decay"
    CORRECT_BIAS = "correct_bias"
    PARAMS = "params"
    STEP = "step"  # Global step.
    EMA_GRADIENT = "exp_avg"  # Exponential moving average of gradient values.
    EMA_SQUARED_GRADIENT = "exp_avg_sq"  # Exponential moving average of squared gradient values.


class AdamWOptimizer(Optimizer):
    r"""Adam with decoupled weight decay optimizer (i.e., AdamW optimizer).

    Reference:
        http://mcneela.github.io/machine_learning/2019/09/03/Writing-Your-Own-Optimizers-In-Pytorch.html.

    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float, optional, defaults to 1e-3):
            Learning rate.
        betas (Tuple[float, float], optional, defaults to (0.9, 0.999)):
            Coefficients used for computing running averages of gradient and its square.
        eps (float, optional, defaults to 1e-8):
            Term added to the denominator to improve numerical stability.
        weight_decay (float, optional, defaults to 1e-2):
            Weight decay coefficient.
        correct_bias (boolean, optional, defaults to False):
            Whether to correct 1st and 2nd beta bias.
            NOTE: set it as `False` when training BERT model.
    """

    def __init__(self,
                 params: Iterable[Parameter],
                 lr: float = 1e-3,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8,
                 weight_decay: float = 1e-2,
                 correct_bias: bool = False):
        # Check the validation of configurations.
        if lr < 0.0:
            raise ValueError("Invalid learning rate {} - should be >= 0.0.".format(lr))
        if len(betas) != 2:
            raise ValueError("Invalid length of betas Tuple {} - should be equal to 2.".format(len(betas)))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid value of `betas`'s 0-th element {} -should be in [0.0, 1.0).".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid value of `betas`'s 1-th element {} -should be in [0.0, 1.0).".format(betas[1]))
        if not eps >= 0:
            raise ValueError("Invalid eps {} -should be >= 0.0".format(eps))
        # A dictionary mapping parameter names to their default values, used for each parameter group.
        defaults = {
            AdamWOptimKeys.LR: lr,
            AdamWOptimKeys.BETAS: betas,
            AdamWOptimKeys.EPS: eps,
            AdamWOptimKeys.WEIGHT_DECAY: weight_decay,
            AdamWOptimKeys.CORRECT_BIAS: correct_bias
        }
        super(AdamWOptimizer, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group[AdamWOptimKeys.PARAMS]:
                if p.grad is None:
                    continue

                # Perform optimization step.
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients.')

                state: Dict = self.state[p]

                # State initialization.
                if len(state) == 0:
                    # Global training step.
                    state[AdamWOptimKeys.STEP] = 0
                    # Exponential moving average of gradient values.
                    state[AdamWOptimKeys.EMA_GRADIENT] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state[AdamWOptimKeys.EMA_SQUARED_GRADIENT] = torch.zeros_like(p.data)

                exp_avg: torch.Tensor = state[AdamWOptimKeys.EMA_GRADIENT]
                exp_avg_sq: torch.Tensor = state[AdamWOptimKeys.EMA_SQUARED_GRADIENT]
                beta1, beta2 = group[AdamWOptimKeys.BETAS]

                state[AdamWOptimKeys.STEP] += 1

                # Decay the first and second moment running average coefficient.
                # In-place operations to update the averages at the same time.
                exp_avg.mul_(other=beta1).add_(other=grad, alpha=1 - beta1)
                exp_avg_sq.mul_(other=beta2).addcmul_(tensor1=grad, tensor2=grad, value=1 - beta2)
                denom = exp_avg_sq.sqrt().add_(other=group[AdamWOptimKeys.EPS])

                step_size = group[AdamWOptimKeys.LR]
                if group[AdamWOptimKeys.CORRECT_BIAS] is True:
                    # NOTE: Don't perform bias correction for BERT model.
                    bias_correction1 = 1 - beta1 ** state[AdamWOptimKeys.STEP]
                    bias_correction2 = 1 - beta2 ** state[AdamWOptimKeys.STEP]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(tensor1=exp_avg, tensor2=denom, value=-step_size)

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version).
                if group[AdamWOptimKeys.WEIGHT_DECAY] > 0.0:
                    p.data.add_(other=p.data, alpha=-group[AdamWOptimKeys.LR] * group[AdamWOptimKeys.WEIGHT_DECAY])

        return loss
