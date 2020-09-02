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
import torch
from torch import nn
from torch.nn import functional as F


class Swish(nn.Module):
    """Define Swish non-linear activation function.

    Reference::
        [Howard et al., 2019](Searching for MobileNetV3)
        <https://arxiv.org/pdf/1905.02244.pdf>.
    """
    @classmethod
    def forward(cls, x: torch.Tensor) -> torch.Tensor:
        return x.sigmoid().mul(other=x)


class Gelu(nn.Module):
    """Define Gelu non-linear activation function."""
    @classmethod
    def forward(cls, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(input=x)


class Relu(nn.Module):
    """Define Relu non-linear activation function."""
    @classmethod
    def forward(cls, x: torch.Tensor) -> torch.Tensor:
        return F.relu(input=x, inplace=False)

ACT2FN = {
    "relu": Relu,
    "gelu": Gelu,
    "swish": Swish
}

def activation_fn(act_fn_str: str):
    if not isinstance(act_fn_str, str):
        raise TypeError("act_fn_str ({} here) must be str type".format(type(act_fn_str)))
    if act_fn_str not in ACT2FN.keys():
        raise ValueError(
            "act_fn_str: {} is not in valid activations string list: {}".format(act_fn_str, list(ACT2FN.keys()))
        )
    return ACT2FN.get(act_fn_str)