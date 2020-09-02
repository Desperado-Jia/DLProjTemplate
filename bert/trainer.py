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
"""PyTorch trainer for training model.
There exists three kinds of **SINGLE-MACHINE-MULTI-GPUS** training methods as following:
    * nn.parallel.DataParallel
    * nn.parallel.DistributedDataParallel
    * horovod
"""
from tqdm import tqdm
from typing import Dict
import torch
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from utils.metric_utils import MetricSummarizer, RunningAverageValue
from utils.keyword_utils import BertModeling4PreTrainingCriterionKeys


def tensors_dict_to_device(tensors_dict: Dict[str, torch.Tensor])


def train_one_epoch(dataloader: DataLoader,
                    model: nn.Module,
                    criterion: nn.Module,
                    optimizer: Optimizer,
                    device: torch.device,
                    epoch: int,
                    max_norm: float = 0.0):
    model.train(mode=True)
    criterion.train(mode=True)

    metric_summarizer = MetricSummarizer(delimiter="\t")
    metric_summarizer.register_meter(
        name=BertModeling4PreTrainingCriterionKeys.TOTAL_LOSS,
        meter=RunningAverageValue(window_size=20)
    )
    # metric_summarizer.register_meter(name=)

    for batch_samples, batch_targets in tqdm(iterable=dataloader, desc="Epoch: %d" % epoch):


