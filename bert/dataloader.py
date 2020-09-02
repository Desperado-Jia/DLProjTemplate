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
"""Define datsets, dataloaders and samplers for BERT model."""
import random
import numpy as np
from typing import Tuple, Dict, Optional
import torch
from torch.utils.data import Dataset
from utils.keyword_utils import (
    BertModeling4PreTrainingSampleKeys,
    BertModeling4PreTrainingTargetKeys
)


class TemplateBertPreTrainingDataset(Dataset):
    """Create a template fake PyTorch :class:`torch.utils.data.Dataset` for BERT
    pre-training task, i.e., random valid dataset for BERT pre-training.
    """
    def __init__(self,
                 n: int,
                 seq_length: int,
                 vocab_size: int,
                 vocab_type_size: int,
                 seed: Optional[int] = None):
        super(TemplateBertPreTrainingDataset, self).__init__()
        self.n = n
        # Set random seed for reproducibility.
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed=seed)
        # Create total samples and targets.
        self.tokenIds = np.random.randint(
            low=0, high=vocab_size, size=(n, seq_length), dtype=np.long
        )
        self.tokenTypeIds = np.random.randint(
            low=0, high=vocab_type_size, size=(n, seq_length), dtype=np.long
        )
        self.languageModelingTargets = np.random.randint(
            low=0, high=vocab_type_size, size=(n, seq_length), dtype=np.long
        )
        self.sequenceRelationshipTargets = self.all_sequence_relationship_targets = np.random.randint(
            low=0, high=2, size=(n, ), dtype=np.long
        )

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, item: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        token_ids = torch.tensor(
            data=self.tokenIds[item], dtype=torch.long, requires_grad=False
        )
        token_type_ids = torch.tensor(
            data=self.tokenTypeIds[item], dtype=torch.long, requires_grad=False
        )
        language_modeling_targets = torch.tensor(
            data=self.languageModelingTargets[item], dtype=torch.long, requires_grad=False
        )
        sequence_relationship_targets = torch.tensor(
            data=self.sequenceRelationshipTargets[item], dtype=torch.long, requires_grad=False
        )

        # The terms in `sample_dict` and `target_dict` are subset of
        # `BertModeling4PreTrainingSampleKeys` and `BertModeling4PreTrainingTargetKeys`.
        # e.g., there doesn't exit `position_ids` and `attention_mask` terms in `sample_dict`.
        sample_dict = dict()
        target_dict = dict()
        sample_dict.setdefault(BertModeling4PreTrainingSampleKeys.TOKEN_IDS,
                               token_ids)
        sample_dict.setdefault(BertModeling4PreTrainingSampleKeys.TOKEN_TYPE_IDS,
                               token_type_ids)
        target_dict.setdefault(BertModeling4PreTrainingTargetKeys.LANGUAGE_MODELING_TARGETS,
                               language_modeling_targets)
        target_dict.setdefault(BertModeling4PreTrainingTargetKeys.SEQUENCE_RELATIONSHIP_TARGETS,
                               sequence_relationship_targets)
        return (sample_dict, target_dict)
