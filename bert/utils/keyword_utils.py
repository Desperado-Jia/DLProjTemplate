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
"""keywords of dictionaries for XXX (e.g., BERT) modeling, including following five enum-like classes:
* XXXModelingSampleKeys:
    Declare all keys in **input** samples dictionary of modeling
    (e.g., :obj:`BertModeling4PreTraining`).
* XXXModelingTargetKeys:
    Declare all keys in **input** targets dictionary of criterion
    (e.g., :obj:`BertModeling4PreTrainingCriterion`).
    The reason why we use a dictionary to contain target tensors is the task may be a multi-objective
    supervised training task, each key corresponds to a kind of target tensors for a specific training
    objective.
* XXXModelingBackboneKeys:
    Declare all keys in **output** dictionary of backbone model inference
    (e.g., :obj:`BertModeling`).
    Namely, the backbone neural network is the features extractor.
* XXXModelingPredictionHeadKeys:
    Declare all keys in **output** dictionary of prediction head inference
    (e.g., :obj:`BertModeling4PreTrainingHead`).
    Usually this class contains all logit prediction tensors, and the number of item in
    :class:`XXXModelingPredictionHeadKeys` equals to :class:`XXXModelingTargetKeys`.
    Namely, the head neural networks are discriminators.
* XXXModelingCriterionKeys:
    Declare all keys in **output** losses dictionary of criterion
    (e.g., :obj:`BertModeling4PreTrainingCriterion`).
    Usually the number of items in this class is once greater than both
    :class:`XXXModelingPredictionHeadKeys` and :class:`XXXModelingTargetKeys`, because
    it contains the total loss term.
Nearly all supervised deep learning task can use this template for centralized management of dataflow.
Note that the first and second enum-like classes are for data, the others are for model and criterion.

Besides, we can also add some standardized k-v mapping during modeling and training/evaluation
(i.e, :obj:`ClsLossKeys`).
"""


# ***************BERT for Pre-Training.***************
class BertModeling4PreTrainingSampleKeys:
    """An enum-like class for centralized management of sample tensors in argument
    :dict:`samples_dict` of forward function, :obj:`BertModeling4PreTraining`.

    `TOKEN_IDS`: Corresponds to mini-batch token indices tensor of shape []
    """
    TOKEN_IDS = "token_ids"
    TOKEN_TYPE_IDS = "token_type_ids"
    POSITION_IDS = "position_ids"
    # Corresponds to a 4D tensor of shape
    # [batch_size, num_attention_heads, from_seq_length, to_seq_length].
    # NOT 0-1 tensor, mask element is represented by `-inf`.
    ATTENTION_MASK = "attention_mask"


class BertModeling4PreTrainingTargetKeys:
    """An enum-like class for centralized management of target tensors in argument
    :dict:`targets_dict` of forward function, :obj:`BertModeling4PreTrainingCriterion`.
    """
    LANGUAGE_MODELING_TARGETS = "language_modeling_targets"
    SEQUENCE_RELATIONSHIP_TARGETS = "sequence_relationship_targets"


class BertModeling4PreTrainingBackboneKeys:
    """An enum-like class for centralized management of output tensors during the inference
    process of backbone neural network, i.e., :obj:`BertModeling`.
    """
    EMBEDDINGS = "embeddings"
    TOKEN_EMBEDDINGS = "token_embeddings"
    ENCODER_HIDDEN_ENCODINGS = "encoder_hidden_encodings"
    ENCODER_ATTENTION_PROBABILITIES = "encoder_attention_probs"
    ENCODER_HIDDEN_ENCODINGS_TUPLE = "encoder_hidden_encodings_tuple"
    ENCODER_ATTENTION_PROBABILITIES_TUPLE = "encoder_attention_probs_tuple"
    POOLER_HIDDEN_ENCODINGS = "pooler_hidden_encodings"


class BertModeling4PreTrainingPredictionHeadKeys:
    """An enum-like class for centralized management of output tensors during the inference
    process of head neural network, i.e., :obj:`BertModeling4PreTrainingHead`.
    """
    LANGUAGE_MODELING_LOGITS = "language_modeling_logits"
    SEQUENCE_RELATIONSHIP_LOGITS = "sequence_relationship_logits"


class BertModeling4PreTrainingCriterionKeys:
    """An enum-like class for centralized management of output losses tensors of criterion,
    i.e., :obj:`BertModeling4PreTrainingCriterion`.
    """
    TOTAL_LOSS = "total_loss"
    LANGUAGE_MODELING_LOSS = "language_modeling_loss"
    SEQUENCE_RELATIONSHIP_LOSS = "sequence_relationship_loss"


class ClsLossKeys:
    """An enum-like class for key-value mapping when perform classification loss computation.

    IGNORE_INDEX: (int).
        When computing loss for multi-classes classification task, the loss value of target `IGNORE_INDEX`
        will be ignored, the loss is only computed from the (sample, target) pair with target value
        NOT equal to `IGNORE_INDEX`.
    """
    IGNORE_INDEX = -100
