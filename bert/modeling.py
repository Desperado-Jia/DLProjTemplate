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
"""PyTorch BERT modeling. The Modeling for pre-training and all downstream tasks include
three components:
 * Embedding module;
 * Transformer module (i.e., BERT encoder);
 * Prediction head module.
"""
import torch
from torch import nn
import math
from typing import Dict, Tuple, Optional
from configuration import BertConfig
from utils.activation_utils import activation_fn
from utils.keyword_utils import (
    ClsLossKeys,
    BertModeling4PreTrainingSampleKeys,
    BertModeling4PreTrainingTargetKeys,
    BertModeling4PreTrainingBackboneKeys,
    BertModeling4PreTrainingPredictionHeadKeys,
    BertModeling4PreTrainingCriterionKeys
)


class AttentionMaskModeling(object):
    """BERT attention mask computation, including sequence mask and padding mask."""

    def __init__(self):
        super(AttentionMaskModeling, self).__init__()

    @staticmethod
    def get_3d_binary_sequence_mask():
        raise NotImplementedError

    @staticmethod
    def get_3d_binary_padding_mask():
        raise NotImplementedError

    @staticmethod
    def get_4d_attention_mask():
        raise NotImplementedError


BertLayerNorm = nn.LayerNorm


class BertEmbeddings(nn.Module):
    """Construct the embeddings for tokens (i.e., words), positions and token types.

    Example::
        ```python
        torch.manual_seed(seed=2021)
        # Load configuration of model from a specific JSON file.
        config = BertConfig.from_json(json_dir="settings/config.json")
        # Create input tensors
        BATCH_SIZE = 12
        SEQ_LENGTH = 128
        token_ids = torch.randint(
            low=0, high=config.vocab_size, size=[BATCH_SIZE, SEQ_LENGTH], dtype=torch.long, requires_grad=False
        )
        token_type_ids = torch.randint(
            low=0, high=config.vocab_type_size, size=[BATCH_SIZE, SEQ_LENGTH], dtype=torch.long, requires_grad=False
        )
        # Instantiate the Bert embeddings module.
        module = BertEmbeddings(config=config)
        # Get the output of module.
        output_tuple = module.forward(token_ids=token_ids, token_type_ids=token_type_ids, position_ids=None)
        # Accessing the output tensors.
        embeddings = output_tuple[0]
        ...
        ```
    """

    def __init__(self, config: BertConfig):
        super(BertEmbeddings, self).__init__()
        self.max_position_embeddings = config.max_position_embeddings
        # Define the token embeddings table.
        self.token_embeddings = nn.Embedding(
            num_embeddings=config.vocab_size, embedding_dim=config.hidden_size, padding_idx=0
        )
        # Define the positional embeddings table.
        self.position_embeddings = nn.Embedding(
            num_embeddings=config.max_position_embeddings, embedding_dim=config.hidden_size
        )
        # Define the token type embeddings table.
        self.token_type_embeddings = nn.Embedding(
            num_embeddings=config.vocab_type_size, embedding_dim=config.hidden_size
        )

        # Layer normalization.
        self.layer_norm = BertLayerNorm(
            normalized_shape=config.hidden_size, eps=config.layer_norm_eps, elementwise_affine=True
        )
        # Dropout.
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob, inplace=False)
        # Positional indices tensor is contiguous in memory and exported when serialized.
        self.register_buffer(
            name="position_ids",
            tensor=torch.arange(
                start=0, end=config.max_position_embeddings, step=1, dtype=torch.int64, requires_grad=False
            ).unsqueeze(dim=0)  # [1, `max position embeddings`]
        )

    def forward(self,
                token_ids: torch.Tensor,
                token_type_ids: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                return_token_embeddings: bool = False,
                return_token_type_embeddings: bool = False,
                return_position_embeddings: bool = False) -> \
            Tuple[
                torch.Tensor,
                Optional[torch.Tensor],
                Optional[torch.Tensor],
                Optional[torch.Tensor]
            ]:
        """
        :param token_ids: (torch.Tensor of shape [batch_size, seq_length]).
            Mini-batch token indices in sequences.
        :param token_type_ids: (torch.Tensor of shape [batch_size, seq_length] or None).
            Mini-batch token type indices in sequences.
            If None, all tokens in mini-batch belongs to the 0-th token type, use `torch.zeros`.
        :param position_ids: (torch.Tensor of shape [batch_size, seq_length] or None).
            Mini-batch positional indices of sequences.
            If None, all tokens belongs to the positions in `token_ids`, use `self.position_ids`.
        :param return_token_embeddings: (bool, optinal, defaults to False).
            Whether to return token embeddings.
            If None, token embeddings term in output object is None.
        :param return_token_type_embeddings: (bool, optional, defaults to False).
            Whether to return the token type embeddings.
            If None, token type embeddings term in output object is None.
        :param return_position_embeddings: (bool, optional, defaults to False).
            Whether to return the position embeddings.
            If None, position embeddings term in output object is None.

        :return output_tuple: (tuple of tensors with following order)
            * embeddings: (torch.Tensor of shape [batch_size, seq_length, hidden_size]).
                Mini-batch linear combination of token embeddings, token type embeddings and positional embeddings.
            * token_embeddings: (torch.Tensor of shape [batch_size, seq_length, hidden_size] or None, optional,
                defaults to None).
                Mini-batch token embeddings.
            * token_type_embeddings: (torch.Tensor of shape [batch_size, seq_length, hidden_size] or None, optional,
                defaults to None).
                Mini-batch token type embeddings.
            * position_embeddings: (torch.Tensor of shape [batch_size, seq_length, hidden_size] or None, optional,
                defaults to None).
                Mini-batch positional embeddings.
        """
        input_shape = token_ids.size()
        seq_length = input_shape[1]
        if seq_length > self.max_position_embeddings:
            raise ValueError(
                "The seq_length (%d) of current mini-batch is not less than max_position_embeddings (%d)."
                % (seq_length, self.max_position_embeddings)
            )

        # Preprocess for `token_type_ids` and `position_ids`.
        if token_type_ids is None:
            token_type_ids = torch.zeros(
                size=input_shape, dtype=torch.long, device=self.position_ids.device, requires_grad=False
            )
        if position_ids is None:
            position_ids = self.position_ids[:, : seq_length]

        # Perform embedding lookup operations.
        # all tensors are of shape [batch_size, seq_length, hidden_size].
        token_embeddings = self.token_embeddings.forward(input=token_ids)
        token_type_embeddings = self.token_type_embeddings.forward(input=token_type_ids)
        position_embeddings = self.position_embeddings.forward(input=position_ids)

        # Combine above three types embeddings (element-wise addition) and perform layer normalization and dropout.
        embeddings = token_embeddings + token_type_embeddings + position_embeddings
        embeddings = self.layer_norm.forward(input=embeddings)
        embeddings = self.dropout.forward(input=embeddings)

        output_tuple = tuple()
        output_tuple += (embeddings,)
        output_tuple += (token_embeddings if return_token_embeddings is True else None, )
        output_tuple += (token_type_embeddings if return_token_type_embeddings is True else None, )
        output_tuple += (position_embeddings if return_position_embeddings is True else None, )

        return output_tuple


class BertSelfAttention(nn.Module):
    """Construct the multi-headed self-attention layer of BERT.

    Example::
        ```python
        torch.manual_seed(seed=2021)
        # Load configuration of model from a specific JSON file.
        config = BertConfig.from_json(json_dir="settings/config.json")
        # Create input tensors
        BATCH_SIZE = 12
        FROM_SEQ_LENGTH = 256
        TO_SEQ_LENGTH = 64
        from_hidden_encodings = torch.rand(
            size=[BATCH_SIZE, FROM_SEQ_LENGTH, config.hidden_size], dtype=torch.float32, requires_grad=False
        )
        to_hidden_encodings = torch.rand(
            size=[BATCH_SIZE, TO_SEQ_LENGTH, config.hidden_size], dtype=torch.float32, requires_grad=False
        )
        attention_mask_4d = None
        # Instantiate the Bert embeddings module.
        module = BertSelfAttention(config=config)
        # Get the output tensors.
        output_tuple = module.forward(
            from_hidden_encodings=from_hidden_encodings,
            to_hidden_encodings=to_hidden_encodings,
            attention_mask_4d=attention_mask_4d,
            return_attention_probs=True
        )
        # Accessing the output tensors.
        hidden_encodings = output_tuple[0]
        attention_probs = output_tuple[1]
        ```
    """

    def __init__(self, config: BertConfig):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention heads (%d)"
                % (config.hidden_size, config.num_attention_heads)
            )
        # We don't specify per-head attention size manually in arguments, instead, we use
        # `hidden_size / num_attention_heads` as per-head attention size. Therefore,
        # `attention_all_head_size` == `hidden_size`.
        self.num_attention_heads = config.num_attention_heads
        self.per_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.per_head_size * self.num_attention_heads

        # Define the multi-headed query projection (hidden space -> multi-headed query space).
        # [hidden_size, attention_all_head_size].
        self.query = nn.Linear(in_features=config.hidden_size, out_features=self.all_head_size, bias=True)
        # Define the multi-headed key projection (hidden space -> multi-headed key space).
        # [hidden_size, attention_all_head_size].
        self.key = nn.Linear(in_features=config.hidden_size, out_features=self.all_head_size, bias=True)
        # Define the multi-headed value projection (hidden space -> multi-headed value space).
        # [hidden_size, attention_all_head_size].
        self.value = nn.Linear(in_features=config.hidden_size, out_features=self.all_head_size, bias=True)

        # Dropout
        self.dropout = nn.Dropout(p=config.attention_probs_dropout_prob, inplace=False)

    @staticmethod
    def reshape_3d_to_2d(x: torch.Tensor) -> torch.Tensor:
        """Reshape a tensor of shape [batch_size, seq_length, hidden_size] into a tensor of shape
        [batch_size * seq_length, hidden_size].
        """
        if x.ndimension() != 3:
            raise ValueError("Dimension of `x`: (%d), it must be 3" % x.ndimension())
        input_shape = list(x.size())
        output_shape = [input_shape[0] * input_shape[1], input_shape[2]]
        # [batch_size * seq_length, hidden_size].
        return x.reshape(shape=output_shape)

    @staticmethod
    def reshape_and_transpose_2d_to_4d(x: torch.Tensor,
                                       seq_length: int,
                                       num_attention_heads: int,
                                       per_head_size: int) -> torch.Tensor:
        """Reshape and transpose a tensor of shape [batch_size * seq_length, all_head_size] into a tensor of shape
        [batch_size, num_attention_heads, seq_length, per_head_size].
        NOTE: num_attention_heads * per_head_size == all_head_size.
        """
        if x.ndimension() != 2:
            raise ValueError("Dimension of `x`: (%d), it must be 2" % x.ndimension())
        src_size = list(x.size())
        batch_size = int(src_size[0] / seq_length)
        # [batch_size, num_attention_heads, seq_length, per_head_size]
        x_4d = x. \
            view(size=[batch_size, seq_length, num_attention_heads, per_head_size]). \
            permute(dims=[0, 2, 1, 3])
        return x_4d

    def forward(self,
                from_hidden_encodings: torch.Tensor,
                to_hidden_encodings: torch.Tensor,
                attention_mask_4d: Optional[torch.Tensor],
                return_attention_probs: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        :param from_hidden_encodings: (torch.Tensor of shape [batch_size, from_seq_length, hidden_size]).
            Mini-batch hidden contextual encodings for computing query encodings later.
        :param to_hidden_encodings: (torch.Tensor of shape [batch_size, to_seq_length, hidden_size]).
            Mini-batch hidden contextual encodings for computing key encodings and value encodings later.
            NOTE:
                * In encoder-decoder attention, from_hidden_encodings comes from decoder part and to_hidden_encodings
                comes from encoder part, `from_seq_length` may not be equal to `to_seq_length`;
                * In encoder-encoder attention, both from_hidden_encodings and to_hidden_encodings come from encoder,
                `from_seq_length` must be equal to `to_seq_length` because they both come from the same hidden
                encodings;
                * In decoder-decoder attention, both from_hidden_encodings and to_hidden_encodings come from decoder,
                `from_seq_length` must be equal to `to_seq_length` because they both come from the same hidden
                encodings.
        :param attention_mask_4d: (torch.Tensor of shape
            [batch_size, num_attention_heads, from_seq_length, to_seq_length]
            or None, optional, defaults to None).
            Precomputed attention mask 4d tensor in which `-inf` represents mask.
            If None, don't have mask element.
        :param return_attention_probs: (bool, optional, defaults to False).
             Whether to return attention probs tensor of shape
            [batch_size, num_attention_heads, from_seq_length, to_seq_length] in the output tensors.

        :return output_tuple: (tuple of (hidden_encodings, attention_probs or None)).
            * hidden_encodings: (torch,Tensor of shape [batch_size, from_seq_length, hidden_size]).
                Mini-batch contextual hidden encodings after performed by a BERT attention layer.
            * attention_probs: (torch.Tensor of shape [batch_size, num_attention_heads, from_seq_length, to_seq_length]
                or None, optional, defaults to None).
                Mini-batch attention probabilities.
                If None, it means don't return the attention probabilities. This term is controlled by argument
                `return_attention_probs`.
        """
        from_seq_length = from_hidden_encodings.size()[1]
        to_seq_length = to_hidden_encodings.size()[1]

        # [batch_size * seq_length, hidden_size].
        from_hidden_encodings_2d = self.reshape_3d_to_2d(x=from_hidden_encodings)
        to_hidden_encodings_2d = self.reshape_3d_to_2d(x=to_hidden_encodings)

        # Compute query hidden encodings based on `from_hidden_encodings_2d`.
        # Compute key hidden encodings and value hidden encodings based on `to_hidden_encodings_2d`.
        # [batch_size * from_seq_length, all_head_size].
        # NOTE: all_head_size == hidden_size.
        query_hidden_encodings_2d = self.query.forward(input=from_hidden_encodings_2d)
        # [batch_size * to_seq_length, all_head_size].
        key_hidden_encodings_2d = self.key.forward(input=to_hidden_encodings_2d)
        value_hidden_encodings_2d = self.value.forward(input=to_hidden_encodings_2d)

        # Reshape and transpose 2D hidden encodings into 4D hidden encodings.
        query_hidden_encodings_4d = self.reshape_and_transpose_2d_to_4d(
            x=query_hidden_encodings_2d,
            seq_length=from_seq_length,
            num_attention_heads=self.num_attention_heads,
            per_head_size=self.per_head_size
        )  # [batch_size, num_attention_heads, from_seq_length, per_head_size].
        key_hidden_encodings_4d = self.reshape_and_transpose_2d_to_4d(
            x=key_hidden_encodings_2d,
            seq_length=to_seq_length,
            num_attention_heads=self.num_attention_heads,
            per_head_size=self.per_head_size
        )  # [batch_size, num_attention_heads, to_seq_length, per_head_size].
        value_hidden_encodings_4d = self.reshape_and_transpose_2d_to_4d(
            x=value_hidden_encodings_2d,
            seq_length=to_seq_length,
            num_attention_heads=self.num_attention_heads,
            per_head_size=self.per_head_size
        )  # [batch_size, num_attention_heads, to_seq_length, per_head_size].

        # Compute attention scores based on `query_hidden_encodings_4d` and `key_hidden_encodings_4d`.
        # Output shape is [batch_size, num_attention_heads, from_seq_length, to_seq_length].
        attention_scores = query_hidden_encodings_4d.matmul(other=key_hidden_encodings_4d.transpose(dim0=-1, dim1=-2))
        attention_scores = attention_scores / math.sqrt(self.num_attention_heads)
        # Apply attention mask operation, which was precomputed for all Bert attention layers.
        # In `attention_mask`, `-inf` means mask.
        if attention_mask_4d is not None:
            # [batch_size, num_attention_heads, from_seq_length, to_seq_length].
            attention_scores += attention_mask_4d
        # [batch_size, num_attention_heads, from_seq_length, to_seq_length].
        attention_probs = nn.functional.softmax(input=attention_scores, dim=-1)
        # Dropout for attention probabilities.
        # [batch_size, num_attention_heads, from_seq_length, to_seq_length].
        attention_probs = self.dropout.forward(input=attention_probs)

        # Compute hidden contextual encodings based on `attention_probs` and `value_hidden_encodings_4d`.
        # [batch_size, num_attention_heads, from_seq_length, per_head_size].
        hidden_encodings = attention_probs.matmul(other=value_hidden_encodings_4d)
        # [batch_size, from_seq_length, num_attention_heads, per_head_size].
        hidden_encodings = hidden_encodings.permute(dims=[0, 2, 1, 3]).contiguous()

        output_shape = [hidden_encodings.size()[0], hidden_encodings.size()[1], self.all_head_size]
        # [batch_size, from_seq_length, all_head_size].
        hidden_encodings = hidden_encodings.view(size=output_shape)

        output_tuple = tuple()
        output_tuple += (hidden_encodings, )
        output_tuple += (attention_probs if return_attention_probs is True else None, )
        return output_tuple


class BertSelfOutput(nn.Module):
    """Construct the output sublayer in attention layer, i.e., further feature extraction of hidden contextual
    encodings, including perform a one-layer dense neural network, layer normalization and dropout.
    """

    def __init__(self, config: BertConfig):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size, bias=True)
        self.layer_norm = BertLayerNorm(
            normalized_shape=config.hidden_size, eps=config.layer_norm_eps, elementwise_affine=True
        )
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob, inplace=False)

    def forward(self,
                hidden_encodings_after_self_attention: torch.Tensor,
                hidden_encodings_before_self_attention: torch.Tensor) -> torch.Tensor:
        """
        :param hidden_encodings_after_self_attention: (torch.Tensor of shape
            [batch_size, from_seq_length, hidden_size]).
            Mini-batch hidden encodings after performed by `BertSelfAttention` module in current attention layer.
        :param hidden_encodings_before_self_attention: (torch.Tensor of shape
            [batch_size, from_seq_length, hidden_size]).
            Mini-batch hidden encodings before performed by `BertSelfAttention` module in current attention layer.

        :return hidden_encodings: (torch,Tensor of shape [batch_size, from_seq_length, hidden_size]).
            Mini-batch contextual hidden encodings after performed by a BERT attention output layer.
        """
        # [batch_size, from_seq_length, hidde_size].
        hidden_encodings = self.dense.forward(input=hidden_encodings_after_self_attention)
        hidden_encodings = self.dropout.forward(input=hidden_encodings)
        hidden_encodings = self.layer_norm.forward(input=hidden_encodings + hidden_encodings_before_self_attention)

        return hidden_encodings


class BertAttention(nn.Module):
    """Construct attention layer for Bert model, including:
    multi-headed self attention sub-layer (:obj:`BertSelfAttention`) and output sub-layer
    (:obj:`BertSelfOutput`).

    Example::
        ```python
        torch.manual_seed(seed=2021)
        # Load configuration of model from a specific JSON file.
        config = BertConfig.from_json(json_dir="settings/config.json")
        # Create input tensors
        BATCH_SIZE = 32
        FROM_SEQ_LENGTH = 256
        TO_SEQ_LENGTH = 64
        from_hidden_encodings = torch.rand(
            size=[BATCH_SIZE, FROM_SEQ_LENGTH, config.hidden_size], dtype=torch.float32, requires_grad=False
        )
        to_hidden_encodings = torch.rand(
            size=[BATCH_SIZE, TO_SEQ_LENGTH, config.hidden_size], dtype=torch.float32, requires_grad=False
        )
        attention_mask_4d = None
        # Instantiate the Bert embeddings module.
        module = BertAttention(config=config)
        # Get the output tensors.
        output_tuple = module.forward(
            from_hidden_encodings=from_hidden_encodings,
            to_hidden_encodings=to_hidden_encodings,
            attention_mask_4d=attention_mask_4d,
            return_attention_probs=True
        )
        # Accessing the output tensors.
        hidden_encodings = output_tuple[0]
        attention_probs = output_tuple[1]
        ```
    """

    def __init__(self, config: BertConfig):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config=config)  # Self attention module.
        self.output = BertSelfOutput(config=config)  # Self output module.

    def forward(self,
                from_hidden_encodings: torch.Tensor,
                to_hidden_encodings: torch.Tensor,
                attention_mask_4d: Optional[torch.Tensor],
                return_attention_probs: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        :param from_hidden_encodings: (torch.Tensor of shape [batch_size, from_seq_length, hidden_size]).
            Mini-batch hidden contextual encodings for computing query encodings later.
        :param to_hidden_encodings: (torch.Tensor of shape [batch_size, to_seq_length, hidden_size]).
            Mini-batch hidden contextual encodings for computing key encodings and value encodings later.
            NOTE:
                * In encoder-decoder attention, from_hidden_encodings comes from decoder part and to_hidden_encodings
                comes from encoder part, `from_seq_length` may not be equal to `to_seq_length`;
                * In encoder-encoder attention, both from_hidden_encodings and to_hidden_encodings come from encoder,
                `from_seq_length` must be equal to `to_seq_length` because they both come from the same hidden
                encodings;
                * In decoder-decoder attention, both from_hidden_encodings and to_hidden_encodings come from decoder,
                `from_seq_length` must be equal to `to_seq_length` because they both come from the same hidden
                encodings.
        :param attention_mask_4d: (torch.Tensor of shape
            [batch_size, num_attention_heads, from_seq_length, to_seq_length]
            or None, optional, defaults to None).
            Precomputed attention mask 4d tensor in which `-inf` represents mask.
            If None, don't have mask element.
        :param return_attention_probs: (bool, optional, defaults to False).
             Whether to return attention probs tensor of shape
            [batch_size, num_attention_heads, from_seq_length, to_seq_length] in the output tensors.

        :return output_tuple: (tuple of (:torch.Tensor:`hidden_encodings`, :torch.Tensor:`attention_probs` or None)).
            * hidden_encodings: (torch,Tensor of shape [batch_size, from_seq_length, hidden_size]).
                Mini-batch contextual hidden encodings after performed by a BERT attention layer.
            * attention_probs: (torch.Tensor of shape [batch_size, num_attention_heads, from_seq_length, to_seq_length]
                or None, optional, defaults to None).
                Mini-batch attention probabilities.
                If None, it means don't return the attention probabilities. This term is controlled by argument
                `return_attention_probs`.
        """
        self_attention_output_tuple = self.self.forward(
            from_hidden_encodings=from_hidden_encodings,
            to_hidden_encodings=to_hidden_encodings,
            attention_mask_4d=attention_mask_4d,
            return_attention_probs=return_attention_probs
        )
        hidden_encodings_after_self_attention = self_attention_output_tuple[0]
        attention_probs = self_attention_output_tuple[1]

        self_output_hidden_encodings = self.output.forward(
            hidden_encodings_after_self_attention=hidden_encodings_after_self_attention,
            hidden_encodings_before_self_attention=from_hidden_encodings  # NOTE: NOT `to_hidden_encodings`.
        )

        output_tuple = tuple()
        output_tuple += (self_output_hidden_encodings, )
        output_tuple += (attention_probs, )
        return output_tuple


class BertFFNIntermediate(nn.Module):
    """Construct the intermediate sub-layer of a feed-forward layer.

    Example usage::
        ```python
        torch.manual_seed(seed=2021)
        # Load configuration of model from a specific JSON file.
        config = BertConfig.from_json(json_dir="settings/config.json")
        # Create input tensors
        BATCH_SIZE = 32
        FROM_SEQ_LENGTH = 256
        TO_SEQ_LENGTH = 64
        hidden_encodings = torch.rand(
            size=[BATCH_SIZE, FROM_SEQ_LENGTH, config.hidden_size], dtype=torch.float32, requires_grad=False
        )
        module = BertFFNIntermediate(config=config)
        intermediate_hidden_encodings = module.forward(hidden_encodings=hidden_encodings)
        ```
    """

    def __init__(self, config: BertConfig):
        super(BertFFNIntermediate, self).__init__()
        self.dense = nn.Linear(in_features=config.hidden_size, out_features=config.intermediate_size, bias=True)
        self.activation = activation_fn(act_fn_str=config.hidden_act)

    def forward(self, hidden_encodings: torch.Tensor) -> torch.Tensor:
        """
        :param hidden_encodings: (torch,Tensor of shape [batch_size, from_seq_length, hidden_size]).
            Mini-batch hidden contextual encodings before performed by intermediate layer.
        :return intermediate_hidden_encodings: (torch,Tensor of shape [batch_size, seq_length, intermediate_size]).
            Mini-batch intermediate contextual hidden encodings after performed by a BERT feed forward intermediate
            layer.
        """
        # [batch_size, from_seq_length, intermediate_size].
        intermediate_hidden_encodings = self.dense.forward(input=hidden_encodings)
        intermediate_hidden_encodings = self.activation.forward(x=intermediate_hidden_encodings)
        return intermediate_hidden_encodings


class BertFFNOutput(nn.Module):
    """Construct the output sub-layer of a feed-forward layer."""

    def __init__(self, config: BertConfig):
        super(BertFFNOutput, self).__init__()
        self.dense = nn.Linear(in_features=config.intermediate_size, out_features=config.hidden_size, bias=True)
        self.layer_norm = BertLayerNorm(
            normalized_shape=config.hidden_size, eps=config.layer_norm_eps, elementwise_affine=True
        )
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob, inplace=False)

    def forward(self,
                before_intermediate_hidden_encodings: torch.Tensor,
                after_intermediate_hidden_encodings: torch.Tensor) -> torch.Tensor:
        """
        :param before_intermediate_hidden_encodings: (torch,Tensor of shape [batch_size, from_seq_length, hidden_size]).
            Mini-batch hidden contextual encodings before performed by intermediate layer.
        :param after_intermediate_hidden_encodings: (torch,Tensor of shape
            [batch_size, from_seq_length, intermediate_size]).
            Mini-batch hidden contextual encodings after performed by intermediate layer.
        :return ffn_out_hidden_encodings: (torch,Tensor of shape [batch_size, seq_length, hidden_size]).
            Mini-batch contextual hidden encodings after performed by a BERT feed forward output layer.
        """
        # [batch_size, from_seq_length, hidden_size].
        hidden_encodings = self.dense.forward(input=after_intermediate_hidden_encodings)
        hidden_encodings = self.dropout.forward(input=hidden_encodings)
        ffn_out_hidden_encodings = self.layer_norm.forward(
            input=before_intermediate_hidden_encodings + hidden_encodings
        )
        return ffn_out_hidden_encodings


class BertFFN(nn.Module):
    """Construct the feed-forward layer of a bert layer.

    Example usage::
        ```python
        torch.manual_seed(seed=2021)
        # Load configuration of model from a specific JSON file.
        config = BertConfig.from_json(json_dir="settings/config.json")
        # Create input tensors
        BATCH_SIZE = 32
        FROM_SEQ_LENGTH = 256
        TO_SEQ_LENGTH = 64
        hidden_encodings = torch.rand(
            size=[BATCH_SIZE, FROM_SEQ_LENGTH, config.hidden_size], dtype=torch.float32, requires_grad=False
        )
        module = BertFFN(config=config)
        ffn_hidden_encodings = module.forward(hidden_encodings=hidden_encodings)
        ```
    """

    def __init__(self, config: BertConfig):
        super(BertFFN, self).__init__()
        self.intermediate = BertFFNIntermediate(config=config)
        self.output = BertFFNOutput(config=config)

    def forward(self, hidden_encodings: torch.Tensor) -> torch.Tensor:
        """
        :param hidden_encodings: (torch,Tensor of shape [batch_size, from_seq_length, hidden_size]).
            Mini-batch hidden contextual encodings before performed by feed-forward layer.
        :return output_hidden_encodings: (torch,Tensor of shape [batch_size, seq_length, hidden_size]).
            Mini-batch contextual hidden encodings after performed by a BERT feed forward layer.
        """
        # [batch_size, from_seq_length, intermediate_size].
        intermediate_hidden_encodings = self.intermediate.forward(hidden_encodings=hidden_encodings)
        # [batch_size, from_seq_length, hidden_size].
        output_hidden_encodings = self.output.forward(
            before_intermediate_hidden_encodings=hidden_encodings,
            after_intermediate_hidden_encodings=intermediate_hidden_encodings
        )
        return output_hidden_encodings


class BertLayer(nn.Module):
    """Construct a BERT layer, including an attention layer (an object of `BertAttention`) and a feed-forward layer
    (an object of `BertFFN`).

    Example::
        ```python
        torch.manual_seed(seed=2021)
        # Load configuration of model from a specific JSON file.
        config = BertConfig.from_json(json_dir="settings/config.json")
        # Create input tensors
        BATCH_SIZE = 32
        FROM_SEQ_LENGTH = 256
        TO_SEQ_LENGTH = 64
        from_hidden_encodings = torch.rand(
            size=[BATCH_SIZE, FROM_SEQ_LENGTH, config.hidden_size], dtype=torch.float32, requires_grad=False
        )
        to_hidden_encodings = torch.rand(
            size=[BATCH_SIZE, TO_SEQ_LENGTH, config.hidden_size], dtype=torch.float32, requires_grad=False
        )
        attention_mask_4d = None
        # Instantiate the Bert embeddings module.
        module = BertLayer(config=config)
        # Get the output tensors.
        output_tuple = module.forward(
            from_hidden_encodings=from_hidden_encodings,
            to_hidden_encodings=to_hidden_encodings,
            attention_mask_4d=attention_mask_4d,
            return_attention_probs=True
        )
        # Accessing the output tensors.
        hidden_encodings = output_tuple[0]
        attention_probs = output_tuple[1]
        ```
    """

    def __init__(self, config: BertConfig):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config=config)
        self.ffn = BertFFN(config=config)

    def forward(self,
                from_hidden_encodings: torch.Tensor,
                to_hidden_encodings: torch.Tensor,
                attention_mask_4d: Optional[torch.Tensor],
                return_attention_probs: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        :param from_hidden_encodings: (torch.Tensor of shape [batch_size, from_seq_length, hidden_size]).
            Mini-batch hidden contextual encodings for computing query encodings later.
        :param to_hidden_encodings: (torch.Tensor of shape [batch_size, to_seq_length, hidden_size]).
            Mini-batch hidden contextual encodings for computing key encodings and value encodings later.
            NOTE:
                * In encoder-decoder attention, from_hidden_encodings comes from decoder part and to_hidden_encodings
                comes from encoder part, `from_seq_length` may not be equal to `to_seq_length`;
                * In encoder-encoder attention, both from_hidden_encodings and to_hidden_encodings come from encoder,
                `from_seq_length` must be equal to `to_seq_length` because they both come from the same hidden
                encodings;
                * In decoder-decoder attention, both from_hidden_encodings and to_hidden_encodings come from decoder,
                `from_seq_length` must be equal to `to_seq_length` because they both come from the same hidden
                encodings.
        :param attention_mask_4d: (torch.Tensor of shape
            [batch_size, num_attention_heads, from_seq_length, to_seq_length]
            or None, optional, defaults to None).
            Precomputed attention mask 4d tensor in which `-inf` represents mask.
            If None, don't have mask element.
        :param return_attention_probs: (bool, optional, defaults to False).
             Whether to return attention probs tensor of shape
            [batch_size, num_attention_heads, from_seq_length, to_seq_length] in the output tensors.

        :return bert_layer_output_tuple: (tuple of
            (:torch.Tensor:`ffn_hidden_encodings`, :torch.Tensor:`attention_probabilities` or None)).
            * ffn_hidden_encodings: (torch,Tensor of shape [batch_size, from_seq_length, hidden_size]).
                Mini-batch contextual hidden encodings after performed by a BERT layer.
            * attention_probabilities: (torch.Tensor of shape
                [batch_size, num_attention_heads, from_seq_length, to_seq_length] or None, optional, defaults to None).
                Mini-batch attention probabilities.
                If None, it means don't return the attention probabilities. This term is controlled by argument
                `return_attention_probs`.
        """
        attention_output_tuple = self.attention.forward(
            from_hidden_encodings=from_hidden_encodings,
            to_hidden_encodings=to_hidden_encodings,
            attention_mask_4d=attention_mask_4d,
            return_attention_probs=return_attention_probs
        )
        attention_hidden_encodings = attention_output_tuple[0]
        attention_probabilities = attention_output_tuple[1]

        # [batch_size, from_seq_length, hidden_size].
        ffn_hidden_encodings = self.ffn.forward(hidden_encodings=attention_hidden_encodings)

        bert_layer_output_tuple = tuple()
        bert_layer_output_tuple += (ffn_hidden_encodings,)
        bert_layer_output_tuple += (attention_probabilities,)
        return bert_layer_output_tuple


class BertEncoder(nn.Module):
    """Construct a BERT encoder, namely a stack of BERT layers.

    Example::
        ```python
        torch.manual_seed(seed=2021)
        # Load configuration of model from a specific JSON file.
        config = BertConfig.from_json(json_dir="settings/config.json")
        # Create input tensors
        BATCH_SIZE = 32
        FROM_SEQ_LENGTH = 256
        TO_SEQ_LENGTH = 64
        embeddings = torch.rand(
            size=[BATCH_SIZE, FROM_SEQ_LENGTH, config.hidden_size], dtype=torch.float32, requires_grad=False
        )
        attention_mask_4d = None
        # Instantiate the Bert embeddings module.
        model = BertEncoder(config=config)
        # Get the output tensors.
        output_tuple = model.forward(
            embeddings=embeddings,
            attention_mask_4d=attention_mask_4d
        )
        # Accessing the output tensors.
        last_encoder_layer_hidden_encodings = output_tuple[0]
        last_encoder_layer_attention_probs = output_tuple[1]
        all_hidden_encodings_tuple = output_tuple[2]
        all_attention_probs_tuple = output_tuple[3]
        ...
        ```
    """

    def __init__(self, config: BertConfig):
        super(BertEncoder, self).__init__()
        self.layers = nn.ModuleList(modules=[BertLayer(config=config) for _ in range(config.num_hidden_layers)])

    def forward(self,
                embeddings: torch.Tensor,
                attention_mask_4d: Optional[torch.Tensor],
                return_all_hidden_encodings: bool = False,
                return_all_attention_probs: bool = False) -> \
            Tuple[
                torch.Tensor,
                torch.Tensor,
                Optional[Tuple[torch.Tensor]],
                Optional[Tuple[torch.Tensor]]
            ]:
        """
        :param embeddings: (torch.Tensor of shape [batch_size, seq_length, hidden_size]).
            Mini-batch linear combination of token embeddings, token type embeddings and positional embeddings.
        :param attention_mask_4d: (torch.Tensor of shape
            [batch_size, num_attention_heads, from_seq_length, to_seq_length]
            or None, optional, defaults to None).
            Precomputed attention mask 4d tensor in which `-inf` represents mask.
            If None, don't have mask element.
        :param return_all_hidden_encodings: (bool, optional, defaults to False).
            Whether to return contextual hidden encodings which each BERT layer outputs in a tuple.
        :param return_all_attention_probs: (bool, optional, defaults to False).
            Whether to return attention probabilities which each BERT layer computes in a tuple.

        :return encoder_output_tuple: (tuple of torch.Tensor or None with following order).
            * hidden_encodings: (torch.Tensor of shape [batch_size, seq_length, hidden_size]).
                Mini-batch contextual hidden encodings which last BERT encoder layer outputs.
            * attention_probs: (torch.Tensor of shape
                [batch_size, num_attention_heads, seq_length, seq_length]).
                Mini-batch multi-headed self-attention probabilities which last BERT encoder layer computes.
            * all_hidden_encodings_tuple: (Tuple of torch.Tensor of shape
                [batch_size, seq_length, hidden_size] or None, optional, defaults to None).
                Mini-batch contextual hidden encodings which each BERT encoder layer outputs.
                If argument in module `return_all_layers_hidden_encodings` is True, it's a tuple of length
                `config.num_hidden_layers`, else, None.
            * all_attention_probs_tuple: (Tuple of torch.Tensor of shape
                [batch_size, num_attention_heads, seq_length, seq_length] or None, optional, defaults to None).
                Mini-batch multi-headed self-attention probabilities which each BERT encoder layer computes.
                If argument in module `return_all_layers_attention_probs` is True, it's a tuple of length
                `config.num_hidden_layers`, else, None.
        """
        all_hidden_encodings_tuple = tuple() if return_all_hidden_encodings is True else None
        all_attention_probs_tuple = tuple() if return_all_attention_probs is True else None
        hidden_encodings = embeddings
        # Solving the problem in PyCharm IDE:
        # Local variable 'attention_probs' might be referenced before assignment.
        attention_probs: Optional[torch.Tensor] = None
        for i, layer in enumerate(self.layers):
            assert isinstance(layer, BertLayer)
            if i < len(self.layers) - 1:
                return_attention_probs = True if all_attention_probs_tuple is not None else False
            else:
                return_attention_probs = True  # Make sure to output last attention probs tensor.

            layer_output_tuple = layer.forward(
                from_hidden_encodings=hidden_encodings,
                to_hidden_encodings=hidden_encodings,
                attention_mask_4d=attention_mask_4d,
                return_attention_probs=return_attention_probs
            )

            hidden_encodings = layer_output_tuple[0]
            attention_probs = layer_output_tuple[1]
            if all_hidden_encodings_tuple is not None:
                all_hidden_encodings_tuple += (hidden_encodings,)
            if all_attention_probs_tuple is not None:
                all_attention_probs_tuple += (attention_probs,)

        encoder_output_tuple = tuple()
        encoder_output_tuple += (hidden_encodings,)  # last encoder-layer's output contextual hidden encodings.
        encoder_output_tuple += (attention_probs,)  # last encoder-layer's attention probs.
        encoder_output_tuple += (all_hidden_encodings_tuple,)  # all encoder layers' hidden encodings or None.
        encoder_output_tuple += (all_attention_probs_tuple,)  # all encoder layers' attention probs or None

        return encoder_output_tuple


class BertPooler(nn.Module):
    def __init__(self, config: BertConfig):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size, bias=True)
        self.activation = nn.Tanh()

    def forward(self, hidden_encodings: torch.Tensor, pooling_pos_in_seq: int = 0) -> torch.Tensor:
        """
        :param hidden_encodings: (torch.Tensor of shape [batch_size, seq_length, hidden_size]).
        :param pooling_pos_in_seq: (int, optional, default to 0).
        :return pooled_specific_token_hidden_encodings: (torch.Tensor of shape [batch_size, hidden_size]).
            Mini-batch pooled hidden encodings.
        """
        seq_length = hidden_encodings.size()[1]
        if pooling_pos_in_seq > seq_length - 1 or pooling_pos_in_seq < -seq_length:
            raise ValueError(
                "taken_position_in_seq must be in [0,...,%d] or [-1,...,-%d]" %
                (pooling_pos_in_seq - 1, -pooling_pos_in_seq)
            )
        # [batch_size, hidden_size].
        # Take `taken_position_in_seq`-th token in sequence of each sample.
        specific_token_hidden_encodings = hidden_encodings[:, pooling_pos_in_seq, :]
        pooled_specific_token_hidden_encodings = self.dense.forward(input=specific_token_hidden_encodings)
        pooled_specific_token_hidden_encodings = self.activation.forward(input=pooled_specific_token_hidden_encodings)
        return pooled_specific_token_hidden_encodings


class BertInitialModeling(nn.Module):
    """BERT Modeling initialization."""

    def __init__(self, config: BertConfig):
        super(BertInitialModeling, self).__init__()
        self.config = config

    def init_parameters(self):
        self.apply(fn=self._init_parameters)

    def _init_parameters(self, module: nn.Module):
        """Helper function for :func:`init_parameters`."""
        if isinstance(module, nn.Embedding):
            # Slightly different from the TF version which uses `truncated_normal` for initialization.
            # NOTE: module.weight is an :obj:`Parameter`, and module.weight.data is an :obj:`Tensor`.
            torch.nn.init.normal_(tensor=module.weight.data, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.Linear):
            torch.nn.init.normal_(tensor=module.weight.data, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(tensor=module.bias.data)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(tensor=module.weight.data)
            torch.nn.init.zeros_(tensor=module.bias.data)
        else:
            # We don't need to consider other kinds of module such as nn.Dropout.
            pass


class BertModeling(BertInitialModeling):
    """BERT model, inherit from :class:`BertInitialModeling` for modeling initialization.

    Example usage::
        ```python
        torch.manual_seed(seed=2021)
        # Load configuration of model from a specific JSON file.
        config = BertConfig.from_json(json_dir="settings/config.json")
        # Create input tensors
        BATCH_SIZE = 32
        FROM_SEQ_LENGTH = 256
        TO_SEQ_LENGTH = 64
        token_ids = torch.randint(
            low=0, high=config.vocab_size,
            size=[BATCH_SIZE, FROM_SEQ_LENGTH], dtype=torch.long, requires_grad=False
        )
        attention_mask_4d = None
        # Instantiate the Bert embeddings module.
        samples_dict = {
            BertModeling4PreTrainingSampleKeys.TOKEN_IDS: token_ids,
            BertModeling4PreTrainingSampleKeys.TOKEN_TYPE_IDS: None,
            BertModeling4PreTrainingSampleKeys.POSITION_IDS: None,
            BertModeling4PreTrainingSampleKeys.ATTENTION_MASK: None
        }

        model = BertModeling(config=config)
        # Get the output tensors.
        output_dict = model.forward(samples_dict=samples_dict, return_token_embeddings=False)
        # Accessing the output tensors.
        encoder_hidden_encodings = output_dict.get(BertModeling4PreTrainingBackboneKeys.ENCODER_HIDDEN_ENCODINGS)
        pooler_hidden_encodings = output_dict.get(BertModeling4PreTrainingBackboneKeys.POOLER_HIDDEN_ENCODINGS)        ```
        ...
        ```
    """

    def __init__(self, config: BertConfig):
        super(BertModeling, self).__init__(config=config)  # Adds a class property `self.config`.
        self.embeddings = BertEmbeddings(config=config)
        self.encoder = BertEncoder(config=config)
        self.pooler = BertPooler(config=config)
        # Inherit the method `init_parameters()` from the parent :class:`BertInitialModeling`.
        self.init_parameters()

    def forward(self, samples_dict: Dict[str, Optional[torch.Tensor]], **kwargs) -> Dict[str, Optional[torch.Tensor]]:
        """
        :param samples_dict: (dict of tensors or None as following).
            * token_ids: (torch.Tensor of shape [batch_size, seq_length]).
                Mini-batch sequences of token indices.
                Each token index must be in range [0, ..., config.vocab_size - 1].
            * token_type_ids: (torch.Tensor of shape [batch_size, seq_length] or None, optional, defaults to None).
                Mini-batch sequences of token type indices.
                Each token type index must be in range [0, ..., config.vocab_type_size - 1].
                If None, all tokens in mini-batch sequences belong to the 0-th token type.
            * position_ids: (torch.Tensor of shape [batch_size, seq_length] or None, optional, defaults to None).
                Mini-batch positional indices of tokens in sequence.
                Each positional index must be in range [0, ..., seq_length - 1] in current batch.
                If None, all tokens belong to the positions in `token_ids`, i.e., [0, 1, ..., seq_length - 1].
            * attention_mask: (torch.Tensor of shape [batch_size, 1, from_seq_length, to_seq_length] or None,
                optional, defaults to None).
                Mini-batch PRE-COMPUTED attention mask tensor.
                If not None, the (i, j, k, l)-th element in :torch.Tensor:`attention_mask_4d` means:
                i-th sample in current mini-batch, j-th attention head, when encode k-th token in current sequence,
                whether to consider l-th token in sequence.
                If element value takes `-inf`, DON'T consider, else, consider the affect of l-th token in sequence on
                k-th token in sequence when encodes k-th token in current sequence.
                If None, don't perform mask operations when computes attention scores in each encoder layer.
        :param kwargs:
            * return_token_embeddings: (bool, optinal, defaults to False).
                Whether to return token embeddings.
                If None, token embeddings term in output object is None.
            * return_token_type_embeddings: (bool, optional, defaults to False).
                Whether to return the token type embeddings.
                If None, token type embeddings term in output object is None.
            * return_position_embeddings: (bool, optional, defaults to False).
                Whether to return the position embeddings.
                If None, position embeddings term in output object is None.
            * return_all_layers_hidden_encodings: (bool, optional, defaults to False).
                Whether to return contextual hidden encodings which each BERT encoder layer outputs in a tuple.
            * return_all_layers_attention_probs: (bool, optional, defaults to False).
                Whether to return attention probabilities which each BERT encoder layer computes in a tuple.
            * pooling_pos_in_seq: (int, optional, default to 0).
                Taken i-th token in sequence before computing pooler hidden encodings in BERT pooler layer.
        """
        if samples_dict.get(BertModeling4PreTrainingSampleKeys.TOKEN_IDS) is None:
            raise ValueError("{} in `samples_dict` "
                             "CANNOT be a None.".format(BertModeling4PreTrainingSampleKeys.TOKEN_IDS))
        input_shape = samples_dict.get(BertModeling4PreTrainingSampleKeys.TOKEN_IDS).size()  # [batch_size, seq_length].
        device = samples_dict.get(BertModeling4PreTrainingSampleKeys.TOKEN_IDS).device
        # Get and update `samples_dict`.
        # If :obj:`token_type_ids` is None, all tokens in mini-batch sequences belong to the 0-th token type.
        if samples_dict.get(BertModeling4PreTrainingSampleKeys.TOKEN_TYPE_IDS) is None:
            # key `BertModeling4PreTrainingSampleKeys.TOKEN_TYPE_IDS` is not in samples_dict.keys().
            # Namely, :torch.Tensor:`token_type_ids` is not in original :dict:`samples_dict` which dataloader output.
            samples_dict[BertModeling4PreTrainingSampleKeys.TOKEN_TYPE_IDS] = torch.zeros(
                size=input_shape, dtype=torch.long, device=device, requires_grad=False
            )
        # If :obj:`position_ids` is None, all tokens belong to the positions in `token_ids`,
        # i.e., [0, 1, ..., seq_length - 1].
        if samples_dict.get(BertModeling4PreTrainingSampleKeys.POSITION_IDS) is None:
            # key `BertModeling4PreTrainingSampleKeys.POSITION_IDS` is not in samples_dict.keys().
            # Namely :torch.Tensor:`position_ids` is not in original :dict:`samples_dict` which dataloader output.
            samples_dict[BertModeling4PreTrainingSampleKeys.POSITION_IDS] = \
                self.embeddings.position_ids[:, : input_shape[1]] # [1, seq_length].

        # Get optional arguments in forward function.
        return_token_embeddings: bool = kwargs.pop("return_token_embeddings", False)
        return_token_type_embeddings: bool = kwargs.pop("return_token_type_embeddings", False)
        return_position_embeddings: bool = kwargs.pop("return_position_embeddings", False)
        return_all_hidden_encodings: bool = kwargs.pop("return_all_hidden_encodings", False)
        return_all_attention_probs: bool = kwargs.pop("return_all_attention_probs", False)
        pooling_pos_in_seq: int = kwargs.pop("pooling_pos_in_seq", 0)

        # -----Embeddings layer-----
        embeddings_module_output_tuple = self.embeddings.forward(
            token_ids=samples_dict.get(BertModeling4PreTrainingSampleKeys.TOKEN_IDS),
            token_type_ids=samples_dict.get(BertModeling4PreTrainingSampleKeys.TOKEN_TYPE_IDS),
            position_ids=samples_dict.get(BertModeling4PreTrainingSampleKeys.POSITION_IDS),
            return_token_embeddings=return_token_embeddings,
            return_token_type_embeddings=return_token_type_embeddings,
            return_position_embeddings=return_position_embeddings
        )

        embeddings = embeddings_module_output_tuple[0]
        token_embeddings = embeddings_module_output_tuple[1]
        # token_type_embeddings = embeddings_module_output_tuple[2]
        # position_embeddings = embeddings_module_output_tuple[3]

        # -----Encoder layer-----
        encoder_output_tuple = self.encoder.forward(
            embeddings=embeddings,
            attention_mask_4d=samples_dict.get(BertModeling4PreTrainingSampleKeys.ATTENTION_MASK),
            return_all_hidden_encodings=return_all_hidden_encodings,
            return_all_attention_probs=return_all_attention_probs
        )
        final_encoder_hidden_encodings = encoder_output_tuple[0]
        final_attention_probs = encoder_output_tuple[1]
        encoder_layers_hidden_encodings_tuple = encoder_output_tuple[2]
        encoder_layers_attention_probs_tuple = encoder_output_tuple[3]

        # -----Pooler layer-----
        pooler_hidden_encodings = self.pooler.forward(
            hidden_encodings=final_encoder_hidden_encodings,
            pooling_pos_in_seq=pooling_pos_in_seq
        )

        # **********Get the output dict of BertModeling.**********
        output_dict = dict()
        # Must return
        output_dict.setdefault(BertModeling4PreTrainingBackboneKeys.EMBEDDINGS,
                               embeddings)
        # Optional return
        output_dict.setdefault(BertModeling4PreTrainingBackboneKeys.TOKEN_EMBEDDINGS,
                               token_embeddings)
        # Must return
        output_dict.setdefault(BertModeling4PreTrainingBackboneKeys.ENCODER_HIDDEN_ENCODINGS,
                               final_encoder_hidden_encodings)
        # Must return
        output_dict.setdefault(BertModeling4PreTrainingBackboneKeys.ENCODER_ATTENTION_PROBABILITIES,
                               final_attention_probs)
        # Optional return
        output_dict.setdefault(BertModeling4PreTrainingBackboneKeys.ENCODER_HIDDEN_ENCODINGS_TUPLE,
                               encoder_layers_hidden_encodings_tuple)
        # Optional return
        output_dict.setdefault(BertModeling4PreTrainingBackboneKeys.ENCODER_ATTENTION_PROBABILITIES_TUPLE,
                               encoder_layers_attention_probs_tuple)
        # Must return
        output_dict.setdefault(BertModeling4PreTrainingBackboneKeys.POOLER_HIDDEN_ENCODINGS,
                               pooler_hidden_encodings)
        return output_dict


class BertPredictionTransformHead(nn.Module):
    """Further feature extraction layer after BERT encoder or pooler.
    It support both 3D input tensor and 2D input tensor.
    """

    def __init__(self, config: BertConfig):
        super(BertPredictionTransformHead, self).__init__()
        self.dense = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size, bias=True)
        self.activation = activation_fn(act_fn_str=config.hidden_act)
        self.layer_norm = BertLayerNorm(
            normalized_shape=config.hidden_size, eps=config.layer_norm_eps, elementwise_affine=True
        )

    def forward(self, hidden_encodings: torch.Tensor) -> torch.Tensor:
        """
        :param hidden_encodings: (torch.Tensor of shape [batch_size, seq_length, hidden_size] or of shape
            [batch_size, hidden_size]).
            if ndim of :torch.Tensor:`hidden_encodings` is 3, it means the output contextual hidden encodings of
            encoder layer, if 2, means the output contextual hidden encodings of pooler layer.
        :return output_hidden_encodings: (torch.Tensor of shape [batch_size, seq_length, hidden_size] or of shape
            [batch_size, hidden_size]).
        """
        # ndim == 2:
        # :torch.Tensor:`hidden_encodings` represents the mini-batch pooler contextual hidden encodings.
        # [batch_size, hidden_size].
        # ndim == 3:
        # :torch.Tensor:`hidden_encodings` represents the mini-batch encoder contextual hidden encodings.
        # [batch_size, seq_length, hidden_size].
        if hidden_encodings.ndimension() not in [2, 3]:
            raise NotImplementedError(
                "input `hidden_encodings` with ndim {} is invalid, "
                "it only support ndim of 2 or 3.".format(hidden_encodings.ndimension())
            )
        output_hidden_encodings = self.dense.forward(input=hidden_encodings)
        output_hidden_encodings = self.activation.forward(x=output_hidden_encodings)
        output_hidden_encodings = self.layer_norm.forward(input=output_hidden_encodings)
        return output_hidden_encodings


class BertLanguageModelingPredictionHead(nn.Module):
    """Language modeling prediction head after :nn.Module:`BertModeling`, for computing the mini-batch
    lm logits."""

    def __init__(self, config: BertConfig):
        super(BertLanguageModelingPredictionHead, self).__init__()
        self.transform = BertPredictionTransformHead(config=config)
        # The output weights are the same as the token embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(in_features=config.hidden_size, out_features=config.vocab_size, bias=False)
        self.bias = nn.Parameter(data=torch.zeros(size=[config.vocab_size], requires_grad=True))
        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`.
        self.decoder.bias = self.bias

    def forward(self, sequence_hidden_encodings: torch.Tensor) -> torch.Tensor:
        # [batch_size, seq_length, hidden_size].
        output_hidden_encodings = self.transform.forward(hidden_encodings=sequence_hidden_encodings)
        # [batch_size, seq_length, vocab_size].
        output_hidden_encodings = self.decoder.forward(input=output_hidden_encodings)
        return output_hidden_encodings


class BertModeling4PreTrainingHead(BertInitialModeling):
    """Integrates multi task logits output functions for computing loss functions into a single
    criterion class later.
    """

    def __init__(self, config: BertConfig):
        super(BertModeling4PreTrainingHead, self).__init__(config=config)
        self.language_modeling_prediction_head = BertLanguageModelingPredictionHead(config=config)
        # Sequence relationship prediction head after :nn.Module:`BertModeling`, for computing the mini-batch
        # seqs relationship logits.
        self.sequence_relationship_prediction_head = nn.Linear(
            in_features=config.hidden_size, out_features=2, bias=True
        )
        self.init_parameters()

    def forward(self,
                encoder_hidden_encodings: torch.Tensor,
                pooler_hidden_encodings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # [batch_size, seq_length, vocab_size].
        language_modeling_prediction_logits = self.language_modeling_prediction_head.forward(
            sequence_hidden_encodings=encoder_hidden_encodings
        )
        # [batch_size, 2].
        sequence_relationship_prediction_logits = self.sequence_relationship_prediction_head.forward(
            input=pooler_hidden_encodings
        )
        output_tuple = tuple()
        output_tuple += (language_modeling_prediction_logits,)
        output_tuple += (sequence_relationship_prediction_logits,)
        return output_tuple


class BertModeling4PreTraining(BertInitialModeling):
    """BERT for pre-training model, including an :obj:`BertModeling` and :obj:`Bert4PreTrainingHeads`.

    Example usage::
        ```python
        torch.manual_seed(seed=2021)
        # Load configuration of model from a specific JSON file.
        config = BertConfig.from_json(json_dir="settings/config.json")
        # Create input tensors
        BATCH_SIZE = 32
        FROM_SEQ_LENGTH = 256
        TO_SEQ_LENGTH = 64
        token_ids = torch.randint(
            low=0, high=config.vocab_size,
            size=[BATCH_SIZE, FROM_SEQ_LENGTH], dtype=torch.long, requires_grad=False
        )
        attention_mask = None
        # Instantiate the Bert embeddings module.
        samples_dict = {
            BertModeling4PreTrainingSampleKeys.TOKEN_IDS: token_ids,
            BertModeling4PreTrainingSampleKeys.TOKEN_TYPE_IDS: None,
            BertModeling4PreTrainingSampleKeys.POSITION_IDS: None,
            BertModeling4PreTrainingSampleKeys.ATTENTION_MASK: attention_mask
        }

        model = BertModeling4PreTraining(config=config)
        output_dict = model.forward(samples_dict=samples_dict)
        language_modeling_logits = output_dict.get(
            BertModeling4PreTrainingPredictionHeadKeys.LANGUAGE_MODELING_LOGITS
        )
        sequence_relationship_logits = output_dict.get(
            BertModeling4PreTrainingPredictionHeadKeys.SEQUENCE_RELATIONSHIP_LOGITS
        )

        a = model.bert.embeddings.token_embeddings.weight.data
        b = model.head.language_modeling_prediction_head.decoder.weight.data
        ```
    """

    def __init__(self, config: BertConfig):
        super(BertModeling4PreTraining, self).__init__(config=config)
        self.backbone = BertModeling(config=config)
        self.head = BertModeling4PreTrainingHead(config=config)
        self.init_parameters()
        self.init_lm_weight_with_vocab_embeddings()

    def init_lm_weight_with_vocab_embeddings(self):
        """Initialize the language modeling head prediction weight with vocabulary embeddings."""
        self.head.language_modeling_prediction_head.decoder.weight.data = \
            self.backbone.embeddings.token_embeddings.weight.data

    def forward(self, samples_dict: Dict[str, Optional[torch.Tensor]], **kwargs):
        bert_output_dict = self.backbone.forward(samples_dict=samples_dict, **kwargs)
        encoder_hidden_encodings = bert_output_dict.get(
            BertModeling4PreTrainingBackboneKeys.ENCODER_HIDDEN_ENCODINGS
        )
        pooler_hidden_encodings = bert_output_dict.get(
            BertModeling4PreTrainingBackboneKeys.POOLER_HIDDEN_ENCODINGS
        )
        # [batch_size, seq_length, vocab_size] and [batch_size, 2].
        language_modeling_logits, sequence_relationship_logits = self.head.forward(
            encoder_hidden_encodings=encoder_hidden_encodings, pooler_hidden_encodings=pooler_hidden_encodings
        )
        output_dict = dict()
        output_dict.setdefault(BertModeling4PreTrainingPredictionHeadKeys.LANGUAGE_MODELING_LOGITS,
                               language_modeling_logits)
        output_dict.setdefault(BertModeling4PreTrainingPredictionHeadKeys.SEQUENCE_RELATIONSHIP_LOGITS,
                               sequence_relationship_logits)
        return output_dict


class BertModeling4PreTrainingCriterion(nn.Module):
    """Criterion of :obj:`BertModeling4PreTraining`. It includes both language modeling loss computation and sequences
    pair relationship loss computation.
    """

    def __init__(self, config: BertConfig):
        super(BertModeling4PreTrainingCriterion, self).__init__()
        self.config = config
        self.criterion = nn.CrossEntropyLoss(ignore_index=ClsLossKeys.IGNORE_INDEX)

    def forward(self,
                inputs_dict: Dict[str, torch.Tensor],
                targets_dict: Dict[str, torch.Tensor]):
        """
        :param inputs_dict: (dict of logit output tensors as following). All valid keys defined in
            :class:`BertModeling4PreTrainingPredictionHeadKeys`.
            language_modeling_logits: (torch.Tensor of shape [batch_size, seq_length, vocab_size]).
            sequence_relationship_logits: (torch.Tensor of shape [batch_size, 2]).
        :param targets_dict: (dict of pretraining label tensors as following). All valid keys defined in
            :class:`BertModeling4PreTrainingTargetKeys`.
            language_modeling_labels: (torch.Tensor of shape [batch_size, seq_length] or None, optional,
                defaults to None).
                Labels for computing the masked language modeling loss.
                Indices should be in `-100`(i.e., IGNORE_INDEX) or [0, ..., config.vocab_size - 1].
                Labels with indices set to `-100` are ignored (masked), the loss is only computed for the tokens with
                labels in `[0, ..., config.vocab_size - 1]`.
                See ignore index setting in `ClsLossKeys`.
            sequence_relationship_labels: (torch.Tensor of shape [batch_size, 2] or None, optional, defaults to None).
                Labels for computing the next sequence prediction (classification) loss.
                Input should be a sequence pair.
                Indices should be in `[0, 1]`.
                `0` indicates sequence B is a random sequence,
                `1` indicates sequence B is a continuation of sequence A.
        :return output_dict: (Dict of Optional[torch.Tensor] as following).
            All valid keys defined in
            :class:`BertModeling4PreTrainingCriterionKeys`.
            * total_loss
            * language_modeling_loss
            * sequence_relationship_loss
        """
        total_loss = None
        # Declare inputs from input_dict.
        # [batch_size, seq_length, vocab_size].
        language_modeling_logits = inputs_dict.get(
            BertModeling4PreTrainingPredictionHeadKeys.LANGUAGE_MODELING_LOGITS
        )
        # [batch_size, 2].
        sequence_relationship_logits = inputs_dict.get(
            BertModeling4PreTrainingPredictionHeadKeys.SEQUENCE_RELATIONSHIP_LOGITS
        )

        # Declare targets from target_dict.
        # [batch_size, seq_length].
        language_modeling_targets = targets_dict.get(
            BertModeling4PreTrainingTargetKeys.LANGUAGE_MODELING_TARGETS
        )
        # [batch_size].
        sequence_relationship_targets = targets_dict.get(
            BertModeling4PreTrainingTargetKeys.SEQUENCE_RELATIONSHIP_TARGETS
        )
        if language_modeling_targets is not None and sequence_relationship_targets is not None:
            language_modeling_loss = self.criterion.forward(
                # [batch_size * seq_length, vocab_size].
                input=language_modeling_logits.view(size=[-1, self.config.vocab_size]),
                # [batch_size * seq_length].
                target=language_modeling_targets.view(size=[-1])
            )
            sequence_relationship_loss = self.criterion.forward(
                input=sequence_relationship_logits,  # [batch_size, 2].
                target=sequence_relationship_targets  # [batch_size].
            )
            total_loss = language_modeling_loss + sequence_relationship_loss
        elif language_modeling_targets is None and sequence_relationship_targets is None:
            language_modeling_loss = None
            sequence_relationship_loss = None
            print(
                "{} and {} in `target_dict` are both None".format(
                    BertModeling4PreTrainingTargetKeys.LANGUAGE_MODELING_TARGETS,
                    BertModeling4PreTrainingTargetKeys.SEQUENCE_RELATIONSHIP_TARGETS
                )
            )
        else:
            raise NotImplementedError(
                "{} and {} should both be not None or both be None".format(
                    BertModeling4PreTrainingTargetKeys.LANGUAGE_MODELING_TARGETS,
                    BertModeling4PreTrainingTargetKeys.SEQUENCE_RELATIONSHIP_TARGETS
                )
            )
        # Create the output dict.
        output_dict = dict()
        output_dict.setdefault(BertModeling4PreTrainingCriterionKeys.TOTAL_LOSS,
                               total_loss)
        output_dict.setdefault(BertModeling4PreTrainingCriterionKeys.LANGUAGE_MODELING_LOSS,
                               language_modeling_loss)
        output_dict.setdefault(BertModeling4PreTrainingCriterionKeys.SEQUENCE_RELATIONSHIP_LOSS,
                               sequence_relationship_loss)
        return output_dict
