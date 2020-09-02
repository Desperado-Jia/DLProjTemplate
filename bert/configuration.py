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
from utils.configuration_utils import BaseConfig


class BertConfig(BaseConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`BertModel`.
    It is used to instantiate a Bert model according to the specified arguments, defining the model architecture.

    Args:
        vocab_size (int):
            Vocabulary size of tokens.
        vocab_type_size (int):
            Vocabulary size of token types.
        hidden_size (int):
            Dimensionality of each encoding layer and pooling layer (i.e., embedding size or contextual encoding size).
        num_attention_heads (int):
            Number of self-attention heads in each attention layer in the Transformer encoder.
        intermediate_size (int):
            The size of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        num_hidden_layers (int):
            Number of hidden layers in the Transformer encoder.
        max_position_embeddings (int, optional, defaults to 512):
            The maximum sequence length that this model might ever be used with.
            Typically set this to something large just in case (e.g., 512 or 1024 or 2048).
        hidden_act (str, optional, defaults to `gelu`):
            The non-linear activation function (function or string) in encoder and pooler.
        hidden_dropout_prob (float, optional, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (float, optional, defaults to 0.1):
            The dropout probability for attention probabilities in encoder.
        layer_norm_eps (float, optional, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        initializer_range (float, optional, defaults to 0.02):
            The standard deviation of the truncated normal initializer for initializing all weight trainable variables.
        pad_token_id (int, optional, defaults to 0):
            The id of the `padding` token.
        pooling_position_in_sequence (int, optional, defaults to 0):
            The pooling position in a sequence when pooler layer outputs.
        bidirectionality (false):
            Whether to use bi-directional encoder. If true, perform sequence mask for computing attention mask.
    """
    def __init__(self,
                 vocab_size: int,
                 vocab_type_size: int,
                 hidden_size: int,
                 num_attention_heads: int,
                 intermediate_size: int,
                 num_hidden_layers: int,
                 max_position_embeddings: int = 512,
                 hidden_act: str = "gelu",
                 hidden_dropout_prob: float = 0.1,
                 attention_probs_dropout_prob: float = 0.1,
                 layer_norm_eps: float = 1e-12,
                 initializer_range: float = 0.02,
                 pad_token_id: int = 0,
                 pooling_position_in_sequence: int = 0,
                 bidirectionality: bool = False,
                 **kwargs):
        super(BertConfig, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.vocab_type_size = vocab_type_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.max_position_embeddings = max_position_embeddings
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        self.pad_token_id = pad_token_id
        self.pooling_position_in_sequence = pooling_position_in_sequence
        self.bidirectionality = bidirectionality

