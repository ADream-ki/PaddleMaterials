# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CrystaLLM - Crystal Structure Generation with Language Models.

This module contains the core model implementation for CrystaLLM.
For training scripts and utilities, see structure_generation_crystallm/.
"""

from ppmat.models.crystallm.config import CrystaLLMConfig
from ppmat.models.crystallm.data_models import Atom, Structure, Molecule
from ppmat.models.crystallm.tokenizer import CIFTokenizer
from ppmat.models.crystallm.model import CrystaLLM
from ppmat.models.crystallm.transformer import (
    LayerNorm, MultiHeadAttention, FeedForward, TransformerBlock,
    TokenEmbedding, PositionalEmbedding
)

__all__ = [
    "CrystaLLMConfig",
    "Atom",
    "Structure",
    "Molecule",
    "CIFTokenizer",
    "CrystaLLM",
    "LayerNorm",
    "MultiHeadAttention",
    "FeedForward",
    "TransformerBlock",
    "TokenEmbedding",
    "PositionalEmbedding",
]
