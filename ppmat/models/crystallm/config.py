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

from dataclasses import dataclass
from typing import Optional


@dataclass
class CrystaLLMConfig:
    """Configuration class for CrystaLLM model.
    
    This configuration class is used to instantiate a CrystaLLM model according to the specified arguments,
    defining the model architecture. Instantiating a configuration with the defaults will yield a similar
    configuration to that of the CrystaLLM small model.
    
    Configuration objects inherit from PretrainedConfig and can be used to control the model outputs.
    
    Args:
        vocab_size (int, optional, defaults to 371):
            Vocabulary size of the model. Defines the number of different tokens that can be represented
            by the inputs_ids passed when calling CrystaLLM.
        
        block_size (int, optional, defaults to 1024):
            The maximum sequence length that this model might ever be used with. Typically set to something large
            just in case (e.g., 512 or 1024 or 2048).
        
        n_layer (int, optional, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        
        n_head (int, optional, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        
        n_embd (int, optional, defaults to 768):
            Dimensionality of the embeddings and hidden states.
        
        dropout (float, optional, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        
        bias (bool, optional, defaults to True):
            Whether the linear and layer norm layers have a bias term.
        
        initializer_range (float, optional, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        
        layer_norm_eps (float, optional, defaults to 1e-5):
            The epsilon used by the layer normalization layers.
    """
    
    vocab_size: int = 371
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-5
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
            )
        
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")
        
        if self.block_size <= 0:
            raise ValueError(f"block_size must be positive, got {self.block_size}")
        
        if self.n_layer <= 0:
            raise ValueError(f"n_layer must be positive, got {self.n_layer}")
        
        if self.n_head <= 0:
            raise ValueError(f"n_head must be positive, got {self.n_head}")
        
        if self.n_embd <= 0:
            raise ValueError(f"n_embd must be positive, got {self.n_embd}")
    
    @classmethod
    def from_dict(cls, config_dict):
        """Create a config from a dictionary."""
        return cls(**config_dict)
    
    def to_dict(self):
        """Convert config to dictionary."""
        return {
            "vocab_size": self.vocab_size,
            "block_size": self.block_size,
            "n_layer": self.n_layer,
            "n_head": self.n_head,
            "n_embd": self.n_embd,
            "dropout": self.dropout,
            "bias": self.bias,
            "initializer_range": self.initializer_range,
            "layer_norm_eps": self.layer_norm_eps,
        }


# Predefined configurations for different model sizes
# These configurations match the PyTorch reference implementation
CRYSTALLM_SMALL_CONFIG = CrystaLLMConfig(
    vocab_size=371,
    block_size=1024,
    n_layer=12,  # Aligned with PyTorch small model
    n_head=12,
    n_embd=768,
    dropout=0.0,  # For pretraining
    bias=True,  # Match PyTorch default
)

CRYSTALLM_MEDIUM_CONFIG = CrystaLLMConfig(
    vocab_size=371,
    block_size=1024,
    n_layer=12,
    n_head=12,
    n_embd=768,
    dropout=0.0,
    bias=True,
)

CRYSTALLM_LARGE_CONFIG = CrystaLLMConfig(
    vocab_size=371,
    block_size=2048,
    n_layer=24,
    n_head=24,  # Aligned with PyTorch large model
    n_embd=1536,
    dropout=0.0,
    bias=True,
)
