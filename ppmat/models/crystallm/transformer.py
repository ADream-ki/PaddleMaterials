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

"""Transformer components for CrystaLLM model."""

import math
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ppmat.models.crystallm.config import CrystaLLMConfig


class LayerNorm(nn.Layer):
    """Custom LayerNorm implementation for precise alignment with PyTorch.
    
    This implementation uses the same epsilon (1e-5) as the original CrystaLLM
    to ensure numerical alignment during precision verification.
    """
    
    def __init__(self, ndim: int, bias: bool = True, eps: float = 1e-5):
        """Initialize LayerNorm.
        
        Args:
            ndim: Dimensionality of the input tensor.
            bias: Whether to add a learnable bias term.
            eps: Small value to prevent division by zero.
        """
        super().__init__()
        self.weight = self.create_parameter(
            shape=[ndim],
            default_initializer=nn.initializer.Constant(1.0)
        )
        self.bias = None
        if bias:
            self.bias = self.create_parameter(
                shape=[ndim],
                default_initializer=nn.initializer.Constant(0.0)
            )
        self.eps = eps
    
    def forward(self, x):
        """Apply layer normalization.
        
        Args:
            x: Input tensor of shape (..., ndim).
        
        Returns:
            Normalized tensor of the same shape.
        """
        return F.layer_norm(x, normalized_shape=self.weight.shape, 
                           weight=self.weight, bias=self.bias, epsilon=self.eps)


# Removed RotaryPositionalEmbedding - using learnable embeddings instead


class MultiHeadAttention(nn.Layer):
    """Multi-head causal self-attention layer.
    
    Implements scaled dot-product attention with multiple heads and causal masking.
    """
    
    def __init__(self, config: CrystaLLMConfig):
        """Initialize multi-head attention.
        
        Args:
            config: CrystaLLMConfig instance.
        """
        super().__init__()
        assert config.n_embd % config.n_head == 0, \
            f"n_embd ({config.n_embd}) must be divisible by n_head ({config.n_head})"
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout_p = config.dropout
        
        # Linear projections for Q, K, V
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias_attr=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias_attr=config.bias)
        
        # Dropout layers
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # Register causal mask buffer
        self.register_buffer(
            "causal_mask",
            paddle.tril(paddle.ones([config.block_size, config.block_size])).unsqueeze(0).unsqueeze(0)
        )
    
    def forward(self, x):
        """Apply multi-head attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, n_embd).
        
        Returns:
            Output tensor of shape (batch_size, seq_length, n_embd).
        """
        B, T, C = x.shape
        
        # Project to Q, K, V
        qkv = self.c_attn(x)
        q, k, v = paddle.split(qkv, 3, axis=-1)
        
        # Reshape for multi-head attention
        # (B, T, n_embd) -> (B, T, n_head, head_dim) -> (B, n_head, T, head_dim)
        q = q.reshape([B, T, self.n_head, self.head_dim]).transpose([0, 2, 1, 3])
        k = k.reshape([B, T, self.n_head, self.head_dim]).transpose([0, 2, 1, 3])
        v = v.reshape([B, T, self.n_head, self.head_dim]).transpose([0, 2, 1, 3])
        
        # Compute attention scores
        scores = paddle.matmul(q, k.transpose([0, 1, 3, 2])) / math.sqrt(self.head_dim)
        
        # Apply causal mask
        mask = self.causal_mask[:, :, :T, :T]
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax
        attn_weights = F.softmax(scores, axis=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention to values
        attn_output = paddle.matmul(attn_weights, v)
        
        # Reshape back to (B, T, n_embd)
        attn_output = attn_output.transpose([0, 2, 1, 3]).reshape([B, T, C])
        
        # Final projection
        output = self.c_proj(attn_output)
        output = self.resid_dropout(output)
        
        return output


def gelu(x):
    """Custom GELU activation function matching PyTorch reference.
    
    This uses the tanh approximation as specified in the original GPT-2 paper
    and CrystaLLM PyTorch implementation, NOT the standard GELU.
    
    Reference: "Gaussian Error Linear Units (GELUs)", https://arxiv.org/abs/1606.08415
    
    Args:
        x: Input tensor.
    
    Returns:
        Tensor after applying GELU activation.
    """
    return 0.5 * x * (1.0 + paddle.tanh(
        math.sqrt(2.0 / math.pi) * (x + 0.044715 * paddle.pow(x, 3.0))
    ))


class FeedForward(nn.Layer):
    """Feed-forward network (MLP) layer.
    
    Implements a two-layer feed-forward network with GELU activation.
    """
    
    def __init__(self, config: CrystaLLMConfig):
        """Initialize feed-forward network.
        
        Args:
            config: CrystaLLMConfig instance.
        """
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias_attr=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias_attr=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        """Apply feed-forward network.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, n_embd).
        
        Returns:
            Output tensor of shape (batch_size, seq_length, n_embd).
        """
        x = self.c_fc(x)
        x = gelu(x)  # Use custom GELU instead of F.gelu
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Layer):
    """Transformer block combining attention and feed-forward layers.
    
    Implements a single Transformer block with:
    - Layer normalization (pre-LN)
    - Multi-head causal self-attention
    - Feed-forward network
    - Residual connections
    """
    
    def __init__(self, config: CrystaLLMConfig):
        """Initialize Transformer block.
        
        Args:
            config: CrystaLLMConfig instance.
        """
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias, eps=config.layer_norm_eps)
        self.attn = MultiHeadAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias, eps=config.layer_norm_eps)
        self.mlp = FeedForward(config)
    
    def forward(self, x):
        """Apply Transformer block.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, n_embd).
        
        Returns:
            Output tensor of shape (batch_size, seq_length, n_embd).
        """
        # Attention with residual connection
        x = x + self.attn(self.ln_1(x))
        
        # Feed-forward with residual connection
        x = x + self.mlp(self.ln_2(x))
        
        return x


class TokenEmbedding(nn.Layer):
    """Token embedding layer.
    
    Maps token IDs to embedding vectors.
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int):
        """Initialize token embedding.
        
        Args:
            vocab_size: Size of vocabulary.
            embedding_dim: Dimensionality of embeddings.
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
    
    def forward(self, x):
        """Get embeddings for token IDs.
        
        Args:
            x: Token ID tensor of shape (batch_size, seq_length).
        
        Returns:
            Embedding tensor of shape (batch_size, seq_length, embedding_dim).
        """
        return self.embedding(x)


class PositionalEmbedding(nn.Layer):
    """Learnable positional embedding layer (aligned with original CrystaLLM).
    
    Uses learnable positional embeddings instead of fixed sinusoidal encoding,
    matching the original PyTorch implementation.
    """
    
    def __init__(self, max_seq_length: int, embedding_dim: int):
        """Initialize positional embedding.
        
        Args:
            max_seq_length: Maximum sequence length.
            embedding_dim: Dimensionality of embeddings.
        """
        super().__init__()
        # Learnable positional embedding (same as original CrystaLLM)
        self.pos_embedding = nn.Embedding(max_seq_length, embedding_dim)
    
    def forward(self, x):
        """Add positional embeddings.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, embedding_dim).
        
        Returns:
            Tensor with positional embeddings added.
        """
        B, T, C = x.shape
        
        # Create position indices on the same device as input
        # This is critical for GPU training to avoid device mismatch errors
        pos = paddle.arange(0, T, dtype='int64')
        
        # Ensure pos is on the same device as x
        if x.place.is_gpu_place():
            pos = pos.cuda(x.place.gpu_device_id())
        
        pos = pos.unsqueeze(0)  # (1, T)
        
        # Get positional embeddings
        pos_emb = self.pos_embedding(pos)  # (1, T, C)
        
        # Add to input
        return x + pos_emb
