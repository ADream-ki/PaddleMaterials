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

"""CrystaLLM model implementation in PaddlePaddle."""

import math
import os
import json
from typing import Optional, Tuple

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ppmat.models.crystallm.config import CrystaLLMConfig
from ppmat.models.crystallm.transformer import (
    LayerNorm, TokenEmbedding, PositionalEmbedding, TransformerBlock
)


class CrystaLLM(nn.Layer):
    """CrystaLLM: A GPT-2 style Transformer model for crystal structure generation.
    
    This model implements a causal language model based on the Transformer architecture,
    specifically designed for generating crystal structures in CIF format.
    """
    
    def __init__(self, config: CrystaLLMConfig):
        """Initialize CrystaLLM model.
        
        Args:
            config: CrystaLLMConfig instance containing model hyperparameters.
        """
        super().__init__()
        self.config = config
        
        # Token embedding
        self.token_embedding = TokenEmbedding(
            vocab_size=config.vocab_size,
            embedding_dim=config.n_embd
        )
        
        # Positional embedding
        self.pos_embedding = PositionalEmbedding(
            max_seq_length=config.block_size,
            embedding_dim=config.n_embd
        )
        
        # Dropout after embeddings
        self.embedding_dropout = nn.Dropout(config.dropout)
        
        # Transformer blocks
        self.transformer_blocks = nn.LayerList([
            TransformerBlock(config) for _ in range(config.n_layer)
        ])
        
        # Final layer normalization
        self.final_ln = LayerNorm(config.n_embd, bias=config.bias, eps=config.layer_norm_eps)
        
        # Language model head (output projection)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias_attr=False)
        
        # Weight tying: share weights between token embedding and output layer
        self.lm_head.weight = self.token_embedding.embedding.weight
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for layer in self.sublayers():
            if isinstance(layer, nn.Linear):
                # Normal initialization for linear layers
                nn.initializer.Normal(std=self.config.initializer_range)(layer.weight)
                if layer.bias is not None:
                    nn.initializer.Constant(0.0)(layer.bias)
            elif isinstance(layer, nn.Embedding):
                # Normal initialization for embeddings
                nn.initializer.Normal(std=self.config.initializer_range)(layer.weight)
        
        # Special initialization for residual projections (GPT-2 paper)
        for i, block in enumerate(self.transformer_blocks):
            # Scale the residual projection weights
            scale = 1.0 / math.sqrt(2 * self.config.n_layer)
            nn.initializer.Normal(std=self.config.initializer_range * scale)(
                block.attn.c_proj.weight
            )
            nn.initializer.Normal(std=self.config.initializer_range * scale)(
                block.mlp.c_proj.weight
            )
    
    def forward(
        self,
        input_ids: paddle.Tensor,
        targets: Optional[paddle.Tensor] = None
    ) -> Tuple[paddle.Tensor, Optional[paddle.Tensor]]:
        """Forward pass of the model.
        
        Args:
            input_ids: Token IDs of shape (batch_size, seq_length).
            targets: Target token IDs for computing loss (optional).
        
        Returns:
            Tuple of (logits, loss):
            - logits: Predicted logits of shape (batch_size, seq_length, vocab_size).
            - loss: Cross-entropy loss (None if targets not provided).
        """
        B, T = input_ids.shape
        
        # Check sequence length
        if T > self.config.block_size:
            raise ValueError(
                f"Sequence length {T} exceeds block size {self.config.block_size}"
            )
        
        # Token embedding
        x = self.token_embedding(input_ids)
        
        # Add positional embedding
        x = self.pos_embedding(x)
        
        # Apply embedding dropout
        x = self.embedding_dropout(x)
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Apply final layer normalization
        x = self.final_ln(x)
        
        # Compute logits and loss
        if targets is not None:
            # Training: compute logits for all positions
            logits = self.lm_head(x)
            # Reshape for cross-entropy loss
            logits_flat = logits.reshape([-1, self.config.vocab_size])
            targets_flat = targets.reshape([-1])
            # Compute cross-entropy loss with ignore_index=-1
            loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=-1, reduction='mean')
        else:
            # Inference: only compute logits for the last position (optimization)
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        
        return logits, loss
    
    def generate(
        self,
        input_ids: paddle.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> paddle.Tensor:
        """Generate new tokens autoregressively.
        
        Args:
            input_ids: Initial token IDs of shape (batch_size, seq_length).
            max_new_tokens: Maximum number of new tokens to generate.
            temperature: Temperature for sampling (higher = more random).
            top_k: If set, only sample from top-k most likely tokens.
            top_p: If set, only sample from tokens with cumulative probability <= top_p.
        
        Returns:
            Generated token IDs of shape (batch_size, seq_length + max_new_tokens).
        """
        self.eval()
        
        # Get newline token ID for CIF file end detection
        try:
            from ppmat.models.crystallm import CIFTokenizer
            tokenizer = CIFTokenizer()
            newline_id = tokenizer.token_to_id.get("\n", None)
        except:
            newline_id = None
        
        prev_id = None
        
        for _ in range(max_new_tokens):
            # Crop input to block size if necessary
            if input_ids.shape[1] > self.config.block_size:
                input_ids_cond = input_ids[:, -self.config.block_size:]
            else:
                input_ids_cond = input_ids
            
            # Forward pass
            with paddle.no_grad():
                logits, _ = self(input_ids_cond)
            
            # Get logits for last token
            logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                indices_to_remove = logits < paddle.topk(logits, top_k)[0][:, -1:]
                logits[indices_to_remove] = float('-inf')
            
            # Apply top-p filtering (nucleus sampling)
            if top_p is not None:
                sorted_logits, sorted_indices = paddle.sort(logits, descending=True, axis=-1)
                cumsum_probs = paddle.cumsum(F.softmax(sorted_logits, axis=-1), axis=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumsum_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False  # Keep at least one token
                
                # Scatter sorted tensors back to original indexing
                for batch_idx in range(logits.shape[0]):
                    remove_mask = sorted_indices_to_remove[batch_idx]
                    remove_indices = sorted_indices[batch_idx][remove_mask]
                    logits[batch_idx, remove_indices] = float('-inf')
            
            # Sample from distribution
            probs = F.softmax(logits, axis=-1)
            next_token = paddle.multinomial(probs, num_samples=1)
            
            # Append to sequence
            input_ids = paddle.concat([input_ids, next_token], axis=1)
            
            # Stop on double newline (CIF file end detection)
            if newline_id is not None and prev_id is not None:
                if prev_id == newline_id and next_token.item() == newline_id:
                    break
            prev_id = next_token.item() if newline_id is not None else None
        
        return input_ids
    
    def get_num_params(self, non_embedding: bool = True) -> int:
        """Get total number of parameters.
        
        Args:
            non_embedding: If True, exclude positional embedding parameters.
        
        Returns:
            Total number of parameters.
        """
        num_params = sum(p.numel() for p in self.parameters())
        
        if non_embedding:
            # Subtract positional embedding parameters
            num_params -= self.pos_embedding.pos_embedding.weight.numel()
        
        return num_params
    
    def estimate_mfu(self, fwdbwd_per_iter: int, dt: float) -> float:
        """Estimate model FLOPs utilization (MFU).
        
        Args:
            fwdbwd_per_iter: Number of forward-backward passes per iteration.
            dt: Time per iteration in seconds.
        
        Returns:
            MFU as a fraction of peak FLOPS.
        """
        # Estimate number of FLOPs per iteration
        # See PaLM paper Appendix B: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        
        # Express throughput as ratio of A100 bfloat16 peak FLOPS
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu
    
    def save_pretrained(self, save_path: str) -> None:
        """Save model and config to disk.
        
        Args:
            save_path: Directory to save model.
        """
        os.makedirs(save_path, exist_ok=True)
        
        # Save model parameters
        paddle.save(
            self.state_dict(),
            os.path.join(save_path, "model.pdparams")
        )
        
        # Save config
        config_dict = self.config.to_dict()
        with open(os.path.join(save_path, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def from_pretrained(cls, load_path: str) -> 'CrystaLLM':
        """Load model and config from disk.
        
        Args:
            load_path: Directory containing model files.
        
        Returns:
            CrystaLLM model instance.
        """
        # Load config
        config_file = os.path.join(load_path, "config.json")
        with open(config_file, "r") as f:
            config_dict = json.load(f)
        config = CrystaLLMConfig.from_dict(config_dict)
        
        # Create model
        model = cls(config)
        
        # Load parameters
        model_file = os.path.join(load_path, "model.pdparams")
        state_dict = paddle.load(model_file)
        model.set_state_dict(state_dict)
        
        return model
    
    def configure_optimizers(
        self,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.01,
        betas: Tuple[float, float] = (0.9, 0.999)
    ) -> paddle.optimizer.Optimizer:
        """Configure optimizer for training.
        
        This implementation matches the PyTorch reference by separating parameters
        into decay/no_decay sets based on module types, and properly handles
        weight tying between token embedding and LM head.
        
        Args:
            learning_rate: Learning rate.
            weight_decay: Weight decay coefficient.
            betas: Betas for Adam optimizer.
        
        Returns:
            Configured optimizer.
        """
        # Separate parameters into decay/no_decay sets
        decay = set()
        no_decay = set()
        
        # Module types that should have weight decay
        whitelist_weight_modules = (nn.Linear,)
        # Module types that should NOT have weight decay
        blacklist_weight_modules = (nn.LayerNorm, nn.Embedding)
        
        for module_name, module in self.named_sublayers():
            for param_name, param in module.named_parameters(include_sublayers=False):
                full_param_name = f"{module_name}.{param_name}" if module_name else param_name
                
                if param_name.endswith('bias'):
                    # All biases will not be decayed
                    no_decay.add(full_param_name)
                elif param_name.endswith('weight') and isinstance(module, whitelist_weight_modules):
                    # Weights of whitelist modules will be weight decayed
                    decay.add(full_param_name)
                elif param_name.endswith('weight') and isinstance(module, blacklist_weight_modules):
                    # Weights of blacklist modules will NOT be weight decayed
                    no_decay.add(full_param_name)
        
        # Special case: lm_head.weight is tied with token_embedding.embedding.weight
        # Remove lm_head.weight from decay set to avoid double counting
        if 'lm_head.weight' in decay:
            decay.remove('lm_head.weight')
        
        # Create parameter dictionary
        param_dict = {pn: p for pn, p in self.named_parameters()}
        
        # Validate that we considered every parameter
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, f"Parameters in both sets: {inter_params}"
        assert len(param_dict.keys() - union_params) == 0, \
            f"Parameters not in either set: {param_dict.keys() - union_params}"
        
        # Create optimizer groups
        optim_groups = [
            {
                'params': [param_dict[pn] for pn in sorted(list(decay))],
                'weight_decay': weight_decay
            },
            {
                'params': [param_dict[pn] for pn in sorted(list(no_decay))],
                'weight_decay': 0.0
            },
        ]
        
        optimizer = paddle.optimizer.AdamW(
            learning_rate=learning_rate,
            parameters=optim_groups,
            beta1=betas[0],
            beta2=betas[1],
        )
        
        return optimizer
