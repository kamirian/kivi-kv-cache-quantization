"""
LLaMA attention module with KIVI KV-cache, and the replacement helper.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaRotaryEmbedding

from .cache import KIVICache


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat key/value heads for Grouped Query Attention (GQA).
    hidden_states: [batch, num_key_value_heads, slen, head_dim]
    returns:       [batch, num_attention_heads, slen, head_dim]
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    """Apply rotary position embeddings to query and key tensors."""
    cos = cos.unsqueeze(1)  # [bs, 1, seq_len, head_dim]
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def reset_all_kivi_caches(model):
    """Reset KIVI caches for all attention layers in a LLaMA model."""
    for layer in model.model.layers:
        attn = layer.self_attn
        if hasattr(attn, "reset_kivi_cache"):
            attn.reset_kivi_cache()


# ---------------------------------------------------------------------------
# LlamaAttentionWithKIVI
# ---------------------------------------------------------------------------

class LlamaAttentionWithKIVI(nn.Module):
    """
    Modified LlamaAttention that uses a KIVI cache instead of the standard
    HuggingFace past_key_values cache.
    """

    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads "
                f"(got hidden_size={self.hidden_size}, num_heads={self.num_heads})."
            )

        # Linear projections
        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim,
            bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            self.hidden_size, self.hidden_size,
            bias=config.attention_bias
        )

        # Rotary embeddings
        self.rotary_emb = LlamaRotaryEmbedding(config=config)

        # KIVI cache hyperparameters
        self.use_kivi = True
        self.kivi_num_bits = 2
        self.kivi_group_size = 32
        self.kivi_residual_length = 128
        self.kivi_cache = None  # Initialized per head on first use

    def _init_kivi_cache(self):
        """Initialize one KIVICache instance per KV head."""
        if self.kivi_cache is None:
            self.kivi_cache = [
                KIVICache(
                    num_bits=self.kivi_num_bits,
                    group_size=self.kivi_group_size,
                    residual_length=self.kivi_residual_length,
                )
                for _ in range(self.num_key_value_heads)
            ]

    def reset_kivi_cache(self):
        """Clear the KIVI cache (call between independent sequences)."""
        self.kivi_cache = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional["Cache"] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        is_prefill = cache_position is not None and cache_position.numel() > 1
        is_decode  = cache_position is not None and cache_position.numel() == 1

        # HF passes the real attention mask via kwargs when using use_cache=True
        hf_mask = kwargs.get("attention_mask", None)

        bsz, q_len, _ = hidden_states.size()

        # Compute Q, K, V projections
        query_states = self.q_proj(hidden_states)
        key_states   = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape to [batch, seq_len, num_heads, head_dim] then transpose
        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        # Apply rotary position embeddings
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, None
        )

        # ---- KIVI cache logic ----
        if past_key_values is not None and self.use_kivi:

            if bsz != 1:
                raise NotImplementedError("KIVI currently only supports batch_size=1")

            self._init_kivi_cache()

            if is_prefill:
                # ---- PREFILL PHASE ----
                for kv_head_idx in range(self.num_key_value_heads):
                    head_keys   = key_states[0, kv_head_idx, :, :].contiguous()
                    head_values = value_states[0, kv_head_idx, :, :].contiguous()
                    self.kivi_cache[kv_head_idx].prefill(head_keys, head_values)

                # Use standard attention for prefill (full KV for accuracy)
                key_states   = repeat_kv(key_states, self.num_key_value_groups)
                value_states = repeat_kv(value_states, self.num_key_value_groups)

                attn_output = self._standard_attention(
                    query_states, key_states, value_states, hf_mask
                )

            else:
                # ---- DECODE PHASE ----
                attn_outputs = []

                for q_head_idx in range(self.num_heads):
                    kv_head_idx = q_head_idx // self.num_key_value_groups

                    head_query = query_states[0, q_head_idx, :, :].contiguous()
                    head_key   = key_states[0, kv_head_idx, :, :].contiguous()
                    head_value = value_states[0, kv_head_idx, :, :].contiguous()

                    self.kivi_cache[kv_head_idx].update(head_key, head_value)

                    head_output = self.kivi_cache[kv_head_idx].compute_attention(
                        head_query,
                        use_split=False,
                        attention_mask=hf_mask,
                    )
                    attn_outputs.append(head_output)

                # [num_heads, 1, head_dim] -> [1, num_heads, 1, head_dim]
                attn_output = torch.stack(attn_outputs, dim=0).unsqueeze(0)

        else:
            # ---- STANDARD ATTENTION (no KIVI) ----
            key_states   = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

            attn_output = self._standard_attention(
                query_states, key_states, value_states, hf_mask
            )

        # Reshape: [bsz, num_heads, seq_len, head_dim] -> [bsz, seq_len, hidden_size]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, None

    def _standard_attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Standard scaled dot-product attention with causal masking and
        optional external attention mask.
        """
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))
        attn_weights = attn_weights / math.sqrt(self.head_dim)

        # Causal mask
        bsz, num_heads, q_len, _ = attn_weights.shape
        kv_len = key_states.shape[-2]

        causal_mask = torch.full(
            (q_len, kv_len),
            float("-inf"),
            device=attn_weights.device,
            dtype=attn_weights.dtype,
        )
        causal_mask = torch.triu(causal_mask, diagonal=1)
        attn_weights = attn_weights + causal_mask

        # External attention mask (e.g., padding mask from HF)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)

        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )

        attn_output = torch.matmul(attn_weights, value_states)
        return attn_output


# ---------------------------------------------------------------------------
# Replacement helper
# ---------------------------------------------------------------------------

def replace_llama_attention_with_kivi(model):
    """
    Replace all LlamaAttention layers in a LlamaForCausalLM with
    KIVI-enabled versions, copying all weights.

    Args:
        model: LlamaForCausalLM

    Returns:
        Modified model with KIVI attention layers.
    """
    print("Replacing attention layers with KIVI...")

    for layer_idx, layer in enumerate(model.model.layers):
        original_attn = layer.self_attn
        kivi_attn = LlamaAttentionWithKIVI(model.config, layer_idx=layer_idx)

        # Copy weights
        kivi_attn.q_proj.weight.data = original_attn.q_proj.weight.data.clone()
        kivi_attn.k_proj.weight.data = original_attn.k_proj.weight.data.clone()
        kivi_attn.v_proj.weight.data = original_attn.v_proj.weight.data.clone()
        kivi_attn.o_proj.weight.data = original_attn.o_proj.weight.data.clone()

        if hasattr(original_attn.q_proj, 'bias') and original_attn.q_proj.bias is not None:
            kivi_attn.q_proj.bias.data = original_attn.q_proj.bias.data.clone()
            kivi_attn.k_proj.bias.data = original_attn.k_proj.bias.data.clone()
            kivi_attn.v_proj.bias.data = original_attn.v_proj.bias.data.clone()
            kivi_attn.o_proj.bias.data = original_attn.o_proj.bias.data.clone()

        layer.self_attn = kivi_attn
        print(f"  Layer {layer_idx}: done")

    print("All attention layers replaced with KIVI.")
    return model
