"""
KIVI KV-cache manager.

KIVICache stores keys and values in a mixed-precision format:
  - An older "grouped" portion is 2-bit quantized.
  - A recent "residual" portion is kept at full precision (FP16/FP32).
"""

import math
import torch

from .quantization import quantize_per_channel, quantize_per_token, dequantize


class KIVICache:
    def __init__(self, num_bits=2, group_size=32, residual_length=128):
        self.num_bits = num_bits
        self.group_size = group_size          # G in the paper
        self.residual_length = residual_length  # R in the paper

        # Grouped parts (quantized)
        self.key_grouped_quant = None
        self.key_grouped_scale = None
        self.key_grouped_zero = None

        self.value_grouped_quant = None
        self.value_grouped_scale = None
        self.value_grouped_zero = None

        # Residual parts (full precision)
        self.key_residual = None    # Leftover tokens that don't form a complete group
        self.value_residual = None  # Last R tokens (sliding window)

    # ------------------------------------------------------------------
    # Prefill
    # ------------------------------------------------------------------

    def prefill(self, key_states, value_states):
        """Initialize cache during prefill phase."""
        num_tokens = key_states.shape[0]

        # Ensure R is divisible by G (required by paper)
        assert self.residual_length % self.group_size == 0, \
            "residual_length (R) must be divisible by group_size (G)"

        # KEY CACHE: r = l % G  (paper uses G, not R — PDF typo corrected)
        r = num_tokens % self.group_size
        num_grouped_tokens = num_tokens - r

        if num_grouped_tokens > 0:
            key_grouped = key_states[:num_grouped_tokens]
            self.key_grouped_quant, self.key_grouped_scale, self.key_grouped_zero = \
                quantize_per_channel(key_grouped, self.num_bits, self.group_size)
            self.key_residual = key_states[num_grouped_tokens:]
        else:
            self.key_grouped_quant = None
            self.key_grouped_scale = None
            self.key_grouped_zero = None
            self.key_residual = key_states

        # VALUE CACHE: XVr = XV[l_prompt - R:]
        if num_tokens > self.residual_length:
            value_grouped = value_states[:-self.residual_length]
            self.value_grouped_quant, self.value_grouped_scale, self.value_grouped_zero = \
                quantize_per_token(value_grouped, self.num_bits, self.group_size)
            self.value_residual = value_states[-self.residual_length:]
        else:
            self.value_grouped_quant = None
            self.value_grouped_scale = None
            self.value_grouped_zero = None
            self.value_residual = value_states

    # ------------------------------------------------------------------
    # Decode update
    # ------------------------------------------------------------------

    def update(self, new_key, new_value):
        """Update cache during autoregressive decoding (one token at a time)."""

        # ---- KEY CACHE (Algorithm 1 exact) ----
        if self.key_residual is None:
            self.key_residual = new_key
        else:
            self.key_residual = torch.cat([self.key_residual, new_key], dim=0)

        # Flush when residual reaches exactly R tokens
        if self.key_residual.shape[0] == self.residual_length:
            key_quant, key_scale, key_zero = quantize_per_channel(
                self.key_residual, self.num_bits, self.group_size
            )

            if self.key_grouped_quant is not None:
                self.key_grouped_quant = torch.cat([self.key_grouped_quant, key_quant], dim=0)
                self.key_grouped_scale = torch.cat([self.key_grouped_scale, key_scale], dim=0)
                self.key_grouped_zero  = torch.cat([self.key_grouped_zero,  key_zero],  dim=0)
            else:
                self.key_grouped_quant = key_quant
                self.key_grouped_scale = key_scale
                self.key_grouped_zero  = key_zero

            self.key_residual = None

        # ---- VALUE CACHE ----
        if self.value_residual is None:
            self.value_residual = new_value
        else:
            self.value_residual = torch.cat([self.value_residual, new_value], dim=0)

        # If residual exceeds R, quantize the oldest tokens
        if self.value_residual.shape[0] > self.residual_length:
            num_to_quantize = self.value_residual.shape[0] - self.residual_length
            to_quantize = self.value_residual[:num_to_quantize]

            value_quant, value_scale, value_zero = quantize_per_token(
                to_quantize, self.num_bits, self.group_size
            )

            if self.value_grouped_quant is not None:
                self.value_grouped_quant = torch.cat([self.value_grouped_quant, value_quant], dim=0)
                self.value_grouped_scale = torch.cat([self.value_grouped_scale, value_scale], dim=0)
                self.value_grouped_zero  = torch.cat([self.value_grouped_zero,  value_zero],  dim=0)
            else:
                self.value_grouped_quant = value_quant
                self.value_grouped_scale = value_scale
                self.value_grouped_zero  = value_zero

            self.value_residual = self.value_residual[-self.residual_length:]

    # ------------------------------------------------------------------
    # Internal dequantization helpers
    # ------------------------------------------------------------------

    def _dequantize_per_channel(self, X_quant, s_X, z_X):
        """
        Dequantize key cache.
        X_quant: [num_groups, group_size, D]
        s_X:     [num_groups, 1, D]
        z_X:     [num_groups, 1, D]
        Returns: [num_groups, group_size, D]
        """
        return X_quant * s_X + z_X

    def _dequantize_per_token(self, X_quant, s_X, z_X):
        """
        Dequantize value cache.
        X_quant: [T, num_groups, group_size]
        s_X:     [T, num_groups, 1]
        z_X:     [T, num_groups, 1]
        Returns: [T, num_groups, group_size]
        """
        return X_quant * s_X + z_X

    # ------------------------------------------------------------------
    # Public accessors (full-precision reconstruction)
    # ------------------------------------------------------------------

    def get_keys(self):
        """
        Reconstruct full key cache: [T_total, D].
        Concatenates dequantized grouped keys and full-precision residual.
        """
        parts = []

        if self.key_grouped_quant is not None:
            key_grouped = self._dequantize_per_channel(
                self.key_grouped_quant,
                self.key_grouped_scale,
                self.key_grouped_zero,
            )
            Ng, G, D = key_grouped.shape
            key_grouped = key_grouped.reshape(Ng * G, D)
            parts.append(key_grouped)

        if self.key_residual is not None:
            parts.append(self.key_residual)

        if not parts:
            return None

        return torch.cat(parts, dim=0)

    def get_values(self):
        """
        Reconstruct full value cache: [T_total, D].
        Concatenates dequantized grouped values and full-precision residual.
        """
        parts = []

        if self.value_grouped_quant is not None:
            value_grouped = self._dequantize_per_token(
                self.value_grouped_quant,
                self.value_grouped_scale,
                self.value_grouped_zero,
            )
            Tg, Ngv, G = value_grouped.shape
            D = Ngv * G
            value_grouped = value_grouped.reshape(Tg, D)
            parts.append(value_grouped)

        if self.value_residual is not None:
            parts.append(self.value_residual)

        if not parts:
            return None

        return torch.cat(parts, dim=0)

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def key_lengths(self):
        """Returns (grouped_len, residual_len) for the key cache."""
        grouped_len = 0
        if self.key_grouped_quant is not None:
            Ng, G, D = self.key_grouped_quant.shape
            grouped_len = Ng * G

        residual_len = 0
        if self.key_residual is not None:
            residual_len = self.key_residual.shape[0]

        return grouped_len, residual_len

    def value_lengths(self):
        """Returns (grouped_len, residual_len) for the value cache."""
        grouped_len = 0
        if self.value_grouped_quant is not None:
            Tg, Ngv, G = self.value_grouped_quant.shape
            grouped_len = Tg

        residual_len = 0
        if self.value_residual is not None:
            residual_len = self.value_residual.shape[0]

        return grouped_len, residual_len

    def get_memory_stats(self):
        """
        Calculate theoretical memory usage statistics.

        Note: Theoretical assumes values are bit-packed.
        In practice, PyTorch stores quantized tensors as full-precision,
        so actual GPU memory will be higher until bit-packing is implemented.
        """
        stats = {
            'key_residual_bytes': 0,
            'value_residual_bytes': 0,
            'key_quant_bytes': 0,
            'value_quant_bytes': 0,
            'key_metadata_bytes': 0,
            'value_metadata_bytes': 0,
        }

        if self.key_residual is not None:
            stats['key_residual_bytes'] = (
                self.key_residual.element_size() * self.key_residual.nelement()
            )
        if self.value_residual is not None:
            stats['value_residual_bytes'] = (
                self.value_residual.element_size() * self.value_residual.nelement()
            )

        if self.key_grouped_quant is not None:
            num_elements = self.key_grouped_quant.nelement()
            stats['key_quant_bytes'] = (num_elements * self.num_bits) / 8  # bits -> bytes
            stats['key_metadata_bytes'] = (
                self.key_grouped_scale.element_size() * self.key_grouped_scale.nelement() +
                self.key_grouped_zero.element_size() * self.key_grouped_zero.nelement()
            )

        if self.value_grouped_quant is not None:
            num_elements = self.value_grouped_quant.nelement()
            stats['value_quant_bytes'] = (num_elements * self.num_bits) / 8
            stats['value_metadata_bytes'] = (
                self.value_grouped_scale.element_size() * self.value_grouped_scale.nelement() +
                self.value_grouped_zero.element_size() * self.value_grouped_zero.nelement()
            )

        stats['total_bytes'] = sum(stats.values())
        stats['total_mb'] = stats['total_bytes'] / (1024 * 1024)

        return stats

    def get_memory_stats_gpu_memory(self):
        """
        Calculate actual GPU memory usage statistics (no bit-packing assumption).
        Quantized tensors are counted at their actual stored precision.
        """
        stats = {
            'key_residual_bytes': 0,
            'value_residual_bytes': 0,
            'key_quant_bytes': 0,
            'value_quant_bytes': 0,
            'key_metadata_bytes': 0,
            'value_metadata_bytes': 0,
        }

        if self.key_residual is not None:
            stats['key_residual_bytes'] = (
                self.key_residual.element_size() * self.key_residual.nelement()
            )
        if self.value_residual is not None:
            stats['value_residual_bytes'] = (
                self.value_residual.element_size() * self.value_residual.nelement()
            )

        if self.key_grouped_quant is not None:
            stats['key_quant_bytes'] = (
                self.key_grouped_quant.element_size() * self.key_grouped_quant.nelement()
            )
            stats['key_metadata_bytes'] = (
                self.key_grouped_scale.element_size() * self.key_grouped_scale.nelement() +
                self.key_grouped_zero.element_size() * self.key_grouped_zero.nelement()
            )

        if self.value_grouped_quant is not None:
            stats['value_quant_bytes'] = (
                self.value_grouped_quant.element_size() * self.value_grouped_quant.nelement()
            )
            stats['value_metadata_bytes'] = (
                self.value_grouped_scale.element_size() * self.value_grouped_scale.nelement() +
                self.value_grouped_zero.element_size() * self.value_grouped_zero.nelement()
            )

        stats['total_bytes'] = sum(stats.values())
        stats['total_mb'] = stats['total_bytes'] / (1024 * 1024)

        return stats

    # ------------------------------------------------------------------
    # Attention computation
    # ------------------------------------------------------------------

    def compute_attention(self, query, use_split=False, attention_mask=None):
        """
        Compute attention following Algorithm 1.

        Args:
            query: [1, D] or [batch, D]
            use_split: If True, use optimized split attention (avoids full
                       dequantization). If False, use simple full-dequant method.
            attention_mask: optional additive mask [1, T]

        Returns:
            output: [1, D] or [batch, D]
        """
        if use_split:
            return self._compute_attention_split(query, attention_mask=attention_mask)
        else:
            return self._compute_attention_simple(query, attention_mask=attention_mask)

    def _compute_attention_simple(self, query, attention_mask=None):
        """Simple attention: fully dequantize, then standard scaled dot-product."""
        keys = self.get_keys()      # [T, D]
        values = self.get_values()  # [T, D]
        D = keys.shape[1]

        scores = torch.matmul(query, keys.T) / math.sqrt(D)  # [1, T]

        if attention_mask is not None:
            scores = scores + attention_mask

        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, values)  # [1, D]
        return output

    def _compute_attention_split(self, query, attention_mask=None):
        """
        Split attention (Equation 3 from Algorithm 1).
        Computes attention on grouped and residual parts separately —
        more memory-efficient than full dequantization.
        """
        scores_grouped = None
        scores_residual = None
        scale = math.sqrt(query.size(-1))

        # Scores for grouped (quantized) keys
        if self.key_grouped_quant is not None:
            keys_grouped = self._dequantize_per_channel(
                self.key_grouped_quant,
                self.key_grouped_scale,
                self.key_grouped_zero,
            )
            Ng, G, D = keys_grouped.shape
            keys_grouped = keys_grouped.reshape(Ng * G, D)
            scores_grouped = torch.matmul(query, keys_grouped.T) / scale

        # Scores for residual (full-precision) keys
        if self.key_residual is not None:
            scores_residual = torch.matmul(query, self.key_residual.T) / scale

        # Concatenate all scores
        if scores_grouped is not None and scores_residual is not None:
            scores = torch.cat([scores_grouped, scores_residual], dim=-1)
        elif scores_grouped is not None:
            scores = scores_grouped
        elif scores_residual is not None:
            scores = scores_residual
        else:
            return None

        if attention_mask is not None:
            scores = scores + attention_mask

        attn_weights = torch.softmax(scores, dim=-1)

        # Split weights according to VALUE cache structure
        value_grouped_len = 0
        if self.value_grouped_quant is not None:
            Tg, Ngv, G = self.value_grouped_quant.shape
            value_grouped_len = Tg

        value_residual_len = 0
        if self.value_residual is not None:
            value_residual_len = self.value_residual.shape[0]

        total_value_len = value_grouped_len + value_residual_len

        if attn_weights.shape[-1] != total_value_len:
            raise RuntimeError(
                f"Attention weights length ({attn_weights.shape[-1]}) doesn't match "
                f"total value cache length ({total_value_len}). "
                f"Key and value caches are misaligned!"
            )

        if value_grouped_len > 0 and value_residual_len > 0:
            attn_grouped  = attn_weights[:, :value_grouped_len]
            attn_residual = attn_weights[:, value_grouped_len:]
        elif value_grouped_len > 0:
            attn_grouped  = attn_weights
            attn_residual = None
        else:
            attn_grouped  = None
            attn_residual = attn_weights

        output = None

        if attn_grouped is not None and self.value_grouped_quant is not None:
            values_grouped = self._dequantize_per_token(
                self.value_grouped_quant,
                self.value_grouped_scale,
                self.value_grouped_zero,
            )
            Tg, Ngv, G = values_grouped.shape
            D = Ngv * G
            values_grouped = values_grouped.reshape(Tg, D)
            output = torch.matmul(attn_grouped, values_grouped)

        if attn_residual is not None and self.value_residual is not None:
            residual_output = torch.matmul(attn_residual, self.value_residual)
            if output is not None:
                output = output + residual_output
            else:
                output = residual_output

        return output
