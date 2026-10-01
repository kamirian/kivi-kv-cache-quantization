"""
KIVI quantization primitives.

Per-token quantization is used for the value cache.
Per-channel quantization is used for the key cache.
"""

import torch


def quantize_per_token(X, num_bits=2, group_size=32):
    """
    Quantize along token dimension (for value cache).
    X: shape [num_tokens, hidden_dim]  i.e. [T, D]

    Groups along the hidden dimension.
    Returns: (X_quant, s_X, z_X)
      X_quant: [T, num_groups, group_size]
      s_X:     [T, num_groups, 1]
      z_X:     [T, num_groups, 1]
    """
    num_groups = X.shape[1] // group_size
    X_grouped = X[:, :num_groups * group_size].reshape(
        X.shape[0], num_groups, group_size
    )  # [T, num_groups, group_size]

    # Per-token statistics
    z_X = X_grouped.min(dim=-1, keepdim=True)[0]   # zero-point
    max_X = X_grouped.max(dim=-1, keepdim=True)[0]
    s_X = (max_X - z_X) / (2 ** num_bits - 1)      # scale

    X_quant = torch.round((X_grouped - z_X) / s_X)

    return X_quant, s_X, z_X


def quantize_per_channel(X, num_bits=2, group_size=32):
    """
    Quantize along channel dimension (for key cache).
    X: shape [num_tokens, hidden_dim]  i.e. [T, D]

    Groups along the token dimension.
    Returns: (X_quant, s_X, z_X)
      X_quant: [num_groups, group_size, D]
      s_X:     [num_groups, 1, D]
      z_X:     [num_groups, 1, D]
    """
    num_groups = X.shape[0] // group_size
    X_grouped = X[:num_groups * group_size, :].reshape(
        num_groups, group_size, X.shape[1]
    )  # [num_groups, group_size, D]

    # Per-channel statistics
    z_X = X_grouped.min(dim=1, keepdim=True)[0]    # zero-point
    max_X = X_grouped.max(dim=1, keepdim=True)[0]
    s_X = (max_X - z_X) / (2 ** num_bits - 1)      # scale

    X_quant = torch.round((X_grouped - z_X) / s_X)

    return X_quant, s_X, z_X


def dequantize(X_quant, s_X, z_X):
    """Dequantize back to float."""
    return X_quant * s_X + z_X
