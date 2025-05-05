# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/)
# Copyright 2025 Datadog, Inc.

import os
import sys

import pytest
import torch
from beartype import beartype
from einops import rearrange

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from helper_functions import set_default_dtype, skip_if_no_xformers

from model.transformer import Transformer
from model.util import make_batched_block_mask

skip_if_no_xformers()
set_default_dtype()

DEVICE = torch.get_default_device()
DTYPE = torch.get_default_dtype()

# Test parameters
BATCH = 2
VARIATE = 3
SEQ_LEN = 10
EMBED_DIM = 64
NUM_HEADS = 8
HEAD_DIM = EMBED_DIM // NUM_HEADS
DROPOUT = 0.0  # Fixed dropout, but model is always in eval mode.


def generate_id_mask(batch, variate, seq_len):
    # Generate random lengths that sum up to variate
    ids = torch.arange(0, variate, device=DEVICE, dtype=torch.int)
    ids = (ids // 2).clamp(max=variate - 1)
    return ids.unsqueeze(0).unsqueeze(-1).expand(batch, -1, seq_len)


@pytest.fixture()
@beartype
def mock_inputs(request):
    """Create mock input data."""
    # Initialize Transformer
    transformer = Transformer(
        num_layers=1,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        mlp_hidden_dim=128,
        dropout=DROPOUT,
        spacewise_every_n_layers=1,
        spacewise_first=True,
        use_memory_efficient_attention=True,
    ).eval()

    # Generate id_mask
    id_mask = generate_id_mask(BATCH, VARIATE, SEQ_LEN)

    timewise_attention_mask = None

    # Generate tensor-based spacewise_attention_mask_tensor (train mode)
    spacewise_attention_mask_tensor = transformer._get_mask(
        num_heads=NUM_HEADS,
        dtype=DTYPE,
        id_mask=id_mask,  # Provide the id_mask for spacewise masks
    ).contiguous()

    return (
        timewise_attention_mask,
        spacewise_attention_mask_tensor,
        id_mask,
        transformer,
    )


@pytest.mark.cuda
@beartype
def test_spacewise_mask(mock_inputs):
    """Test spacewise attention mask generation using old boolean (BHND) and new (BNHD for xformers) implementations."""
    _, spacewise_attention_mask_tensor, id_mask, _ = mock_inputs

    if spacewise_attention_mask_tensor is not None:
        assert spacewise_attention_mask_tensor.max() == 0.0
        assert spacewise_attention_mask_tensor.min() == -float("inf")

    # Generate spacewise mask using old implementation
    old_mask = make_batched_block_mask(id_mask.transpose(-1, -2))
    old_mask = rearrange(old_mask, "batch seq_len variate1 variate2 -> (batch seq_len) 1 variate1 variate2")

    # Convert the boolean old_mask to float mask: True -> 0.0, False -> -inf (because float masks are additive)
    converted_old_mask = torch.where(old_mask, 0.0, -float("inf"))

    assert (
        spacewise_attention_mask_tensor[:, :1, :VARIATE, :VARIATE].shape == converted_old_mask.shape
    ), "Spacewise mask shape mismatch."
    assert torch.equal(
        spacewise_attention_mask_tensor[:, :1, :VARIATE, :VARIATE], converted_old_mask
    ), "Spacewise mask values mismatch."


@pytest.mark.cuda
@beartype
def test_swiglu_equivalence(monkeypatch):
    """
    Test that the SwiGLU implementation inside Transformer produces equivalent results
    when using xFormers and the Python fallback.
    """
    torch.manual_seed(42)

    # Generate input tensor
    x = torch.randn(BATCH, SEQ_LEN, EMBED_DIM)

    from model.transformer import XFORMERS_SWIGLU_AVAILABLE as xformers_available

    if not xformers_available:
        print("xformers function not available on this system. This test is invalid.")

    # Transformer using xFormers SwiGLU (if available)
    transformer_xformers = Transformer(
        num_layers=1,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        mlp_hidden_dim=2 * EMBED_DIM,
        dropout=DROPOUT,
        spacewise_every_n_layers=1,
        spacewise_first=True,
        use_memory_efficient_attention=True,
    ).eval()

    # Get the original xFormers model state dict
    original_state_dict = transformer_xformers.layers[0].state_dict()

    # Force XFORMERS_AVAILABLE = False for Python fallback
    with monkeypatch.context() as m:
        m.setattr("model.util.XFORMERS_RMSNORM_AVAILABLE", False)
        m.setattr("model.transformer.XFORMERS_SWIGLU_AVAILABLE", False)
        from model.transformer import (
            XFORMERS_SWIGLU_AVAILABLE as xformers_not_available,
        )
        from model.transformer import Transformer as TransformerFallback

        assert xformers_not_available == False

        transformer_fallback = TransformerFallback(
            num_layers=1,
            embed_dim=EMBED_DIM,
            num_heads=NUM_HEADS,
            mlp_hidden_dim=2 * EMBED_DIM,
            dropout=DROPOUT,
            spacewise_every_n_layers=1,
            spacewise_first=True,
            use_memory_efficient_attention=False,
        ).eval()

    # Remap xFormers' state dict to match the fallback SwiGLU format
    remapped_state_dict = {
        key.replace("mlp.0.w12.weight", "mlp.0.weight")
        .replace("mlp.0.w12.bias", "mlp.0.bias")
        .replace("mlp.0.w3.weight", "mlp.2.weight")
        .replace("mlp.0.w3.bias", "mlp.2.bias"): value
        for key, value in original_state_dict.items()
    }

    # Load the remapped state into the fallback model
    transformer_fallback.layers[0].load_state_dict(remapped_state_dict)

    # Fetch the MLPs
    mlp_xformers = transformer_xformers.layers[0].mlp
    mlp_fallback = transformer_fallback.layers[0].mlp

    # Compute outputs with no_grad
    with torch.no_grad():
        output_xformers = mlp_xformers(x)
        output_fallback = mlp_fallback(x)

    # Validate output shape
    assert output_xformers.shape == output_fallback.shape, "MLP output shape mismatch"

    # Validate numerical closeness
    delta = torch.abs(output_xformers - output_fallback).mean().item()
    assert (
        delta < torch.finfo(DTYPE).eps
    ), f"SwiGLU outputs differ by {delta:.6f} between xFormers and Python implementations."
