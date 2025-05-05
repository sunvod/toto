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
from torch.nn.functional import scaled_dot_product_attention

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from helper_functions import set_default_dtype, skip_if_no_xformers

from model.attention import SpaceWiseMultiheadAttention, TimeWiseMultiheadAttention
from model.transformer import Transformer
from model.util import KVCache

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


@pytest.fixture(params=[(use_kv_cache) for use_kv_cache in [True, False]])
@beartype
def mock_inputs(request):
    """Create mock input data."""
    use_kv_cache = request.param

    inputs = torch.randn(BATCH, VARIATE, SEQ_LEN, EMBED_DIM, device=DEVICE, dtype=DTYPE)

    # Initialize Transformer
    transformer = Transformer(
        num_layers=6,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        mlp_hidden_dim=128,
        dropout=DROPOUT,
        spacewise_every_n_layers=3,
        spacewise_first=True,
    ).eval()

    # Generate id_mask
    id_mask = generate_id_mask(BATCH, VARIATE, SEQ_LEN)

    # Conditional generation of timewise_attention_mask
    timewise_attention_mask = None

    # Generate tensor-based spacewise_attention_mask_tensor (train mode)
    spacewise_attention_mask_tensor = transformer._get_mask(
        num_heads=NUM_HEADS,
        dtype=DTYPE,
        id_mask=id_mask,  # Provide the id_mask for spacewise masks
    ).contiguous()

    spacewise_attention_mask_blockdiag = None
    kv_cache = None
    if use_kv_cache:
        kv_cache = KVCache(
            batch_size=BATCH,
            num_variates=VARIATE,
            transformer_layers=transformer.layers,
            num_layers=6,
            embed_dim=EMBED_DIM,
            num_heads=NUM_HEADS,
            max_seq_len=SEQ_LEN,
            device=DEVICE,
            dtype=DTYPE,
        )

    return (
        inputs,
        timewise_attention_mask,
        spacewise_attention_mask_tensor,
        spacewise_attention_mask_blockdiag,
        kv_cache,
    )


@pytest.mark.cuda
@beartype
def test_timewise_attention(mock_inputs):
    """Test TimeWiseMultiheadAttention with attention masks and KV cache."""
    inputs, timewise_attention_mask, _, _, kv_cache = mock_inputs

    model = (
        TimeWiseMultiheadAttention(
            embed_dim=EMBED_DIM,
            num_heads=NUM_HEADS,
            dropout=DROPOUT,
            rotary_emb=None,
            use_memory_efficient_attention=True,
        )
        .to(DEVICE)
        .eval()
    )

    # Generate QKV for PyTorch scaled_dot_product_attention
    timewise_inputs = rearrange(inputs, "batch variate seq_len embed_dim -> (batch variate) seq_len embed_dim")
    qkv = model.wQKV(timewise_inputs)
    q, k, v = rearrange(
        qkv,
        "batch_X_variate seq_len (qkv head_dim n_heads) -> qkv batch_X_variate n_heads seq_len head_dim",
        qkv=3,
        head_dim=HEAD_DIM,
        n_heads=NUM_HEADS,
    ).unbind(dim=0)

    # PyTorch scaled_dot_product_attention. We call contiguous because floating point errors
    # cause small differences between contiguous and non-contiguous tensors when calculating attention.
    if timewise_attention_mask is None:
        pytorch_output = scaled_dot_product_attention(q.contiguous(), k.contiguous(), v.contiguous(), is_causal=True)
    else:
        pytorch_output = scaled_dot_product_attention(
            q.contiguous(), k.contiguous(), v.contiguous(), attn_mask=timewise_attention_mask[:, :, :SEQ_LEN, :SEQ_LEN]
        )

    pytorch_output = rearrange(
        pytorch_output,
        "(batch variate) n_heads seq_len head_dim -> batch variate seq_len (n_heads head_dim)",
        batch=BATCH,
        variate=VARIATE,
    )

    pytorch_output = model.wO(pytorch_output)

    # xFormers memory_efficient_attention
    with torch.no_grad():
        model_output = model(
            layer_idx=0,
            inputs=inputs,
            attention_mask=timewise_attention_mask,
            kv_cache=kv_cache,
        )

    # Validate output shape
    assert model_output.shape == (BATCH, VARIATE, SEQ_LEN, EMBED_DIM), "Output shape mismatch for TimeWise attention."

    # Validate numerical closeness
    delta = torch.abs(pytorch_output - model_output).mean().item()
    assert delta < torch.finfo(DTYPE).eps, f"TimeWise attention outputs differ by {delta:.6f}"


@pytest.mark.cuda
@beartype
def test_spacewise_attention_tensor(mock_inputs):
    """Test SpaceWiseMultiheadAttention with attention masks."""
    inputs, _, spacewise_attention_mask_tensor, _, kv_cache = mock_inputs

    model = (
        SpaceWiseMultiheadAttention(
            embed_dim=EMBED_DIM,
            num_heads=NUM_HEADS,
            dropout=DROPOUT,
            rotary_emb=None,
            use_memory_efficient_attention=True,
        )
        .to(DEVICE)
        .eval()
    )
    # Generate QKV for PyTorch scaled_dot_product_attention
    spacewise_inputs = rearrange(
        inputs,
        "batch variate seq_len embed_dim -> (batch seq_len) variate embed_dim",
    )
    qkv = model.wQKV(spacewise_inputs)
    q, k, v = rearrange(
        qkv,
        "batch_X_seq_len variate (qkv head_dim n_heads) -> qkv batch_X_seq_len n_heads variate head_dim",
        qkv=3,
        head_dim=HEAD_DIM,
        n_heads=NUM_HEADS,
    ).unbind(dim=0)

    # PyTorch scaled_dot_product_attention
    pytorch_output = scaled_dot_product_attention(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        attn_mask=spacewise_attention_mask_tensor[:, :, :VARIATE, :VARIATE],
    )
    pytorch_output = rearrange(
        pytorch_output,
        "(batch seq_len) n_heads variate head_dim -> batch variate seq_len (n_heads head_dim)",
        batch=BATCH,
    )

    pytorch_output = model.wO(pytorch_output)

    # xFormers memory_efficient_attention
    with torch.no_grad():
        model_output = model(
            layer_idx=0,
            inputs=inputs,
            attention_mask=spacewise_attention_mask_tensor,
            kv_cache=kv_cache,
        )

    # Validate output shape
    assert model_output.shape == (BATCH, VARIATE, SEQ_LEN, EMBED_DIM), "Output shape mismatch for SpaceWise attention."

    delta = torch.abs(pytorch_output - model_output).mean().item()
    assert delta < torch.finfo(DTYPE).eps, f"SpaceWise attention outputs differ by {delta:.6f}"


@pytest.mark.parametrize(
    "attention_class",
    [
        TimeWiseMultiheadAttention,
        SpaceWiseMultiheadAttention,
    ],
)
@pytest.mark.cuda
@beartype
def test_memory_efficient_attention_equivalence(attention_class):

    inputs = torch.randn(BATCH, VARIATE, SEQ_LEN, EMBED_DIM, device=DEVICE, dtype=DTYPE)

    spacewise_first = attention_class == SpaceWiseMultiheadAttention
    spacewise_every_n_layers = 1 if spacewise_first else -1

    # Initialize Transformer
    transformer = Transformer(
        num_layers=1,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        mlp_hidden_dim=128,
        dropout=0.0,
        spacewise_every_n_layers=spacewise_every_n_layers,
        spacewise_first=spacewise_first,
        use_memory_efficient_attention=True,
    ).eval()

    # Conditional generation of timewise_attention_mask
    timewise_attention_mask_efficient = None
    timewise_attention_mask_standard = None

    id_mask = generate_id_mask(BATCH, VARIATE, SEQ_LEN)

    transformer.use_memory_efficient_attention = True
    # Generate tensor-based spacewise_attention_mask_tensor (train mode)
    spacewise_attention_mask_tensor_efficient = transformer._get_mask(
        num_heads=NUM_HEADS,
        dtype=DTYPE,
        id_mask=id_mask,  # Provide the id_mask for spacewise masks
    ).contiguous()

    transformer.use_memory_efficient_attention = False
    # Generate tensor-based spacewise_attention_mask_tensor (train mode)
    spacewise_attention_mask_tensor_standard = transformer._get_mask(
        num_heads=NUM_HEADS,
        dtype=DTYPE,
        id_mask=id_mask,  # Provide the id_mask for spacewise masks
    ).contiguous()

    if attention_class == TimeWiseMultiheadAttention:
        attention_mask_efficient = timewise_attention_mask_efficient
        attention_mask_standard = timewise_attention_mask_standard
    else:
        attention_mask_efficient = spacewise_attention_mask_tensor_efficient
        attention_mask_standard = spacewise_attention_mask_tensor_standard

    model = (
        attention_class(
            embed_dim=EMBED_DIM,
            num_heads=NUM_HEADS,
            dropout=0.0,
            rotary_emb=None,
            use_memory_efficient_attention=True,
        )
        .to(DEVICE)
        .eval()
    )
    with torch.no_grad():
        model.use_memory_efficient_attention = True
        output_efficient = model(0, inputs, attention_mask_efficient, None)
        model.use_memory_efficient_attention = False
        output_standard = model(0, inputs, attention_mask_standard, None)

    assert (
        output_efficient.shape == output_standard.shape
    ), "Output shapes differ between efficient and standard attention."
    assert (
        torch.abs(output_efficient - output_standard).mean().item() < torch.finfo(DTYPE).eps
    ), "Outputs differ between efficient and standard attention."
