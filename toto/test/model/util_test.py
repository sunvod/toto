# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/)
# Copyright 2025 Datadog, Inc.

import os
import sys

import pytest
import torch
from beartype import beartype

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from helper_functions import set_default_dtype, skip_if_no_xformers

skip_if_no_xformers()
set_default_dtype()

from model.util import RMSNorm

DEVICE = torch.get_default_device()
DTYPE = torch.get_default_dtype()


@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("include_weight", [True, False])
@pytest.mark.cuda
@beartype
def test_RMSNorm_equivalence(training, include_weight, monkeypatch):
    """
    Test RMSNorm equivalence by forcing Triton off (via environment variable)
    to compare the Python fallback implementation with the xFormers one.
    """
    torch.manual_seed(42)

    dim = 64
    batch_size = 2
    seq_len = 10

    # Input tensor
    x = torch.randn(batch_size, seq_len, dim, dtype=DTYPE, device=DEVICE)

    # Standard RMSNorm (uses xFormers if available)
    RMSNorm_xformers = RMSNorm(dim, include_weight=include_weight)

    # Explicitly set training mode
    if training:
        RMSNorm_xformers.train()
    else:
        RMSNorm_xformers.eval()

    # Compute output BEFORE disabling Triton
    with torch.no_grad():
        from model.util import XFORMERS_RMSNORM_AVAILABLE as xformers_available

        assert xformers_available == True
        output_xformers = RMSNorm_xformers(x)

    # Temporarily disable Triton using monkeypatch (resets after test)
    monkeypatch.setenv("XFORMERS_FORCE_DISABLE_TRITON", "1")

    # Now apply the monkey patch for XFORMERS_AVAILABLE
    with monkeypatch.context() as m:
        m.setattr("model.util.XFORMERS_RMSNORM_AVAILABLE", False)  # Ensure xFormers is disabled

        from model.util import XFORMERS_RMSNORM_AVAILABLE as xformers_available_fallback
        from model.util import RMSNorm as RMSNormPythonFallback

        assert xformers_available_fallback == False

        RMSNorm_python = RMSNormPythonFallback(dim, include_weight=include_weight)

        # Explicitly set training mode for the Python fallback
        if training:
            RMSNorm_python.train()
        else:
            RMSNorm_python.eval()

    # Compute Python fallback output AFTER patching
    with torch.no_grad():
        output_python = RMSNorm_python(x)

    # Validate output shape
    assert output_xformers.shape == output_python.shape, "Output shapes do not match"

    # Validate numerical closeness
    delta = torch.abs(output_xformers - output_python).mean().item()
    assert (
        delta < torch.finfo(DTYPE).eps
    ), f"RMSNorm outputs differ by {delta:.6f} between xFormers and Python implementations"
