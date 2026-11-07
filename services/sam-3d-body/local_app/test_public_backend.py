import importlib.util

import numpy as np
import torch

from public_backend import checkpoint_state, weights_available
from vendor.compat import LayerNorm2d_op, PositionEmbeddingRandom, optimized_attention


def test_helpers():
    q = torch.randn(2, 4, 5, 8)
    mask = torch.ones(2, 5, 5, dtype=torch.bool)
    expected = torch.nn.functional.scaled_dot_product_attention(q, q, q, attn_mask=mask[:, None])
    actual = optimized_attention(q, q, q, 4, mask=mask)
    torch.testing.assert_close(actual, expected.transpose(1, 2).flatten(2))
    pe = PositionEmbeddingRandom(8)
    assert pe((7, 9)).shape == (1, 16, 7, 9)
    assert pe._encode(torch.rand(2, 3, 2)).shape == (2, 3, 16)
    norm = LayerNorm2d_op(torch.nn)(8)
    assert norm.eps == 1e-5
    assert norm(torch.randn(2, 8, 7, 9)).shape == (2, 8, 7, 9)


def test_no_comfy_dependency():
    from vendor.model import SAM3DBody
    assert SAM3DBody is not None
    assert importlib.util.find_spec("comfy") is None


def test_checkpoint_contract():
    assert weights_available()
    state, omitted = checkpoint_state()
    assert len(state) == 1206
    assert len(omitted) == 423
    assert state["backbone.layer.0.attention.q_proj.weight"].dtype == torch.bfloat16
    assert state["mhr.base_shape"].dtype == torch.float32
    assert state["decoder.layers.0.ffn.layers.0.weight"].dtype == torch.float32
    assert state["mhr.skel_joint_parents"].dtype == torch.int32
    assert not any(".layers.0.0." in name for name in state)
