"""Small PyTorch equivalents for inference-only framework helpers."""

import torch
from torch import nn
from torch.nn import functional as F
from sam_3d_body.models.decoders.prompt_encoder import PositionEmbeddingRandom as _PositionEmbeddingRandom


class PositionEmbeddingRandom(_PositionEmbeddingRandom):
    def _encode(self, coords):
        matrix = self.positional_encoding_gaussian_matrix.to(device=coords.device, dtype=torch.float32)
        projected = (2 * torch.pi * (2 * coords.float() - 1)) @ matrix
        return torch.cat([projected.sin(), projected.cos()], dim=-1).to(coords.dtype)

    def forward(self, size):
        h, w = size
        grid = torch.ones((h, w), device=self.positional_encoding_gaussian_matrix.device)
        xy = torch.stack([(grid.cumsum(1) - 0.5) / w, (grid.cumsum(0) - 0.5) / h], dim=-1)
        return self._encode(xy).permute(2, 0, 1).unsqueeze(0)


def cast_to_input(tensor, reference, copy=False):
    return tensor.to(device=reference.device, dtype=reference.dtype, copy=copy)


class DINOv3ViTLayerScale(nn.Module):
    def __init__(self, dim, device=None, dtype=None, operations=None):
        super().__init__()
        self.lambda1 = nn.Parameter(torch.empty(dim, device=device, dtype=dtype))

    def forward(self, x):
        return x * self.lambda1


def LayerNorm2d_op(operations):
    class LayerNorm2d(operations.LayerNorm):
        def __init__(self, channels, eps=1e-5, **kwargs):
            super().__init__(channels, eps=eps, **kwargs)

        def forward(self, x):
            return super().forward(x.movedim(1, -1)).movedim(-1, 1)

    return LayerNorm2d


def optimized_attention(q, k, v, heads, mask=None, skip_reshape=True,
                        skip_output_reshape=False, low_precision_attention=False):
    if not skip_reshape:
        raise ValueError("Only pre-split attention is supported.")
    if mask is not None and mask.ndim == 3:
        mask = mask.unsqueeze(1)
    result = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    if skip_output_reshape:
        return result
    return result.transpose(1, 2).flatten(2)


def optimized_attention_for_device(device, mask=False):
    return optimized_attention
