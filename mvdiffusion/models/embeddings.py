import torch
import torch.nn as nn
import numpy as np
import math
from typing import List, Optional, Tuple, Union
from ipdb import set_trace as st
from torch import Tensor
from einops import rearrange

def rope(pos: Tensor, dim: int, theta: int):
# -> Tensor:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    # st()
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()


def apply_rope(xq: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    # xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    # xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq)
# , xk_out.reshape(*xk.shape).type_as(xk)

class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int):
                #  axes_dim: list[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        # self.axes_dim = axes_dim

    def forward(self, ids: Tensor, axes_dim: list[int]) -> Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )
        return emb.unsqueeze(1)