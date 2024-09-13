import torch
import torch.nn as nn
import numpy as np
import math
from typing import List, Optional, Tuple, Union
from ipdb import set_trace as st
from torch import Tensor
from einops import rearrange, repeat

def get_rope_ids(target: Tensor, mode: str, num_domains: int = 5):
    
    bs, l, _ = target.shape
    if mode == "query":
        h = w = int(math.sqrt(l))
    elif mode == "key":
        h = w = int(math.sqrt(l//num_domains))
    # print(mode, h, w)
    
    img_ids = torch.zeros(h, w, 3)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w)[None, :]
    
    # domain
    img_ids = repeat(img_ids, "h w c -> d (h w) c", d=num_domains)
    img_ids = img_ids.clone()  # Ensure no memory overlap
    img_ids[..., 0] = img_ids[..., 0] + torch.arange(num_domains)[:, None]
    
    if mode == "query":
        img_ids = torch.repeat_interleave(img_ids, bs//num_domains, dim=0)
    elif mode == "key": # all domains are contained in the sequence
        img_ids = repeat(img_ids, "d l c -> b (d l) c", b=bs)
        
    return img_ids.to(target.device)

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