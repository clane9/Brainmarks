"""
Patch embedding baseline "model" to be used with attentive probe.

Patchifies flat map inputs and adds sin-cos position embeddings.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange

from brainmarks.models.base import Embeddings
from brainmarks.models.registry import register_model
import brainmarks.nisc as nisc


class PatchEmbed(nn.Module):
    __space__ = "flat"
    pos_embed: torch.Tensor

    def __init__(
        self,
        num_frames: int = 16,
        patch_size: int = 16,
        t_patch_size: int = 4,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.patch_size = patch_size
        self.t_patch_size = t_patch_size
        self.img_size = (224, 560)
        self.in_chans = 1
        self.embed_dim = t_patch_size * patch_size * patch_size

        H, W = self.img_size
        T = num_frames
        p, pt = patch_size, t_patch_size
        t, h, w = T // pt, H // p, W // p

        weight = get_3d_sincos_pos_embed(
            embed_dim=self.embed_dim,
            grid_size=(h, w),
            grid_depth=t,
            uniform_power=True,
        )
        self.register_buffer("pos_embed", torch.from_numpy(weight).float())

    def extra_repr(self):
        return f"'{self.__space__}', {self.num_frames}, {self.patch_size}, {self.t_patch_size}"

    def forward(self, batch: dict[str, Tensor]) -> Embeddings:
        x = batch["bold"]

        B, C, T, H, W = x.shape
        assert (C, H, W) == (self.in_chans, *self.img_size)
        assert T >= self.num_frames

        # truncate frames
        x = x[:, :, : self.num_frames]

        # patchify
        pt, p = self.t_patch_size, self.patch_size
        x = rearrange(x, "b c (t u) (h p) (w q) -> b (t h w) (c u p q)", u=pt, p=p, q=p)

        # pos embed
        x = x + self.pos_embed
        return None, None, x


class Transform:
    def __init__(self):
        self.norm = "frame"
        self.clip_vmax = 3.0
        self.target_tr = 1.0

        resampler = nisc.flat_resampler_fslr64k_224_560()
        self.mask = torch.tensor(resampler.mask_)

    def __call__(self, sample: dict[str, Tensor]) -> dict[str, Tensor]:
        bold = sample["bold"]
        tr = float(sample["tr"])

        # temporal resample
        if abs(tr - self.target_tr) > 0.1:
            bold = resample_to_tr(bold, tr=tr, target_tr=self.target_tr, mode="linear")

        # sample-wise normalization
        if self.norm:
            dim = {"frame": 1, "global": None}[self.norm]
            bold = normalize(bold, dim=dim)

        # clipping
        if self.clip_vmax and self.clip_vmax > 0:
            bold = torch.clamp(bold, min=-self.clip_vmax, max=self.clip_vmax)

        # unmask masked input
        T, D = bold.shape
        bold_ = torch.zeros(1, T, *self.mask.shape)
        bold_[..., self.mask] = bold
        bold = bold_
        return {**sample, "bold": bold}


def normalize(x: torch.Tensor, dim: int | None = None, eps: float = 1e-6) -> torch.Tensor:
    mean = x.mean(dim=dim, keepdim=True)
    std = x.std(dim=dim, keepdim=True)
    x = (x - mean) / (std + eps)
    return x


def resample_to_tr(x: Tensor, tr: float, target_tr: float, mode: str = "linear") -> Tensor:
    T, D = x.shape
    x = x.t().unsqueeze(0)  # [1, D, T]
    x = F.interpolate(x, size=round(tr * T / target_tr), mode=mode)
    x = x.squeeze(0).t()
    return x


# sincos pos embed utils from vjepa2, but fixed the confusing meshgrid indexing


def get_3d_sincos_pos_embed(embed_dim, grid_size, grid_depth, cls_token=False, uniform_power=False):
    """
    grid_size: tuple of int of the grid height and width
    grid_depth: int of the grid depth
    returns:
        pos_embed: [grid_depth*grid_height*grid_width, embed_dim] (w/o cls_token)
                or [1+grid_depth*grid_height*grid_width, embed_dim] (w/ cls_token)
    """
    grid_d = np.arange(grid_depth, dtype=float)
    grid_h = np.arange(grid_size[0], dtype=float)
    grid_w = np.arange(grid_size[1], dtype=float)
    grid_d, grid_h, grid_w = np.meshgrid(grid_d, grid_h, grid_w, indexing="ij")

    if not uniform_power:
        h_embed_dim = embed_dim // 4
        w_embed_dim = embed_dim // 4
        d_embed_dim = embed_dim // 2
    else:
        h_embed_dim = w_embed_dim = d_embed_dim = int(np.ceil(embed_dim / 6) * 2)

    emb_h = get_1d_sincos_pos_embed_from_grid(h_embed_dim, grid_h)  # (T*H*W, D1)
    emb_w = get_1d_sincos_pos_embed_from_grid(w_embed_dim, grid_w)  # (T*H*W, D2)
    emb_d = get_1d_sincos_pos_embed_from_grid(d_embed_dim, grid_d)  # (T*H*W, D3)
    pos_embed = np.concatenate([emb_d, emb_h, emb_w], axis=1)
    pos_embed = pos_embed[:, :embed_dim]
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    returns: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


@register_model
def patch_embed(**kwargs) -> tuple[Transform, PatchEmbed]:
    transform = Transform()
    model = PatchEmbed(**kwargs)
    return transform, model
