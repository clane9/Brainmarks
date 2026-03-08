import math
import urllib.request
from pathlib import Path
from typing import Iterable

import gdown
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm

from fmri_fm_eval.models.base import Embeddings
from fmri_fm_eval.models.registry import register_model

try:
    import brainharmonix.libs.model as model
    import brainharmonix.libs.position_embedding as pos_embeds
    from brainharmonix.configs.harmonizer.stage0_embed import conf_embed_downstream
except ImportError as exc:
    raise ImportError(
        "brainharmonix not installed. Please install the brainharmonix extra "
        "with `uv sync --extra brainharmonix`."
    ) from exc


BRAIN_HARMONY_CACHE_DIR = Path.home() / ".cache" / "fmri-fm-eval" / "brain-harmony"


class BrainHarmonixFTransform:
    # constant values copied from original config
    # https://github.com/hzlab/Brain-Harmony/blob/453edc18aed68d834401159f81757297d0c5281f/configs/harmonizer/stage0_embed/conf_embed_downstream.py#L103-L104
    # duration of "standard" patch in seconds
    # real patch size is adjusted based on tr so that each patch matches this duration
    standard_time = 48 * 0.735
    # standard number of temporal patches in a sequence
    target_num_patches = 18

    def __init__(self, coord_normalize: bool = False):
        self.coord_normalize = coord_normalize

    def __call__(self, sample: dict[str, Tensor]) -> dict[str, Tensor]:
        bold = sample["bold"]  # (T, D) - z-score normalized data
        mean = sample["mean"]  # (1, D)
        std = sample["std"]  # (1, D)
        tr = float(sample["tr"])  # float - repetition time

        # Convert z-scored data back to raw signal
        if not self.coord_normalize:
            bold = bold * std + mean

        # https://github.com/hzlab/Brain-Harmony/blob/453edc18aed68d834401159f81757297d0c5281f/datasets/datasets.py#L258
        # apply global robust scaling
        assert self.global_stats_ is not None, "global_stats_ is None; call fit()"
        median, iqr = self.global_stats_
        bold = (bold - median) / iqr

        # transpose [T, D] -> [D, T]
        bold = bold.T.contiguous()
        D, T = bold.shape

        # pad to expected sequence length
        # note, they support a flexible patch size but a fixed expected number of
        # patches and sequence duration
        # https://github.com/hzlab/Brain-Harmony/blob/453edc18aed68d834401159f81757297d0c5281f/datasets/datasets.py#L696-L702
        patch_size = round(self.standard_time / tr)
        num_patches = math.ceil(T / patch_size)
        target_pad_length = self.target_num_patches * patch_size

        if target_pad_length > T:
            bold = F.pad(bold, (0, target_pad_length - T))
        else:
            bold = bold[:, :target_pad_length]

        # attention mask for invalid patches
        # https://github.com/hzlab/Brain-Harmony/blob/453edc18aed68d834401159f81757297d0c5281f/datasets/datasets.py#L277
        attn_mask = torch.ones(D, self.target_num_patches, dtype=torch.bool)
        attn_mask[:, num_patches:] = 0
        attn_mask = attn_mask.flatten()

        # [C, D, T]
        bold = bold.unsqueeze(0)

        return {
            **sample,
            "bold": bold,
            "attn_mask": attn_mask,
            "patch_size": patch_size,
        }

    def fit(self, train_dataset: Iterable[dict[str, Tensor]]) -> None:
        """
        Precompute global stats on training dataset
        """
        # brainharmony computes ROI median and IQR over per-sample ROI *means*, and uses
        # these for sample normalization.
        # their default kwargs:
        #   norm="all_robust_scaling"
        #   preprocess=None
        # https://github.com/hzlab/Brain-Harmony/blob/453edc18aed68d834401159f81757297d0c5281f/datasets/datasets.py#L193-L196
        #
        # QUESTIONS:
        #   - the downstream ABIDE_I_fMRI_Dataset doesn't include any explicit
        #     normalization. how are data processed for downstream eval?
        all_bold = []
        all_bold_mean = []
        for sample in tqdm(train_dataset):
            # get per ROI means, already in the sample
            bold = sample["bold"]  # (T, D)
            mean = sample["mean"]  # (1, D)
            all_bold.append(bold)
            all_bold_mean.append(mean.squeeze(0))

        # compute median and iqr over ROI means
        # note, it is a bit strange to scale only by the iqr of the *mean*. this will be
        # like 1 / sqrt(T) * iqr of the actual data. if the eval time series are much
        # different sequence length than the training data, you will get a significant
        # scale difference.
        all_bold = torch.cat(all_bold)
        all_bold_mean = torch.stack(all_bold_mean)
        q = torch.tensor([0.25, 0.5, 0.75], dtype=all_bold_mean.dtype)
        if not self.coord_normalize:
            q1, median, q3 = torch.quantile(all_bold_mean, q, dim=0)
        else:
            # when using coordinate wise normalization, we follow their preprocess=True (direct) mode
            # and compute normalization stats over the full data
            # https://github.com/hzlab/Brain-Harmony/blob/453edc18aed68d834401159f81757297d0c5281f/datasets/datasets.py#L134-L136
            q1, median, q3 = torch.quantile(all_bold, q, dim=0)
        iqr = q3 - q1
        self.global_stats_ = median, iqr


class BrainHarmonixFWrapper(nn.Module):
    __space__ = "schaefer400"

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, batch: dict[str, Tensor]) -> Embeddings:
        x = batch["bold"]
        # note, we're assuming all samples in the batch have the same patch size
        # if they don't, we should get a collate error earlier
        # though we might want instead to just resample
        patch_size = batch["patch_size"][0].item()
        attn_mask = batch["attn_mask"]

        # forward pass
        # https://github.com/hzlab/Brain-Harmony/blob/453edc18aed68d834401159f81757297d0c5281f/modules/harmonizer/stage0_embed/embedding_downstream.py#L157
        patch_embeds = self.encoder(x, patch_size, attention_mask=attn_mask)
        # We don't have cls embeds or reg embeds
        return None, None, patch_embeds


def get_pos_embed(name, **kwargs):
    return getattr(pos_embeds, name)(kwargs["model_args"])


def get_encoder(pos_embed, cls_token, name, attn_mode="sdpa", **kwargs):
    return getattr(model, name)(
        pos_embed=pos_embed, cls_token=cls_token, attn_mode=attn_mode, **kwargs
    )


def fetch_brain_harmonix_checkpoint() -> Path:
    """Download harmonix-f/model.pth from Google Drive with caching."""

    # File ID from Brain-Harmony Readme -https://drive.google.com/drive/folders/12MkUAOcegU60YVlK8u8_Owmgk4eQVheB, github.com/hzlab/Brain-Harmony
    file_id = "1M4SHZx4L09d8jvP_-kgEHPqeDoNqWtqB"
    cache_dir = BRAIN_HARMONY_CACHE_DIR
    cached_file = cache_dir / "harmonix-f" / "model.pth"
    cached_file.parent.mkdir(exist_ok=True, parents=True)

    if not cached_file.exists():
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, str(cached_file), quiet=False)

    return cached_file


def fetch_brain_harmonix_pos_embeds() -> tuple[Path, Path]:
    # Download the CSVs from the OG repo that we need for the pos embed
    pos_emb_dir = BRAIN_HARMONY_CACHE_DIR / "pos_emb"
    pos_emb_dir.mkdir(exist_ok=True, parents=True)

    geo_harm_file = pos_emb_dir / "schaefer400_roi_eigenmodes.csv"
    download_file(
        "https://raw.githubusercontent.com/hzlab/Brain-Harmony/refs/heads/main/brainharmony_pos_embed/schaefer400_roi_eigenmodes.csv",
        geo_harm_file,
    )

    gradient_file = pos_emb_dir / "gradient_mapping_400.csv"
    download_file(
        "https://raw.githubusercontent.com/hzlab/Brain-Harmony/refs/heads/main/brainharmony_pos_embed/gradient_mapping_400.csv",
        gradient_file,
    )
    return geo_harm_file, gradient_file


def download_file(url: str, path: Path) -> None:
    if path.exists():
        return
    urllib.request.urlretrieve(url, path)


@register_model
def brain_harmonix_f(**kwargs):
    geo_harm_file, gradient_file = fetch_brain_harmonix_pos_embeds()

    config = conf_embed_downstream.get_config()
    config.pos_embed.model_args.gradient = str(gradient_file)
    config.pos_embed.model_args.geo_harm = str(geo_harm_file)

    fmri_encoder_pos_embed = get_pos_embed(**config.pos_embed)
    fmri_encoder = get_encoder(fmri_encoder_pos_embed, None, **config.encoder)

    ckpt = fetch_brain_harmonix_checkpoint()
    state_dict = torch.load(ckpt, map_location="cpu", weights_only=True)

    prefix = "encoder_ema."
    state_dict = {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}

    # Drop pos_embed parameters that are fixed untrained sincos pos embeddings
    # They have mismatching size compared to the checkpoint and mess up loading
    # https://github.com/hzlab/Brain-Harmony/blob/453edc18aed68d834401159f81757297d0c5281f/modules/harmonizer/stage0_embed/embedding_downstream.py#L105C5-L115C65
    state_dict.pop("pos_embed.emb_h_encoder")
    state_dict.pop("pos_embed.emb_h_decoder")

    missing_keys, unexpected_keys = fmri_encoder.load_state_dict(state_dict, strict=False)
    assert missing_keys == [
        "pos_embed.emb_h_encoder",  # mismatched size sincos pos embeds
        "pos_embed.emb_h_decoder",
        "pos_embed.geo_harm",  # gradient buffers I added that aren't in the original ckpt
        "pos_embed.gradient",
    ]
    assert not unexpected_keys

    transform = BrainHarmonixFTransform(**kwargs)
    model = BrainHarmonixFWrapper(fmri_encoder)
    return transform, model
