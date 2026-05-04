"""

NeuroSTORM: Towards a general-purpose foundation model for fMRI analysis

"""

import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from brainmarks.models.base import Embeddings
from brainmarks.models.registry import register_model
from pathlib import Path
import numpy as np
import torch
from brainmarks import nisc
from einops import rearrange
import templateflow.api as tflow
from huggingface_hub import hf_hub_download


try:
    from neurostorm.models.lightning_model import LightningModel

except ImportError as exc:
    raise ImportError(
        "neurostorm not installed. Please install the optional neurostorm extra with `uv sync --extra neurostorm`"
    ) from exc


NEUROSTORM_VARIANTS = {
    "0.8": "fmrifound/pt_fmrifound_mae_ratio0.8.ckpt",
    "0.5": "fmrifound/pt_fmrifound_mae_ratio0.5.ckpt",
}


def fetch_neurostorm_checkpoint(variant: str) -> Path:
    repo_id = "zxcvb20001/NeuroSTORM"
    filename = NEUROSTORM_VARIANTS[variant]
    return Path(
        hf_hub_download(
            repo_id=repo_id, filename=filename, revision="e96ad7bcbf393d9e4c655f3064a5e42b89ca1664"
        )
    )


# Dummy datamodule to initialize LitClassifier
class _DummyTrainDataset:
    target_values = np.zeros((32, 1), dtype=np.float32)


class _DummyDataModule:
    def __init__(self):
        self.train_dataset = _DummyTrainDataset()


class NeuroStormWrapper(nn.Module):
    __space__: str = "mni"

    def __init__(self, variant: str) -> None:
        super().__init__()

        self.ckpt_path = fetch_neurostorm_checkpoint(variant)

        ckpt = torch.load(self.ckpt_path, map_location="cpu")

        # patch hyperparameters
        hparams = ckpt["hyper_parameters"]
        hparams["print_flops"] = False  # missing required key
        # overwrite model name, current checkpoint has "swin4d_mae" which is not a valid model name here: https://github.com/CUHK-AIM-Group/NeuroSTORM/blob/main/models/load_model.py but the keys match the 'neurostorm' model
        hparams["model"] = "neurostorm"
        model = LightningModel(**hparams, data_module=_DummyDataModule())

        # load weights
        state_dict = ckpt["state_dict"]
        model.load_state_dict(state_dict, strict=True)

        self.backbone = model.model
        self.expected_seq_len = 20
        self.max_windows = 8

    def forward_encoder(self, x):
        # patch method since original code always applies mask
        # https://github.com/CUHK-AIM-Group/NeuroSTORM/blob/5bb4f7c844ed7544f95cd934eece69b390a55ea4/models/neurostorm.py#L1191C1-L1205C1
        x = self.backbone.patch_embed(x)

        for i in range(self.backbone.num_layers):
            x = self.backbone.pos_embeds[i](x)
            x = self.backbone.layers[i](x.contiguous())

        return x

    def forward(self, batch: dict[str, Tensor]) -> Embeddings:
        x = batch["bold"]
        B, C, H, W, D, T = x.shape

        # handle sliding windows
        num_windows = min(T // self.expected_seq_len, self.max_windows)
        T = num_windows * self.expected_seq_len
        x = rearrange(x[..., :T], "b c x y z (w t) -> (b w) c x y z t", w=num_windows)

        # feats have shape (B, channels, H, W, D, T) (B, 288, 2, 2, 2, 20)
        feats = self.forward_encoder(x)

        feats = rearrange(
            feats, "(b w) c x y z t -> b (w x y z t) c", w=num_windows
        )  # convert to (B, patches, channels)

        return Embeddings(cls_embeds=None, reg_embeds=None, patch_embeds=feats)


class NeuroStormTransform:
    """
    0. Unnormalize voxelwize z-scored data
    1. temporal resampling
    2. global z-score normalization
    3. pad/crop to expected sequence length t=20
    4. unmask input to full 4D volume
    5. spatial crop/pad to (96, 96, 96)
    6. reshape to expected shape (C, H, W, D, T)
    """

    def __init__(self, coord_normalize: bool = False):
        self.coord_normalize = coord_normalize

        # Mask calculation from brainmarks.readers
        roi_path = tflow.get(
            "MNI152NLin6Asym", desc="brain", resolution=2, suffix="mask", extension="nii.gz"
        )
        mask = nisc.read_mni152_2mm_data(roi_path) > 0  # (Z, Y, X)

        self.mask = torch.from_numpy(mask)
        self.mask_shape = mask.shape

        # NeuroSTORM input size is (H, W, D, T) = (X, Y, Z, T) = (96, 96, 96, 20)
        # https://github.com/CUHK-AIM-Group/NeuroSTORM/blob/49dd063e48a635d66653e3b02e752256f6813621/README.md?plain=1#L297
        self.expected_seq_len = 20
        self.spatial_target = 96

        # target temporal resampling
        # https://github.com/CUHK-AIM-Group/NeuroSTORM/blob/5bb4f7c844ed7544f95cd934eece69b390a55ea4/datasets/preprocessing_volume.py#L54C1-L69C26
        self.target_tr = 0.8

    def __call__(self, sample: dict[str, Tensor]) -> dict[str, Tensor]:
        """
        Transform bold volumes to model input format.

        sample dicts requires keys:
            - bold: (T,V) normalized bold signal,
            - mean: (1,V) mean of bold signal,
            - std: (1,V) standard deviation of bold signal

        sample dict is modified in place:
            - bold: (C, H, W, D, T)

        """
        # unnormalize
        if not self.coord_normalize:
            bold = sample["bold"] * sample["std"] + sample["mean"]
        else:
            bold = sample["bold"]
        tr = float(sample["tr"])

        # temporal resampling
        # nb, we break from original authors and resample while in sparse (T, V) format.
        # this is a ~5x efficiency gain.
        # https://github.com/CUHK-AIM-Group/NeuroSTORM/blob/5bb4f7c844ed7544f95cd934eece69b390a55ea4/datasets/preprocessing_volume.py#L54C1-L69C26
        if abs(tr - self.target_tr) >= 0.1:
            bold = resample_to_target_tr(bold, tr, self.target_tr)

        # every negative value is set to 0
        # https://github.com/CUHK-AIM-Group/NeuroSTORM/blob/5bb4f7c844ed7544f95cd934eece69b390a55ea4/datasets/preprocessing_volume.py#L119
        bold = bold.clip(min=0.0)

        # global normalization
        # nb, normalization on sparse (T, V) for efficiency
        # https://github.com/CUHK-AIM-Group/NeuroSTORM/blob/5bb4f7c844ed7544f95cd934eece69b390a55ea4/datasets/preprocessing_volume.py#L123C1-L126C54
        bold = (bold - bold.mean()) / bold.std()

        # Pad if too short - repeat mean (consistent with other models)
        T = len(bold)
        if T < self.expected_seq_len:
            mean = bold.mean(dim=0).repeat(self.expected_seq_len - T, 1)
            bold = torch.cat([bold, mean], dim=0)
            T = self.expected_seq_len

        # Crop to fixed number of non-overlapping windows
        num_windows = T // self.expected_seq_len
        T = num_windows * self.expected_seq_len
        bold = bold[:T, :]

        # unflatten
        # background is filled with min value
        # https://github.com/CUHK-AIM-Group/NeuroSTORM/blob/5bb4f7c844ed7544f95cd934eece69b390a55ea4/datasets/preprocessing_volume.py#L130C5-L132C54
        T, V = bold.shape
        Z, Y, X = self.mask_shape
        mask = self.mask.to(device=bold.device)
        fill_value = bold.min().item()
        volume = torch.full((T, Z, Y, X), fill_value, device=bold.device)
        volume[:, mask] = bold
        volume = rearrange(volume, "t z y x -> t x y z")

        # flip x axis. the provided MNI data are in RAS orientation, but the model
        # expects HCP (FSL) convention LAS.
        # https://github.com/CUHK-AIM-Group/NeuroSTORM/blob/5bb4f7c844ed7544f95cd934eece69b390a55ea4/datasets/preprocessing_volume.py#L76
        volume = torch.flip(volume, (1,))

        # center crop or pad
        # equivalent to select_middle_96 -> pad_to_96
        # nb the crop for y axis is different from swift. this is intentional and follows the neurostorm select_middle_96
        # https://github.com/CUHK-AIM-Group/NeuroSTORM/blob/5bb4f7c844ed7544f95cd934eece69b390a55ea4/datasets/preprocessing_volume.py#L12
        # https://github.com/CUHK-AIM-Group/NeuroSTORM/blob/5bb4f7c844ed7544f95cd934eece69b390a55ea4/datasets/fmri_datasets.py#L12C1-L21C13
        assert (X, Y, Z) == (91, 109, 91), "unexpected volume shape"
        volume = F.pad(volume, (3, 2, -6, -7, 3, 2), value=fill_value)

        # rearrange to (C, X, Y, Z, T)
        volume = rearrange(volume, "t x y z -> 1 x y z t")

        sample["bold"] = volume
        return sample


def resample_to_target_tr(
    x: Tensor,
    tr: float,
    target_tr: float,
    mode: str = "linear",
) -> Tensor:
    # x: [T, D]
    x = F.interpolate(
        x.T.unsqueeze(0),
        size=round(float(tr) * len(x) / float(target_tr)),
        mode=mode,
    )  # [1, D, T]
    return x.squeeze(0).T


@register_model
def neurostorm(**kwargs) -> tuple[NeuroStormTransform, NeuroStormWrapper]:
    return NeuroStormTransform(**kwargs), NeuroStormWrapper(variant="0.5")


@register_model
def neurostorm_0p8(**kwargs) -> tuple[NeuroStormTransform, NeuroStormWrapper]:
    return NeuroStormTransform(**kwargs), NeuroStormWrapper(variant="0.8")
