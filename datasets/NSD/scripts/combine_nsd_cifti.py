"""Combine NSD 32k_fs_LR gifti and MNI152 NIfTI into fslr91k CIFTI dtseries.

For each run, reads the local surface gifti and the (potentially remote) MNI152 NIfTI,
stitches them into the grayordinate structure of the Schaefer400+Tian S3 fslr91k template,
and saves a dtseries.nii.
"""

import argparse
import logging
import re
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.orientations import io_orientation, ornt_transform

import fmri_fm_eval.nisc as nisc

logging.basicConfig(
    format="[%(levelname)s %(asctime)s]: %(message)s",
    level=logging.INFO,
    datefmt="%y-%m-%d %H:%M:%S",
)
logging.getLogger("nibabel").setLevel(logging.ERROR)
logging.getLogger("botocore").setLevel(logging.ERROR)

_logger = logging.getLogger(__name__)

ROOT = Path(__file__).parents[1]
NSD_ROOT = ROOT / "data/NSD"

# Upsampled 1.0s TR for high-res func1mm data.
# https://cvnlab.slite.page/p/vjWTghPTb3/Time-series-data
NSD_TR = 1.0
NSD_TOTAL_RUNS = 3600

FSLR91K_NUM_VERTICES = 91282


def main(args):
    # Load the Schaefer400+TianS3 fslr91k dlabel to get the target grayordinate structure.
    # This defines 91282 grayordinates: cortical surface vertices + subcortical voxels.
    template_path = nisc.fetch_schaefer_tian(400, 3, space="fslr91k")
    template_img = nib.load(template_path)
    bm_axis = template_img.header.get_axis(1)  # BrainModelAxis
    assert len(bm_axis) == FSLR91K_NUM_VERTICES

    # Surface: CIFTI column indices and gifti vertex selector (both hemispheres).
    surf_ids, surf_mask = nisc.get_cifti_surf_indices(template_img)

    # Volume: (cifti_column_indices, voxel_ijk) for each subcortical structure.
    # model.voxel contains IJK coordinates in the CIFTI's LAS reference space.
    vol_structures = []
    full_indices = np.arange(FSLR91K_NUM_VERTICES)
    for _name, slc, model in bm_axis.iter_structures():
        if model.volume_shape is not None:
            indices = full_indices[slc]
            vol_structures.append((indices, model.voxel))

    _logger.info("Template: n_surf=%d, n_vol_structs=%d", len(surf_ids), len(vol_structures))

    # Discover all runs from the local gifti tree.
    root = Path(args.root)
    gifti_paths = sorted(
        root.glob("**/32k_fs_LR/timeseries/timeseries_session??_run??.lh.func.gii")
    )
    assert len(gifti_paths) == NSD_TOTAL_RUNS, f"unexpected number of runs {len(gifti_paths)}"

    out_root = Path(args.out_root)

    for gifti_path in gifti_paths:
        process_run(
            gifti_path=gifti_path,
            root=root,
            out_root=out_root,
            bm_axis=bm_axis,
            surf_ids=surf_ids,
            surf_mask=surf_mask,
            vol_structures=vol_structures,
            overwrite=args.overwrite,
        )


def parse_nsd_metadata(path: Path) -> dict[str, str | int]:
    match = re.search(r"(subj[0-9]+)/.*/.*_session([0-9]+)_run([0-9]+)\.", str(path))
    sub = match.group(1)
    ses = int(match.group(2))
    run = int(match.group(3))
    return sub, ses, run


def process_run(
    gifti_path: Path,
    *,
    root: Path,
    out_root: Path,
    bm_axis,
    surf_ids: np.ndarray,
    surf_mask: np.ndarray,
    vol_structures: list,
    overwrite: bool = False,
):
    sub, ses, run = parse_nsd_metadata(gifti_path)
    out_path = (
        out_root
        / f"nsddata_timeseries/ppdata/{sub}/fslr91k/timeseries"
        / f"timeseries_session{ses:02d}_run{run:02d}.dtseries.nii"
    )
    if out_path.exists() and not overwrite:
        _logger.info("Output %s exists; skipping.", out_path)
        return

    # Read surface gifti (loads both hemispheres), shape (T, 64984).
    gifti_data = nisc.read_gifti_surf_data(gifti_path)
    T = len(gifti_data)

    # Fetch MNI152 NIfTI (may be remote).
    nii_rel = (
        f"nsddata_timeseries/ppdata/{sub}/MNI152/timeseries/"
        f"timeseries_session{ses:02d}_run{run:02d}.nii.gz"
    )
    nii_path = root / nii_rel

    # Reorient NIfTI to LAS so voxel IJK aligns with the CIFTI volumetric reference.
    # The CIFTI affine is LAS [[-2,0,0,90],...]; the NSD MNI NIfTI is RAS [[2,0,0,-90],...].
    # This is a pure axis flip on x — no interpolation.
    nii_img: nib.Nifti1Image = nib.load(nii_path)
    ornt_in = io_orientation(nii_img.affine)
    ornt_out = io_orientation(bm_axis.affine)
    nii_img = nii_img.as_reoriented(ornt_transform(ornt_in, ornt_out))
    nii_data = nii_img.get_fdata(dtype=np.float32)  # (X, Y, Z, T)

    # Build combined (T, D) grayordinate array.
    series = np.zeros((T, FSLR91K_NUM_VERTICES), dtype=np.float32)

    # Fill cortical surface from gifti.
    series[:, surf_ids] = gifti_data[:, surf_mask]

    # Fill subcortical voxels from NIfTI.
    for col_ids, voxel_ijk in vol_structures:
        i, j, k = voxel_ijk[:, 0], voxel_ijk[:, 1], voxel_ijk[:, 2]
        series[:, col_ids] = nii_data[i, j, k, :].T

    # Write fslr91k CIFTI dtseries.
    series_axis = nib.cifti2.SeriesAxis(start=0.0, step=NSD_TR, size=T)
    header = nib.cifti2.Cifti2Header.from_axes((series_axis, bm_axis))
    out_img = nib.cifti2.Cifti2Image(series, header=header)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out_img, out_path)
    _logger.info("Done: %s", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=str, default=str(NSD_ROOT))
    parser.add_argument("--out-root", type=str, default=str(NSD_ROOT))
    parser.add_argument("--overwrite", "-x", action="store_true", default=False)
    args = parser.parse_args()
    sys.exit(main(args))
