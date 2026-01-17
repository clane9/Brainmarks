import argparse
import logging
import os
import sys
import tempfile
from contextlib import contextmanager
from functools import partialmethod
from pathlib import Path

import nibabel as nib
import numpy as np
from cloudpathlib import AnyPath, CloudPath
from nilearn.image import resample_img
from nsdcode import NSDmapdata
from tqdm import tqdm

# Disable tqdm by default
# https://stackoverflow.com/a/67238486
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

logging.basicConfig(
    format="[%(levelname)s %(asctime)s]: %(message)s",
    level=logging.WARNING,
    datefmt="%y-%m-%d %H:%M:%S",
)

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

ROOT = Path(__file__).parents[1]

BATCH_SIZE = 16

# NSD MNI 1mm output affine
# From official template: s3://natural-scenes-dataset/nsddata/templates/MNI152_T1_1mm.nii.gz
# Orientation: LAS (Left-Anterior-Superior)
NSD_MNI_1MM_SHAPE = (182, 218, 182)
NSD_MNI_1MM_AFFINE = np.array(
    [
        [-1.0, 0.0, 0.0, 90.0],
        [0.0, 1.0, 0.0, -126.0],
        [0.0, 0.0, 1.0, -72.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

# Target MNI152 FSL 2mm (LAS for consistency)
MNI152_2MM_SHAPE = (91, 109, 91)
MNI152_2MM_AFFINE = np.array(
    [
        [-2.0, 0.0, 0.0, 90.0],
        [0.0, 2.0, 0.0, -126.0],
        [0.0, 0.0, 2.0, -72.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def main(
    path: str | Path,
    nsd_dir: str | Path | None = None,
    out_dir: str | Path | None = None,
    overwrite: bool = False,
):
    path = AnyPath(path)

    # Parse info from the path.
    # Example paths:
    #     natural-scenes-dataset/nsddata_timeseries/ppdata/subj01/func1mm/timeseries/timeseries_session01_run01.nii.gz
    #     natural-scenes-dataset/nsddata_betas/ppdata/subj01/func1mm/betas_fithrf/betas_session30.nii.gz
    #     s3://bucket/nsddata_timeseries/ppdata/subj01/func1mm/timeseries/timeseries_session01_run01.nii.gz
    subid = int(path.parts[-4][-2:])  # subj01 -> 1
    func_res = path.parts[-3]  # func1mm
    assert func_res == "func1mm", "Only func1mm data supported"
    sourcespace = "func1pt0"

    # Determine NSD directory for transforms (must be local).
    if nsd_dir is None:
        if isinstance(path, CloudPath):
            raise ValueError("Must specify --nsd-dir when input is a cloud path")
        nsd_dir = path.parents[5]
    nsd_dir = Path(nsd_dir)

    # Prepare output path.
    out_dir = AnyPath(out_dir) if out_dir else path.parents[5]
    out_base = path.relative_to(path.parents[5])
    out_base = out_base.replace(func_res, "MNI152NLin6Asym_res-2")
    out_path = AnyPath(out_dir / out_base)

    if out_path.exists() and not overwrite:
        _logger.info("Output %s exists; skipping.", out_path)
        return

    # Download input if remote, process, and upload output if remote.
    with tempfile.TemporaryDirectory(prefix="nsd-") as tmpdir:
        tmpdir = Path(tmpdir)

        # Download input if needed.
        if isinstance(path, CloudPath):
            local_input = tmpdir / "input.nii.gz"
            _logger.info("Downloading %s", path)
            path.download_to(local_input)
        else:
            local_input = Path(path)

        # Process the file.
        local_output = tmpdir / "output.nii.gz"
        _process_file(local_input, local_output, nsd_dir, subid, sourcespace, tmpdir)

        # Upload output if needed.
        if isinstance(out_path, CloudPath):
            _logger.info("Uploading to %s", out_path)
            out_path.upload_from(local_output)
        else:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            local_output.rename(out_path)

        _logger.info("Done: %s", out_path)


def _process_file(
    input_path: Path, output_path: Path, nsd_dir: Path, subid: int, sourcespace: str, tmpdir: Path
):
    """Process a single NSD file: native volume -> MNI 2mm."""
    # Load memory mapped volume time series.
    img = nib.load(input_path)
    nvols = img.shape[-1]

    # Process in batches to avoid loading full 4D volume in memory.
    nsd = NSDmapdata(nsd_dir)
    frames = []

    for ii in tqdm(range(0, nvols, BATCH_SIZE), disable=False):
        batch = img.slicer[..., ii : ii + BATCH_SIZE].get_fdata()

        tmppath = Path(tmpdir) / f"batch_{ii:03d}.nii.gz"

        # NSD native volume to MNI.
        # we let NSD save as nifti so we don't have to worry about getting
        # orientation correct.
        with suppress_print():
            nsd.fit(
                subid,
                sourcespace,
                "MNI",
                batch,
                interptype="cubic",
                badval=0,
                outputfile=str(tmppath),
            )

        # resample the batch to MNI 2mm space (LAS)
        batch_img = nib.load(tmppath)
        batch_img = resample_img(
            batch_img,
            target_affine=MNI152_2MM_AFFINE,
            target_shape=MNI152_2MM_SHAPE,
            interpolation="continuous",
            force_resample=True,
            copy_header=True,
        )
        batch = batch_img.get_fdata()
        batch = batch.astype(np.float32)
        frames.append(batch)
        tmppath.unlink()

    # Concatenate all frames.
    data = np.concatenate(frames, axis=-1)
    img_mni = nib.Nifti1Image(data, MNI152_2MM_AFFINE)

    # Save output.
    nib.save(img_mni, output_path)
    _logger.info("Processed: %s -> %s %s", input_path.name, output_path, img_mni.shape)


@contextmanager
def suppress_print():
    devnull = open(os.devnull, "w")
    old_stdout = sys.stdout
    try:
        sys.stdout = devnull
        yield
    finally:
        sys.stdout = old_stdout
        devnull.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Resample NSD func1mm data to MNI152NLin6Asym 2mm space."
    )
    parser.add_argument("path", type=str, help="Input path (local or s3://)")
    parser.add_argument(
        "--nsd-dir",
        type=str,
        default=None,
        help="Local NSD directory containing transforms (required for S3 input)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory (local or s3://). Defaults to same root as input.",
    )
    parser.add_argument("--overwrite", "-x", action="store_true", default=False)
    args = parser.parse_args()
    main(**vars(args))
