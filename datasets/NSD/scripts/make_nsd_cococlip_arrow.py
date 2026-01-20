import argparse
import logging
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import datasets as hfds
import numpy as np
import pandas as pd
from cloudpathlib import AnyPath, CloudPath, S3Client, S3Path
from boto3.s3.transfer import TransferConfig

import fmri_fm_eval.nisc as nisc
import fmri_fm_eval.readers as readers

# use smaller writer batch size to avoid OverflowError on very large mni data
# https://github.com/huggingface/datasets/issues/6422
hfds.config.DEFAULT_MAX_BATCH_SIZE = 256

logging.basicConfig(
    format="[%(levelname)s %(asctime)s]: %(message)s",
    level=logging.INFO,
    datefmt="%y-%m-%d %H:%M:%S",
)
logging.getLogger("nibabel").setLevel(logging.ERROR)
logging.getLogger("botocore").setLevel(logging.ERROR)  # quiet aws credential log msg

_logger = logging.getLogger(__name__)

ROOT = Path(__file__).parents[1]
NSD_ROOT = ROOT / "data/NSD"

# Upsampled 1.0s TR for high-res func1mm data.
# https://cvnlab.slite.page/p/vjWTghPTb3/Time-series-data
NSD_TR = 1.0

# clip length
NUM_FRAMES = 16

# split of subjects and sessions
# we hold out two subjects for validation and test
# we also include an in-distribution test set "testid"
# nb, sessions are 1-indexed in the metadata (from NSD filenames)
SUB_SES_SPLITS = {
    "train": [
        ("subj01", (0, 35)),
        ("subj02", (0, 35)),
        ("subj03", (0, 27)),
        ("subj06", (0, 27)),
        ("subj07", (0, 35)),
        ("subj08", (0, 25)),
    ],
    "validation": [
        ("subj04", (0, 30)),
    ],
    "test": [
        ("subj05", (0, 30)),  # nb, subj05 still has 10 more sessions
    ],
    "testid": [
        ("subj01", (35, 40)),
        ("subj02", (35, 40)),
        ("subj03", (27, 32)),
        ("subj06", (27, 32)),
        ("subj07", (35, 40)),
        ("subj08", (25, 30)),
    ],
}


def main(args):
    valid_spaces = readers.VOLUME_SPACES | {"fslr64k", "schaefer400", "flat"}
    assert args.space in valid_spaces, f"{args.space} not supported"

    out_root = AnyPath(args.out_root or (ROOT / "data/processed"))
    outdir = out_root / f"nsd-cococlip.{args.space}.arrow"

    upload_client = get_s3_client()
    if isinstance(outdir, CloudPath):
        outdir = CloudPath(outdir, client=upload_client)

    _logger.info("Generating dataset: %s", outdir)
    if outdir.exists():
        _logger.info("Output %s exists; exiting.", outdir)
        return

    trial_df = pd.read_parquet(ROOT / "metadata/nsd_trial_metadata.parquet")
    include_trial_ids = np.load(ROOT / "metadata/nsd_include_trial_ids.npy")
    trial_df = trial_df.loc[include_trial_ids]

    run_splits = {}
    for split in SUB_SES_SPLITS:
        run_splits[split] = []
        for sub, (start, end) in SUB_SES_SPLITS[split]:
            for ses_id in range(start, end):
                ses = ses_id + 1  # nsd sessions are 1-indexed
                ses_df = trial_df.query(f"sub == '{sub}' and ses == {ses}")
                for run in ses_df["run"].unique().tolist():
                    run_splits[split].append((sub, ses, run))
        _logger.info(f"{split}: num runs = {len(run_splits[split])}")

    logits = np.load(ROOT / "data/nsd_clip_coco_logits.npy")
    # nb, targets are shape (n, 80) (excluding background category 0)
    # so if you're used to classic coco category ids these will be off by 1
    targets = np.argmax(logits, axis=1)

    # load the data reader for the target space and look up the data dimension.
    # all readers return a bold data array of shape (n_samples, dim).
    reader = readers.READER_DICT[args.space]()
    dim = readers.DATA_DIMS[args.space]

    # root can be local or remote.
    root = AnyPath(args.root or NSD_ROOT)

    # the bold data are scaled to mean 0, stdev 1 and then truncated to float16 to save
    # space. but we keep the mean and std to reverse this since some models need this.
    # note, the mean and std are computed over the entire run and are redundant across
    # clips from the same run.
    features = hfds.Features(
        {
            "sub": hfds.Value("string"),
            "ses": hfds.Value("int32"),
            "run": hfds.Value("int32"),
            "trial_id": hfds.Value("int32"),
            "nsd_id": hfds.Value("int32"),
            "category_id": hfds.Value("int32"),
            "path": hfds.Value("string"),
            "start": hfds.Value("int32"),
            "end": hfds.Value("int32"),
            "n_frames": hfds.Value("int32"),
            "tr": hfds.Value("float32"),
            "bold": hfds.Array2D(shape=(None, dim), dtype="float16"),
            "mean": hfds.Array2D(shape=(1, dim), dtype="float32"),
            "std": hfds.Array2D(shape=(1, dim), dtype="float32"),
        }
    )

    # generate the datasets with huggingface. cache to a temp dir to save space.
    with tempfile.TemporaryDirectory(prefix="huggingface-") as tmpdir:
        dataset_dict = {}
        for split, runs in run_splits.items():
            dataset_dict[split] = hfds.Dataset.from_generator(
                generate_samples,
                features=features,
                gen_kwargs={
                    "runs": runs,
                    "root": root,
                    "trial_df": trial_df,
                    "targets": targets,
                    "reader": reader,
                    "is_volume": args.space in readers.VOLUME_SPACES,
                },
                num_proc=args.num_proc,
                split=hfds.NamedSplit(split),
                cache_dir=tmpdir,
                # otherwise fingerprint crashes on mni space, ig bc of hashing the reader
                fingerprint=f"nsd-cococlip-{args.space}-{split}",
            )
        dataset = hfds.DatasetDict(dataset_dict)

        if isinstance(outdir, S3Path):
            _logger.info("Saving to s3: %s", outdir)
            tmp_outdir = Path(tmpdir) / outdir.name
            # in theory save_to_disk should support s3, but idk why it wasn't working
            dataset.save_to_disk(tmp_outdir, max_shard_size="300MB")
            outdir.upload_from(tmp_outdir)
        else:
            _logger.info("Saving locally: %s", outdir)
            dataset.save_to_disk(outdir, max_shard_size="300MB")


def generate_samples(
    runs: list[tuple[str, int, int]],
    *,
    root: Path,
    trial_df: pd.DataFrame,
    targets: np.ndarray,
    reader: readers.Reader,
    is_volume: bool = False,
):
    for (sub, ses, run), path, fullpath in prefetch(root, runs, is_volume=is_volume):
        series = reader(fullpath)
        series, mean, std = nisc.scale(series)

        run_df = trial_df.query(f"sub == '{sub}' and ses == {ses} and run == {run}")
        for _, event in run_df.iterrows():
            start = int(event["onset"] / NSD_TR)
            end = start + NUM_FRAMES
            if end > len(series):
                continue
            clip = series[start:end]
            target = targets[event["nsd_id"]]

            sample = {
                "sub": sub,
                "ses": ses,
                "run": run,
                "trial_id": event["trial_id"],
                "nsd_id": event["nsd_id"],
                "category_id": target,
                "path": str(path),
                "start": start,
                "end": end,
                "n_frames": len(clip),
                "tr": NSD_TR,
                "bold": clip.astype(np.float16),
                "mean": mean.astype(np.float32),
                "std": std.astype(np.float32),
            }
            yield sample


def prefetch(
    root: AnyPath, runs: list[tuple[str, int, int]], is_volume: bool, *, max_workers: int = 1
):
    """Prefetch files from remote storage."""

    with tempfile.TemporaryDirectory(prefix="prefetch-") as tmpdir:

        def fn(run_tuple: tuple[str, int, int]):
            sub, ses, run = run_tuple
            if is_volume:
                path = f"{sub}/MNI152/timeseries/timeseries_session{ses:02d}_run{run:02d}.nii.gz"
            else:
                # nb, .lh path is passed but read_gifti_surf_data loads both hemispheres
                path = f"{sub}/32k_fs_LR/timeseries/timeseries_session{ses:02d}_run{run:02d}.lh.func.gii"

            fullpath = root / "nsddata_timeseries/ppdata" / path
            if isinstance(fullpath, CloudPath):
                tmppath = Path(tmpdir) / path
                tmppath.parent.mkdir(parents=True, exist_ok=True)
                fullpath = fullpath.download_to(tmppath)
            return run_tuple, path, fullpath

        with ThreadPoolExecutor(max_workers) as executor:
            futures = [executor.submit(fn, run_tuple) for run_tuple in runs]

            for future in futures:
                path, fullpath = future.result()
                yield path, fullpath

                if str(fullpath).startswith(tmpdir):
                    fullpath.unlink()


def get_s3_client():
    config = TransferConfig(
        multipart_threshold=8 * 1024 * 1024,
        multipart_chunksize=8 * 1024 * 1024,
        max_concurrency=10,
        use_threads=True,
    )

    if "R2_ACCESS_KEY_ID" in os.environ:
        client = S3Client(
            aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
            endpoint_url=os.environ["R2_ENDPOINT_URL_S3"],
            boto3_transfer_config=config,
        )
    else:
        client = S3Client(boto3_transfer_config=config)
    return client


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--out-root", type=str, default=None)
    parser.add_argument(
        "--space", type=str, default="schaefer400", choices=list(readers.READER_DICT)
    )
    parser.add_argument("--num_proc", "-j", type=int, default=32)
    args = parser.parse_args()
    sys.exit(main(args))
