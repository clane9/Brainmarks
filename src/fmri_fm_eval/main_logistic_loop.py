# This source code is licensed under the Apache License, Version 2.0

import argparse
import datetime
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn.utils
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

import fmri_fm_eval.utils as ut
import fmri_fm_eval.version
from fmri_fm_eval.datasets.base import HFDataset
from fmri_fm_eval.datasets.registry import create_dataset, list_datasets
from fmri_fm_eval.models.registry import create_model, list_models

DEFAULT_CONFIG = Path(__file__).parent / "config/default_logistic_loop.yaml"


def main(args: DictConfig):
    # setup
    ut.init_distributed_mode(args)
    assert not args.distributed, "distributed logistic eval not supported"
    device = torch.device(args.device)
    ut.random_seed(args.seed)

    if not args.get("name"):
        args.name = (
            f"{args.name_prefix}/{args.model}/{args.representation}__logistic_loop/{args.dataset}"
        )
    args.output_dir = f"{args.output_root}/{args.name}"
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_cfg_path = output_dir / "config.yaml"
    if out_cfg_path.exists():
        prev_cfg = OmegaConf.load(out_cfg_path)
        assert args == prev_cfg, "current config doesn't match previous config"
    else:
        OmegaConf.save(args, out_cfg_path)

    ut.setup_for_distributed(log_path=output_dir / "log.txt")

    print("fMRI foundation model logistic probe loop eval")
    print(f"version: {fmri_fm_eval.version.__version__}")
    print(ut.get_sha())
    print(f"cwd: {Path.cwd()}")
    print(f"start: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("config:", OmegaConf.to_yaml(args), sep="\n")

    # backbone model
    print(f"creating frozen backbone model: {args.model}")
    transform, backbone = create_model(args.model, **(args.model_kwargs or {}))
    backbone.requires_grad_(False)
    backbone.to(device)
    print(f"backbone:\n{backbone}")

    # dataset
    print(f"creating dataset: {args.dataset} ({backbone.__space__})")
    dataset_dict = create_dataset(
        args.dataset, space=backbone.__space__, **(args.dataset_kwargs or {})
    )
    for split, ds in dataset_dict.items():
        print(f"{split} (n={len(ds)}):\n{ds}\n")
    train_dataset: HFDataset = dataset_dict["train"]

    if hasattr(transform, "fit"):
        print("fitting transform on training dataset")
        transform.fit(train_dataset)

    if transform is not None:
        for split, ds in dataset_dict.items():
            ds.compose(transform)

    # extract features
    print("extracting features for all splits")
    start_time = time.monotonic()
    features_dict, targets_dict = extract_features(args, backbone, dataset_dict, device)
    extract_time = time.monotonic() - start_time
    print(f"feature extraction time: {datetime.timedelta(seconds=int(extract_time))}")

    for split, features in features_dict.items():
        print(f"{split} features: {features.shape}")

    all_features = np.concatenate(list(features_dict.values()))
    all_targets = np.concatenate(list(targets_dict.values()))

    train_log = output_dir / "train_log.json"
    if train_log.exists():
        with train_log.open() as f:
            table = [json.loads(line.strip()) for line in f.readlines()]
        start_trial = table[-1]["trial_id"] + 1
        print(f"resuming from trial {start_trial}")
    else:
        table = []
        start_trial = 0
        print(f"starting from trial {start_trial}")

    for trial_id in range(start_trial, args.n_trials):
        random_state = sklearn.utils.check_random_state(args.seed + trial_id)
        X_train, X_test, y_train, y_test = train_test_split(
            all_features,
            all_targets,
            train_size=0.7,
            random_state=random_state,
            stratify=all_targets,
        )
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        clf = LogisticRegressionCV(
            Cs=args.Cs,
            cv=args.cv_folds,
            scoring="accuracy",
            max_iter=args.max_iter,
            random_state=random_state,
            n_jobs=args.num_workers,
        )

        clf.fit(X_train, y_train)
        acc_test = clf.score(X_test, y_test)

        record = {
            "model": args.model,
            "repr": args.representation,
            "clf": "logistic",
            "dataset": args.dataset,
            "trial_id": trial_id,
            "C": float(clf.C_[0]),
            "split": "test",
            "acc": acc_test,
        }
        table.append(record)

        print(json.dumps(record))
        with train_log.open("a") as f:
            print(json.dumps(record), file=f)

    table = pd.DataFrame.from_records(table)
    summary = (
        table.groupby(["model", "repr", "clf", "dataset"])
        .agg({"trial_id": "count", "acc": ["mean", "std"]})
        .reset_index()
    )
    summary.columns = ["model", "repr", "clf", "dataset", "n_trials", "acc_mean", "acc_std"]
    summary_fmt = summary.to_markdown(index=False, floatfmt=".5g")
    print(f"eval results:\n\n{summary_fmt}\n\n")
    table.to_csv(output_dir / "eval_table.csv", index=False)

    total_time = time.monotonic() - start_time
    print(f"done! total time: {datetime.timedelta(seconds=int(total_time))}")


@torch.inference_mode()
def extract_features(
    args: DictConfig,
    backbone: nn.Module,
    dataset_dict: dict[str, HFDataset],
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    backbone.eval()
    print_freq = args.get("print_freq", 20)

    features_dict = {}
    targets_dict = {}

    for split, dataset in dataset_dict.items():
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False,
        )

        metric_logger = ut.MetricLogger(delimiter="  ")
        header = f"extract ({split})"

        all_features = []
        all_targets = []

        for batch in metric_logger.log_every(loader, print_freq, header, len(loader)):
            batch = ut.send_data(batch, device)
            target = batch.pop("target")

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.amp):
                cls_embeds, reg_embeds, patch_embeds = backbone(batch)

            all_embeds = {"cls": cls_embeds, "reg": reg_embeds, "patch": patch_embeds}
            embeds = all_embeds[args.representation]

            # average over sequence dimension: (n, l, d) -> (n, d)
            if embeds.ndim == 3:
                embeds = embeds.mean(dim=1)

            all_features.append(embeds.cpu().float().numpy())
            all_targets.append(target.cpu().numpy())

        features_dict[split] = np.concatenate(all_features, axis=0)
        targets_dict[split] = np.concatenate(all_targets, axis=0)

    return features_dict, targets_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model",
        type=str,
        help=f"[{', '.join(list_models())}]",
    )
    parser.add_argument("representation", type=str, help="[cls, reg, patch]")
    parser.add_argument(
        "dataset",
        type=str,
        help=f"[{', '.join(list_datasets())}]",
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--overrides", type=str, default=None, nargs="+")
    args = parser.parse_args()
    cfg = OmegaConf.load(DEFAULT_CONFIG)
    if args.config:
        cfg = OmegaConf.unsafe_merge(cfg, OmegaConf.load(args.config))
    if args.overrides:
        cfg = OmegaConf.unsafe_merge(cfg, OmegaConf.from_dotlist(args.overrides))
    cfg.model = args.model
    cfg.representation = args.representation
    cfg.dataset = args.dataset
    main(cfg)
