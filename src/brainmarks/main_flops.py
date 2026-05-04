import argparse
import datetime
import json
import importlib.metadata
import time
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils import flop_counter
from torch.utils.data import DataLoader

import brainmarks.utils as ut
from brainmarks.datasets.base import HFDataset
from brainmarks.datasets.registry import create_dataset, list_datasets
from brainmarks.models.registry import create_model, list_models

DEFAULT_CONFIG = Path(__file__).parent / "config/default_probe.yaml"


def main(args: DictConfig):
    # setup
    ut.init_distributed_mode(args)
    assert not args.distributed, "distributed probe eval not supported"
    device = torch.device(args.device)
    ut.random_seed(args.seed)

    print("fMRI foundation model flops benchmark")
    print(f"version: {importlib.metadata.version('brainmarks')}")
    print(ut.get_sha())
    print(f"cwd: {Path.cwd()}")
    print(f"start: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("config:", OmegaConf.to_yaml(args), sep="\n")

    # backbone model
    print(f"creating frozen backbone model: {args.model}")
    transform, backbone = create_model(args.model, **(args.model_kwargs or {}))
    backbone.requires_grad_(False)
    backbone.eval()
    backbone.to(device)
    print(f"backbone:\n{backbone}")
    num_params = sum(p.numel() for p in backbone.parameters())
    print(f"backbone params: {num_params / 1e6:.1f}M")

    # dataset
    print(f"creating dataset: {args.dataset} ({backbone.__space__})")
    dataset_dict = create_dataset(
        args.dataset, space=backbone.__space__, **(args.dataset_kwargs or {})
    )

    for split, ds in dataset_dict.items():
        print(f"{split} (n={len(ds)}):\n{ds}\n")
    train_dataset: HFDataset = dataset_dict["train"]

    sample = train_dataset[0]
    input_shape = list(sample["bold"].shape)
    frames_per_sample = input_shape[0]
    input_size_bytes = sample["bold"].numel() * 4
    print(f"input size: {input_shape}")

    if hasattr(transform, "fit"):
        print("fitting transform on training dataset")
        transform.fit(train_dataset)
    train_dataset.compose(transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor if args.num_workers else None,
        drop_last=True,
    )

    batch = next(iter(train_loader))
    batch = ut.send_data(batch, device)
    with torch.inference_mode():
        with flop_counter.FlopCounterMode(display=False) as counter:
            backbone(batch)
            num_flops = counter.get_total_flops() / args.batch_size
    print(f"total flops: {num_flops / 1e9:.1f}G")

    warmup_steps = max(args.num_workers, 1)
    total_steps = args.steps_per_epoch

    print("benchmarking data loading")
    train_iter = iter(ut.infinite_data_wrapper(train_loader))
    for ii in range(-warmup_steps, total_steps):
        batch = next(train_iter)
        batch = ut.send_data(batch, device)
        if ii == 0:
            t0 = time.perf_counter()
            total_samples = 0
        elif ii > 0:
            step = ii + 1
            elapsed = time.perf_counter() - t0
            total_samples += args.batch_size
            load_sps = total_samples / elapsed
            load_fps = (total_samples * frames_per_sample) / elapsed
            load_mbs = (total_samples * input_size_bytes) / elapsed / 1e6
            if step % 20 == 0 or step == total_steps:
                print(
                    f"load step {step}/{total_steps} | {load_sps:.0f} samp/s | {load_mbs:.0f} MB/s"
                )

    print("benchmarking forward pass")
    train_iter = iter(ut.infinite_data_wrapper(train_loader))
    backbone.eval()
    with torch.inference_mode():
        for ii in range(-warmup_steps, total_steps):
            batch = next(train_iter)
            batch = ut.send_data(batch, device)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.amp):
                backbone(batch)
            torch.cuda.synchronize()
            t_step = time.perf_counter() - t0
            if ii == 0:
                total_t_step = 0.0
                total_samples = 0
            elif ii > 0:
                step = ii + 1
                total_t_step += t_step
                total_samples += args.batch_size
                fwd_sps = total_samples / total_t_step
                fwd_fps = (total_samples * frames_per_sample) / total_t_step
                fwd_tflops = total_samples * num_flops / total_t_step / 1e12
                if step % 20 == 0 or step == total_steps:
                    print(
                        f"fwd step {step}/{total_steps} | {fwd_sps:.1f} sample/s | {fwd_tflops:.0f} Tflop/s"
                    )

    result = {
        "model": args.model,
        "dataset": args.dataset,
        "input_shape": input_shape,
        "params": num_params,
        "gflop": num_flops / 1e9,
        "load_sps": load_sps,
        "load_fps": load_fps,
        "load_mbs": load_mbs,
        "fwd_sps": fwd_sps,
        "fwd_fps": fwd_fps,
        "fwd_tflops": fwd_tflops,
    }

    print("result:\n----")
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model",
        type=str,
        help=f"[{', '.join(list_models())}]",
    )
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
    cfg.dataset = args.dataset
    main(cfg)
