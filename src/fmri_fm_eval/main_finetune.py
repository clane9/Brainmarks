# This source code is licensed under the Apache License, Version 2.0
#
# References:
# main_probe.py: adapted for LoRA fine-tuning with peft

import argparse
import datetime
import json
import importlib.metadata
import math
import re
import time
from functools import partial
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import sklearn.metrics
import torch
import torch.nn as nn
import wandb
from cloudpathlib import S3Path
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, WeightedRandomSampler

import fmri_fm_eval.utils as ut
from fmri_fm_eval.classifiers import create_classifier, list_classififiers
from fmri_fm_eval.datasets.base import HFDataset
from fmri_fm_eval.datasets.registry import create_dataset, list_datasets
from fmri_fm_eval.models.registry import create_model, list_models

DEFAULT_CONFIG = Path(__file__).parent / "config/default_finetune.yaml"

METRICS = {
    "acc": sklearn.metrics.accuracy_score,
    "f1": partial(sklearn.metrics.f1_score, average="macro"),
}


def main(args: DictConfig):
    # setup
    ut.init_distributed_mode(args)
    assert not args.distributed, "distributed fine-tune eval not supported"
    device = torch.device(args.device)
    ut.random_seed(args.seed)

    if not args.get("name"):
        args.name = (
            f"{args.name_prefix}/"
            f"{args.dataset}__{args.model}__{args.representation}__{args.classifier}"
        )
    args.output_dir = f"{args.output_root}/{args.name}"
    output_dir = Path(args.output_dir)

    # remote backup location
    if args.remote_root:
        args.remote_dir = f"{args.remote_root}/{args.name}"
        if S3Path(args.remote_dir).exists():
            ut.rsync(args.remote_dir, args.output_dir)
    else:
        args.remote_dir = None

    output_dir.mkdir(parents=True, exist_ok=True)
    out_cfg_path = output_dir / "config.yaml"
    if out_cfg_path.exists():
        prev_cfg = OmegaConf.load(out_cfg_path)
        assert args == prev_cfg, "current config doesn't match previous config"
    else:
        OmegaConf.save(args, out_cfg_path)

    if args.wandb:
        wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=args.name,
            notes=args.notes,
            config=OmegaConf.to_container(args),
        )

    ut.setup_for_distributed(log_path=output_dir / "log.txt")

    print("fMRI foundation model fine-tune eval")
    print(f"version: {importlib.metadata.version('fmri-fm-eval')}")
    print(ut.get_sha())
    print(f"cwd: {Path.cwd()}")
    print(f"start: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("config:", OmegaConf.to_yaml(args), sep="\n")

    # backbone model
    print(f"creating backbone model: {args.model}")
    transform, backbone = create_model(args.model, **(args.model_kwargs or {}))

    if args.finetune_mode == "lora":
        print(f"applying LoRA (r={args.lora.r}, alpha={args.lora.lora_alpha})")
        lora_cfg = LoraConfig(
            r=args.lora.r,
            lora_alpha=args.lora.lora_alpha,
            lora_dropout=args.lora.lora_dropout,
            target_modules=args.lora.target_modules,
            exclude_modules=args.lora.get("exclude_modules"),
        )
        backbone = get_peft_model(backbone, lora_cfg)
    elif args.finetune_mode == "full_ft":
        print("full fine-tuning: unfreezing all backbone parameters")
        backbone.requires_grad_(True)
        if args.full_ft.freeze_layers:
            freeze_backbone_params(backbone, args.full_ft.freeze_layers)
    else:
        raise ValueError(f"unknown finetune_mode: {args.finetune_mode}")

    trainable_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    all_param = sum(p.numel() for p in backbone.parameters())
    print(
        f"trainable params: {trainable_params:,d} || "
        f"all params: {all_param:,d} || "
        f"trainable%: {100 * trainable_params / all_param:.4f}"
    )

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
    args.num_classes = train_dataset.num_classes

    if hasattr(transform, "fit"):
        print("fitting transform on training dataset")
        transform.fit(train_dataset)

    if transform is not None:
        for split, ds in dataset_dict.items():
            ds.compose(transform)

    # balanced class sampling for imbalanced classes
    if args.balanced_sampling:
        weights = 1 / (train_dataset.label_counts / train_dataset.label_counts.max())
        print(f"sampling with balanced class weights: {np.round(weights, 2)}")
        weights = weights[train_dataset.target_ids]
        train_sampler = WeightedRandomSampler(weights, num_samples=len(train_dataset))
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor if args.num_workers else None,
        drop_last=True,
    )

    eval_loaders_dict = {}
    for split, dataset in dataset_dict.items():
        eval_loaders_dict[split] = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor if args.num_workers else None,
            drop_last=False,
        )
    val_loader = eval_loaders_dict["validation"]

    # classifier head
    print("running backbone on example batch to get embedding dim")
    embed_dim = get_embedding_dim(args, backbone, train_dataset, device)
    print(f"embedding feature dim ({args.representation}): {embed_dim}")

    print(f"creating classifier head: {args.classifier}")
    classifier = create_classifier(
        args.classifier,
        in_dim=embed_dim,
        out_dim=args.num_classes,
        **(args.classifier_kwargs or {}),
    )

    model = FineTuneModel(backbone, args.representation, classifier)
    model.to(device)

    num_backbone_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    num_head_params = sum(p.numel() for p in classifier.parameters())
    print(f"trainable backbone params: {num_backbone_params / 1e6:.3f}M")
    print(f"classifier head params: {num_head_params / 1e6:.3f}M")

    # optimizer
    print("setting up optimizer")
    total_batch_size = args.batch_size * args.accum_iter
    print(
        f"total batch size: {total_batch_size} = "
        f"{args.batch_size} bs per gpu x {args.accum_iter} accum"
    )

    param_groups = make_param_groups(backbone, classifier, args)
    print(f"backbone lr: {args.lr:.2e}")
    print(f"head lr: {args.lr * args.head_lr_scale:.2e}")
    ut.update_lr(param_groups, args.lr)
    ut.update_wd(param_groups, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups)

    # lr schedule
    if not args.steps_per_epoch:
        args.steps_per_epoch = len(train_loader) // args.accum_iter
    total_steps = args.epochs * args.steps_per_epoch
    warmup_steps = args.warmup_epochs * args.steps_per_epoch
    no_decay = args.get("no_decay", False)

    delay_epochs = args.get("delay_epochs", 0)
    delay_steps = delay_epochs * args.steps_per_epoch

    head_lr_schedule = make_lr_schedule(args.lr, total_steps, warmup_steps, no_decay=no_decay)
    backbone_lr_schedule = make_lr_schedule(
        args.lr, total_steps, warmup_steps, delay_steps=delay_steps, no_decay=no_decay
    )
    lr_schedules = {"backbone": backbone_lr_schedule, "head": head_lr_schedule}

    print(f"full schedule: epochs = {args.epochs} (steps = {total_steps}) (decay = {not no_decay})")
    print(f"warmup: epochs = {args.warmup_epochs} (steps = {warmup_steps})")
    if delay_steps > 0:
        print(f"backbone delay: epochs = {delay_epochs} (steps = {delay_steps})")

    # load checkpoint/resume training
    ckpt_meta = load_model(args, model, optimizer)
    best_score = ckpt_meta["best_score"] if ckpt_meta else float("-inf")

    # training loss
    criterion = nn.CrossEntropyLoss()

    print(f"start training for {args.epochs} epochs")
    log_wandb = args.wandb and ut.is_main_process()
    start_time = time.monotonic()
    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(
            args,
            model,
            criterion,
            train_loader,
            optimizer,
            lr_schedules,
            epoch,
            device,
        )

        val_stats = evaluate(
            args,
            model,
            criterion,
            val_loader,
            epoch,
            device,
            eval_name="validation",
        )

        if log_wandb:
            wandb.log(val_stats, (epoch + 1) * args.steps_per_epoch)

        cv_score = get_cv_score(args, val_stats)
        val_scores_fmt = "  ".join(
            f"{metric}: {val_stats[f'validation/{metric}']:.3f}"
            for metric in ["loss"] + args.metrics
        )
        print(f"cv: [{epoch}]  {val_scores_fmt}")

        merged_stats = {"epoch": epoch, **train_stats, **val_stats}
        with (output_dir / "train_log.json").open("a") as f:
            print(json.dumps(merged_stats), file=f)

        is_best = cv_score > best_score
        if is_best:
            best_score = cv_score
        meta = {
            "score": cv_score,
            "epoch": epoch,
            "is_best": is_best,
            "best_score": best_score,
        }

        save_model(
            args,
            epoch,
            model,
            optimizer,
            meta=meta,
            is_best=is_best,
        )

        if args.remote_dir:
            print(f"backing up to remote: {args.remote_dir}")
            ut.rsync(args.remote_dir, output_dir)

    for ckpt_label in ["last", "best"]:
        evaluate_checkpoint(
            args, model, criterion, eval_loaders_dict, device, ckpt_label, output_dir, log_wandb
        )

    table = pd.read_csv(output_dir / "eval_table.csv")
    table_fmt = table.to_markdown(index=False, floatfmt=".5g")
    print(f"eval results:\n\n{table_fmt}\n\n")

    total_time = time.monotonic() - start_time
    print(f"done! total time: {datetime.timedelta(seconds=int(total_time))}")

    if args.remote_dir:
        print(f"backing up to remote: {args.remote_dir}")
        ut.rsync(args.remote_dir, output_dir)


class FineTuneModel(nn.Module):
    def __init__(self, backbone: nn.Module, representation: str, classifier: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.representation = representation
        self.classifier = classifier

    def forward(self, batch):
        cls_embeds, reg_embeds, patch_embeds = self.backbone(batch)
        all_embeds = {"cls": cls_embeds, "reg": reg_embeds, "patch": patch_embeds}
        embeds = all_embeds[self.representation]
        return self.classifier(embeds)


@torch.inference_mode()
def get_embedding_dim(
    args: DictConfig,
    backbone: nn.Module,
    dataset: torch.utils.data.Dataset,
    device: torch.device,
):
    loader = DataLoader(dataset, batch_size=1)
    example_batch = next(iter(loader))
    example_batch = ut.send_data(example_batch, device)

    cls_embeds, reg_embeds, patch_embeds = backbone(example_batch)
    all_embeds = {"cls": cls_embeds, "reg": reg_embeds, "patch": patch_embeds}
    embeds = all_embeds[args.representation]
    embed_dim = embeds.shape[-1]
    return embed_dim


def freeze_backbone_params(backbone: nn.Module, patterns: list[str]):
    """Freeze backbone params whose names match any of the given regex patterns."""
    compiled = [re.compile(p) for p in patterns]
    frozen = []
    unfrozen = []
    for name, param in backbone.named_parameters():
        if any(pat.search(name) for pat in compiled):
            param.requires_grad_(False)
            frozen.append(name)
        elif param.requires_grad:
            unfrozen.append(name)
    total = sum(1 for _ in backbone.parameters())
    print(f"froze {len(frozen)}/{total} backbone params matching {patterns}")
    print("frozen:\n  " + "\n  ".join(frozen))
    print("unfrozen:\n  " + "\n  ".join(unfrozen))


def make_param_groups(backbone: nn.Module, classifier: nn.Module, args: DictConfig):
    head_lr_multiplier = args.head_lr_scale
    groups = {}

    for name, param in backbone.named_parameters():
        if not param.requires_grad:
            continue
        wd_multiplier = 0.0 if (name.endswith(".bias") or "norm" in name) else 1.0
        key = ("backbone", 1.0, wd_multiplier)
        if key not in groups:
            groups[key] = {
                "params": [],
                "name": "backbone",
                "lr_multiplier": 1.0,
                "wd_multiplier": wd_multiplier,
            }
        groups[key]["params"].append(param)

    for name, param in classifier.named_parameters():
        if not param.requires_grad:
            continue
        wd_multiplier = 0.0 if (name.endswith(".bias") or "norm" in name) else 1.0
        key = ("head", head_lr_multiplier, wd_multiplier)
        if key not in groups:
            groups[key] = {
                "params": [],
                "name": "head",
                "lr_multiplier": head_lr_multiplier,
                "wd_multiplier": wd_multiplier,
            }
        groups[key]["params"].append(param)

    param_groups = list(groups.values())
    return param_groups


def make_lr_schedule(
    base_lr: float,
    total_steps: int,
    warmup_steps: int,
    no_decay: bool = False,
    delay_steps: int = 0,
):
    delay = np.zeros(delay_steps)
    active_steps = max(total_steps - delay_steps, 0)
    warmup = np.linspace(0.0, 1.0, warmup_steps)
    decay_steps = max(active_steps - warmup_steps, 0)
    if not no_decay:
        decay = np.cos(np.linspace(0, np.pi, decay_steps))
        decay = (decay + 1) / 2
    else:
        decay = np.ones(decay_steps)
    lr_schedule = base_lr * np.concatenate([delay, warmup, decay])
    lr_schedule = lr_schedule[:total_steps]
    return lr_schedule


def train_one_epoch(
    args: DictConfig,
    model: FineTuneModel,
    criterion: nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    lr_schedules: dict[str, Sequence[float]],
    epoch: int,
    device: torch.device,
):
    model.train()
    use_cuda = device.type == "cuda"
    log_wandb = args.wandb and ut.is_main_process()
    print_freq = args.get("print_freq", 20) if not args.debug else 1
    epoch_num_batches = args.steps_per_epoch * args.accum_iter if not args.debug else 10

    metric_logger = ut.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", ut.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("head_lr", ut.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"train: [{epoch}]"

    data_loader = ut.infinite_data_wrapper(data_loader)
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(
        metric_logger.log_every(data_loader, print_freq, header, epoch_num_batches)
    ):
        batch = ut.send_data(batch, device)

        global_step = epoch * args.steps_per_epoch + (batch_idx + 1) // args.accum_iter
        need_update = (batch_idx + 1) % args.accum_iter == 0
        if need_update:
            for group in optimizer.param_groups:
                schedule = lr_schedules[group["name"]]
                group["lr"] = float(schedule[global_step - 1] * group["lr_multiplier"])
            lr = lr_schedules["backbone"][global_step - 1]
            head_lr = lr_schedules["head"][global_step - 1]

        target = batch.pop("target")

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.amp):
            pred = model(batch)
            loss = criterion(pred, target)

        if need_update:
            loss_value = loss.item()
            if not math.isfinite(loss_value):
                raise RuntimeError(f"Loss is {loss_value}, stopping training")

        (loss / args.accum_iter).backward()

        if need_update:
            grad = nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
            optimizer.zero_grad()

        if need_update:
            log_metric_dict = {
                "lr": lr,
                "head_lr": head_lr,
                "loss": loss_value,
                "grad": grad.item(),
            }
            metric_logger.update(**log_metric_dict)

            if log_wandb:
                wandb.log({f"train/{k}": v for k, v in log_metric_dict.items()}, global_step)

        if need_update and use_cuda:
            torch.cuda.synchronize()

    print(f"{header} Summary:", metric_logger)

    stats = {f"train/{k}": meter.global_avg for k, meter in metric_logger.meters.items()}
    return stats


@torch.inference_mode()
def evaluate(
    args: DictConfig,
    model: FineTuneModel,
    criterion: nn.Module,
    data_loader: Iterable,
    epoch: int,
    device: torch.device,
    eval_name: str,
):
    model.eval()
    use_cuda = device.type == "cuda"
    print_freq = args.get("print_freq", 20) if not args.debug else 1
    epoch_num_batches = len(data_loader)
    if args.debug:
        epoch_num_batches = min(epoch_num_batches, 10)

    metric_logger = ut.MetricLogger(delimiter="  ")
    header = f"eval ({eval_name}): [{epoch}]"

    logits = []
    targets = []

    for batch_idx, batch in enumerate(
        metric_logger.log_every(data_loader, print_freq, header, epoch_num_batches)
    ):
        batch = ut.send_data(batch, device)
        target = batch.pop("target")

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.amp):
            logit = model(batch)

        logits.append(logit.cpu().float())
        targets.append(target.cpu())

        if use_cuda:
            torch.cuda.synchronize()

    logits = torch.cat(logits)
    targets = torch.cat(targets)

    total_loss = criterion(logits, targets)
    stats = {"loss": total_loss.item()}

    preds = torch.argmax(logits, dim=1).numpy()
    targets = targets.numpy()

    for metric in args.metrics:
        metric_fn = METRICS[metric]
        stats[metric] = metric_fn(targets, preds)

    stats = {f"{eval_name}/{k}": v for k, v in stats.items()}
    return stats


def evaluate_checkpoint(
    args: DictConfig,
    model: FineTuneModel,
    criterion: nn.Module,
    eval_loaders_dict: dict[str, DataLoader],
    device: torch.device,
    ckpt_label: str,
    output_dir: Path,
    log_wandb: bool,
):
    ckpt_path = output_dir / f"checkpoint-{ckpt_label}.pth"
    print(f"evaluating {ckpt_label} checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.backbone.load_state_dict(ckpt["backbone"], strict=False)
    model.classifier.load_state_dict(ckpt["classifier"])
    ckpt_meta = ckpt["meta"]
    print(f"eval model info:\n{json.dumps(ckpt_meta)}")

    header = {
        "model": args.model,
        "repr": args.representation,
        "clf": args.classifier,
        "dataset": args.dataset,
        "ckpt": ckpt_label,
        "epoch": ckpt_meta["epoch"],
        "lr": args.lr,
        "head_lr": args.lr * args.head_lr_scale,
    }
    eval_stats = {
        f"eval/{ckpt_label}/epoch": header["epoch"],
        f"eval/{ckpt_label}/lr": header["lr"],
        f"eval/{ckpt_label}/head_lr": header["head_lr"],
    }
    table = []

    for split, loader in eval_loaders_dict.items():
        stats = evaluate(
            args,
            model,
            criterion,
            loader,
            args.epochs,
            device,
            eval_name=split,
        )
        record = {**header, "split": split}

        log_prefix = f"eval/{ckpt_label}/{split}"
        record["loss"] = eval_stats[f"{log_prefix}/loss"] = stats[f"{split}/loss"]
        for metric in args.metrics:
            score = stats[f"{split}/{metric}"]
            record[metric] = eval_stats[f"{log_prefix}/{metric}"] = score

        table.append(record)

    table = pd.DataFrame.from_records(table)
    table.to_csv(output_dir / f"eval_table_{ckpt_label}.csv", index=False)

    with (output_dir / f"eval_log_{ckpt_label}.json").open("w") as f:
        print(json.dumps(eval_stats), file=f)

    if log_wandb:
        wandb.log(eval_stats, args.epochs * args.steps_per_epoch)

    preferred = "best" if args.get("early_stopping", True) else "last"
    if ckpt_label == preferred:
        table.to_csv(output_dir / "eval_table.csv", index=False)
        eval_stats = {k.replace(f"/{ckpt_label}", ""): v for k, v in eval_stats.items()}
        with (output_dir / "eval_log.json").open("w") as f:
            print(json.dumps(eval_stats), file=f)
        if log_wandb:
            wandb.log(eval_stats, args.epochs * args.steps_per_epoch)


def get_cv_score(args: DictConfig, stats: dict[str, float]):
    metric = args.cv_metric
    if metric.startswith("neg_"):
        sign = -1
        metric = metric[4:]
    else:
        sign = 1
    return sign * stats[f"validation/{metric}"]


def save_model(args, epoch, model, optimizer, meta=None, is_best=None):
    output_dir = Path(args.output_dir)
    last_checkpoint_path = output_dir / "checkpoint-last.pth"
    best_checkpoint_path = output_dir / "checkpoint-best.pth"

    trainable_names = {n for n, p in model.backbone.named_parameters() if p.requires_grad}
    backbone_state = {k: v for k, v in model.backbone.state_dict().items() if k in trainable_names}
    to_save = {
        "backbone": backbone_state,
        "classifier": model.classifier.state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": OmegaConf.to_container(args),
        "epoch": epoch,
        "meta": meta,
        "is_best": is_best,
    }

    print(f"saving checkpoint {last_checkpoint_path}")
    safe_save(to_save, last_checkpoint_path)
    if is_best:
        print(f"saving best checkpoint {best_checkpoint_path}")
        safe_save(to_save, best_checkpoint_path)


def safe_save(obj, path):
    path = Path(path)
    tmp_path = path.parent / f".tmp-{path.name}"
    torch.save(obj, tmp_path)
    tmp_path.rename(path)


def load_model(args, model, optimizer):
    ckpt_path = Path(args.output_dir) / "checkpoint-last.pth"

    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.backbone.load_state_dict(ckpt["backbone"], strict=False)
        model.classifier.load_state_dict(ckpt["classifier"])
        optimizer.load_state_dict(ckpt["optimizer"])
        args.start_epoch = ckpt["epoch"] + 1
        meta = ckpt["meta"]
        print(f"loaded model and optimizer state, resuming training from {args.start_epoch}")
    else:
        args.start_epoch = 0
        meta = None

    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model",
        type=str,
        help=f"[{', '.join(list_models())}]",
    )
    parser.add_argument("representation", type=str, help="[cls, reg, patch]")
    parser.add_argument(
        "classifier",
        type=str,
        help=f"[{', '.join(list_classififiers())}]",
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
    cfg.representation = args.representation
    cfg.classifier = args.classifier
    cfg.dataset = args.dataset
    main(cfg)
