from __future__ import annotations

import copy
import csv
import itertools
import json
import os
import random
import time
from dataclasses import dataclass
from dataclasses import fields as dataclass_fields

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import yaml
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from diagnostics.pretrain_tests import run_pretrain_tests
from onn_config import AppConfig
from onn_layers import FTconvlayer, replace_conv2d_with_jtc
from onn_training_state import (
    EpochDataset,
    EpochSampler,
    capture_rank_states,
    restore_rng,
    validate_resume_config,
)

DISTORTION_STRENGTH_FIELDS = [
    f.name
    for f in dataclass_fields(AppConfig)
    if f.name.endswith("_distortion_strength")
]


@dataclass(frozen=True)
class DistributedContext:
    enabled: bool = False
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def _init_distributed(config: AppConfig) -> tuple[DistributedContext, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    enable_ddp = bool(config.enable_ddp) and world_size > 1
    if not enable_ddp:
        return DistributedContext(), torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    if not dist.is_initialized():
        from datetime import timedelta

        # device_id: bind collectives explicitly (removes the NCCL
        # "guessing device ID ... can cause a hang" path). timeout: a
        # wedged rendezvous/collective becomes an error in 30 min instead
        # of a silent walltime burn (pair with
        # TORCH_NCCL_ASYNC_ERROR_HANDLING=1 in the launcher).
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            timeout=timedelta(minutes=30),
            device_id=device if device.type == "cuda" else None,
        )
    return (
        DistributedContext(
            enabled=True,
            rank=int(dist.get_rank()),
            local_rank=local_rank,
            world_size=int(dist.get_world_size()),
        ),
        device,
    )


def _cleanup_distributed(ctx: DistributedContext) -> None:
    if ctx.enabled and dist.is_initialized():
        dist.destroy_process_group()


def _barrier(ctx: DistributedContext) -> None:
    if ctx.enabled and dist.is_initialized():
        dist.barrier()


def _log(ctx: DistributedContext, message: object) -> None:
    if ctx.is_main:
        print(message)


def _read_status_kib(pid: int, key: str) -> int:
    try:
        with open(f"/proc/{pid}/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith(f"{key}:"):
                    parts = line.split()
                    return int(parts[1]) if len(parts) >= 2 else 0
    except OSError:
        return 0
    return 0


def _child_pids(pid: int) -> list[int]:
    try:
        with open(f"/proc/{pid}/task/{pid}/children", "r", encoding="utf-8") as f:
            return [int(value) for value in f.read().split()]
    except OSError:
        return []


def _process_tree_rss_gib() -> float:
    seen: set[int] = set()
    stack = [os.getpid()]
    rss_kib = 0
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        rss_kib += _read_status_kib(pid, "VmRSS")
        stack.extend(_child_pids(pid))
    return rss_kib / (1024**2)


def _print_memory_snapshot(
    ctx: DistributedContext,
    device: torch.device,
    label: str,
) -> None:
    cpu_rss_gib = _process_tree_rss_gib()
    message = (
        f"[MEM][rank={ctx.rank} local_rank={ctx.local_rank}][{label}] "
        f"process_tree_rss={cpu_rss_gib:.2f} GiB"
    )
    if device.type == "cuda":
        try:
            allocated = torch.cuda.memory_allocated(device) / (1024**3)
            reserved = torch.cuda.memory_reserved(device) / (1024**3)
            peak_allocated = torch.cuda.max_memory_allocated(device) / (1024**3)
            peak_reserved = torch.cuda.max_memory_reserved(device) / (1024**3)
            total = torch.cuda.get_device_properties(device).total_memory / (1024**3)
            message += (
                f" | cuda_alloc={allocated:.2f} GiB"
                f" cuda_reserved={reserved:.2f} GiB"
                f" peak_alloc={peak_allocated:.2f} GiB"
                f" peak_reserved={peak_reserved:.2f} GiB"
                f" gpu_total={total:.2f} GiB"
            )
        except RuntimeError as exc:
            message += f" | cuda_memory_unavailable={exc}"
    print(message, flush=True)


def _reset_jtc_pd_range_regularizers(model: nn.Module) -> None:
    for module in model.modules():
        reset_fn = getattr(module, "reset_pd_range_regularization", None)
        if callable(reset_fn):
            reset_fn()


def _jtc_pd_range_regularization_loss(model: nn.Module) -> torch.Tensor | None:
    total: torch.Tensor | None = None
    for module in model.modules():
        loss_fn = getattr(module, "weighted_pd_range_regularization_loss", None)
        if not callable(loss_fn):
            continue
        loss = loss_fn()
        if loss is None:
            continue
        total = loss if total is None else total + loss
    return total


def _local_memory_row(device: torch.device) -> torch.Tensor:
    values = [
        _process_tree_rss_gib(),
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        values[1] = torch.cuda.memory_allocated(device) / (1024**3)
        values[2] = torch.cuda.memory_reserved(device) / (1024**3)
        values[3] = torch.cuda.max_memory_allocated(device) / (1024**3)
        values[4] = torch.cuda.max_memory_reserved(device) / (1024**3)
        values[5] = torch.cuda.get_device_properties(device).total_memory / (1024**3)
    tensor_device = device if device.type == "cuda" else torch.device("cpu")
    return torch.tensor(values, dtype=torch.float64, device=tensor_device)


def _peak_memory_summary(
    ctx: DistributedContext,
    device: torch.device,
    label: str,
    *,
    print_summary: bool = True,
) -> dict[str, object]:
    row = _local_memory_row(device)
    if ctx.enabled:
        gathered = [torch.zeros_like(row) for _ in range(ctx.world_size)]
        dist.all_gather(gathered, row)
        rows = torch.stack(gathered).cpu().tolist()
    else:
        rows = [row.cpu().tolist()]

    keys = [
        "process_tree_rss_gib",
        "cuda_allocated_gib",
        "cuda_reserved_gib",
        "cuda_peak_allocated_gib",
        "cuda_peak_reserved_gib",
        "cuda_total_gib",
    ]
    per_rank = [
        {"rank": rank, **{key: float(value) for key, value in zip(keys, values)}}
        for rank, values in enumerate(rows)
    ]
    summary = {
        "label": label,
        "per_rank": per_rank,
        "max_process_tree_rss_gib": max(
            row["process_tree_rss_gib"] for row in per_rank
        ),
        "sum_process_tree_rss_gib": sum(
            row["process_tree_rss_gib"] for row in per_rank
        ),
        "max_cuda_allocated_gib": max(row["cuda_allocated_gib"] for row in per_rank),
        "max_cuda_reserved_gib": max(row["cuda_reserved_gib"] for row in per_rank),
        "max_cuda_peak_allocated_gib": max(
            row["cuda_peak_allocated_gib"] for row in per_rank
        ),
        "max_cuda_peak_reserved_gib": max(
            row["cuda_peak_reserved_gib"] for row in per_rank
        ),
        "cuda_total_gib": max(row["cuda_total_gib"] for row in per_rank),
    }
    if print_summary and ctx.is_main and device.type == "cuda":
        peak_alloc = [round(row["cuda_peak_allocated_gib"], 2) for row in per_rank]
        peak_reserved = [round(row["cuda_peak_reserved_gib"], 2) for row in per_rank]
        print(
            f"[MEM][summary][{label}] "
            f"max_peak_alloc={summary['max_cuda_peak_allocated_gib']:.2f} GiB "
            f"max_peak_reserved={summary['max_cuda_peak_reserved_gib']:.2f} GiB "
            f"max_current_alloc={summary['max_cuda_allocated_gib']:.2f} GiB "
            f"gpu_total={summary['cuda_total_gib']:.2f} GiB "
            f"per_rank_peak_alloc={peak_alloc} "
            f"per_rank_peak_reserved={peak_reserved}",
            flush=True,
        )
    return summary


def _unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def _is_jtc_cache_key(key: str) -> bool:
    return key.startswith("_jtc_cache.") or "._jtc_cache." in key


def load_model_state(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    """Load all persistent model state; derived JTC caches are rebuilt locally."""
    target = _unwrap_model(model)
    if state_dict and all(key.startswith("module.") for key in state_dict):
        state_dict = {
            key.removeprefix("module."): value for key, value in state_dict.items()
        }
    state_dict = {
        key: value for key, value in state_dict.items() if not _is_jtc_cache_key(key)
    }
    expected = {key for key in target.state_dict() if not _is_jtc_cache_key(key)}
    missing = sorted(expected - state_dict.keys())
    unexpected = sorted(state_dict.keys() - expected)
    if missing or unexpected:
        raise RuntimeError(
            f"Incompatible model state: missing keys {missing}; unexpected keys {unexpected}"
        )
    # Only lazy caches may be absent; gain/calibration buffers are mandatory.
    target.load_state_dict(state_dict, strict=False)


# -------------------------------
#  Utility helpers
# -------------------------------


def _replicate_channels(image: torch.Tensor) -> torch.Tensor:
    """Expand a single-channel image to 3 channels (MNIST -> RGB models)."""
    return image.expand(3, -1, -1)


def _dataset_transforms(dataset: str) -> tuple[transforms.Compose, transforms.Compose]:
    if dataset == "cifar10":
        train_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
                transforms.ToTensor(),
            ]
        )
        test_transform = transforms.Compose([transforms.ToTensor()])
        return train_transform, test_transform
    if dataset == "mnist":
        # Pad 28->32 and replicate to 3 channels so model architectures are
        # unchanged. No horizontal flip: digits are not mirror-invariant.
        train_transform = transforms.Compose(
            [
                transforms.Pad(2),
                transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
                transforms.ToTensor(),
                transforms.Lambda(_replicate_channels),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.Pad(2),
                transforms.ToTensor(),
                transforms.Lambda(_replicate_channels),
            ]
        )
        return train_transform, test_transform
    raise ValueError(f"Unsupported dataset: {dataset}")


def get_data_loaders(
    batch_size: int,
    *,
    dataset: str = "cifar10",
    seed: int = 0,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    download: bool = True,
    num_workers: int = 8,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int | None = 2,
) -> tuple[
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
]:
    """Create train, train-eval, and test dataloaders for the dataset."""

    train_transform, test_transform = _dataset_transforms(dataset)
    dataset_cls = {
        "cifar10": torchvision.datasets.CIFAR10,
        "mnist": torchvision.datasets.MNIST,
    }[dataset]

    full_trainset_aug = dataset_cls(
        root="./data", train=True, download=download, transform=train_transform
    )
    full_trainset_plain = dataset_cls(
        root="./data", train=True, download=download, transform=test_transform
    )
    testset = dataset_cls(
        root="./data", train=False, download=download, transform=test_transform
    )

    full_trainset_aug = EpochDataset(full_trainset_aug, seed)
    train_sampler = EpochSampler(
        full_trainset_aug,
        num_replicas=world_size if distributed else 1,
        rank=rank if distributed else 0,
        seed=int(seed),
        shuffle=True,
        drop_last=False,
    )
    train_eval_sampler = (
        DistributedSampler(
            full_trainset_plain,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
        if distributed
        else None
    )
    test_sampler = (
        DistributedSampler(
            testset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
        if distributed
        else None
    )

    num_workers = max(0, int(num_workers))
    loader_kwargs: dict[str, object] = {
        "num_workers": num_workers,
        "pin_memory": bool(pin_memory),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(persistent_workers)
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = int(prefetch_factor)

    trainloader = torch.utils.data.DataLoader(
        full_trainset_aug,
        batch_size=batch_size,
        sampler=train_sampler,
        generator=torch.Generator().manual_seed(seed),
        **loader_kwargs,
    )
    train_eval_loader = torch.utils.data.DataLoader(
        full_trainset_plain,
        batch_size=batch_size,
        shuffle=False,
        sampler=train_eval_sampler,
        generator=torch.Generator().manual_seed(seed + 1),
        **loader_kwargs,
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        sampler=test_sampler,
        generator=torch.Generator().manual_seed(seed + 2),
        **loader_kwargs,
    )
    return trainloader, train_eval_loader, testloader


# -------------------------------
#  Model definition
# -------------------------------


def _norm_divisor(x: torch.Tensor, mode: str) -> torch.Tensor:
    """Return a scalar divisor for max-norm style normalization."""
    if mode == "max":
        return x.max()
    if mode == "second_largest":
        flat = x.reshape(-1)
        if flat.numel() < 2:
            return flat.max()
        top2 = torch.topk(flat, k=2).values
        denom = top2[1]
        # If the second-largest is zero (e.g. a single nonzero outlier),
        # fall back to the maximum to avoid division by ~0.
        if denom.item() <= 0:
            denom = top2[0]
        return denom
    raise ValueError(f"Unknown max_norm_mode: {mode}")


class FFTConvNet(nn.Module):
    """Configurable 7-layer FFTConv network.

    The number of identical intermediate blocks (originally 5) can be varied
    through `config.num_identical_layers`."""

    def __init__(self, config: AppConfig):
        super().__init__()
        self.max_norm_mode = config.max_norm_mode
        self.input_gain = float(config.fftconvnet_input_gain)

        # Stem
        self.conv1 = FTconvlayer(
            3,
            8,
            config=config,
            kernel_size=8,
            hv_concat=True,
        )
        self.bn1 = nn.BatchNorm2d(16)
        self.maxpool1 = nn.MaxPool2d(2)

        # Second block (fixed)
        self.conv2 = FTconvlayer(
            16,
            16,
            config=config,
            kernel_size=8,
            hv_concat=True,
        )
        self.bn2 = nn.BatchNorm2d(32)
        self.maxpool2 = nn.MaxPool2d(2)

        # Configurable sequence of identical blocks
        class _MaxNorm(nn.Module):
            def __init__(self, mode: str):
                super().__init__()
                self.mode = mode

            def forward(self, x: torch.Tensor):
                denom = _norm_divisor(x, self.mode).clamp_min(1e-12)
                return x / denom

        blocks = []
        for _ in range(config.num_identical_layers):
            seq = [
                FTconvlayer(
                    32,
                    16,
                    config=config,
                    kernel_size=8,
                    hv_concat=True,
                ),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
            ]
            if config.normalize_blocks:
                seq.append(_MaxNorm(self.max_norm_mode))
            blocks.append(nn.Sequential(*seq))
        self.blocks = nn.Sequential(*blocks)

        self.classifier = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.Linear(256, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_gain != 1.0:
            x = x * self.input_gain
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool1(x)
        x = F.relu(x)
        x = x / _norm_divisor(x, self.max_norm_mode).clamp_min(1e-12)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.maxpool2(x)
        x = F.relu(x)
        x = x / _norm_divisor(x, self.max_norm_mode).clamp_min(1e-12)

        x = self.blocks(x)
        x = self.classifier(x)
        return x


def _build_resnet_cifar(
    config: AppConfig,
    layers: list[int],
) -> nn.Module:
    model = torchvision.models.resnet.ResNet(
        torchvision.models.resnet.BasicBlock,
        layers,
        num_classes=10,
    )
    model.conv1 = nn.Conv2d(
        3,
        64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )
    model.maxpool = nn.Identity()
    if config.conv_backend != "pytorch":
        replace_conv2d_with_jtc(
            model,
            config=config,
            max_jtc_shots=config.jtc_max_shots,
            assume_nonnegative_input=config.jtc_assume_nonnegative_input,
        )
    return model


def build_model(config: AppConfig) -> nn.Module:
    if config.model_arch == "fftconvnet":
        model = FFTConvNet(config)
    elif config.model_arch == "resnet11":
        model = _build_resnet_cifar(config, [1, 1, 1, 2])
    elif config.model_arch == "resnet18":
        model = _build_resnet_cifar(config, [2, 2, 2, 2])
    else:
        raise ValueError(f"Unsupported model_arch: {config.model_arch}")
    from onn_jtc_conv2d import configure_jtc_runtime

    configure_jtc_runtime(model, config)
    return model


# -------------------------------
#  Training / evaluation helpers
# -------------------------------


def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> float:
    _, acc = evaluate_metrics(
        model, dataloader, device, criterion=None, max_batches=max_batches
    )
    print(f"Accuracy: {acc:.3f}%")
    return acc


def evaluate_metrics(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion: nn.Module | None,
    max_batches: int | None = None,
    use_autocast: bool = False,
) -> tuple[float | None, float]:
    """Compute (avg_loss, accuracy). If criterion is None, avg_loss is None."""
    total_loss, correct, total = _evaluate_metric_totals(
        model,
        dataloader,
        device,
        criterion,
        max_batches=max_batches,
        use_autocast=use_autocast,
    )
    acc = 100.0 * correct / max(total, 1)
    if criterion is None:
        return None, acc
    return total_loss / max(total, 1), acc


def evaluate_metrics_distributed(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion: nn.Module | None,
    ctx: DistributedContext,
    max_batches: int | None = None,
    use_autocast: bool = False,
) -> tuple[float | None, float]:
    """Compute metrics over all DDP ranks without rank-0-only evaluation."""
    total_loss, correct, total = _evaluate_metric_totals(
        model,
        dataloader,
        device,
        criterion,
        max_batches=max_batches,
        use_autocast=use_autocast,
    )
    values = torch.tensor(
        [float(total_loss), float(correct), float(total)],
        dtype=torch.float64,
        device=device,
    )
    if ctx.enabled:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    total_loss_value, correct_value, total_value = values.tolist()
    acc = 100.0 * correct_value / max(total_value, 1.0)
    if criterion is None:
        return None, float(acc)
    return float(total_loss_value / max(total_value, 1.0)), float(acc)


def _evaluate_metric_totals(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion: nn.Module | None,
    max_batches: int | None = None,
    use_autocast: bool = False,
) -> tuple[float, int, int]:
    """Return summed loss, correct predictions, and sample count for one rank."""
    model.eval()
    correct = torch.zeros((), dtype=torch.float64, device=device)
    total = 0
    total_loss = torch.zeros((), dtype=torch.float64, device=device)
    with torch.inference_mode():
        for batch_idx, (images, labels) in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            images, labels = (
                images.to(device, non_blocking=True),
                labels.to(device, non_blocking=True),
            )
            with autocast(device_type=device.type, enabled=use_autocast):
                outputs = model(images)
                loss = criterion(outputs, labels) if criterion is not None else None
            if loss is not None:
                total_loss += loss.double() * labels.size(0)
            predicted = outputs.argmax(dim=1)
            total += int(labels.size(0))
            correct += (predicted == labels).sum()
    return float(total_loss.item()), int(correct.item()), total


def run_full_strength_inference(
    model: nn.Module,
    config: AppConfig,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    ctx: DistributedContext,
) -> float:
    if not DISTORTION_STRENGTH_FIELDS:
        return float("nan")

    distortion_config = copy.deepcopy(config)
    for field in DISTORTION_STRENGTH_FIELDS:
        setattr(distortion_config, field, 1.0)

    ref_model = build_model(distortion_config).to(device)
    load_model_state(ref_model, model.state_dict())

    try:
        if ctx.is_main:
            print(
                "[INFO] Running inference with all distortion strength parameters "
                "set to 1.0"
            )
        _, acc = evaluate_metrics_distributed(
            ref_model,
            dataloader,
            device,
            criterion=None,
            ctx=ctx,
            max_batches=config.max_eval_batches,
        )
    finally:
        if device.type == "cuda":
            ref_model.to("cpu")
            torch.cuda.empty_cache()
        del ref_model
    return acc


def _atomic_save(state, filename):
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    tmp_filename = f"{filename}.tmp-{os.getpid()}"
    try:
        torch.save(state, tmp_filename)
        os.replace(tmp_filename, filename)
    finally:
        if os.path.exists(tmp_filename):
            os.unlink(tmp_filename)


def _model_snapshot(model):
    return {
        name: value.detach().cpu().clone()
        for name, value in _unwrap_model(model).state_dict().items()
        if "_jtc_cache." not in name
    }


def _save_best_model(state, config, accuracy, epoch, snapshot, filename):
    """Export the actual best weights; this artifact is for inference/fine-tuning."""
    _atomic_save(
        {
            "model_state_dict": state,
            "config": vars(config),
            "best_accuracy": accuracy,
            "best_epoch": epoch,
            "epoch": epoch,
            "best_snapshot": snapshot,
            "checkpoint_kind": "best_model",
        },
        filename,
    )


def save_checkpoint(
    model: nn.Module,
    config: AppConfig,
    best_acc: float,
    filename: str,
    *,
    epoch: int = -1,
    optimizer: optim.Optimizer | None = None,
    scheduler: optim.lr_scheduler.LRScheduler | None = None,
    scaler: GradScaler | None = None,
    best_epoch: int = -1,
    best_snapshot: dict[str, object] | None = None,
    checkpoint_kind: str = "epoch",
    epoch_complete: bool = True,
    batch_idx: int = 0,
    global_step: int = 0,
    rank_states: list | None = None,
    data_seed: int,
    best_model_state_dict: dict | None = None,
) -> None:
    """Save replayable training state and the actual historical best weights."""
    if rank_states is None:
        if dist.is_initialized():
            raise ValueError(
                "Capture rank states collectively before saving a DDP checkpoint"
            )
        rank_states = capture_rank_states(next(model.parameters()).device)
    if best_epoch >= 0 and best_model_state_dict is None:
        raise ValueError("A checkpoint with a historical best must include its weights")
    from onn_shotreport import observed_shot_plans

    state: dict[str, object] = {
        "shot_plans": observed_shot_plans(_unwrap_model(model)),
        "training_state_version": 2,
        "rank_states": rank_states,
        "data_seed": int(data_seed),
        "best_model_state_dict": best_model_state_dict,
        "model_state_dict": _model_snapshot(model),
        "best_accuracy": float(best_acc),
        "best_epoch": int(best_epoch),
        "best_snapshot": best_snapshot or {},
        "epoch": int(epoch),
        "checkpoint_kind": str(checkpoint_kind),
        "epoch_complete": bool(epoch_complete),
        "batch_idx": int(batch_idx),
        "global_step": int(global_step),
        "config": vars(config),
    }
    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None:
        state["scaler_state_dict"] = scaler.state_dict()
    _atomic_save(state, filename)


def load_training_checkpoint(
    filename: str,
    model: nn.Module,
    device: torch.device,
    *,
    config: AppConfig,
    optimizer: optim.Optimizer | None = None,
    scheduler: optim.lr_scheduler.LRScheduler | None = None,
    scaler: GradScaler | None = None,
) -> dict[str, object]:
    ckpt = torch.load(filename, map_location="cpu", weights_only=False)
    if ckpt.get("training_state_version") != 2:
        raise ValueError(
            "Exact resume requires training_state_version=2 (RNG, data seed, and best weights). "
            "Use pretrained_weights for model-only initialization from older or best-model files."
        )
    validate_resume_config(ckpt["config"], config)
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    if len(ckpt["rank_states"]) != world_size:
        raise ValueError(
            "Exact resume requires the checkpoint's original DDP world size"
        )
    load_model_state(model, ckpt["model_state_dict"])
    for name, target in (
        ("optimizer", optimizer),
        ("scheduler", scheduler),
        ("scaler", scaler),
    ):
        if target is not None:
            key = f"{name}_state_dict"
            if key not in ckpt:
                raise ValueError(f"Exact resume requires saved {name} state")
            target.load_state_dict(ckpt[key])
    restore_rng(ckpt["rank_states"][rank]["rng"], device)
    return ckpt


# -------------------------------
#  Public API
# -------------------------------


def train_onn_model(config: AppConfig) -> float:
    """Entry-point used by `onn_main.py`.

    * If `config.eval_only` is True, we load `config.pretrained_weights` and run
      a single evaluation pass.
    * Otherwise we train from scratch and export the best checkpoint as well as
      the model structure under `config.output_dir`.

    Returns the final test accuracy (percent) for training or eval-only runs.
    """

    if config.seed is not None:
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

    ctx, device = _init_distributed(config)
    completed_successfully = False
    try:
        if config.resume_checkpoint and config.pretrained_weights:
            raise ValueError(
                "Use either resume_checkpoint for exact resume or "
                "pretrained_weights for model-only fine-tuning/eval, not both."
            )

        # -------------------------------------------------------------
        #  Pre-training diagnostics (plots & quick sanity checks)
        # -------------------------------------------------------------
        if config.run_pretrain_tests or config.pretrain_tests_only:
            if ctx.is_main:
                run_pretrain_tests(config)
            _barrier(ctx)
            if config.pretrain_tests_only:
                _log(
                    ctx,
                    "[INFO] Pretrain tests only requested; exiting without training.",
                )
                return float("nan")

        if ctx.is_main:
            os.makedirs(config.output_dir, exist_ok=True)
        _barrier(ctx)

        max_train_batches = config.max_train_batches
        max_eval_batches = config.max_eval_batches

        # Build model and optimization state.
        model: nn.Module = build_model(config).to(device)
        criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
        if config.optimizer == "sgd":
            optimizer = optim.SGD(
                model.parameters(),
                lr=config.learning_rate,
                momentum=config.momentum,
                weight_decay=config.weight_decay,
                nesterov=config.momentum > 0,
            )
        else:
            optimizer = optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
        warmup = max(int(config.lr_warmup_epochs), 0)
        cosine = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(int(config.num_epochs) - warmup, 1)
        )
        if warmup > 0:
            scheduler = optim.lr_scheduler.SequentialLR(
                optimizer,
                [
                    optim.lr_scheduler.LinearLR(
                        optimizer, start_factor=0.1, total_iters=warmup
                    ),
                    cosine,
                ],
                milestones=[warmup],
            )
        else:
            scheduler = cosine
        scaler = GradScaler("cuda") if device.type == "cuda" else None

        start_epoch = 0
        resume_batch_idx = 0
        global_step = 0
        best_test_acc = -1.0
        best_test_epoch = -1
        best_test_snapshot: dict[str, object] = {}
        best_model_state = None
        resume_state = None

        if config.resume_checkpoint:
            resume_state = load_training_checkpoint(
                config.resume_checkpoint,
                model,
                device,
                config=config,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
            )
            best_model_state = resume_state["best_model_state_dict"]
            checkpoint_epoch = int(resume_state.get("epoch", -1))
            epoch_complete = bool(resume_state.get("epoch_complete", True))
            resume_batch_idx = (
                0 if epoch_complete else int(resume_state.get("batch_idx", 0))
            )
            start_epoch = checkpoint_epoch + 1 if epoch_complete else checkpoint_epoch
            global_step = int(resume_state.get("global_step", 0))
            best_test_acc = float(resume_state.get("best_accuracy", -1.0))
            best_test_epoch = int(
                resume_state.get("best_epoch", resume_state.get("epoch", -1))
            )
            best_test_snapshot = dict(resume_state.get("best_snapshot", {}) or {})
            _log(
                ctx,
                f"[INFO] Resumed training checkpoint {config.resume_checkpoint} "
                f"from epoch {start_epoch}"
                + (f", batch {resume_batch_idx}" if resume_batch_idx > 0 else ""),
            )

        data_seed = (
            int(resume_state["data_seed"])
            if resume_state is not None
            else int(
                config.seed
                if config.seed is not None
                else torch.initial_seed() % (2**31)
            )
        )
        if ctx.enabled:
            shared_seed = [data_seed]
            dist.broadcast_object_list(shared_seed, src=0)
            data_seed = shared_seed[0]
        # Build dataset loaders. In DDP, rank 0 downloads first to avoid races.
        if ctx.enabled and not ctx.is_main:
            _barrier(ctx)
        trainloader, train_eval_loader, testloader = get_data_loaders(
            config.batch_size,
            dataset=config.dataset,
            seed=data_seed,
            distributed=ctx.enabled,
            rank=ctx.rank,
            world_size=ctx.world_size,
            download=(not ctx.enabled or ctx.is_main),
            num_workers=config.dataloader_num_workers,
            pin_memory=config.dataloader_pin_memory,
            persistent_workers=config.dataloader_persistent_workers,
            prefetch_factor=config.dataloader_prefetch_factor,
        )
        if ctx.enabled and ctx.is_main:
            _barrier(ctx)
        if config.memory_report_interval:
            _print_memory_snapshot(ctx, device, "after_dataloaders")

        if config.pretrained_weights:
            pretrained = torch.load(
                config.pretrained_weights, map_location="cpu", weights_only=False
            )
            load_model_state(model, pretrained["model_state_dict"])
            del pretrained
            _log(
                ctx,
                f"[INFO] Loaded pretrained model weights: {config.pretrained_weights}",
            )

        # ---------------- Evaluation-only path ----------------
        if config.eval_only:
            if not config.pretrained_weights and not config.resume_checkpoint:
                raise ValueError(
                    "--eval-only set but neither --pretrained-weights nor "
                    "--resume-checkpoint was provided"
                )
            _, test_acc = evaluate_metrics_distributed(
                model,
                testloader,
                device,
                criterion=None,
                ctx=ctx,
                max_batches=max_eval_batches,
            )
            if ctx.is_main:
                print(f"Test accuracy: {test_acc:.2f}%")
            return test_acc

        if ctx.enabled:
            ddp_kwargs = (
                {"device_ids": [ctx.local_rank], "output_device": ctx.local_rank}
                if device.type == "cuda"
                else {}
            )
            model = DistributedDataParallel(model, **ddp_kwargs)
            _log(
                ctx,
                f"[INFO] DDP enabled: world_size={ctx.world_size}, "
                f"local_batch_size={config.batch_size}, "
                f"effective_batch_size={config.batch_size * ctx.world_size}",
            )
        if config.memory_report_interval:
            _print_memory_snapshot(ctx, device, "after_model_setup")

        # Per-epoch metrics (written as independent CSVs per split)
        train_metrics_path = os.path.join(config.output_dir, "train_metrics.csv")
        test_metrics_path = os.path.join(config.output_dir, "test_metrics.csv")

        def _init_csv(path: str, fieldnames: list[str]) -> None:
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()

        def _append_csv(
            path: str, row: dict[str, object], fieldnames: list[str]
        ) -> None:
            with open(path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writerow(row)

        metrics_fields = ["epoch", "loss", "accuracy", "lr"]
        if ctx.is_main:
            resume_metrics = start_epoch > 0
            if not (resume_metrics and os.path.exists(train_metrics_path)):
                _init_csv(train_metrics_path, metrics_fields)
            if not (resume_metrics and os.path.exists(test_metrics_path)):
                _init_csv(test_metrics_path, metrics_fields)

        train_loss: float | None = None
        train_eval_loss: float | None = None
        train_acc: float | None = None
        last_epoch_test_acc = float("nan")
        latest_ckpt_path = os.path.join(config.output_dir, "latest_checkpoint.pth")
        best_ckpt_path = os.path.join(config.output_dir, "fftconv_checkpoint.pth")
        pre_eval_ckpt_path = os.path.join(config.output_dir, "pre_eval_checkpoint.pth")
        time_ckpt_path = os.path.join(config.output_dir, "time_checkpoint.pth")
        checkpoint_time_interval_seconds = (
            float(config.checkpoint_time_interval_minutes) * 60.0
        )
        last_time_checkpoint_at = time.monotonic()

        if resume_state is not None:
            restore_rng(resume_state["rank_states"][ctx.rank]["rng"], device)
        checkpoint_state = {
            "data_seed": data_seed,
            "best_model_state_dict": best_model_state,
            "rank_states": capture_rank_states(device),
        }

        for epoch in range(start_epoch, config.num_epochs):
            sampler = getattr(trainloader, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)

            recal = int(getattr(config, "jtc_gain_recal_epochs", 0) or 0)
            if (
                recal > 0
                and epoch > 0
                and not (epoch == start_epoch and resume_batch_idx > 0)
                and epoch % recal == 0
                and config.jtc_output_gain_mode in {"calibrated", "calibrate_freeze"}
            ):
                from onn_jtc_conv2d import JTCConv2d as _JTCConv2d

                for module in _unwrap_model(model).modules():
                    if isinstance(module, _JTCConv2d):
                        module.reset_gain_calibration()

            model.train()
            running_loss = torch.zeros((), dtype=torch.float64, device=device)
            train_samples = 0
            if epoch == start_epoch and resume_batch_idx > 0:
                progress = resume_state["rank_states"][ctx.rank]
                running_loss.fill_(progress["running_loss"])
                train_samples = progress["train_samples"]
            skip_batches = resume_batch_idx if epoch == start_epoch else 0

            if max_train_batches is not None:
                total_batches = min(int(max_train_batches), len(trainloader))
                stop_batch = int(max_train_batches)
            else:
                total_batches = len(trainloader)
                stop_batch = None
            skip_batches = min(max(int(skip_batches), 0), total_batches)
            sampler.start_index = skip_batches * config.batch_size
            remaining_batches = (
                None if stop_batch is None else max(0, stop_batch - skip_batches)
            )
            train_iter = itertools.islice(trainloader, remaining_batches)

            show_progress = bool(config.show_progress) and ctx.is_main
            progress_iter = (
                tqdm(
                    train_iter,
                    total=total_batches,
                    initial=skip_batches,
                    desc=f"Epoch {epoch}/{config.num_epochs - 1}",
                )
                if show_progress
                else train_iter
            )
            for batch_idx, (inputs, labels) in enumerate(
                progress_iter,
                start=skip_batches + 1,
            ):
                inputs, labels = (
                    inputs.to(device, non_blocking=True),
                    labels.to(device, non_blocking=True),
                )

                with autocast(device_type=device.type, enabled=scaler is not None):
                    _reset_jtc_pd_range_regularizers(model)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    range_loss = _jtc_pd_range_regularization_loss(model)
                    if range_loss is not None:
                        loss = loss + range_loss

                if scaler is not None:
                    scaler.scale(loss).backward()
                    if config.grad_clip_norm > 0:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(
                            model.parameters(), config.grad_clip_norm
                        )
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                else:
                    loss.backward()
                    if config.grad_clip_norm > 0:
                        nn.utils.clip_grad_norm_(
                            model.parameters(), config.grad_clip_norm
                        )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                batch_size = int(labels.size(0))
                running_loss += loss.detach().double() * batch_size
                train_samples += batch_size
                global_step += 1
                if show_progress:
                    progress_iter.set_postfix({"loss": f"{loss.item():.3f}"})
                if (
                    config.memory_report_interval
                    and batch_idx % int(config.memory_report_interval) == 0
                ):
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    _print_memory_snapshot(
                        ctx,
                        device,
                        f"epoch={epoch} batch={batch_idx}/{total_batches}",
                    )
                checkpoint_due = False
                if checkpoint_time_interval_seconds > 0:
                    due = torch.tensor(
                        int(
                            time.monotonic() - last_time_checkpoint_at
                            >= checkpoint_time_interval_seconds
                        ),
                        device=device,
                    )
                    if ctx.enabled:
                        dist.broadcast(due, src=0)
                    checkpoint_due = bool(due)
                if checkpoint_due:
                    checkpoint_state["rank_states"] = capture_rank_states(
                        device, running_loss, train_samples
                    )
                    if ctx.is_main:
                        save_checkpoint(
                            model,
                            config,
                            best_test_acc,
                            time_ckpt_path,
                            epoch=epoch,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            scaler=scaler,
                            best_epoch=best_test_epoch,
                            best_snapshot=best_test_snapshot,
                            checkpoint_kind="time",
                            epoch_complete=False,
                            batch_idx=batch_idx,
                            global_step=global_step,
                            **checkpoint_state,
                        )
                        print(
                            "[INFO] Saved time checkpoint "
                            f"at epoch {epoch}, batch {batch_idx}/{total_batches}: "
                            f"{time_ckpt_path}",
                            flush=True,
                        )
                    last_time_checkpoint_at = time.monotonic()
            resume_batch_idx = 0

            resumed_pre_eval = (
                resume_state is not None
                and epoch == start_epoch
                and resume_state["checkpoint_kind"] == "pre_eval"
            )
            if not resumed_pre_eval:
                scheduler.step()

            local_stats = torch.stack(
                [
                    running_loss,
                    torch.tensor(
                        float(train_samples), dtype=torch.float64, device=device
                    ),
                ]
            )
            if ctx.enabled:
                dist.all_reduce(local_stats, op=dist.ReduceOp.SUM)
            train_loss = float(local_stats[0].item() / max(local_stats[1].item(), 1.0))

            lr = float(optimizer.param_groups[0]["lr"])
            checkpoint_state["rank_states"] = capture_rank_states(
                device, running_loss, train_samples
            )
            if ctx.is_main:
                save_checkpoint(
                    model,
                    config,
                    best_test_acc,
                    pre_eval_ckpt_path,
                    epoch=epoch,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    best_epoch=best_test_epoch,
                    best_snapshot=best_test_snapshot,
                    checkpoint_kind="pre_eval",
                    epoch_complete=False,
                    batch_idx=total_batches,
                    global_step=global_step,
                    **checkpoint_state,
                )
                print(
                    "[INFO] Saved pre-eval checkpoint "
                    f"for epoch {epoch}: {pre_eval_ckpt_path}",
                    flush=True,
                )

            eval_model = _unwrap_model(model)
            eval_autocast = scaler is not None
            run_train_eval = config.train_eval_interval > 0 and (
                (epoch + 1) % int(config.train_eval_interval) == 0
                or epoch == config.num_epochs - 1
            )
            if run_train_eval:
                train_eval_loss, train_acc = evaluate_metrics_distributed(
                    eval_model,
                    train_eval_loader,
                    device,
                    criterion,
                    ctx,
                    max_batches=max_eval_batches,
                    use_autocast=eval_autocast,
                )
            else:
                train_eval_loss, train_acc = None, None
            test_loss, test_acc = evaluate_metrics_distributed(
                eval_model,
                testloader,
                device,
                criterion,
                ctx,
                max_batches=max_eval_batches,
                use_autocast=eval_autocast,
            )
            last_epoch_test_acc = float(test_acc)

            if ctx.is_main:
                _append_csv(
                    train_metrics_path,
                    {
                        "epoch": epoch,
                        "loss": f"{train_loss:.6f}",
                        "accuracy": (
                            f"{float(train_acc):.6f}" if train_acc is not None else ""
                        ),
                        "lr": f"{lr:.8f}",
                    },
                    metrics_fields,
                )
                _append_csv(
                    test_metrics_path,
                    {
                        "epoch": epoch,
                        "loss": (
                            f"{float(test_loss):.6f}" if test_loss is not None else ""
                        ),
                        "accuracy": f"{float(test_acc):.6f}",
                        "lr": f"{lr:.8f}",
                    },
                    metrics_fields,
                )

            if test_acc > best_test_acc:
                best_test_acc = float(test_acc)
                best_test_epoch = int(epoch)
                best_test_snapshot = {
                    "epoch": epoch,
                    "train_acc": float(train_acc) if train_acc is not None else None,
                    "train_eval_loss": (
                        float(train_eval_loss) if train_eval_loss is not None else None
                    ),
                    "train_loss": float(train_loss),
                    "test_acc": float(test_acc),
                    "test_loss": float(test_loss) if test_loss is not None else None,
                }
                if ctx.is_main:
                    best_model_state = _model_snapshot(model)
                    checkpoint_state["best_model_state_dict"] = best_model_state
                    _save_best_model(
                        best_model_state,
                        config,
                        best_test_acc,
                        best_test_epoch,
                        best_test_snapshot,
                        best_ckpt_path,
                    )

            if ctx.is_main:
                save_checkpoint(
                    model,
                    config,
                    best_test_acc,
                    latest_ckpt_path,
                    epoch=epoch,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    best_epoch=best_test_epoch,
                    best_snapshot=best_test_snapshot,
                    checkpoint_kind="latest",
                    epoch_complete=True,
                    batch_idx=total_batches,
                    global_step=global_step,
                    **checkpoint_state,
                )
                if config.checkpoint_interval and (
                    (epoch + 1) % int(config.checkpoint_interval) == 0
                    or epoch == config.num_epochs - 1
                ):
                    epoch_ckpt = os.path.join(
                        config.output_dir, f"checkpoint_epoch_{epoch:04d}.pth"
                    )
                    save_checkpoint(
                        model,
                        config,
                        best_test_acc,
                        epoch_ckpt,
                        epoch=epoch,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        best_epoch=best_test_epoch,
                        best_snapshot=best_test_snapshot,
                        checkpoint_kind="epoch",
                        epoch_complete=True,
                        batch_idx=total_batches,
                        global_step=global_step,
                        **checkpoint_state,
                    )

                print(
                    {
                        "epoch": epoch,
                        "test_acc": f"{test_acc:.2f}",
                        "best_test_acc": f"{best_test_acc:.2f}",
                        "loss": f"{train_loss:.3f}",
                    }
                )

            restore_rng(checkpoint_state["rank_states"][ctx.rank]["rng"], device)
            _peak_memory_summary(ctx, device, f"epoch={epoch}")

        distortion_acc = None
        if config.run_full_strength_inference and DISTORTION_STRENGTH_FIELDS:
            distortion_acc = run_full_strength_inference(
                _unwrap_model(model), config, testloader, device, ctx
            )
            if ctx.is_main:
                print(
                    "[INFO] Accuracy with all distortion strengths set to 1.0: "
                    f"{distortion_acc:.2f}%"
                )

        final_memory_summary = _peak_memory_summary(ctx, device, "final")

        if ctx.is_main:
            if not os.path.exists(latest_ckpt_path):
                save_checkpoint(
                    model,
                    config,
                    best_test_acc,
                    latest_ckpt_path,
                    epoch=max(start_epoch - 1, -1),
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    best_epoch=best_test_epoch,
                    best_snapshot=best_test_snapshot,
                    checkpoint_kind="latest",
                    epoch_complete=True,
                    batch_idx=0,
                    global_step=global_step,
                    **checkpoint_state,
                )
            if best_model_state is not None:
                _save_best_model(
                    best_model_state,
                    config,
                    best_test_acc,
                    best_test_epoch,
                    best_test_snapshot,
                    best_ckpt_path,
                )

            torch.save(
                _unwrap_model(model),
                os.path.join(config.output_dir, "fftconv_full_model.pth"),
            )

            with open(os.path.join(config.output_dir, "final_config.yaml"), "w") as f:
                yaml.dump(vars(config), f)

            from onn_shotreport import observed_shot_plans

            shot_report = observed_shot_plans(_unwrap_model(model))
            with open(os.path.join(config.output_dir, "shot_plans.json"), "w") as f:
                json.dump(shot_report, f, indent=2)

            summary = {
                "shot_plans_file": "shot_plans.json",
                "best_test": best_test_snapshot,
                "final": {
                    "epoch": (config.num_epochs - 1) if config.num_epochs > 0 else -1,
                    "train_acc": float(train_acc) if train_acc is not None else None,
                    "train_eval_loss": (
                        float(train_eval_loss) if train_eval_loss is not None else None
                    ),
                    "train_loss": float(train_loss) if train_loss is not None else None,
                    "test_acc": (
                        float(last_epoch_test_acc) if config.num_epochs > 0 else None
                    ),
                },
                "distributed": {
                    "enabled": ctx.enabled,
                    "world_size": ctx.world_size,
                },
                "memory": final_memory_summary,
            }
            with open(
                os.path.join(config.output_dir, "metrics_summary.json"), "w"
            ) as f:
                json.dump(summary, f, indent=2, sort_keys=True)

            print(
                "Training finished. Last epoch test accuracy: "
                f"{last_epoch_test_acc:.2f}% | Best test accuracy: "
                f"{best_test_acc:.2f}% (epoch {best_test_epoch}). "
                f"Best checkpoint saved to {best_ckpt_path}; latest checkpoint saved "
                f"to {latest_ckpt_path}."
            )

        _barrier(ctx)
        completed_successfully = True
        return last_epoch_test_acc
    finally:
        if not completed_successfully and config.memory_report_interval:
            try:
                _print_memory_snapshot(ctx, device, "final_on_exit")
            except Exception:
                pass
        _cleanup_distributed(ctx)
