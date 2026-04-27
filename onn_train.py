from __future__ import annotations

import copy
import csv
import itertools
import json
import os
import yaml
from dataclasses import fields as dataclass_fields

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from diagnostics.pretrain_tests import run_pretrain_tests
from onn_config import AppConfig
from onn_layers import FTconvlayer

DISTORTION_STRENGTH_FIELDS = [
    f.name
    for f in dataclass_fields(AppConfig)
    if f.name.endswith("_distortion_strength")
]


# -------------------------------
#  Utility helpers
# -------------------------------


def get_data_loaders(
    batch_size: int,
) -> tuple[
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
]:
    """Create CIFAR-10 train, train-eval, and test dataloaders."""

    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
            transforms.ToTensor(),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    full_trainset_aug = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=train_transform
    )
    full_trainset_plain = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=test_transform
    )
    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=test_transform
    )

    trainloader = torch.utils.data.DataLoader(
        full_trainset_aug,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    train_eval_loader = torch.utils.data.DataLoader(
        full_trainset_plain,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
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
        self.max_norm_mode = str(getattr(config, "max_norm_mode", "max") or "max")

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
                nn.ReLU(inplace=True),
            ]
            if getattr(config, "normalize_blocks", False):
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
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = F.relu(x)
        x = x / _norm_divisor(x, self.max_norm_mode).clamp_min(1e-12)

        x = self.conv2(x)
        x = self.maxpool2(x)
        x = F.relu(x)
        x = x / _norm_divisor(x, self.max_norm_mode).clamp_min(1e-12)

        x = self.blocks(x)
        x = self.classifier(x)
        return x


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
) -> tuple[float | None, float]:
    """Compute (avg_loss, accuracy). If criterion is None, avg_loss is None."""
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            images, labels = (
                images.to(device, non_blocking=True),
                labels.to(device, non_blocking=True),
            )
            outputs = model(images)
            if criterion is not None:
                loss = criterion(outputs, labels)
                total_loss += float(loss.item()) * labels.size(0)
            predicted = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100.0 * correct / max(total, 1)
    if criterion is None:
        return None, acc
    return total_loss / max(total, 1), acc


def run_full_strength_inference(
    model: nn.Module,
    config: AppConfig,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    if not DISTORTION_STRENGTH_FIELDS:
        return float("nan")

    distortion_config = copy.deepcopy(config)
    for field in DISTORTION_STRENGTH_FIELDS:
        setattr(distortion_config, field, 1.0)

    ref_model = FFTConvNet(distortion_config).to(device)
    ref_model.load_state_dict(model.state_dict())

    try:
        print(
            "[INFO] Running inference with all distortion strength parameters set to 1.0"
        )
        acc = evaluate(
            ref_model, dataloader, device, max_batches=getattr(config, "max_eval_batches", None)
        )
    finally:
        if device.type == "cuda":
            ref_model.to("cpu")
            torch.cuda.empty_cache()
        del ref_model
    return acc


def save_checkpoint(
    model: nn.Module, config: AppConfig, best_acc: float, filename: str
) -> None:
    """Save model state dict together with the config and accuracy."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "best_accuracy": best_acc,
            "config": vars(config),
        },
        filename,
    )


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

    # -------------------------------------------------------------
    #  Pre-training diagnostics (plots & quick sanity checks)
    # -------------------------------------------------------------
    if config.run_pretrain_tests or config.pretrain_tests_only:
        run_pretrain_tests(config)
        if config.pretrain_tests_only:
            print("[INFO] Pretrain tests only requested; exiting without training.")
            return float("nan")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(config.output_dir, exist_ok=True)

    max_train_batches = getattr(config, "max_train_batches", None)
    max_eval_batches = getattr(config, "max_eval_batches", None)

    # Build dataset loaders
    trainloader, train_eval_loader, testloader = get_data_loaders(config.batch_size)

    # Build model
    model = FFTConvNet(config).to(device)

    # Optionally load pretrained weights for fine-tuning
    if not config.eval_only and config.pretrained_weights:
        try:
            ckpt = torch.load(
                config.pretrained_weights, map_location=device, weights_only=False
            )
            state_dict = ckpt.get("model_state_dict", ckpt)
            model.load_state_dict(state_dict)
            print(
                f"[INFO] Loaded pretrained weights for fine-tuning: {config.pretrained_weights}"
            )
        except Exception as e:
            print(
                f"[WARN] Failed to load pretrained weights '{config.pretrained_weights}': {e}. Proceeding without."
            )

    # ---------------- Evaluation-only path ----------------
    if config.eval_only:
        if not config.pretrained_weights:
            raise ValueError("--eval-only set but --pretrained-weights not provided")
        ckpt = torch.load(config.pretrained_weights, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        test_acc = evaluate(model, testloader, device, max_batches=max_eval_batches)
        print(f"Test accuracy: {test_acc:.2f}%")
        return test_acc

    # ---------------- Training path -----------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(int(config.num_epochs), 1)
    )

    # Per-epoch metrics (written as independent CSVs per split)
    train_metrics_path = os.path.join(config.output_dir, "train_metrics.csv")
    test_metrics_path = os.path.join(config.output_dir, "test_metrics.csv")

    def _init_csv(path: str, fieldnames: list[str]) -> None:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()

    def _append_csv(path: str, row: dict[str, object], fieldnames: list[str]) -> None:
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writerow(row)

    metrics_fields = ["epoch", "loss", "accuracy", "lr"]
    _init_csv(train_metrics_path, metrics_fields)
    _init_csv(test_metrics_path, metrics_fields)

    best_test_acc = -1.0
    best_test_epoch = -1
    best_test_snapshot: dict[str, object] = {}
    scaler = GradScaler("cuda") if device.type == "cuda" else None

    train_loss: float | None = None
    train_eval_loss: float | None = None
    train_acc: float | None = None

    last_epoch_test_acc = float("nan")
    for epoch in range(config.num_epochs):
        model.train()
        running_loss = 0.0
        num_train_batches = 0

        if max_train_batches is not None:
            total_batches = min(int(max_train_batches), len(trainloader))
            train_iter = itertools.islice(trainloader, int(max_train_batches))
        else:
            total_batches = len(trainloader)
            train_iter = trainloader

        pbar = tqdm(
            train_iter,
            total=total_batches,
            desc=f"Epoch {epoch}/{config.num_epochs - 1}",
        )
        for inputs, labels in pbar:
            inputs, labels = (
                inputs.to(device, non_blocking=True),
                labels.to(device, non_blocking=True),
            )

            with autocast(device_type=device.type, enabled=scaler is not None):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            else:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item()
            num_train_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.3f}"})

        scheduler.step()

        # Evaluate
        lr = float(optimizer.param_groups[0]["lr"])
        train_loss = float(running_loss / max(num_train_batches, 1))

        train_eval_loss, train_acc = evaluate_metrics(
            model,
            train_eval_loader,
            device,
            criterion,
            max_batches=max_eval_batches,
        )
        test_loss, test_acc = evaluate_metrics(
            model, testloader, device, criterion, max_batches=max_eval_batches
        )
        last_epoch_test_acc = float(test_acc)

        _append_csv(
            train_metrics_path,
            {
                "epoch": epoch,
                "loss": f"{train_loss:.6f}",
                "accuracy": f"{train_acc:.6f}",
                "lr": f"{lr:.8f}",
            },
            metrics_fields,
        )
        _append_csv(
            test_metrics_path,
            {
                "epoch": epoch,
                "loss": f"{float(test_loss):.6f}" if test_loss is not None else "",
                "accuracy": f"{test_acc:.6f}",
                "lr": f"{lr:.8f}",
            },
            metrics_fields,
        )

        if test_acc > best_test_acc:
            best_test_acc = float(test_acc)
            best_test_epoch = int(epoch)
            best_test_snapshot = {
                "epoch": epoch,
                "train_acc": float(train_acc),
                "train_eval_loss": float(train_eval_loss)
                if train_eval_loss is not None
                else None,
                "train_loss": float(train_loss),
                "test_acc": float(test_acc),
                "test_loss": float(test_loss) if test_loss is not None else None,
            }

        print(
            {
                "epoch": epoch,
                "test_acc": f"{test_acc:.2f}",
                "best_test_acc": f"{best_test_acc:.2f}",
                "loss": f"{train_loss:.3f}",
            }
        )

    distortion_acc = None
    if DISTORTION_STRENGTH_FIELDS:
        distortion_acc = run_full_strength_inference(model, config, testloader, device)
        print(
            f"[INFO] Accuracy with all distortion strengths set to 1.0: {distortion_acc:.2f}%"
        )

    # ---------------- Export -------------------------------
    ckpt_path = os.path.join(config.output_dir, "fftconv_checkpoint.pth")
    save_checkpoint(model, config, best_test_acc, ckpt_path)
    # Save full model for structure reference (note: bigger file)
    torch.save(model, os.path.join(config.output_dir, "fftconv_full_model.pth"))

    # Also dump the final config for completeness
    with open(os.path.join(config.output_dir, "final_config.yaml"), "w") as f:
        yaml.dump(vars(config), f)

    summary = {
        "best_test": best_test_snapshot,
        "final": {
            "epoch": (config.num_epochs - 1) if config.num_epochs > 0 else -1,
            "train_acc": float(train_acc) if train_acc is not None else None,
            "train_eval_loss": float(train_eval_loss)
            if train_eval_loss is not None
            else None,
            "train_loss": float(train_loss) if train_loss is not None else None,
            "test_acc": float(last_epoch_test_acc) if config.num_epochs > 0 else None,
        },
    }
    with open(os.path.join(config.output_dir, "metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(
        f"Training finished. Last epoch test accuracy: {last_epoch_test_acc:.2f}% | Best test accuracy: {best_test_acc:.2f}% (epoch {best_test_epoch}). Checkpoint saved to {ckpt_path}."
    )
    return last_epoch_test_acc
