from __future__ import annotations

import copy
import os
import yaml
from dataclasses import fields as dataclass_fields
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from onn_layers import FTconvlayer, FTConv2d
from onn_config import AppConfig
from diagnostics.pretrain_tests import run_pretrain_tests

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
    dataset: str = "cifar10",
    return_meta: bool = False,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader] | Tuple[
    torch.utils.data.DataLoader, torch.utils.data.DataLoader, int, int
]:
    """Create dataset-specific train/test loaders.

    Args:
        batch_size: Mini-batch size.
        dataset: 'cifar10' or 'mnist'.
        return_meta: If True, also return (in_channels, num_classes).
    """

    dataset = dataset.lower()
    if dataset == "cifar10":
        train_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
                transforms.ToTensor(),
            ]
        )
        test_transform = transforms.Compose([transforms.ToTensor()])
        trainset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=train_transform
        )
        testset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=test_transform
        )
        in_channels = 3
        num_classes = 10
    elif dataset == "mnist":
        train_transform = transforms.Compose(
            [
                transforms.RandomRotation(10),
                transforms.Resize(32),
                transforms.ToTensor(),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.Resize(32),
                transforms.ToTensor(),
            ]
        )
        trainset = torchvision.datasets.MNIST(
            root="./data", train=True, download=True, transform=train_transform
        )
        testset = torchvision.datasets.MNIST(
            root="./data", train=False, download=True, transform=test_transform
        )
        in_channels = 1
        num_classes = 10
    else:
        raise ValueError(f"Unsupported dataset '{dataset}'.")

    common_loader_kwargs = dict(
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, **common_loader_kwargs
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, **common_loader_kwargs
    )

    if return_meta:
        return trainloader, testloader, in_channels, num_classes
    return trainloader, testloader


# -------------------------------
#  Model definition
# -------------------------------


class FFTConvNet(nn.Module):
    """Configurable variant of the 7-layer FFTConv network from `old_template.py`.

    The number of identical intermediate blocks (originally 5) can be varied
    through `config.num_identical_layers`."""

    def __init__(self, config: AppConfig, in_channels: int = 3, num_classes: int = 10):
        super().__init__()
        self.num_classes = num_classes

        # Stem
        self.conv1 = FTconvlayer(
            in_channels,
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
            def forward(self, x: torch.Tensor):
                return x / x.max().clamp_min(1e-12)

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
                seq.append(_MaxNorm())
            blocks.append(nn.Sequential(*seq))
        self.blocks = nn.Sequential(*blocks)

        # Classifier (identical to original template)
        self.classifier = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.Linear(256, num_classes),
        )

    # pylint: disable=arguments-differ
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = F.relu(x)
        x = x / x.max().clamp_min(1e-12)

        x = self.conv2(x)
        x = self.maxpool2(x)
        x = F.relu(x)
        x = x / x.max().clamp_min(1e-12)

        x = self.blocks(x)
        x = self.classifier(x)
        return x


class FTVGG11(nn.Module):
    """VGG11-style network built from FTConv2d blocks."""

    def __init__(self, config: AppConfig, in_channels: int = 3, num_classes: int = 10):
        super().__init__()
        self.config = config
        self.num_classes = num_classes

        self.features = nn.Sequential(
            self._conv_block(in_channels, 64),
            nn.MaxPool2d(2),
            self._conv_block(64, 128),
            nn.MaxPool2d(2),
            self._conv_block(128, 256),
            self._conv_block(256, 256),
            nn.MaxPool2d(2),
            self._conv_block(256, 512),
            self._conv_block(512, 512),
            nn.MaxPool2d(2),
            self._conv_block(512, 512),
            self._conv_block(512, 512),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def _conv_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            FTConv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                config=self.config,
                conv_backend=self.config.conv_backend,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


class FTVGG3(nn.Module):
    """Lightweight 3-block VGG-style network for quick experiments."""

    def __init__(self, config: AppConfig, in_channels: int = 3, num_classes: int = 10):
        super().__init__()
        self.config = config
        self.features = nn.Sequential(
            self._conv_block(in_channels, 32),
            nn.MaxPool2d(2),
            self._conv_block(32, 64),
            nn.MaxPool2d(2),
            self._conv_block(64, 128),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, num_classes),
        )

    def _conv_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            FTConv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                config=self.config,
                conv_backend=self.config.conv_backend,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def create_model(
    config: AppConfig, in_channels: int, num_classes: int
) -> nn.Module:
    """Factory for building the selected architecture."""
    arch = (config.model_arch or "fftconvnet").lower()
    if arch == "fftconvnet":
        return FFTConvNet(config, in_channels, num_classes)
    if arch == "ftvgg11":
        return FTVGG11(config, in_channels, num_classes)
    if arch == "ftvgg3":
        return FTVGG3(config, in_channels, num_classes)
    raise ValueError(f"Unsupported model_arch '{config.model_arch}'")


# -------------------------------
#  Training / evaluation helpers
# -------------------------------


def evaluate(
    model: nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device
) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = (
                images.to(device, non_blocking=True),
                labels.to(device, non_blocking=True),
            )
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    print(f"Accuracy: {acc:.3f}%")
    return acc


def run_full_strength_inference(
    model: nn.Module,
    config: AppConfig,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    in_channels: int,
    num_classes: int,
) -> float:
    if not DISTORTION_STRENGTH_FIELDS:
        return float("nan")

    distortion_config = copy.deepcopy(config)
    for field in DISTORTION_STRENGTH_FIELDS:
        setattr(distortion_config, field, 1.0)

    ref_model = create_model(distortion_config, in_channels, num_classes).to(device)
    ref_model.load_state_dict(model.state_dict())

    try:
        print(
            "[INFO] Running inference with all distortion strength parameters set to 1.0"
        )
        acc = evaluate(ref_model, dataloader, device)
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

    # Ensure output directory exists for checkpoints / artifacts
    os.makedirs(config.output_dir, exist_ok=True)

    # -------------------------------------------------------------
    #  Pre-training diagnostics (plots & quick sanity checks)
    # -------------------------------------------------------------
    if config.run_pretrain_tests or config.pretrain_tests_only:
        run_pretrain_tests(config)
        if config.pretrain_tests_only:
            print("[INFO] Pretrain tests only requested; exiting without training.")
            return float("nan")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build dataset loaders and retrieve dataset metadata
    (
        trainloader,
        testloader,
        in_channels,
        num_classes,
    ) = get_data_loaders(
        config.batch_size, dataset=config.dataset, return_meta=True
    )

    # Build model
    model = create_model(config, in_channels, num_classes).to(device)

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
        test_acc = evaluate(model, testloader, device)
        print(f"Test accuracy: {test_acc:.2f}%")
        return test_acc

    # ---------------- Training path -----------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)

    best_acc = 0.0
    scaler = GradScaler() if device.type == "cuda" else None
    checkpoint_dir = os.path.join(config.output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    last_epoch_test_acc = 0.0
    for epoch in range(config.num_epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(trainloader, desc=f"Epoch {epoch}/{config.num_epochs - 1}")
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
            pbar.set_postfix({"loss": f"{loss.item():.3f}"})

        scheduler.step()

        # Evaluate
        test_acc = evaluate(model, testloader, device)
        last_epoch_test_acc = test_acc
        train_acc = evaluate(model, trainloader, device)
        best_acc = max(best_acc, test_acc)
        print(
            {
                "epoch": epoch,
                "train_acc": f"{train_acc:.2f}",
                "test_acc": f"{test_acc:.2f}",
                "best_acc": f"{best_acc:.2f}",
                "loss": f"{running_loss / len(trainloader):.3f}",
            }
        )
        # Save per-epoch checkpoint for recovery and analysis
        epoch_ckpt = os.path.join(checkpoint_dir, f"epoch_{epoch:03d}.pth")
        save_checkpoint(model, config, best_acc, epoch_ckpt)

    distortion_acc = None
    if DISTORTION_STRENGTH_FIELDS:
        distortion_acc = run_full_strength_inference(
            model, config, testloader, device, in_channels, num_classes
        )
        print(
            f"[INFO] Accuracy with all distortion strengths set to 1.0: {distortion_acc:.2f}%"
        )

    # ---------------- Export -------------------------------
    ckpt_path = os.path.join(config.output_dir, "fftconv_checkpoint.pth")
    save_checkpoint(model, config, best_acc, ckpt_path)
    # Save full model for structure reference (note: bigger file)
    torch.save(model, os.path.join(config.output_dir, "fftconv_full_model.pth"))

    # Also dump the final config for completeness
    with open(os.path.join(config.output_dir, "final_config.yaml"), "w") as f:
        yaml.dump(vars(config), f)

    print(
        f"Training finished. Last epoch test accuracy: {last_epoch_test_acc:.2f}% | Best test accuracy: {best_acc:.2f}%. Checkpoint saved to {ckpt_path}."
    )
    return last_epoch_test_acc
