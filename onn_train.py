from __future__ import annotations

import copy
import json
import math
import os
import yaml
from dataclasses import fields as dataclass_fields
from typing import Tuple, Optional

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
from onn_component import JTC
from diagnostics.pretrain_tests import run_pretrain_tests

DISTORTION_STRENGTH_FIELDS = [
    f.name
    for f in dataclass_fields(AppConfig)
    if f.name.endswith("_distortion_strength")
]

VGG_STAGE_CHANNELS = [64, 128, 256, 512, 512]
VGG_VARIANT_CONFIGS: dict[str, list[int]] = {
    "vgg3": [1, 1, 1, 0, 0],
    "vgg5": [1, 1, 2, 1, 0],
    "vgg9": [1, 1, 2, 2, 3],
    "vgg11": [1, 1, 2, 2, 2],
    "vgg13": [2, 2, 2, 2, 2],
    "vgg16": [2, 2, 3, 3, 3],
    "vgg19": [2, 2, 4, 4, 4],
}


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


class FTVGG(nn.Module):
    """FTConv2d-based VGG family with selectable depth (11/13/16/19)."""

    def __init__(
        self,
        config: AppConfig,
        variant: str,
        in_channels: int = 3,
        num_classes: int = 10,
    ):
        super().__init__()
        variant = variant.lower()
        if variant not in VGG_VARIANT_CONFIGS:
            raise ValueError(f"Unknown VGG variant '{variant}'")
        self.config = config
        self.variant = variant
        self.num_classes = num_classes
        feature_layers: list[nn.Module] = []
        curr_in = in_channels
        stage_cfg = VGG_VARIANT_CONFIGS[variant]

        for stage_idx, convs in enumerate(stage_cfg):
            out_channels = VGG_STAGE_CHANNELS[stage_idx]
            if convs <= 0:
                continue
            for _ in range(convs):
                feature_layers.append(
                    self._conv_block(curr_in, out_channels, kernel_size=3)
                )
                curr_in = out_channels
            feature_layers.append(nn.MaxPool2d(2))

        feature_layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.features = nn.Sequential(*feature_layers)
        self.final_channels = curr_in
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.final_channels, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def _conv_block(self, in_channels: int, out_channels: int, kernel_size: int) -> nn.Sequential:
        return nn.Sequential(
            FTConv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
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
    if arch.startswith("ftvgg"):
        variant = getattr(config, "vgg_variant", "vgg11").lower()
        arch_variant_map = {
            "ftvgg": None,
            "ftvgg3": "vgg3",
            "ftvgg5": "vgg5",
            "ftvgg9": "vgg9",
            "ftvgg11": "vgg11",
            "ftvgg13": "vgg13",
            "ftvgg16": "vgg16",
            "ftvgg19": "vgg19",
        }
        mapped_variant = arch_variant_map.get(arch)
        if mapped_variant and variant == "vgg11" and mapped_variant != "vgg11":
            variant = mapped_variant
            config.vgg_variant = variant
        elif mapped_variant is None:
            config.vgg_variant = variant
        return FTVGG(config, config.vgg_variant.lower(), in_channels, num_classes)
    raise ValueError(f"Unsupported model_arch '{config.model_arch}'")


# -------------------------------
#  Training / evaluation helpers
# -------------------------------


def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> float:
    if max_batches is not None and max_batches <= 0:
        print("[INFO] Evaluation skipped (max_eval_batches <= 0).")
        return float("nan")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            images, labels = (
                images.to(device, non_blocking=True),
                labels.to(device, non_blocking=True),
            )
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if max_batches is not None and (batch_idx + 1) >= max_batches:
                break

    if total == 0:
        return float("nan")
    acc = 100 * correct / total
    print(f"Accuracy: {acc:.3f}% (batches={max_batches or 'all'})")
    return acc


def run_full_strength_inference(
    model: nn.Module,
    config: AppConfig,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    in_channels: int,
    num_classes: int,
    max_eval_batches: Optional[int] = None,
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
        acc = evaluate(ref_model, dataloader, device, max_batches=max_eval_batches)
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

    # Helper to switch backend of all FTConv2d layers
    def set_model_backend(mdl, backend_name):
        count = 0
        for m in mdl.modules():
            if isinstance(m, FTConv2d):
                m.conv_backend = backend_name
                # Re-init JTC if needed (e.g. switching from pytorch to jtc_fast)
                if backend_name in ["jtc_emulation", "jtc_fast"] and m.jtc is None:
                    m.jtc = JTC(m.config)
                count += 1
        print(f"[INFO] Switched {count} FTConv2d layers to backend: {backend_name}")

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
            
    # ---------------- Pre-Training (PyTorch) ----------------
    if not config.eval_only and config.pretrain_epochs > 0:
        print(f"\n[PRETRAIN] Starting {config.pretrain_epochs} epochs of PyTorch pretraining...")
        original_backend = config.conv_backend
        set_model_backend(model, "pytorch")
        
        # Create a separate optimizer for pretraining (fresh start)
        pre_optim = optim.AdamW(model.parameters(), lr=config.learning_rate)
        pre_crit = nn.CrossEntropyLoss()
        pre_scaler = GradScaler() if device.type == "cuda" else None
        
        for epoch in range(config.pretrain_epochs):
            model.train()
            running_loss = 0.0
            pbar = tqdm(trainloader, desc=f"[Pretrain] Epoch {epoch+1}/{config.pretrain_epochs}")
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                
                with autocast(device_type=device.type, enabled=pre_scaler is not None):
                    outputs = model(inputs)
                    loss = pre_crit(outputs, labels)
                
                if pre_scaler is not None:
                    pre_scaler.scale(loss).backward()
                    pre_scaler.step(pre_optim)
                    pre_scaler.update()
                    pre_optim.zero_grad()
                else:
                    loss.backward()
                    pre_optim.step()
                    pre_optim.zero_grad()
                
                running_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.3f}"})
            
            # Optional: Quick eval
            if not config.skip_eval:
                acc = evaluate(model, testloader, device, max_batches=config.max_eval_batches)
                print(f"[Pretrain] Epoch {epoch+1} Test Acc: {acc:.2f}%")
        
        print(f"[PRETRAIN] Finished. Switching back to {original_backend}...\n")
        set_model_backend(model, original_backend)
        # Restore config (though create_model used it initially, layers hold their own ref)
        # FTConv2d layers now have original_backend set via helper.

    # ---------------- Evaluation-only path ----------------
    if config.eval_only:
        if not config.pretrained_weights:
            raise ValueError("--eval-only set but --pretrained-weights not provided")
        ckpt = torch.load(config.pretrained_weights, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        test_acc = evaluate(model, testloader, device)
        print(f"Test accuracy: {test_acc:.2f}%")
        return test_acc

    summary_requested = device.type == "cuda" and config.dump_memory_summary
    summary_path = os.path.join(config.output_dir, "memory_summary.txt")

    try:
        # ---------------- Training path -----------------------
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)

        best_acc = 0.0
        scaler = GradScaler() if device.type == "cuda" else None

        last_epoch_test_acc = float("nan") if config.skip_eval else 0.0
        steps_done = 0
        stop_training = False
        for epoch in range(config.num_epochs):
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

            model.train()
            running_loss = 0.0
            steps_this_epoch = 0
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
                steps_this_epoch += 1
                steps_done += 1
                pbar.set_postfix({"loss": f"{loss.item():.3f}"})

                if config.max_train_steps and steps_done >= config.max_train_steps:
                    stop_training = True
                    pbar.set_postfix({"loss": f"{loss.item():.3f}", "note": "step_cap"})
                    break

            scheduler.step()

            avg_loss = running_loss / max(1, steps_this_epoch)
            if device.type == "cuda":
                peak_alloc = torch.cuda.max_memory_allocated(device=device) / (1024**2)
                peak_reserved = torch.cuda.max_memory_reserved(device=device) / (1024**2)
                print(
                    f"[MEM] Epoch {epoch}: peak allocated {peak_alloc:.1f} MB | reserved {peak_reserved:.1f} MB"
                )

            if config.skip_eval:
                test_acc = float("nan")
                train_acc = float("nan")
            else:
                test_acc = evaluate(
                    model,
                    testloader,
                    device,
                    max_batches=config.max_eval_batches,
                )
                last_epoch_test_acc = test_acc
                train_acc = evaluate(
                    model,
                    trainloader,
                    device,
                    max_batches=config.max_eval_batches,
                )
                if not math.isnan(test_acc):
                    best_acc = max(best_acc, test_acc)

            print(
                {
                    "epoch": epoch,
                    "train_acc": f"{train_acc:.2f}",
                    "test_acc": f"{test_acc:.2f}",
                    "best_acc": f"{best_acc:.2f}",
                    "loss": f"{avg_loss:.3f}",
                    "steps": steps_done,
                }
            )

            if stop_training:
                print(
                    f"[INFO] Reached max_train_steps={config.max_train_steps}; stopping training loop."
                )
                break

        distortion_acc = None
        if DISTORTION_STRENGTH_FIELDS and not config.skip_eval:
            distortion_acc = run_full_strength_inference(
                model,
                config,
                testloader,
                device,
                in_channels,
                num_classes,
                max_eval_batches=config.max_eval_batches,
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
    finally:
        if summary_requested:
            os.makedirs(config.output_dir, exist_ok=True)
            try:
                torch.cuda.synchronize()
            except RuntimeError:
                pass
            with open(summary_path, "w", encoding="utf-8") as handle:
                handle.write(
                    torch.cuda.memory_summary(device=device, abbreviated=False)
                )
            print(f"[INFO] Wrote CUDA memory summary to {summary_path}")
            snapshot_path = os.path.join(config.output_dir, "memory_snapshot.json")
            try:
                snapshot = torch.cuda.memory_snapshot()
                with open(snapshot_path, "w", encoding="utf-8") as handle:
                    json.dump(snapshot, handle)
                print(f"[INFO] Wrote CUDA memory snapshot to {snapshot_path}")
            except RuntimeError as exc:
                print(f"[WARN] Unable to capture CUDA memory snapshot: {exc}")
