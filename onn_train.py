from __future__ import annotations

import copy
import os
import yaml
from dataclasses import fields as dataclass_fields
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from onn_config import AppConfig
from diagnostics.pretrain_tests import run_pretrain_tests
from onn_models import FFTConvNet, build_model

DISTORTION_STRENGTH_FIELDS = [
    f.name
    for f in dataclass_fields(AppConfig)
    if f.name.endswith("_distortion_strength")
]


# -------------------------------
#  Utility helpers
# -------------------------------


def get_data_loaders(
    config: AppConfig,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create dataset-specific train/test loaders."""
    dataset = config.dataset.lower()
    batch_size = config.batch_size

    if dataset == "mnist":
        train_transform = transforms.Compose(
            [
                transforms.Pad(2),
                transforms.RandomRotation(10, fill=0),
                transforms.ToTensor(),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.Pad(2),
                transforms.ToTensor(),
            ]
        )
        trainset = torchvision.datasets.MNIST(
            root="./data", train=True, download=True, transform=train_transform
        )
        testset = torchvision.datasets.MNIST(
            root="./data", train=False, download=True, transform=test_transform
        )
        if config.auto_infer_input_channels:
            config.input_channels = 1
    else:
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
        trainset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=train_transform
        )
        testset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=test_transform
        )
        if config.auto_infer_input_channels:
            config.input_channels = 3

    common_loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    trainloader = torch.utils.data.DataLoader(
        trainset,
        shuffle=True,
        **common_loader_kwargs,
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        shuffle=False,
        **common_loader_kwargs,
    )
    return trainloader, testloader


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
) -> float:
    if not DISTORTION_STRENGTH_FIELDS:
        return float("nan")

    distortion_config = copy.deepcopy(config)
    for field in DISTORTION_STRENGTH_FIELDS:
        setattr(distortion_config, field, 1.0)

    ref_model = build_model(distortion_config).to(device)
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

    # -------------------------------------------------------------
    #  Pre-training diagnostics (plots & quick sanity checks)
    # -------------------------------------------------------------
    if config.run_pretrain_tests or config.pretrain_tests_only:
        run_pretrain_tests(config)
        if config.pretrain_tests_only:
            print("[INFO] Pretrain tests only requested; exiting without training.")
            return float("nan")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build dataset loaders
    trainloader, testloader = get_data_loaders(config)

    # Build model
    model = build_model(config).to(device)

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

    distortion_acc = None
    if DISTORTION_STRENGTH_FIELDS:
        distortion_acc = run_full_strength_inference(model, config, testloader, device)
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
