from __future__ import annotations

import os
import yaml
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

from onn_layers import FTconvlayer
from onn_config import AppConfig
from onn_tests import run_pretrain_tests


# -------------------------------
#  Utility helpers
# -------------------------------


def get_data_loaders(
    batch_size: int,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create CIFAR-10 train / test dataloaders with the same augmentation
    pipeline used in the original template."""
    # stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
            transforms.ToTensor(),
            # transforms.Normalize(*stats, inplace=True),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.Normalize(*stats),
        ]
    )

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=train_transform
    )
    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=test_transform
    )

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True
    )
    return trainloader, testloader


# -------------------------------
#  Model definition
# -------------------------------


class FFTConvNet(nn.Module):
    """Configurable variant of the 7-layer FFTConv network from `old_template.py`.

    The number of identical intermediate blocks (originally 5) can be varied
    through `config.num_identical_layers`."""

    def __init__(self, config: AppConfig):
        super().__init__()

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
        blocks = []
        for _ in range(config.num_identical_layers):
            blocks.append(
                nn.Sequential(
                    FTconvlayer(
                        32,
                        16,
                        config=config,
                        kernel_size=8,
                        hv_concat=True,
                    ),
                    nn.ReLU(inplace=True),
                )
            )
        self.blocks = nn.Sequential(*blocks)

        # Classifier (identical to original template)
        self.classifier = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.Linear(256, 10),
        )

    # pylint: disable=arguments-differ
    def forward(self, x):  # type: ignore[override]
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = F.relu(x)
        x = x / x.max()

        x = self.conv2(x)
        x = self.maxpool2(x)
        x = F.relu(x)
        x = x / x.max()

        #x = self.blocks(x)
        x = self.classifier(x)
        return x


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
    return 100 * correct / total


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


def train_onn_model(config: AppConfig):
    """Entry-point used by `onn_main.py`.

    * If `config.eval_only` is True, we load `config.pretrained_weights` and run
      a single evaluation pass.
    * Otherwise we train from scratch and export the best checkpoint as well as
      the model structure under `config.output_dir`."""

    # -------------------------------------------------------------
    #  Pre-training diagnostics (plots & quick sanity checks)
    # -------------------------------------------------------------
    if config.run_pretrain_tests:
        run_pretrain_tests(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build dataset loaders
    trainloader, testloader = get_data_loaders(config.batch_size)

    # Build model
    model = FFTConvNet(config).to(device)

    # ---------------- Evaluation-only path ----------------
    if config.eval_only:
        if not config.pretrained_weights:
            raise ValueError("--eval-only set but --pretrained-weights not provided")
        ckpt = torch.load(config.pretrained_weights, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        test_acc = evaluate(model, testloader, device)
        print(f"Test accuracy: {test_acc:.2f}%")
        return

    # ---------------- Training path -----------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)

    best_acc = 0.0
    for epoch in range(config.num_epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(trainloader, desc=f"Epoch {epoch}/{config.num_epochs - 1}")
        for inputs, labels in pbar:
            inputs, labels = (
                inputs.to(device, non_blocking=True),
                labels.to(device, non_blocking=True),
            )

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.3f}"})

        scheduler.step()

        # Evaluate
        test_acc = evaluate(model, testloader, device)
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

    # ---------------- Export -------------------------------
    ckpt_path = os.path.join(config.output_dir, "fftconv_checkpoint.pth")
    save_checkpoint(model, config, best_acc, ckpt_path)
    # Save full model for structure reference (note: bigger file)
    torch.save(model, os.path.join(config.output_dir, "fftconv_full_model.pth"))

    # Also dump the final config for completeness
    with open(os.path.join(config.output_dir, "final_config.yaml"), "w") as f:
        yaml.dump(vars(config), f)

    print(
        f"Training finished. Best test accuracy: {best_acc:.2f}%. Checkpoint saved to {ckpt_path}."
    )
