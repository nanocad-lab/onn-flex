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
import torch.distributed as dist
import torchvision
import torchvision.transforms as transforms
from torch.amp import autocast, GradScaler
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig,
)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from onn_layers import FTconvlayer, FTConv2d
from onn_config import AppConfig
from diagnostics.pretrain_tests import run_pretrain_tests

DISTORTION_STRENGTH_FIELDS = [
    f.name
    for f in dataclass_fields(AppConfig)
    if f.name.endswith("_distortion_strength")
]


def _is_distributed_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def _is_main_process() -> bool:
    return not _is_distributed_and_initialized() or dist.get_rank() == 0


def _broadcast_object(obj, src: int = 0):
    """Broadcast a Python object from src to all ranks."""
    if not _is_distributed_and_initialized():
        return obj
    obj_list = [obj]
    dist.broadcast_object_list(obj_list, src=src)
    return obj_list[0]


def _init_distributed_if_needed(config: AppConfig) -> Tuple[torch.device, int, int]:
    """Initialize torch.distributed when FSDP is enabled."""
    if not config.enable_fsdp:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return device, 0, 1

    if not torch.cuda.is_available():
        raise RuntimeError("FSDP requires CUDA devices; disable --enable-fsdp or use GPUs.")

    if not dist.is_initialized():
        try:
            dist.init_process_group(backend="nccl")
        except Exception as e:  # pragma: no cover - init errors are environment-specific
            raise RuntimeError(
                "Failed to initialize torch.distributed. "
                "Launch with torchrun --nproc_per_node 4 ... or disable --enable-fsdp."
            ) from e

    local_rank = int(os.environ.get("LOCAL_RANK", dist.get_rank()))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    return device, dist.get_rank(), dist.get_world_size()


# -------------------------------
#  Utility helpers
# -------------------------------


def get_data_loaders(
    batch_size: int,
    dataset: str = "cifar10",
    return_meta: bool = False,
    distributed: bool = False,
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
    def _build_dataset(download_flag: bool):
        if dataset == "cifar10":
            train_transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
                    transforms.ToTensor(),
                ]
            )
            test_transform = transforms.Compose([transforms.ToTensor()])
            train_ds = torchvision.datasets.CIFAR10(
                root="./data",
                train=True,
                download=download_flag,
                transform=train_transform,
            )
            test_ds = torchvision.datasets.CIFAR10(
                root="./data",
                train=False,
                download=download_flag,
                transform=test_transform,
            )
            return train_ds, test_ds, 3, 10
        if dataset == "mnist":
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
            train_ds = torchvision.datasets.MNIST(
                root="./data",
                train=True,
                download=download_flag,
                transform=train_transform,
            )
            test_ds = torchvision.datasets.MNIST(
                root="./data",
                train=False,
                download=download_flag,
                transform=test_transform,
            )
            return train_ds, test_ds, 1, 10
        raise ValueError(f"Unsupported dataset '{dataset}'.")

    if distributed and _is_distributed_and_initialized():
        if _is_main_process():
            trainset, testset, in_channels, num_classes = _build_dataset(True)
            dist.barrier()
        else:
            dist.barrier()
            trainset, testset, in_channels, num_classes = _build_dataset(False)
    else:
        trainset, testset, in_channels, num_classes = _build_dataset(True)

    common_loader_kwargs = dict(
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    train_sampler = DistributedSampler(trainset) if distributed else None
    test_sampler = DistributedSampler(testset, shuffle=False) if distributed else None
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        **common_loader_kwargs,
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False if test_sampler is None else False,
        sampler=test_sampler,
        **common_loader_kwargs,
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
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    distributed: bool = False,
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
    correct_t = torch.tensor(correct, device=device)
    total_t = torch.tensor(total, device=device)
    if distributed and _is_distributed_and_initialized():
        dist.all_reduce(correct_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_t, op=dist.ReduceOp.SUM)
    total_val = max(total_t.item(), 1e-12)
    acc = 100 * correct_t.item() / total_val
    if _is_main_process():
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
    if _is_distributed_and_initialized() and not _is_main_process():
        # Skip auxiliary inference on non-main ranks
        return float("nan")

    if not DISTORTION_STRENGTH_FIELDS:
        return float("nan")

    distortion_config = copy.deepcopy(config)
    for field in DISTORTION_STRENGTH_FIELDS:
        setattr(distortion_config, field, 1.0)

    ref_model = create_model(distortion_config, in_channels, num_classes).to(device)
    if isinstance(model, FSDP):
        full_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_cfg):
            state_dict = model.state_dict()
        ref_model.load_state_dict(state_dict)
    else:
        ref_model.load_state_dict(model.state_dict())

    try:
        print(
            "[INFO] Running inference with all distortion strength parameters set to 1.0"
        )
        eval_loader = dataloader
        if isinstance(dataloader.sampler, DistributedSampler):
            loader_kwargs = dict(
                batch_size=dataloader.batch_size,
                shuffle=False,
                num_workers=dataloader.num_workers,
                pin_memory=dataloader.pin_memory,
            )
            if dataloader.num_workers > 0:
                loader_kwargs["prefetch_factor"] = dataloader.prefetch_factor
                loader_kwargs["persistent_workers"] = dataloader.persistent_workers
            eval_loader = torch.utils.data.DataLoader(
                dataloader.dataset,
                **loader_kwargs,
            )
        acc = evaluate(ref_model, eval_loader, device, distributed=False)
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
    if not _is_main_process():
        return

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    if isinstance(model, FSDP):
        full_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_cfg):
            state_dict = model.state_dict()
    else:
        state_dict = model.state_dict()

    torch.save(
        {
            "model_state_dict": state_dict,
            "best_accuracy": best_acc,
            "config": vars(config),
        },
        filename,
    )


def load_weights_into_model(
    model: nn.Module, weights_path: str, device: torch.device
):
    """Load weights into (possibly FSDP) model and broadcast when distributed."""
    distributed = _is_distributed_and_initialized()

    checkpoint = None
    if distributed:
        if _is_main_process():
            checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
        checkpoint = _broadcast_object(checkpoint, src=0)
    else:
        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)

    state_dict = checkpoint.get("model_state_dict", checkpoint)

    if isinstance(model, FSDP):
        full_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_cfg):
            model.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)

    return checkpoint


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

    # Initialize distributed (FSDP) if requested
    device, rank, world_size = _init_distributed_if_needed(config)
    distributed_training = config.enable_fsdp and _is_distributed_and_initialized()

    # -------------------------------------------------------------
    #  Pre-training diagnostics (plots & quick sanity checks)
    # -------------------------------------------------------------
    if config.run_pretrain_tests or config.pretrain_tests_only:
        if _is_main_process():
            run_pretrain_tests(config)
        if distributed_training:
            dist.barrier()
        if config.pretrain_tests_only:
            if _is_main_process():
                print("[INFO] Pretrain tests only requested; exiting without training.")
            return float("nan")

    # Build dataset loaders and retrieve dataset metadata
    (
        trainloader,
        testloader,
        in_channels,
        num_classes,
    ) = get_data_loaders(
        config.batch_size,
        dataset=config.dataset,
        return_meta=True,
        distributed=distributed_training,
    )

    # Build model
    model = create_model(config, in_channels, num_classes).to(device)
    if config.enable_fsdp:
        model = FSDP(model, device_id=device)
        if _is_main_process():
            print(f"[INFO] FSDP enabled (world_size={world_size}, device={device}).")

    # Optionally load pretrained weights for fine-tuning
    if not config.eval_only and config.pretrained_weights:
        try:
            load_weights_into_model(model, config.pretrained_weights, device)
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
        ckpt = load_weights_into_model(model, config.pretrained_weights, device)
        test_acc = evaluate(model, testloader, device, distributed=distributed_training)
        if _is_main_process():
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
        if isinstance(trainloader.sampler, DistributedSampler):
            trainloader.sampler.set_epoch(epoch)
        if isinstance(testloader.sampler, DistributedSampler):
            testloader.sampler.set_epoch(epoch)
        running_loss = 0.0
        pbar = (
            tqdm(trainloader, desc=f"Epoch {epoch}/{config.num_epochs - 1}")
            if _is_main_process()
            else trainloader
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
            if _is_main_process() and hasattr(pbar, "set_postfix"):
                pbar.set_postfix({"loss": f"{loss.item():.3f}"})

        scheduler.step()

        # Evaluate
        test_acc = evaluate(model, testloader, device, distributed=distributed_training)
        last_epoch_test_acc = test_acc
        train_acc = evaluate(
            model, trainloader, device, distributed=distributed_training
        )
        best_acc = max(best_acc, test_acc)
        if _is_main_process():
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
        if _is_main_process():
            print(
                f"[INFO] Accuracy with all distortion strengths set to 1.0: {distortion_acc:.2f}%"
            )

    # ---------------- Export -------------------------------
    ckpt_path = os.path.join(config.output_dir, "fftconv_checkpoint.pth")
    save_checkpoint(model, config, best_acc, ckpt_path)
    # Save full model for structure reference when not sharded
    if not isinstance(model, FSDP) and _is_main_process():
        torch.save(model, os.path.join(config.output_dir, "fftconv_full_model.pth"))
    elif _is_main_process():
        print(
            "[WARN] Skipping full-model serialization under FSDP; use checkpoints for loading weights."
        )

    # Also dump the final config for completeness
    if _is_main_process():
        with open(os.path.join(config.output_dir, "final_config.yaml"), "w") as f:
            yaml.dump(vars(config), f)

        print(
            f"Training finished. Last epoch test accuracy: {last_epoch_test_acc:.2f}% | Best test accuracy: {best_acc:.2f}%. Checkpoint saved to {ckpt_path}."
        )
    if distributed_training:
        dist.barrier()
    return last_epoch_test_acc
