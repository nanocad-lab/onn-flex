from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from jtc_cycle_planner import WIDTH as CIFAR_WIDTH, cycles_for_config
from onn_config import AppConfig
from onn_layers import FTconvlayer


def _find_optimal_lengths(
    lens_size: int,
    kernel_length_candidates: Sequence[int],
    input_length_candidates: Sequence[int],
) -> Tuple[int, int, int, int]:
    """Search for the best (input_len, kernel_len, sep, stride) tuple."""
    best_config: Tuple[int, int, int, int] | None = None
    best_cycles: int | None = None
    best_stride: int | None = None

    for kernel_len in kernel_length_candidates:
        for input_len in input_length_candidates:
            if input_len < kernel_len:
                continue
            max_sep = lens_size - (input_len + kernel_len)
            if max_sep < 0:
                continue
            for sep in range(max_sep + 1):
                result = cycles_for_config(input_len, kernel_len, lens_size, sep)
                if result is None:
                    continue
                passes, total_cycles, effective_stride, _ = result
                if effective_stride <= 0:
                    continue
                if (
                    best_cycles is None
                    or total_cycles < best_cycles
                    or (
                        total_cycles == best_cycles
                        and effective_stride > (best_stride or 0)
                    )
                ):
                    best_cycles = total_cycles
                    best_stride = effective_stride
                    best_config = (input_len, kernel_len, sep, effective_stride)

    if best_config is None:
        raise ValueError(
            f"Unable to find valid JTC configuration for lens_size={lens_size}, "
            f"kernel candidates={kernel_length_candidates}"
        )
    return best_config


def _maybe_plan_jtc_lengths(config: AppConfig, model_name: str) -> None:
    """Adapt JTC lengths for models that benefit from optimized planning."""
    if not getattr(config, "auto_plan_jtc_lengths", True):
        return
    if not model_name.startswith("vgg"):
        return

    lens_size = int(config.jtc_total_field or 64)

    max_input = getattr(config, "input_length", None)
    max_kernel = getattr(config, "kernel_length", None)

    input_cap = min(
        lens_size,
        CIFAR_WIDTH,
        max_input if max_input is not None and max_input > 0 else CIFAR_WIDTH,
    )
    if input_cap < 3:
        raise ValueError(
            "Maximum input_length is too small for auto planning. "
            "Increase --input-length or disable auto_plan_jtc_lengths."
        )
    input_candidates = list(range(3, input_cap + 1))

    kernel_cap = min(
        lens_size,
        max_kernel if max_kernel is not None and max_kernel > 0 else lens_size,
    )
    kernel_candidates = [k for k in [3, 5, 7, 9, 11] if k <= kernel_cap]
    if not kernel_candidates:
        raise ValueError(
            "Maximum kernel_length is too small for auto planning. "
            "Increase --kernel-length or disable auto_plan_jtc_lengths."
        )

    input_len, kernel_len, sep, stride = _find_optimal_lengths(
        lens_size, kernel_candidates, input_candidates
    )

    if (
        config.input_length == input_len
        and config.kernel_length == kernel_len
        and config.jtc_separation == sep
    ):
        return

    config.input_length = input_len
    config.kernel_length = kernel_len
    config.jtc_separation = sep
    config.output_length = input_len + kernel_len - 1
    print(
        f"[JTC Planner] Using optimized lengths for {model_name}: "
        f"input_length={input_len}, kernel_length={kernel_len}, separation={sep}, "
        f"effective_stride={stride}, lens_size={lens_size}"
    )


class FFTConvNet(nn.Module):
    """Configurable variant of the 7-layer FFTConv network from `old_template.py`.

    The number of identical intermediate blocks (originally 5) can be varied
    through `config.num_identical_layers`.
    """

    def __init__(self, config: AppConfig):
        super().__init__()
        self.config = config

        # Stem
        in_ch = max(1, getattr(config, "input_channels", 3))
        self.conv1 = FTconvlayer(
            in_ch,
            8,
            config=config,
            kernel_size=config.input_length,
            hv_concat=True,
        )
        self.bn1 = nn.BatchNorm2d(16)
        self.maxpool1 = nn.MaxPool2d(2)

        # Second block (fixed)
        self.conv2 = FTconvlayer(
            16,
            16,
            config=config,
            kernel_size=config.input_length,
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
                    kernel_size=config.input_length,
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
            nn.Linear(256, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
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


VGG_CONFIGS: Dict[str, List[int | str]] = {
    "vgg11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg13": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        "M",
        512,
        512,
        "M",
        512,
        512,
        "M",
    ],
    "vgg16": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "vgg19": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


def _make_vgg_layers(config: AppConfig, cfg: Sequence[int | str]) -> Tuple[nn.Sequential, int]:
    layers: List[nn.Module] = []
    in_channels = max(1, getattr(config, "input_channels", 3))
    last_channels = in_channels

    for v in cfg:
        if v == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            out_channels = int(v)
            layers.append(
                FTconvlayer(
                    in_channels,
                    out_channels,
                    config=config,
                    kernel_size=config.input_length,
                    hv_concat=False,
                )
            )
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
            last_channels = out_channels

    return nn.Sequential(*layers), last_channels


class VGG(nn.Module):
    """VGG-style network built from FTconvlayer blocks."""

    def __init__(self, config: AppConfig, variant: str):
        super().__init__()
        if variant not in VGG_CONFIGS:
            raise ValueError(f"Unsupported VGG variant '{variant}'")
        self.config = config

        layers, last_channels = _make_vgg_layers(config, VGG_CONFIGS[variant])
        self.features = layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(last_channels, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x


MODEL_REGISTRY: Dict[str, Callable[[AppConfig], nn.Module]] = {
    "fftconv": FFTConvNet,
    "vgg11": lambda cfg: VGG(cfg, "vgg11"),
    "vgg13": lambda cfg: VGG(cfg, "vgg13"),
    "vgg16": lambda cfg: VGG(cfg, "vgg16"),
    "vgg19": lambda cfg: VGG(cfg, "vgg19"),
}


def build_model(config: AppConfig) -> nn.Module:
    """Instantiate a model based on config.model_name."""
    model_name = getattr(config, "model_name", "fftconv")
    builder = MODEL_REGISTRY.get(model_name)
    if builder is None:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available models: {', '.join(sorted(MODEL_REGISTRY))}"
        )

    _maybe_plan_jtc_lengths(config, model_name)
    return builder(config)
