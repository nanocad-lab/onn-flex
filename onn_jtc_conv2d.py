import math
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from onn_component import JTC
from onn_config import AppConfig
from jtc_cycle_planner import compute_contamination_profile
from onn_math import complex_abs_squared, sqrt_nonnegative_with_finite_grad
from onn_quantization import quantize_ste


def _pair(value) -> tuple[int, int]:
    if isinstance(value, tuple):
        return int(value[0]), int(value[1])
    return int(value), int(value)


class JTCConv2d(nn.Module):
    """Conv2d-compatible wrapper backed by JTC/Fourier dot products.

    The public weight/bias layout matches ``torch.nn.Conv2d``. The ideal and
    emulated JTC backends both use the explicit optical-shot decomposition so
    backend selection reflects the user's requested optical model.
    """

    _ROWWISE_FIELD = 256

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] | str = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        config: AppConfig | None = None,
        max_jtc_shots: int = 65536,
        assume_nonnegative_input: bool = False,
    ):
        super().__init__()
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = padding
        self.dilation = _pair(dilation)
        self.groups = int(groups)
        self.padding_mode = padding_mode
        self.config = config if config is not None else AppConfig()
        self.max_jtc_shots = int(max_jtc_shots)
        self.assume_nonnegative_input = bool(assume_nonnegative_input)
        self.weight = nn.Parameter(
            torch.empty(
                self.out_channels,
                self.in_channels // self.groups,
                self.kernel_size[0],
                self.kernel_size[1],
            )
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_channels))
        else:
            self.register_parameter("bias", None)
        self._jtc_cache = nn.ModuleDict()
        self.reset_parameters()

    @classmethod
    def from_conv2d(
        cls,
        conv: nn.Conv2d,
        config: AppConfig,
        max_jtc_shots: int = 65536,
        assume_nonnegative_input: bool = False,
    ) -> "JTCConv2d":
        layer = cls(
            conv.in_channels,
            conv.out_channels,
            conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=conv.bias is not None,
            padding_mode=conv.padding_mode,
            config=config,
            max_jtc_shots=max_jtc_shots,
            assume_nonnegative_input=assume_nonnegative_input,
        )
        layer = layer.to(device=conv.weight.device, dtype=conv.weight.dtype)
        with torch.no_grad():
            layer.weight.copy_(conv.weight)
            if conv.bias is not None and layer.bias is not None:
                layer.bias.copy_(conv.bias)
        return layer

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def _padding_4tuple(self) -> tuple[int, int, int, int]:
        if isinstance(self.padding, str):
            if self.padding != "same":
                raise ValueError("Only padding='same' is supported as a string")
            if self.stride != (1, 1):
                raise ValueError("padding='same' requires stride=1")
            total_h = self.dilation[0] * (self.kernel_size[0] - 1)
            total_w = self.dilation[1] * (self.kernel_size[1] - 1)
            top = total_h // 2
            left = total_w // 2
            return top, total_h - top, left, total_w - left
        pad_h, pad_w = _pair(self.padding)
        return pad_h, pad_h, pad_w, pad_w

    def _conv2d_native(self, x: torch.Tensor) -> torch.Tensor:
        conv_input = x
        padding = self.padding
        if self.padding_mode != "zeros":
            top, bottom, left, right = self._padding_4tuple()
            conv_input = F.pad(
                x,
                (left, right, top, bottom),
                mode=self.padding_mode,
            )
            padding = (0, 0)
        return F.conv2d(
            conv_input,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=padding,
            dilation=self.dilation,
            groups=self.groups,
        )

    def _output_hw(self, h: int, w: int) -> tuple[int, int]:
        top, bottom, left, right = self._padding_4tuple()
        out_h = (
            h
            + top
            + bottom
            - self.dilation[0] * (self.kernel_size[0] - 1)
            - 1
        ) // self.stride[0] + 1
        out_w = (
            w
            + left
            + right
            - self.dilation[1] * (self.kernel_size[1] - 1)
            - 1
        ) // self.stride[1] + 1
        return int(out_h), int(out_w)

    def _apply_padding(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        top, bottom, left, right = self._padding_4tuple()
        if top == bottom and left == right and self.padding_mode == "zeros":
            return x, (top, left)
        if top == bottom and left == right and not isinstance(self.padding, str):
            if self.padding_mode == "zeros":
                return x, (top, left)
            x = F.pad(x, (left, right, top, bottom), mode=self.padding_mode)
            return x, (0, 0)
        if self.padding_mode == "zeros":
            x = F.pad(x, (left, right, top, bottom))
        else:
            x = F.pad(x, (left, right, top, bottom), mode=self.padding_mode)
        return x, (0, 0)

    def _plane_size(self, length: int) -> int:
        sep = int(self.config.jtc_separation)
        configured = int(self.config.jtc_total_field)
        return max(configured, 8 * int(length) + 64, 2 * int(length) + sep + 1)

    def _positive_paired_dot_jtc_ideal(
        self, signal: torch.Tensor, kernel: torch.Tensor
    ) -> torch.Tensor:
        if signal.shape != kernel.shape:
            raise ValueError("signal and kernel must have matching paired shapes")
        shots, length = signal.shape
        if shots == 0:
            return signal.new_empty(0)

        sep = int(self.config.jtc_separation)
        plane_size = self._plane_size(length)
        output = signal.new_empty(shots)
        max_shots = max(1, int(self.max_jtc_shots))
        dot_index = (length + 1) // 2

        for start in range(0, shots, max_shots):
            end = min(start + max_shots, shots)
            signal_chunk = signal[start:end]
            kernel_chunk = kernel[start:end]
            chunk = end - start
            if self.config.dac_bits is not None:
                signal_chunk = quantize_ste(signal_chunk, self.config.dac_bits)
                kernel_chunk = quantize_ste(kernel_chunk, self.config.dac_bits)

            input_plane = torch.zeros(
                chunk,
                plane_size,
                dtype=torch.complex64,
                device=signal.device,
            )
            input_plane[:, :length] = kernel_chunk.to(torch.complex64)
            signal_start = length + sep
            input_plane[:, signal_start : signal_start + length] = signal_chunk.to(
                torch.complex64
            )

            roll_amount = (plane_size // 2) - (2 * length + sep) // 2
            input_plane = torch.roll(input_plane, shifts=roll_amount, dims=-1)

            jft = torch.fft.fft(input_plane, dim=-1)
            jps = complex_abs_squared(torch.fft.fftshift(jft, dim=-1)) / plane_size
            jps = quantize_ste(jps, self.config.adc_bits)
            if self.config.fourier_plane_bits is not None:
                jps = quantize_ste(jps, self.config.fourier_plane_bits)
            jps = quantize_ste(jps, self.config.dac_bits)
            out_mag = torch.abs(
                torch.fft.fftshift(torch.fft.fft(jps, dim=-1), dim=-1)
            )
            out_power = quantize_ste(out_mag * out_mag, self.config.adc_bits)
            out_plane = sqrt_nonnegative_with_finite_grad(out_power)
            same_start = plane_size // 2 + sep + length // 2
            indices = (
                torch.arange(
                    same_start,
                    same_start + 2 * length - 1,
                    device=signal.device,
                )
                % plane_size
            )
            output[start:end] = out_plane[:, indices[dot_index]].to(output.dtype)
        return output

    def _jtc_for_length(self, length: int, device: torch.device) -> JTC:
        key = str(int(length))
        if key not in self._jtc_cache:
            cfg = deepcopy(self.config)
            cfg.input_length = int(length)
            cfg.kernel_length = int(length)
            cfg.output_length = None
            cfg.jtc_total_field = self._plane_size(length)
            self._jtc_cache[key] = JTC(cfg)
        return self._jtc_cache[key].to(device=device)

    def _positive_paired_dot_jtc(
        self, signal: torch.Tensor, kernel: torch.Tensor
    ) -> torch.Tensor:
        jtc = self._jtc_for_length(signal.shape[-1], signal.device)
        output = signal.new_empty(signal.shape[0])
        max_shots = max(1, int(self.max_jtc_shots))
        dot_index = (signal.shape[-1] + 1) // 2
        for start in range(0, signal.shape[0], max_shots):
            end = min(start + max_shots, signal.shape[0])
            full = jtc.forward_paired(signal[start:end], kernel[start:end])
            output[start:end] = full[:, dot_index].to(output.dtype)
        return output

    def _positive_paired_dot(
        self, signal: torch.Tensor, kernel: torch.Tensor, backend: str
    ) -> torch.Tensor:
        if backend == "jtc_ideal":
            return self._positive_paired_dot_jtc_ideal(signal, kernel)
        if backend == "jtc_emulation":
            return self._positive_paired_dot_jtc(signal, kernel)
        raise ValueError(f"Unsupported optical backend: {backend}")

    def _signed_paired_dot(
        self, signal: torch.Tensor, kernel: torch.Tensor, backend: str
    ) -> torch.Tensor:
        if self.assume_nonnegative_input:
            # This is an explicit scheduling assertion for post-ReLU inputs.
            # Do not clamp `signal` here; clamping would change layer math.
            kernel_pos = kernel.clamp_min(0)
            kernel_neg = (-kernel).clamp_min(0)
            pos = self._positive_paired_dot(signal, kernel_pos, backend)
            neg = self._positive_paired_dot(signal, kernel_neg, backend)
            return pos - neg

        signal_pos = signal.clamp_min(0)
        signal_neg = (-signal).clamp_min(0)
        kernel_pos = kernel.clamp_min(0)
        kernel_neg = (-kernel).clamp_min(0)
        pos = self._positive_paired_dot(signal_pos, kernel_pos, backend)
        pos = pos + self._positive_paired_dot(signal_neg, kernel_neg, backend)
        neg = self._positive_paired_dot(signal_pos, kernel_neg, backend)
        neg = neg + self._positive_paired_dot(signal_neg, kernel_pos, backend)
        return pos - neg

    def _rowwise_jtc_config(self, row_width: int, backend: str) -> AppConfig:
        kernel_width = self.kernel_size[1]
        cfg = deepcopy(self.config)
        cfg.input_length = int(row_width)
        cfg.kernel_length = kernel_width
        cfg.output_length = None
        cfg.jtc_total_field = self._ROWWISE_FIELD
        cfg.jtc_separation = int(row_width) - kernel_width
        if backend == "jtc_ideal":
            cfg.loss = 1.0
            cfg.driver_distortion_strength = 0.0
            cfg.pd_distortion_strength = 0.0
            cfg.tia_distortion_strength = 0.0
            cfg.mrm_power_distortion_strength = 0.0
            cfg.mrm_phase_distortion_strength = 0.0
            cfg.lens_distortion_strength = 0.0
            cfg.ler_std_dev = 0.0
            cfg.laser_rin_db = None
            cfg.pd_noise_w = 0.0
            cfg.pd_input_clamp_min_w = None
            cfg.pd_input_clamp_max_w = None
            cfg.scale_output = "none"
        return cfg

    def _rowwise_jtc_for_width(
        self, row_width: int, backend: str, device: torch.device
    ) -> JTC:
        key = f"row_{backend}_{int(row_width)}"
        if key not in self._jtc_cache:
            self._jtc_cache[key] = JTC(self._rowwise_jtc_config(row_width, backend))
        return self._jtc_cache[key].to(device=device)

    def _positive_paired_row_corr(
        self, signal: torch.Tensor, kernel: torch.Tensor, backend: str
    ) -> torch.Tensor:
        jtc = self._rowwise_jtc_for_width(signal.shape[-1], backend, signal.device)
        output = signal.new_empty(signal.shape[0], jtc.output_length)
        max_shots = max(1, int(self.max_jtc_shots))
        for start in range(0, signal.shape[0], max_shots):
            end = min(start + max_shots, signal.shape[0])
            full = jtc.forward_paired(signal[start:end], kernel[start:end])
            output[start:end] = full.to(output.dtype)
        return output

    def _signed_paired_row_corr(
        self, signal: torch.Tensor, kernel: torch.Tensor, backend: str
    ) -> torch.Tensor:
        if self.assume_nonnegative_input:
            # This is an explicit scheduling assertion for post-ReLU inputs.
            # Do not clamp `signal` here; clamping would change layer math.
            kernel_pos = kernel.clamp_min(0)
            kernel_neg = (-kernel).clamp_min(0)
            pos = self._positive_paired_row_corr(signal, kernel_pos, backend)
            neg = self._positive_paired_row_corr(signal, kernel_neg, backend)
            return pos - neg

        signal_pos = signal.clamp_min(0)
        signal_neg = (-signal).clamp_min(0)
        kernel_pos = kernel.clamp_min(0)
        kernel_neg = (-kernel).clamp_min(0)
        pos = self._positive_paired_row_corr(signal_pos, kernel_pos, backend)
        pos = pos + self._positive_paired_row_corr(signal_neg, kernel_neg, backend)
        neg = self._positive_paired_row_corr(signal_pos, kernel_neg, backend)
        neg = neg + self._positive_paired_row_corr(signal_neg, kernel_pos, backend)
        return pos - neg

    def _group_paired_dots(
        self,
        patches: torch.Tensor,
        weight: torch.Tensor,
        backend: str,
    ) -> torch.Tensor:
        # patches: [B, L, D], weight: [O, D]
        batch, num_locations, length = patches.shape
        out_channels = weight.shape[0]
        flat_patches = patches.reshape(batch * num_locations, length)
        total_shots = flat_patches.shape[0] * out_channels
        output = patches.new_empty(total_shots)
        max_shots = max(1, int(self.max_jtc_shots))

        for start in range(0, total_shots, max_shots):
            end = min(start + max_shots, total_shots)
            shot_ids = torch.arange(start, end, device=patches.device)
            patch_ids = torch.div(shot_ids, out_channels, rounding_mode="floor")
            out_ids = shot_ids.remainder(out_channels)
            signal = flat_patches.index_select(0, patch_ids)
            kernel = weight.index_select(0, out_ids)
            output[start:end] = self._signed_paired_dot(signal, kernel, backend)

        return output.reshape(batch, num_locations, out_channels).permute(0, 2, 1)

    def _optical_conv2d(self, x: torch.Tensor, backend: str) -> torch.Tensor:
        out_h, out_w = self._output_hw(x.shape[-2], x.shape[-1])
        x, unfold_padding = self._apply_padding(x)
        patches = F.unfold(
            x,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            padding=unfold_padding,
            stride=self.stride,
        )
        patches = patches.transpose(1, 2).contiguous()

        in_per_group = self.in_channels // self.groups
        out_per_group = self.out_channels // self.groups
        features_per_group = in_per_group * self.kernel_size[0] * self.kernel_size[1]
        group_outputs: list[torch.Tensor] = []
        for group_idx in range(self.groups):
            feature_start = group_idx * features_per_group
            feature_end = feature_start + features_per_group
            out_start = group_idx * out_per_group
            out_end = out_start + out_per_group
            patches_g = patches[:, :, feature_start:feature_end]
            weight_g = self.weight[out_start:out_end].reshape(
                out_per_group, features_per_group
            )
            group_outputs.append(self._group_paired_dots(patches_g, weight_g, backend))

        out = torch.cat(group_outputs, dim=1)
        out = out.reshape(x.shape[0], self.out_channels, out_h, out_w)
        if self.bias is not None:
            out = out + self.bias.reshape(1, -1, 1, 1)
        return out

    def _rowwise_clean_input_limit(self) -> int:
        kernel_width = self.kernel_size[1]
        if kernel_width <= 1:
            return 0
        clean_limit = 0
        for signal_length in range(kernel_width, self._ROWWISE_FIELD + 1):
            sep = signal_length - kernel_width
            _, clean, stride = compute_contamination_profile(
                signal_length,
                kernel_width,
                self._ROWWISE_FIELD,
                sep,
            )
            valid_outputs = signal_length - kernel_width + 1
            if clean == valid_outputs and stride == valid_outputs:
                clean_limit = signal_length
        return clean_limit

    def _can_use_rowwise_conv2d(self, x: torch.Tensor) -> bool:
        if self.kernel_size[0] != self.kernel_size[1]:
            return False
        if self.kernel_size[1] <= 1:
            return False
        if self.dilation != (1, 1):
            return False
        if self.groups != 1:
            return False
        top, bottom, left, right = self._padding_4tuple()
        row_width = x.shape[-1] + left + right
        clean_limit = self._rowwise_clean_input_limit()
        return (
            row_width >= self.kernel_size[1]
            and row_width <= clean_limit
            and x.shape[-2] + top + bottom >= self.kernel_size[0]
        )

    def _rowwise_rows_per_shot(self, row_width: int, out_h: int) -> int:
        guard = self.kernel_size[1] - 1
        clean_limit = self._rowwise_clean_input_limit()
        if clean_limit <= 0:
            return 1
        return max(
            1,
            min(
                int(out_h),
                (clean_limit + guard) // (int(row_width) + guard),
            ),
        )

    def _rowwise_conv2d(self, x: torch.Tensor, backend: str) -> torch.Tensor:
        top, bottom, left, right = self._padding_4tuple()
        if top or bottom or left or right:
            if self.padding_mode == "zeros":
                x_padded = F.pad(x, (left, right, top, bottom))
            else:
                x_padded = F.pad(x, (left, right, top, bottom), mode=self.padding_mode)
        else:
            x_padded = x

        batch, in_channels, _, row_width = x_padded.shape
        out_h, out_w = self._output_hw(x.shape[-2], x.shape[-1])
        kernel_width = self.kernel_size[1]
        valid_start = 0 if kernel_width == 1 else (kernel_width + 1) // 2
        pre_stride_w = row_width - kernel_width + 1
        col_indices = torch.arange(out_w, device=x.device) * self.stride[1]
        row_indices = torch.arange(out_h, device=x.device) * self.stride[0]
        output = x.new_zeros(batch, self.out_channels, out_h, out_w)

        rows_per_shot = self._rowwise_rows_per_shot(row_width, out_h)
        if rows_per_shot > 1:
            return self._rowwise_conv2d_packed(
                x_padded,
                output,
                backend,
                row_width,
                col_indices,
                row_indices,
                rows_per_shot,
            )

        shots_per_out_channel = batch * out_h * in_channels
        max_shots = max(1, int(self.max_jtc_shots))
        out_chunk_size = max(1, max_shots // max(1, shots_per_out_channel))

        for kernel_row in range(self.kernel_size[0]):
            input_rows = row_indices + kernel_row
            row_signals = x_padded.index_select(2, input_rows)
            row_signals = row_signals.permute(0, 2, 1, 3).contiguous()

            for out_start in range(0, self.out_channels, out_chunk_size):
                out_end = min(out_start + out_chunk_size, self.out_channels)
                out_count = out_end - out_start
                total_shots = batch * out_h * in_channels * out_count
                output_flat = output.reshape(-1)
                w_ids = torch.arange(out_w, device=x.device)

                for shot_start in range(0, total_shots, max_shots):
                    shot_end = min(shot_start + max_shots, total_shots)
                    shot_ids = torch.arange(shot_start, shot_end, device=x.device)
                    out_ids = shot_ids.remainder(out_count)
                    channel_ids = torch.div(
                        shot_ids,
                        out_count,
                        rounding_mode="floor",
                    ).remainder(in_channels)
                    row_ids = torch.div(
                        shot_ids,
                        out_count * in_channels,
                        rounding_mode="floor",
                    ).remainder(out_h)
                    batch_ids = torch.div(
                        shot_ids,
                        out_count * in_channels * out_h,
                        rounding_mode="floor",
                    )
                    out_abs = out_start + out_ids

                    signal_chunk = row_signals[batch_ids, row_ids, channel_ids]
                    kernel_chunk = self.weight[out_abs, channel_ids, kernel_row, :]
                    row_corr = self._signed_paired_row_corr(
                        signal_chunk,
                        kernel_chunk,
                        backend,
                    )
                    valid = row_corr[
                        :,
                        valid_start : valid_start + pre_stride_w,
                    ]
                    valid = valid.index_select(1, col_indices)
                    flat_base = (
                        ((batch_ids * self.out_channels + out_abs) * out_h + row_ids)
                        * out_w
                    )
                    flat_indices = flat_base[:, None] + w_ids[None, :]
                    output_flat.index_add_(
                        0,
                        flat_indices.reshape(-1),
                        valid.reshape(-1),
                    )

        if self.bias is not None:
            output = output + self.bias.reshape(1, -1, 1, 1)
        return output

    def _rowwise_conv2d_packed(
        self,
        x_padded: torch.Tensor,
        output: torch.Tensor,
        backend: str,
        row_width: int,
        col_indices: torch.Tensor,
        row_indices: torch.Tensor,
        rows_per_shot: int,
    ) -> torch.Tensor:
        batch, in_channels, _, _ = x_padded.shape
        out_h = output.shape[-2]
        out_w = output.shape[-1]
        kernel_width = self.kernel_size[1]
        valid_start = 0 if kernel_width == 1 else (kernel_width + 1) // 2
        guard = kernel_width - 1
        num_packs = math.ceil(out_h / rows_per_shot)
        packed_width = rows_per_shot * row_width + (rows_per_shot - 1) * guard
        packed_col_indices = torch.cat(
            [
                local_row * (row_width + guard)
                + valid_start
                + col_indices
                for local_row in range(rows_per_shot)
            ],
            dim=0,
        )
        max_shots = max(1, int(self.max_jtc_shots))
        shots_per_out_channel = batch * num_packs * in_channels
        out_chunk_size = max(1, max_shots // max(1, shots_per_out_channel))

        for kernel_row in range(self.kernel_size[0]):
            packed_rows = x_padded.new_zeros(
                batch, num_packs, in_channels, packed_width
            )
            for pack_group, row_start in enumerate(range(0, out_h, rows_per_shot)):
                row_end = min(row_start + rows_per_shot, out_h)
                pack_count = row_end - row_start
                input_rows = row_indices[row_start:row_end] + kernel_row
                row_signals = x_padded.index_select(2, input_rows)
                row_signals = row_signals.permute(0, 2, 1, 3).contiguous()

                for local_row in range(pack_count):
                    signal_start = local_row * (row_width + guard)
                    packed_rows[
                        :,
                        pack_group,
                        :,
                        signal_start : signal_start + row_width,
                    ] = row_signals[:, local_row]

            for out_start in range(0, self.out_channels, out_chunk_size):
                out_end = min(out_start + out_chunk_size, self.out_channels)
                out_count = out_end - out_start
                total_shots = batch * num_packs * in_channels * out_count
                output_flat = output.reshape(-1)
                local_rows = torch.arange(rows_per_shot, device=x_padded.device)
                w_ids = torch.arange(out_w, device=x_padded.device)

                for shot_start in range(0, total_shots, max_shots):
                    shot_end = min(shot_start + max_shots, total_shots)
                    shot_ids = torch.arange(
                        shot_start,
                        shot_end,
                        device=x_padded.device,
                    )
                    out_ids = shot_ids.remainder(out_count)
                    channel_ids = torch.div(
                        shot_ids,
                        out_count,
                        rounding_mode="floor",
                    ).remainder(in_channels)
                    pack_ids = torch.div(
                        shot_ids,
                        out_count * in_channels,
                        rounding_mode="floor",
                    ).remainder(num_packs)
                    batch_ids = torch.div(
                        shot_ids,
                        out_count * in_channels * num_packs,
                        rounding_mode="floor",
                    )
                    out_abs = out_start + out_ids

                    signal_chunk = packed_rows[batch_ids, pack_ids, channel_ids]
                    kernel_chunk = self.weight[out_abs, channel_ids, kernel_row, :]
                    row_corr = self._signed_paired_row_corr(
                        signal_chunk,
                        kernel_chunk,
                        backend,
                    )
                    valid = row_corr.index_select(1, packed_col_indices)
                    valid = valid.reshape(-1, rows_per_shot, out_w)

                    row_ids = pack_ids[:, None] * rows_per_shot + local_rows[None, :]
                    valid_rows = row_ids < out_h
                    flat_base = (
                        ((batch_ids[:, None] * self.out_channels + out_abs[:, None])
                        * out_h
                        + row_ids)
                        * out_w
                    )
                    flat_indices = flat_base[:, :, None] + w_ids[None, None, :]
                    valid_mask = valid_rows[:, :, None].expand(-1, -1, out_w)
                    output_flat.index_add_(
                        0,
                        flat_indices[valid_mask],
                        valid[valid_mask],
                    )

        if self.bias is not None:
            output = output + self.bias.reshape(1, -1, 1, 1)
        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        backend = self.config.conv_backend
        if backend == "pytorch":
            return self._conv2d_native(x)
        if backend not in {"jtc_ideal", "jtc_emulation"}:
            raise ValueError(
                f"Unknown conv_backend: {backend}. "
                "Must be one of: 'pytorch', 'jtc_ideal', 'jtc_emulation'"
            )
        if self.kernel_size == (1, 1):
            return self._conv2d_native(x)
        if self._can_use_rowwise_conv2d(x):
            return self._rowwise_conv2d(x, backend)
        return self._optical_conv2d(x, backend)


def replace_conv2d_with_jtc(
    module: nn.Module,
    config: AppConfig,
    max_jtc_shots: int = 65536,
    assume_nonnegative_input: bool = False,
) -> nn.Module:
    """Recursively replace ``nn.Conv2d`` modules with ``JTCConv2d``.

    The replacement is in-place and preserves each Conv2d module's weight,
    bias, stride, padding, dilation, groups, and padding mode.
    Set assume_nonnegative_input only when every replaced conv receives
    nonnegative activations.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv2d):
            setattr(
                module,
                name,
                JTCConv2d.from_conv2d(
                    child,
                    config=config,
                    max_jtc_shots=max_jtc_shots,
                    assume_nonnegative_input=assume_nonnegative_input,
                ),
            )
        else:
            replace_conv2d_with_jtc(
                child,
                config=config,
                max_jtc_shots=max_jtc_shots,
                assume_nonnegative_input=assume_nonnegative_input,
            )
    return module
