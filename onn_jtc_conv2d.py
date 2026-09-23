import math
from copy import deepcopy

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.utils.checkpoint import checkpoint

from onn_analog_shot import AnalogJTCShot
from onn_component import JTC
from onn_config import AppConfig
from onn_math import complex_abs_squared, sqrt_nonnegative_with_finite_grad
from onn_quantization import quantize_ste
from onn_shotplan import (
    compute_contamination_profile,
    plan_convolution,
    resolve_geometry,
)


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
        self.checkpoint_policy = self.config.jtc_checkpoint_policy
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
        self.shot = AnalogJTCShot(self.config)
        self._shot_plans = {}
        # jtc_analytic calibrated-gain state: running max of the pre-ADC
        # useful-lag outputs (persistent so eval-only runs keep the gain).
        self.register_buffer("_analytic_gain_running_max", torch.ones(()))
        self.register_buffer(
            "_analytic_gain_calibrated", torch.zeros((), dtype=torch.bool)
        )
        self.register_buffer(
            "_analytic_gain_obs_count", torch.zeros((), dtype=torch.long)
        )
        # The analytic readout is ~10 pointwise passes over the per-shot
        # correlation tensor; fusing them dominates this backend's step time.
        self._analytic_chunk = self._make_analytic_chunk()
        self._fourier_chunk = self._make_fourier_chunk()
        self.reset_parameters()

    def _make_analytic_chunk(self):
        if bool(getattr(self.config, "compile_jtc", False)):
            from onn_component import raise_dynamo_recompile_limit

            raise_dynamo_recompile_limit()
            return torch.compile(self._analytic_row_chunk, dynamic=True)
        return self._analytic_row_chunk

    def _make_fourier_chunk(self):
        if bool(getattr(self.config, "compile_jtc", False)):
            from onn_component import raise_dynamo_recompile_limit

            raise_dynamo_recompile_limit()
            return torch.compile(self._fourier_row_chunk, dynamic=True)
        return self._fourier_row_chunk

    def __getstate__(self):
        state = dict(self.__dict__)
        state.pop("_analytic_chunk", None)
        state.pop("_fourier_chunk", None)
        state.pop("_fourier_maps_cache", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._analytic_chunk = self._make_analytic_chunk()
        self._fourier_chunk = self._make_fourier_chunk()

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
            h + top + bottom - self.dilation[0] * (self.kernel_size[0] - 1) - 1
        ) // self.stride[0] + 1
        out_w = (
            w + left + right - self.dilation[1] * (self.kernel_size[1] - 1) - 1
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
            out_mag = torch.abs(torch.fft.fftshift(torch.fft.fft(jps, dim=-1), dim=-1))
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

    def _jtc_on_device(self, jtc: JTC, device: torch.device) -> JTC:
        # Module.to() walks the whole submodule tree even when it is a no-op;
        # this runs once per shot chunk, so skip it when already in place.
        dtype = torch.float64 if self.weight.dtype == torch.float64 else torch.float32
        if (
            jtc.input_field_coeffs.device != device
            or jtc.input_field_coeffs.dtype != dtype
        ):
            jtc.to(device=device, dtype=dtype)
        return jtc

    def _jtc_for_length(self, length: int, device: torch.device) -> JTC:
        if self.config.conv_backend in {"jtc_analytic", "jtc_analog_fourier"}:
            geometry = resolve_geometry(
                "dot",
                length,
                length,
                self.config.jtc_total_field,
                self.config.jtc_separation,
                self.config.jtc_rowwise_geometry,
            )
            return self.shot.components(geometry, self.weight)
        key = str(int(length))
        if key not in self._jtc_cache:
            cfg = deepcopy(self.config)
            cfg.input_length = int(length)
            cfg.kernel_length = int(length)
            cfg.output_length = None
            cfg.jtc_total_field = (
                int(self.config.jtc_total_field)
                if self.config.jtc_rowwise_geometry == "config"
                else self._plane_size(length)
            )
            self._jtc_cache[key] = JTC(cfg)
        return self._jtc_on_device(self._jtc_cache[key], device)

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

    def _analytic_gain_snapshot(self, ref: torch.Tensor) -> torch.Tensor | None:
        """Resolve the gain for this forward, without mutating any state.

        Returns None for per_shot mode (the AGC is computed inside the chunk
        from the shot itself, which is a pure function). The calibrated-mode
        buffer is snapshotted so checkpoint recompute during backward sees
        the same value the forward pass used; the running-max update happens
        outside any checkpointed region (_observe_analytic_max).
        """
        mode = self.config.jtc_output_gain_mode
        if mode == "per_shot":
            return None
        if mode == "fixed":
            # Scheduler inputs may be fp16 under AMP; physical gains can
            # exceed fp16's range and must retain at least fp32 precision.
            dtype = torch.float64 if ref.dtype == torch.float64 else torch.float32
            return torch.full(
                (), float(self.config.jtc_output_gain), device=ref.device, dtype=dtype
            )
        headroom = float(getattr(self.config, "jtc_gain_headroom", 1.0) or 1.0)
        return (
            (1.0 / (self._analytic_gain_running_max * headroom).clamp_min(1e-12))
            .detach()
            .clone()
        )

    def reset_gain_calibration(self) -> None:
        """Re-open calibrate_freeze observation (epoch-wise gain ranging).

        The running max is kept as the prior; the next observed batch
        overwrites it (calibrated flag cleared), then the EMA runs for
        another jtc_gain_freeze_batches layer-forwards before freezing.
        """
        with torch.no_grad():
            self._analytic_gain_obs_count.zero_()
            self._analytic_gain_calibrated.fill_(False)

    def _observe_analytic_max(self, observed_max: torch.Tensor | float) -> None:
        """Observe all shots once per layer forward, for the next forward.

        DDP ranks share the global batch maximum. Chunk size and checkpoint
        recomputation must not affect the population, EMA, or freeze count.
        """
        mode = self.config.jtc_output_gain_mode
        if mode not in {"calibrated", "calibrate_freeze"} or not self.training:
            return
        if mode == "calibrate_freeze" and int(self._analytic_gain_obs_count) >= int(
            self.config.jtc_gain_freeze_batches
        ):
            return  # frozen: a stationary per-layer gain from here on
        with torch.no_grad():
            observed_max = (
                torch.as_tensor(
                    observed_max,
                    device=self._analytic_gain_running_max.device,
                    dtype=self._analytic_gain_running_max.dtype,
                )
                .detach()
                .clone()
            )
            if dist.is_initialized():
                dist.all_reduce(observed_max, op=dist.ReduceOp.MAX)
            batch_max = max(float(observed_max), 1e-12)
            if not bool(self._analytic_gain_calibrated):
                new_max = batch_max
                self._analytic_gain_calibrated.fill_(True)
            else:
                new_max = 0.9 * float(self._analytic_gain_running_max) + 0.1 * batch_max
            self._analytic_gain_running_max.fill_(new_max)
            self._analytic_gain_obs_count += 1

    def _needs_gain_observation(self) -> bool:
        return (
            self.config.conv_backend in {"jtc_analytic", "jtc_analog_fourier"}
            and self.config.jtc_output_gain_mode in {"calibrated", "calibrate_freeze"}
            and self.training
            and (
                self.config.jtc_output_gain_mode == "calibrated"
                or int(self._analytic_gain_obs_count)
                < self.config.jtc_gain_freeze_batches
            )
        )

    def _analytic_paired_valid_corr(
        self, signal: torch.Tensor, kernel: torch.Tensor
    ) -> torch.Tensor:
        """Valid cross-correlation of paired shots via one batched GEMM."""
        kernel_length = kernel.shape[-1]
        windows = signal.unfold(-1, kernel_length, 1)  # (S, W-Kw+1, Kw)
        return torch.einsum("swk,sk->sw", windows, kernel)

    def _analytic_paired_row_corr(
        self,
        signal: torch.Tensor,
        kernel: torch.Tensor,
        jtc: JTC,
        gain: torch.Tensor | None,
        need_vmax: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Analytic replacement for a JTC rowwise correlation shot batch.

        Returns the full (W + Kw - 1)-length correlation layout the JTC
        callers index into; only the valid (clean-lag) region is populated —
        the schedulers never select contaminated positions.

        Runs in float32 regardless of autocast: the physical constants span
        ~1e6, which overflows half precision (the optical backends are
        likewise fp32 because FFTs never autocast).
        """
        with torch.autocast(device_type=signal.device.type, enabled=False):
            a_in, b_in, p, c1, _ = self.shot.constants(jtc)
            f_s = self.shot.input_field(signal.float(), jtc)
            f_k = self.shot.input_field(kernel.float(), jtc)
            corr = self._analytic_paired_valid_corr(f_s, f_k)
            out_valid, v_max = self.shot.correlation_readout(
                corr, p, c1, gain, need_vmax=need_vmax
            )
        kernel_length = kernel.shape[-1]
        # Place the valid conv outputs exactly where the rowwise schedulers
        # read them (_select_rowwise_columns starts at valid_start); the
        # remaining positions are contaminated lags that are never selected.
        valid_start = 0 if kernel_length <= 1 else (kernel_length + 1) // 2
        full = signal.new_zeros(signal.shape[0], signal.shape[-1] + kernel_length - 1)
        full[:, valid_start : valid_start + out_valid.shape[-1]] = out_valid
        return full, v_max

    def _positive_paired_dot(
        self,
        signal: torch.Tensor,
        kernel: torch.Tensor,
        backend: str,
        gain: torch.Tensor | None,
        need_vmax: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if backend == "jtc_ideal":
            return self._positive_paired_dot_jtc_ideal(
                signal, kernel
            ), signal.new_zeros(())
        if backend == "jtc_emulation":
            return self._positive_paired_dot_jtc(signal, kernel), signal.new_zeros(())
        if backend == "jtc_analytic":
            jtc = self._jtc_for_length(signal.shape[-1], signal.device)
            with torch.autocast(device_type=signal.device.type, enabled=False):
                a_in, b_in, p, c1, _ = self.shot.constants(jtc)
                f_s = self.shot.input_field(signal.float(), jtc)
                f_k = self.shot.input_field(kernel.float(), jtc)
                corr = (f_s * f_k).sum(dim=-1, keepdim=True)
                out, v_max = self.shot.correlation_readout(
                    corr, p, c1, gain, need_vmax=need_vmax
                )
                return out.squeeze(-1), v_max
        if backend == "jtc_analog_fourier":
            jtc = self._jtc_for_length(signal.shape[-1], signal.device)
            out, v_max = self.shot.evaluate(signal, kernel, jtc, gain, need_vmax)
            return out[:, (signal.shape[-1] + 1) // 2], v_max
        raise ValueError(f"Unsupported optical backend: {backend}")

    def _signed_paired_dot(
        self,
        signal: torch.Tensor,
        kernel: torch.Tensor,
        backend: str,
        gain: torch.Tensor | None,
        need_vmax: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        kernel_pos = kernel.clamp_min(0)
        kernel_neg = (-kernel).clamp_min(0)
        if self.assume_nonnegative_input:
            # Explicit scheduling assertion for post-ReLU inputs.
            pos, vmax_pos = self._positive_paired_dot(
                signal, kernel_pos, backend, gain, need_vmax
            )
            neg, vmax_neg = self._positive_paired_dot(
                signal, kernel_neg, backend, gain, need_vmax
            )
            return pos - neg, torch.maximum(vmax_pos, vmax_neg)

        signal_pos = signal.clamp_min(0)
        signal_neg = (-signal).clamp_min(0)
        pp, vmax_pp = self._positive_paired_dot(
            signal_pos, kernel_pos, backend, gain, need_vmax
        )
        nn, vmax_nn = self._positive_paired_dot(
            signal_neg, kernel_neg, backend, gain, need_vmax
        )
        pn, vmax_pn = self._positive_paired_dot(
            signal_pos, kernel_neg, backend, gain, need_vmax
        )
        np, vmax_np = self._positive_paired_dot(
            signal_neg, kernel_pos, backend, gain, need_vmax
        )
        vmax = torch.maximum(
            torch.maximum(vmax_pp, vmax_nn), torch.maximum(vmax_pn, vmax_np)
        )
        return (pp + nn) - (pn + np), vmax

    def _positive_paired_row_corr_jtc_ideal(
        self, signal: torch.Tensor, kernel: torch.Tensor
    ) -> torch.Tensor:
        if signal.dim() != 2 or kernel.dim() != 2:
            raise ValueError("row correlation inputs must be 2D")
        if signal.shape[0] != kernel.shape[0]:
            raise ValueError("signal and kernel must have the same shot count")
        shots, signal_length = signal.shape
        kernel_length = kernel.shape[-1]
        if shots == 0:
            return signal.new_empty(0, signal_length + kernel_length - 1)

        jtc = self._rowwise_jtc_for_width(signal_length, "jtc_ideal", signal.device)
        output = signal.new_empty(shots, jtc.output_length)
        max_shots = max(1, int(self.max_jtc_shots))
        indices = jtc.compute_correlation_indices(signal.device)

        for start in range(0, shots, max_shots):
            end = min(start + max_shots, shots)
            signal_chunk = signal[start:end]
            kernel_chunk = kernel[start:end]
            chunk = end - start
            if self.config.dac_bits is not None:
                signal_chunk = quantize_ste(signal_chunk, self.config.dac_bits)
                kernel_chunk = quantize_ste(kernel_chunk, self.config.dac_bits)

            input_plane = signal.new_zeros(chunk, jtc.jtc_total_field)
            input_plane[:, :kernel_length] = kernel_chunk
            signal_start = kernel_length + jtc.jtc_separation
            input_plane[:, signal_start : signal_start + signal_length] = signal_chunk

            jps = complex_abs_squared(torch.fft.fft(input_plane, dim=-1))
            jps = jps / float(jtc.jtc_total_field)
            jps = quantize_ste(jps, self.config.adc_bits)
            if self.config.fourier_plane_bits is not None:
                jps = quantize_ste(jps, self.config.fourier_plane_bits)
            jps = quantize_ste(jps, self.config.dac_bits)

            out_mag = torch.abs(torch.fft.fft(jps, dim=-1))
            out_power = quantize_ste(out_mag * out_mag, self.config.adc_bits)
            out_plane = sqrt_nonnegative_with_finite_grad(out_power)
            output[start:end] = out_plane.index_select(-1, indices).to(output.dtype)

        return output

    def _rowwise_jtc_config(self, row_width: int, backend: str) -> AppConfig:
        kernel_width = self.kernel_size[1]
        cfg = deepcopy(self.config)
        cfg.input_length = int(row_width)
        cfg.kernel_length = kernel_width
        cfg.output_length = None
        if self.config.jtc_rowwise_geometry == "config":
            # Lens-size / separation study mode: the plane geometry comes
            # straight from the config instead of the legacy fixed plane.
            cfg.jtc_total_field = int(self.config.jtc_total_field)
            cfg.jtc_separation = int(self.config.jtc_separation)
        else:
            cfg.jtc_total_field = self._ROWWISE_FIELD
            cfg.jtc_separation = int(row_width) - kernel_width
        if backend == "jtc_ideal":
            cfg.loss = 1.0
            cfg.driver_distortion_strength = 0.0
            cfg.pd_distortion_strength = 0.0
            cfg.tia_distortion_strength = 0.0
            cfg.mrm_amplitude_distortion_strength = 0.0
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
        if backend in {"jtc_analytic", "jtc_analog_fourier"}:
            geometry = resolve_geometry(
                "row",
                row_width,
                self.kernel_size[1],
                self.config.jtc_total_field,
                self.config.jtc_separation,
                self.config.jtc_rowwise_geometry,
            )
            return self.shot.components(geometry, self.weight)
        key = f"row_{backend}_{int(row_width)}"
        if key not in self._jtc_cache:
            self._jtc_cache[key] = JTC(self._rowwise_jtc_config(row_width, backend))
        jtc = self._jtc_on_device(self._jtc_cache[key], device)
        if backend in {"jtc_analytic", "jtc_analog_fourier"}:
            # Resolve Python constants before entering a compiled readout.
            # Calibration no longer runs a separate probe that warmed them.
            self.shot.constants(jtc)
        return jtc

    def _positive_paired_row_corr(
        self,
        signal: torch.Tensor,
        kernel: torch.Tensor,
        backend: str,
        gain: torch.Tensor | None,
        need_vmax: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if backend == "jtc_ideal":
            return self._positive_paired_row_corr_jtc_ideal(
                signal, kernel
            ), signal.new_zeros(())
        if backend == "jtc_analytic":
            jtc = self._rowwise_jtc_for_width(signal.shape[-1], backend, signal.device)
            return self._analytic_paired_row_corr(signal, kernel, jtc, gain, need_vmax)
        if backend == "jtc_analog_fourier":
            jtc = self._rowwise_jtc_for_width(signal.shape[-1], backend, signal.device)
            return self.shot.evaluate(signal, kernel, jtc, gain, need_vmax)
        if backend not in {"jtc_ideal", "jtc_emulation"}:
            raise ValueError(f"Unsupported optical backend: {backend}")
        jtc = self._rowwise_jtc_for_width(signal.shape[-1], backend, signal.device)
        output = signal.new_empty(signal.shape[0], jtc.output_length)
        max_shots = max(1, int(self.max_jtc_shots))
        for start in range(0, signal.shape[0], max_shots):
            end = min(start + max_shots, signal.shape[0])
            full = jtc.forward_paired(signal[start:end], kernel[start:end])
            output[start:end] = full.to(output.dtype)
        return output, signal.new_zeros(())

    def _signed_paired_row_corr(
        self,
        signal: torch.Tensor,
        kernel: torch.Tensor,
        backend: str,
        gain: torch.Tensor | None,
        need_vmax: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        kernel_pos = kernel.clamp_min(0)
        kernel_neg = (-kernel).clamp_min(0)
        if self.assume_nonnegative_input:
            # Explicit scheduling assertion for post-ReLU inputs.
            pos, vmax_pos = self._positive_paired_row_corr(
                signal, kernel_pos, backend, gain, need_vmax
            )
            neg, vmax_neg = self._positive_paired_row_corr(
                signal, kernel_neg, backend, gain, need_vmax
            )
            return pos - neg, torch.maximum(vmax_pos, vmax_neg)

        signal_pos = signal.clamp_min(0)
        signal_neg = (-signal).clamp_min(0)
        pp, vmax_pp = self._positive_paired_row_corr(
            signal_pos, kernel_pos, backend, gain, need_vmax
        )
        nn, vmax_nn = self._positive_paired_row_corr(
            signal_neg, kernel_neg, backend, gain, need_vmax
        )
        pn, vmax_pn = self._positive_paired_row_corr(
            signal_pos, kernel_neg, backend, gain, need_vmax
        )
        np, vmax_np = self._positive_paired_row_corr(
            signal_neg, kernel_pos, backend, gain, need_vmax
        )
        vmax = torch.maximum(
            torch.maximum(vmax_pp, vmax_nn), torch.maximum(vmax_pn, vmax_np)
        )
        return (pp + nn) - (pn + np), vmax

    def _checkpoint_chunks(self) -> bool:
        """Recompute per-chunk optical activations during backward.

        Checkpointing at chunk granularity bounds peak memory by
        ``jtc_max_shots`` worth of pipeline intermediates. A layer-level
        checkpoint does not: its backward re-materializes every chunk of the
        layer at once, which OOMs for wide layers regardless of chunk size.
        The spectra policy additionally retains compact preparation graphs
        across chunks; their memory accumulates through the model forward.
        """
        return (
            self.config.enable_jtc_activation_checkpointing
            and self.training
            and torch.is_grad_enabled()
        )

    def _validate_analytic_geometry(self, row_width: int) -> None:
        """jtc_analytic is exact only when every valid lag is clean.

        In explicit-geometry mode the configured (lens, separation) may put
        autocorrelation or mirror-lobe energy on the extracted lags; the
        closed form omits those terms, so it must refuse rather than be
        silently wrong. Use conv_backend='jtc_analog_fourier' for
        contaminated geometries (it computes the plane physics exactly).
        """
        if self.config.jtc_rowwise_geometry != "config":
            return
        cache = getattr(self, "_analytic_geom_ok", None)
        if cache is None:
            cache = self._analytic_geom_ok = {}
        key = int(row_width)
        if key in cache:
            if not cache[key]:
                raise ValueError(self._geom_error(row_width))
            return
        kernel_width = self.kernel_size[1]
        _, clean, stride = compute_contamination_profile(
            row_width,
            kernel_width,
            int(self.config.jtc_total_field),
            int(self.config.jtc_separation),
        )
        valid = row_width - kernel_width + 1
        ok = clean == valid and stride == valid
        cache[key] = ok
        if not ok:
            raise ValueError(self._geom_error(row_width))

    def _geom_error(self, row_width: int) -> str:
        return (
            "jtc_analytic requires contamination-free valid lags for the "
            f"configured geometry (row_width={row_width}, "
            f"kernel={self.kernel_size[1]}, "
            f"field={self.config.jtc_total_field}, "
            f"sep={self.config.jtc_separation}); use "
            "conv_backend='jtc_analog_fourier' for contaminated geometries"
        )

    def _fourier_closed_maps(self, row_width: int, jtc: JTC, device: torch.device):
        """Mod-field lag maps for the closed-form jtc_analog_fourier path.

        The output plane of the all-affine JTC is v[d] = P * R[d]^2 at every
        non-DC bin, where R is the circular autocorrelation of the composite
        input plane (kernel at 0, signal at kernel_width + separation). R
        decomposes into four linear-lag vectors — cross-correlation, its
        mirror lobe (the same vector re-indexed), and the signal/kernel
        autocorrelations — each aliased mod-field onto the detector.

        The cross+mirror map is emitted as per-window gather layers over the
        padded signal: both lobes multiply the same kernel tap, so the map
        folds into the signal BEFORE the output-channel broadcast and the
        expensive (batch x cout x cin x rows x lags) multiply-accumulate
        stays identical to _analytic_row_chunk. Layer j holds indices
        t_j[e] and weights w_j[e] (weight 0 pads rows with fewer entries;
        weights exceed 1 where two lobe directions land on the same bin).
        The low-rank autocorrelation terms stay as small dense matrices. In
        clean geometry the autocorrelation maps are empty and the gather is
        a single permutation layer — the backend then reproduces
        jtc_analytic exactly.
        """
        cache = getattr(self, "_fourier_maps_cache", None)
        if cache is None:
            cache = self._fourier_maps_cache = {}
        key = (int(row_width), str(device))
        hit = cache.get(key)
        if hit is not None:
            return hit
        n = int(jtc.jtc_total_field)
        sep = int(jtc.jtc_separation)
        kw = int(self.kernel_size[1])
        w = int(row_width)
        if w + kw + sep > n:
            raise ValueError(
                "jtc_analog_fourier geometry is infeasible: row_width "
                f"({w}) + kernel ({kw}) + separation ({sep}) exceeds "
                f"jtc_total_field ({n}); the two apertures do not fit on "
                "the input plane"
            )
        s0 = kw + sep
        num_lags = w + kw - 1  # full cross-correlation == extraction window
        ext = jtc.compute_correlation_indices(torch.device("cpu")).tolist()
        if len(ext) != num_lags:
            raise ValueError(
                f"unexpected extraction window length {len(ext)} (expected {num_lags})"
            )
        if 0 in ext:
            raise ValueError(
                "jtc_analog_fourier closed form cannot model an extraction "
                "window that includes the DC bin; set "
                "jtc_fourier_closed_form=false for this geometry"
            )
        pos = {d: e for e, d in enumerate(ext)}

        # Cross + mirror lobes. The chunk computes the correlation lag
        # x_c[t] = sum_k s_pad[t+k]*k[k] on a (kw-1)-zero-padded row, so
        # gather index t corresponds to relative kernel-vs-signal
        # displacement u = (kw-1) - t.
        rows: list[list[tuple[int, float]]] = [[] for _ in range(num_lags)]
        for t in range(num_lags):
            u = (kw - 1) - t
            for d in ((u - s0) % n, (s0 - u) % n):
                e = pos.get(d)
                if e is not None:
                    for entry in rows[e]:
                        if entry[0] == t:
                            rows[e].remove(entry)
                            rows[e].append((t, entry[1] + 1.0))
                            break
                    else:
                        rows[e].append((t, 1.0))
        depth = max(len(r) for r in rows)
        x_gathers = []
        for j in range(depth):
            t_idx = [r[j][0] if j < len(r) else 0 for r in rows]
            wts = [r[j][1] if j < len(r) else 0.0 for r in rows]
            x_gathers.append(
                (
                    torch.tensor(t_idx, dtype=torch.long, device=device),
                    (
                        None
                        if all(v == 1.0 for v in wts)
                        else torch.tensor(wts, dtype=torch.float32, device=device)
                    ),
                )
            )

        def autocorr_map(width: int):
            entries: dict[tuple[int, int], float] = {}
            lags: list[int] = []
            for lag in range(1, width):
                for d in (lag % n, (-lag) % n):
                    e = pos.get(d)
                    if e is None:
                        continue
                    if lag not in lags:
                        lags.append(lag)
                    key2 = (e, lags.index(lag))
                    entries[key2] = entries.get(key2, 0.0) + 1.0
            if not lags:
                return (), None
            m = torch.zeros(num_lags, len(lags))
            for (e, i), c in entries.items():
                m[e, i] = c
            return tuple(lags), m.to(device)

        s_lags, m_s = autocorr_map(w)
        k_lags, m_k = autocorr_map(kw)
        maps = (num_lags, tuple(x_gathers), s_lags, m_s, k_lags, m_k)
        cache[key] = maps
        return maps

    def _run_shot_chunk(self, fn, *args):
        if fn == self._rowwise_chunk_contrib and self._checkpoint_spectra(
            args[0].shape[0], args[2]
        ):
            # Preparation is retained; _rowwise_chunk_contrib checkpoints
            # only detection. Do not wrap both in another checkpoint.
            return fn(*args)
        if self._checkpoint_chunks():
            # The FFT plane paths need reentrant checkpointing on sm_100:
            # their in-place ops (plane slice assignment) desynchronize the
            # non-reentrant recompute bookkeeping under torch.compile
            # ("different number of tensors saved"), while sm_80 runs clean.
            # Reentrant costs ~5x under compile, so the lag-GEMM path was
            # rewritten strictly functionally (per-primitive cosine GEMMs,
            # no index_add_/slice writes) to checkpoint non-reentrantly like
            # the analytic path, which runs non-reentrant on sm_100 fine.
            reentrant = (
                self.config.conv_backend == "jtc_analog_fourier"
                and self.shot.uses_plane_fft()
            )
            return checkpoint(fn, *args, use_reentrant=reentrant)
        return fn(*args)

    def _paired_dot_chunk(
        self,
        flat_patches: torch.Tensor,
        weight: torch.Tensor,
        patch_ids: torch.Tensor,
        out_ids: torch.Tensor,
        backend: str,
        gain: torch.Tensor | None,
        need_vmax: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        signal = flat_patches.index_select(0, patch_ids)
        kernel = weight.index_select(0, out_ids)
        return self._signed_paired_dot(signal, kernel, backend, gain, need_vmax)

    def _group_paired_dots(
        self,
        patches: torch.Tensor,
        weight: torch.Tensor,
        backend: str,
        gain: torch.Tensor | None,
        need_vmax: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # patches: [B, L, D], weight: [O, D]
        batch, num_locations, length = patches.shape
        out_channels = weight.shape[0]
        flat_patches = patches.reshape(batch * num_locations, length)
        total_shots = flat_patches.shape[0] * out_channels
        output = patches.new_empty(total_shots)
        observed_max = patches.new_zeros(())
        max_shots = max(1, int(self.max_jtc_shots))

        for start in range(0, total_shots, max_shots):
            end = min(start + max_shots, total_shots)
            shot_ids = torch.arange(start, end, device=patches.device)
            patch_ids = torch.div(shot_ids, out_channels, rounding_mode="floor")
            out_ids = shot_ids.remainder(out_channels)
            contrib, v_max = self._run_shot_chunk(
                self._paired_dot_chunk,
                flat_patches,
                weight,
                patch_ids,
                out_ids,
                backend,
                gain,
                need_vmax,
            )
            output[start:end] = contrib
            observed_max = torch.maximum(observed_max, v_max)

        return (
            output.reshape(batch, num_locations, out_channels).permute(0, 2, 1),
            observed_max,
        )

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
        gain = self._analytic_gain_snapshot(x)
        need_vmax = self._needs_gain_observation()
        observed_max = x.new_zeros(())
        for group_idx in range(self.groups):
            feature_start = group_idx * features_per_group
            feature_end = feature_start + features_per_group
            out_start = group_idx * out_per_group
            out_end = out_start + out_per_group
            patches_g = patches[:, :, feature_start:feature_end]
            weight_g = self.weight[out_start:out_end].reshape(
                out_per_group, features_per_group
            )
            group_output, v_max = self._group_paired_dots(
                patches_g, weight_g, backend, gain, need_vmax
            )
            group_outputs.append(group_output)
            observed_max = torch.maximum(observed_max, v_max)

        out = torch.cat(group_outputs, dim=1)
        out = out.reshape(x.shape[0], self.out_channels, out_h, out_w)
        if self.bias is not None:
            out = out + self.bias.reshape(1, -1, 1, 1)
        if need_vmax:
            self._observe_analytic_max(observed_max)
        return out

    def _rowwise_clean_input_limit(self) -> int:
        # This sweep is pure Python over ~250 planner evaluations; it must not
        # run on every forward, so memoize per layer (kernel size is fixed).
        cached = getattr(self, "_rowwise_clean_input_limit_cache", None)
        if cached is not None:
            return cached
        kernel_width = self.kernel_size[1]
        clean_limit = 0
        if kernel_width > 1:
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
        self._rowwise_clean_input_limit_cache = clean_limit
        return clean_limit

    def _can_use_rowwise_conv2d(self, x: torch.Tensor) -> bool:
        if self.kernel_size[1] <= 1:
            return False
        if self.dilation != (1, 1):
            return False
        if self.groups != 1:
            return False
        top, bottom, left, right = self._padding_4tuple()
        row_width = x.shape[-1] + left + right
        if self.config.jtc_rowwise_geometry == "config":
            # Explicit-geometry mode: feasible iff the row fits the plane.
            return (
                row_width >= self.kernel_size[1]
                and row_width + self.kernel_size[1] + int(self.config.jtc_separation)
                <= int(self.config.jtc_total_field)
                and x.shape[-2] + top + bottom >= self.kernel_size[0]
            )
        clean_limit = self._rowwise_clean_input_limit()
        return (
            row_width >= self.kernel_size[1]
            and row_width <= clean_limit
            and x.shape[-2] + top + bottom >= self.kernel_size[0]
        )

    def _rowwise_rows_per_shot(self, row_width: int, out_h: int) -> int:
        if self.config.jtc_rowwise_geometry == "config":
            return 1  # explicit-geometry study mode: one row per plane
        if self.config.conv_backend in {"jtc_analytic", "jtc_analog_fourier"}:
            # Nonlinear transfers make per-shot power levels physical;
            # multi-row packing would change the detector operating point.
            return 1
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

    def _reuse_rowwise_spectra(self, positions, backend):
        return (
            backend == "jtc_analog_fourier"
            and not self.config.jtc_fourier_closed_form
            and self.config.jtc_fourier_lag_gemm
            and self.shot.needs_full_transfers()
            and positions > 1
        )

    def _checkpoint_spectra(self, positions, backend):
        return (
            self.checkpoint_policy == "spectra"
            and self._checkpoint_chunks()
            and self._reuse_rowwise_spectra(positions, backend)
        )

    def _rowwise_chunk_contrib(
        self,
        signal_block: torch.Tensor,
        kernel_template: torch.Tensor,
        backend: str,
        valid_start: int,
        pre_stride_w: int,
        col_indices: torch.Tensor,
        out_w: int,
        gain: torch.Tensor | None,
        need_vmax: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pos_count, in_channels, row_width = signal_block.shape
        out_count, kernel_width = kernel_template.shape[1], kernel_template.shape[2]
        if self._reuse_rowwise_spectra(pos_count, backend):
            jtc = self._rowwise_jtc_for_width(row_width, backend, signal_block.device)
            if self._checkpoint_spectra(pos_count, backend):
                spectra = self.shot.prepare_rowwise(
                    signal_block, kernel_template, jtc, self.assume_nonnegative_input
                )
                wide = (
                    torch.float64
                    if signal_block.dtype == torch.float64
                    else torch.float32
                )
                row_corr, v_max = checkpoint(
                    self.shot._compiled_detect_rowwise,
                    *spectra,
                    jtc,
                    wide,
                    gain,
                    need_vmax,
                    self.assume_nonnegative_input,
                    use_reentrant=False,
                )
            else:
                row_corr, v_max = self.shot._compiled_rowwise(
                    signal_block,
                    kernel_template,
                    jtc,
                    gain,
                    need_vmax,
                    self.assume_nonnegative_input,
                )
        else:
            signal_chunk = (
                signal_block[:, :, None, :]
                .expand(pos_count, in_channels, out_count, row_width)
                .reshape(-1, row_width)
            )
            kernel_chunk = (
                kernel_template[None, :, :, :]
                .expand(pos_count, in_channels, out_count, kernel_width)
                .reshape(-1, kernel_width)
            )
            row_corr, v_max = self._signed_paired_row_corr(
                signal_chunk, kernel_chunk, backend, gain, need_vmax
            )
        valid = self._select_rowwise_columns(
            row_corr,
            valid_start,
            pre_stride_w,
            col_indices,
            out_w,
        )
        return valid.reshape(pos_count, in_channels, out_count, out_w).sum(dim=1), v_max

    def _select_rowwise_columns(
        self,
        row_corr: torch.Tensor,
        valid_start: int,
        pre_stride_w: int,
        col_indices: torch.Tensor,
        out_w: int,
    ) -> torch.Tensor:
        if self.stride[1] == 1:
            return row_corr[:, valid_start : valid_start + out_w]
        valid = row_corr[:, valid_start : valid_start + pre_stride_w]
        return valid.index_select(1, col_indices)

    def _select_packed_rowwise_columns(
        self,
        row_corr: torch.Tensor,
        row_width: int,
        guard: int,
        valid_start: int,
        col_indices: torch.Tensor,
        rows_per_shot: int,
        out_w: int,
        packed_col_indices: torch.Tensor,
    ) -> torch.Tensor:
        if self.stride[1] == 1:
            pieces = []
            for local_row in range(rows_per_shot):
                start = local_row * (row_width + guard) + valid_start
                pieces.append(row_corr[:, start : start + out_w])
            return torch.cat(pieces, dim=1)
        return row_corr.index_select(1, packed_col_indices)

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

        gain = self._analytic_gain_snapshot(x)
        need_vmax = self._needs_gain_observation()
        observed_max = x.new_zeros(())
        rows_per_shot = self._rowwise_rows_per_shot(row_width, out_h)
        if rows_per_shot > 1:
            output_rows, observed_max = self._rowwise_conv2d_packed(
                x_padded,
                backend,
                row_width,
                col_indices,
                row_indices,
                rows_per_shot,
                gain,
                need_vmax,
            )
            output = output_rows.permute(0, 2, 1, 3).contiguous()
            if self.bias is not None:
                output = output + self.bias.reshape(1, -1, 1, 1)
            if need_vmax:
                self._observe_analytic_max(observed_max)
            return output

        output_rows = x.new_zeros(batch, out_h, self.out_channels, out_w)
        output_rows_flat = output_rows.reshape(batch * out_h, self.out_channels, out_w)

        max_shots = max(1, int(self.max_jtc_shots))
        out_chunk_size = max(1, max_shots // max(1, in_channels))
        position_count = batch * out_h

        for kernel_row in range(self.kernel_size[0]):
            if self.stride[0] == 1:
                row_signals = x_padded[:, :, kernel_row : kernel_row + out_h, :]
            else:
                input_rows = row_indices + kernel_row
                row_signals = x_padded.index_select(2, input_rows)
            row_signals = row_signals.permute(0, 2, 1, 3).contiguous()
            row_signals_flat = row_signals.reshape(
                position_count, in_channels, row_width
            )

            for out_start in range(0, self.out_channels, out_chunk_size):
                out_end = min(out_start + out_chunk_size, self.out_channels)
                out_count = out_end - out_start
                shots_per_position = in_channels * out_count
                position_chunk = max(1, max_shots // max(1, shots_per_position))
                weight_chunk = self.weight[out_start:out_end, :, kernel_row, :]
                kernel_template = weight_chunk.permute(1, 0, 2).contiguous()

                for pos_start in range(0, position_count, position_chunk):
                    pos_end = min(pos_start + position_chunk, position_count)
                    signal_block = row_signals_flat[pos_start:pos_end]
                    contrib, v_max = self._run_shot_chunk(
                        self._rowwise_chunk_contrib,
                        signal_block,
                        kernel_template,
                        backend,
                        valid_start,
                        pre_stride_w,
                        col_indices,
                        out_w,
                        gain,
                        need_vmax,
                    )
                    observed_max = torch.maximum(observed_max, v_max)
                    output_rows_flat[pos_start:pos_end, out_start:out_end] = (
                        output_rows_flat[pos_start:pos_end, out_start:out_end] + contrib
                    )

        output = output_rows.permute(0, 2, 1, 3).contiguous()
        if self.bias is not None:
            output = output + self.bias.reshape(1, -1, 1, 1)
        if need_vmax:
            self._observe_analytic_max(observed_max)
        return output

    def _packed_chunk_contrib(
        self,
        signal_block: torch.Tensor,
        kernel_template: torch.Tensor,
        backend: str,
        row_width: int,
        guard: int,
        valid_start: int,
        col_indices: torch.Tensor,
        rows_per_shot: int,
        out_w: int,
        packed_col_indices: torch.Tensor,
        gain: torch.Tensor | None,
        need_vmax: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pos_count, in_channels, packed_width = signal_block.shape
        out_count, kernel_width = kernel_template.shape[1], kernel_template.shape[2]
        signal_chunk = (
            signal_block[:, :, None, :]
            .expand(pos_count, in_channels, out_count, packed_width)
            .reshape(-1, packed_width)
        )
        kernel_chunk = (
            kernel_template[None, :, :, :]
            .expand(pos_count, in_channels, out_count, kernel_width)
            .reshape(-1, kernel_width)
        )
        row_corr, v_max = self._signed_paired_row_corr(
            signal_chunk, kernel_chunk, backend, gain, need_vmax
        )
        valid = self._select_packed_rowwise_columns(
            row_corr,
            row_width,
            guard,
            valid_start,
            col_indices,
            rows_per_shot,
            out_w,
            packed_col_indices,
        )
        contrib = valid.reshape(
            pos_count,
            in_channels,
            out_count,
            rows_per_shot,
            out_w,
        ).sum(dim=1)
        return contrib.permute(0, 2, 1, 3).contiguous(), v_max

    def _rowwise_conv2d_packed(
        self,
        x_padded: torch.Tensor,
        backend: str,
        row_width: int,
        col_indices: torch.Tensor,
        row_indices: torch.Tensor,
        rows_per_shot: int,
        gain: torch.Tensor | None,
        need_vmax: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, in_channels, _, _ = x_padded.shape
        observed_max = x_padded.new_zeros(())
        out_h = len(row_indices)
        out_w = len(col_indices)
        kernel_width = self.kernel_size[1]
        valid_start = 0 if kernel_width == 1 else (kernel_width + 1) // 2
        guard = kernel_width - 1
        num_packs = math.ceil(out_h / rows_per_shot)
        packed_width = rows_per_shot * row_width + (rows_per_shot - 1) * guard
        packed_col_indices = torch.cat(
            [
                local_row * (row_width + guard) + valid_start + col_indices
                for local_row in range(rows_per_shot)
            ],
            dim=0,
        )
        max_shots = max(1, int(self.max_jtc_shots))
        out_chunk_size = max(1, max_shots // max(1, in_channels))
        position_count = batch * num_packs
        output_packed = x_padded.new_zeros(
            batch,
            num_packs,
            rows_per_shot,
            self.out_channels,
            out_w,
        )

        for kernel_row in range(self.kernel_size[0]):
            packed_rows = x_padded.new_zeros(
                batch, num_packs, in_channels, packed_width
            )
            for pack_group, row_start in enumerate(range(0, out_h, rows_per_shot)):
                row_end = min(row_start + rows_per_shot, out_h)
                pack_count = row_end - row_start
                if self.stride[0] == 1:
                    input_start = row_start + kernel_row
                    row_signals = x_padded[
                        :, :, input_start : input_start + pack_count, :
                    ]
                else:
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
                shots_per_position = in_channels * out_count
                position_chunk = max(1, max_shots // max(1, shots_per_position))
                packed_rows_flat = packed_rows.reshape(
                    position_count,
                    in_channels,
                    packed_width,
                )
                weight_chunk = self.weight[out_start:out_end, :, kernel_row, :]
                kernel_template = weight_chunk.permute(1, 0, 2).contiguous()
                output_packed_flat = output_packed.reshape(
                    position_count,
                    rows_per_shot,
                    self.out_channels,
                    out_w,
                )

                for pos_start in range(0, position_count, position_chunk):
                    pos_end = min(pos_start + position_chunk, position_count)
                    signal_block = packed_rows_flat[pos_start:pos_end]
                    contrib, v_max = self._run_shot_chunk(
                        self._packed_chunk_contrib,
                        signal_block,
                        kernel_template,
                        backend,
                        row_width,
                        guard,
                        valid_start,
                        col_indices,
                        rows_per_shot,
                        out_w,
                        packed_col_indices,
                        gain,
                        need_vmax,
                    )
                    observed_max = torch.maximum(observed_max, v_max)
                    output_packed_flat[
                        pos_start:pos_end,
                        :,
                        out_start:out_end,
                        :,
                    ] = (
                        output_packed_flat[
                            pos_start:pos_end,
                            :,
                            out_start:out_end,
                            :,
                        ]
                        + contrib
                    )

        return (
            output_packed.reshape(
                batch,
                num_packs * rows_per_shot,
                self.out_channels,
                out_w,
            )[:, :out_h],
            observed_max,
        )

    def _analytic_row_chunk(
        self,
        fx: torch.Tensor,
        field_kernel: torch.Tensor,
        stride_h: int,
        stride_w: int,
        out_h: int,
        out_w: int,
        p: float,
        c1: float,
        gain: torch.Tensor | None,
        need_vmax: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Readout for one (branch, cout-chunk) over all kernel rows.

        fx: (B, Cin, H, W) padded input field; field_kernel: (O, Cin, Kh, Kw).
        One optical shot is one (b, o, cin, h) row correlation for one kernel
        row: the ADC readout applies per row-shot and the row contributions
        are summed digitally afterwards — never fuse rows before the readout.
        For small Kw the correlation is unrolled shifted multiply-adds so the
        compiler fuses correlation, readout chain, and both digital sums into
        one graph; the per-shot (B, O, Cin, H, W) tensor never hits HBM.
        Pure function: safe to checkpoint (calibration updated by caller).
        """
        fk = field_kernel.to(fx.dtype)
        out_ch, in_ch, kh, kw = fk.shape
        h_span = (out_h - 1) * stride_h + 1
        w_span = (out_w - 1) * stride_w + 1
        total = None
        v_max_total = None
        for r in range(kh):
            corr = None
            for k in range(kw):
                window = fx[
                    :, None, :, r : r + h_span : stride_h, k : k + w_span : stride_w
                ]
                term = window * fk[:, :, r, k].reshape(1, out_ch, in_ch, 1, 1)
                corr = term if corr is None else corr + term
            if corr.dtype != torch.float32:
                corr = corr.float()
            out, v_max = self.shot.correlation_readout(
                corr, p, c1, gain, need_vmax=need_vmax
            )
            contrib = out.sum(dim=2)
            total = contrib if total is None else total + contrib
            v_max_total = (
                v_max if v_max_total is None else torch.maximum(v_max_total, v_max)
            )
        return total, v_max_total

    def _analytic_conv2d(self, x: torch.Tensor) -> torch.Tensor:
        """Direct conv-form JTC: fused correlation + readout per chunk."""
        top, bottom, left, right = self._padding_4tuple()
        if top or bottom or left or right:
            mode = {} if self.padding_mode == "zeros" else {"mode": self.padding_mode}
            x_padded = F.pad(x, (left, right, top, bottom), **mode)
        else:
            x_padded = x
        out_h, out_w = self._output_hw(x.shape[-2], x.shape[-1])
        batch, in_channels, _, row_width = x_padded.shape
        self._validate_analytic_geometry(row_width)
        jtc = self._rowwise_jtc_for_width(row_width, "jtc_analytic", x.device)

        with torch.autocast(device_type=x.device.type, enabled=False):
            a_in, b_in, p, c1, _ = self.shot.constants(jtc)
            weight = self.weight.float()
            w_pos = weight.clamp_min(0)
            w_neg = (-weight).clamp_min(0)
            x_f = x_padded.float()
            if self.assume_nonnegative_input:
                signals = [(1.0, self.shot.input_field(x_f, jtc))]
            else:
                # Signed inputs: x = x_pos - x_neg, each part modulated and
                # shot separately (matches _signed_paired_row_corr).
                signals = [
                    (1.0, self.shot.input_field(x_f.clamp_min(0), jtc)),
                    (-1.0, self.shot.input_field((-x_f).clamp_min(0), jtc)),
                ]
            if self.config.jtc_analytic_gemm_dtype == "bfloat16":
                # Cast once so window slices stay views; casting per chunk
                # would materialize them (and trips an inductor stride bug
                # under compile).
                signals = [(s, fx.bfloat16()) for s, fx in signals]
            branches = [
                (x_sign * w_sign, fx, w_branch)
                for x_sign, fx in signals
                for w_sign, w_branch in ((1.0, w_pos), (-1.0, w_neg))
            ]
            fx = signals[0][1]
            max_shots = max(1, int(self.max_jtc_shots))
            out_chunk = max(1, max_shots // max(1, batch * out_h * in_channels))
            gain = self._analytic_gain_snapshot(fx)
            need_vmax = self._needs_gain_observation()
            observed_max = torch.zeros((), device=x.device)
            output = torch.zeros(
                batch,
                self.out_channels,
                out_h,
                out_w,
                dtype=torch.float32,
                device=x.device,
            )
            for sign, fx_branch, w_branch in branches:
                for o_start in range(0, self.out_channels, out_chunk):
                    o_end = min(o_start + out_chunk, self.out_channels)
                    f_w = self.shot.input_field(w_branch[o_start:o_end], jtc)
                    contrib, v_max = self._run_shot_chunk(
                        self._analytic_chunk,
                        fx_branch,
                        f_w,
                        self.stride[0],
                        self.stride[1],
                        out_h,
                        out_w,
                        p,
                        c1,
                        gain,
                        need_vmax,
                    )
                    observed_max = torch.maximum(observed_max, v_max)
                    output[:, o_start:o_end] = output[:, o_start:o_end] + sign * contrib
            if need_vmax:
                self._observe_analytic_max(observed_max)
        output = output.to(x.dtype)
        if self.bias is not None:
            output = output + self.bias.reshape(1, -1, 1, 1)
        return output

    def _fourier_row_chunk(
        self,
        fx_pad: torch.Tensor,
        field_kernel: torch.Tensor,
        stride_h: int,
        stride_w: int,
        out_h: int,
        out_w: int,
        p: float,
        c1: float,
        gain: torch.Tensor | None,
        need_vmax: bool,
        num_lags: int,
        x_gathers: tuple,
        s_lags: tuple,
        m_s: torch.Tensor | None,
        k_lags: tuple,
        m_k: torch.Tensor | None,
        valid_start: int,
        col_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Closed-form contaminated readout for one (branch, cout-chunk).

        Same shot schedule as _analytic_row_chunk — one shot is one
        (b, o, cin, h) row correlation per kernel row, readout per shot, rows
        summed digitally — but the extracted lag values carry every plane
        term: R = cross + mirror + signal/kernel autocorrelations aliased
        mod-field into the extraction window (see _fourier_closed_maps), so
        v = P * R^2 matches the two-FFT pipeline of _fourier_paired_row_corr
        exactly, wraparound and lobe overlap included. The cross+mirror map
        is applied by gathering the padded signal per kernel tap (no
        output-channel dimension), so the wide multiply-accumulate matches
        the analytic chunk's cost. The readout runs on the full extraction
        window (per-shot AGC and gain calibration see the same values the
        FFT path sees); output columns are selected afterwards. Pure
        function: safe to checkpoint.
        """
        fk = field_kernel.to(fx_pad.dtype)
        out_ch, in_ch, kh, kw = fk.shape
        h_span = (out_h - 1) * stride_h + 1
        total = None
        v_max_total = None
        for r in range(kh):
            rows_blk = fx_pad[:, :, r : r + h_span : stride_h, :]
            corr = None
            for k in range(kw):
                pre = None
                for t_idx, wts in x_gathers:
                    sel = rows_blk.index_select(-1, t_idx + k)
                    if wts is not None:
                        sel = sel * wts.to(sel.dtype)
                    pre = sel if pre is None else pre + sel
                term = pre[:, None] * fk[:, :, r, k].reshape(1, out_ch, in_ch, 1, 1)
                corr = term if corr is None else corr + term
            if corr.dtype != torch.float32:
                corr = corr.float()
            if m_s is not None:
                # unpadded row: fx_pad columns [kw-1, num_lags) hold the field
                row = rows_blk[..., kw - 1 : num_lags]
                width = row.shape[-1]
                a_s = torch.stack(
                    [(row[..., : width - d] * row[..., d:]).sum(-1) for d in s_lags],
                    dim=-1,
                )
                if a_s.dtype != torch.float32:
                    a_s = a_s.float()
                corr = corr + torch.einsum("...t,et->...e", a_s, m_s)[:, None]
            if m_k is not None:
                fk_r = fk[:, :, r, :].float()
                a_k = torch.stack(
                    [(fk_r[..., : kw - d] * fk_r[..., d:]).sum(-1) for d in k_lags],
                    dim=-1,
                )
                corr = corr + torch.einsum("oct,et->oce", a_k, m_k).reshape(
                    1, out_ch, in_ch, 1, num_lags
                )
            out_full, v_max = self.shot.correlation_readout(
                corr, p, c1, gain, need_vmax=need_vmax
            )
            if stride_w == 1:
                out = out_full[..., valid_start : valid_start + out_w]
            else:
                out = out_full.index_select(-1, col_idx)
            contrib = out.sum(dim=2)
            total = contrib if total is None else total + contrib
            v_max_total = (
                v_max if v_max_total is None else torch.maximum(v_max_total, v_max)
            )
        return total, v_max_total

    def _fourier_conv2d(self, x: torch.Tensor) -> torch.Tensor:
        """Direct conv-form jtc_analog_fourier via the closed lag algebra."""
        top, bottom, left, right = self._padding_4tuple()
        if top or bottom or left or right:
            mode = {} if self.padding_mode == "zeros" else {"mode": self.padding_mode}
            x_padded = F.pad(x, (left, right, top, bottom), **mode)
        else:
            x_padded = x
        out_h, out_w = self._output_hw(x.shape[-2], x.shape[-1])
        batch, in_channels, _, row_width = x_padded.shape
        jtc = self._rowwise_jtc_for_width(row_width, "jtc_analog_fourier", x.device)
        maps = self._fourier_closed_maps(row_width, jtc, x.device)

        with torch.autocast(device_type=x.device.type, enabled=False):
            a_in, b_in, p, c1, _ = self.shot.constants(jtc)
            kw = self.kernel_size[1]
            valid_start = 0 if kw <= 1 else (kw + 1) // 2
            col_idx = (
                valid_start + torch.arange(out_w, device=x.device) * self.stride[1]
            )
            weight = self.weight.float()
            w_pos = weight.clamp_min(0)
            w_neg = (-weight).clamp_min(0)
            x_f = x_padded.float()
            if self.assume_nonnegative_input:
                signals = [(1.0, self.shot.input_field(x_f, jtc))]
            else:
                signals = [
                    (1.0, self.shot.input_field(x_f.clamp_min(0), jtc)),
                    (-1.0, self.shot.input_field((-x_f).clamp_min(0), jtc)),
                ]
            if self.config.jtc_analytic_gemm_dtype == "bfloat16":
                signals = [(s, fx.bfloat16()) for s, fx in signals]
            kw_pad = self.kernel_size[1] - 1
            signals = [(s, F.pad(fx, (kw_pad, kw_pad))) for s, fx in signals]
            branches = [
                (x_sign * w_sign, fx, w_branch)
                for x_sign, fx in signals
                for w_sign, w_branch in ((1.0, w_pos), (-1.0, w_neg))
            ]
            fx = signals[0][1]
            max_shots = max(1, int(self.max_jtc_shots))
            out_chunk = max(1, max_shots // max(1, batch * out_h * in_channels))
            gain = self._analytic_gain_snapshot(fx)
            need_vmax = self._needs_gain_observation()
            observed_max = torch.zeros((), device=x.device)
            output = torch.zeros(
                batch,
                self.out_channels,
                out_h,
                out_w,
                dtype=torch.float32,
                device=x.device,
            )
            for sign, fx_pad_branch, w_branch in branches:
                for o_start in range(0, self.out_channels, out_chunk):
                    o_end = min(o_start + out_chunk, self.out_channels)
                    f_w = self.shot.input_field(w_branch[o_start:o_end], jtc)
                    contrib, v_max = self._run_shot_chunk(
                        self._fourier_chunk,
                        fx_pad_branch,
                        f_w,
                        self.stride[0],
                        self.stride[1],
                        out_h,
                        out_w,
                        p,
                        c1,
                        gain,
                        need_vmax,
                        *maps,
                        valid_start,
                        col_idx,
                    )
                    observed_max = torch.maximum(observed_max, v_max)
                    output[:, o_start:o_end] = output[:, o_start:o_end] + sign * contrib
            if need_vmax:
                self._observe_analytic_max(observed_max)
        output = output.to(x.dtype)
        if self.bias is not None:
            output = output + self.bias.reshape(1, -1, 1, 1)
        return output

    def shot_plan(self, input_shape):
        """Plan this layer without running optics or changing calibration state."""
        if self.config.conv_backend not in {"jtc_analytic", "jtc_analog_fourier"}:
            raise ValueError("ShotPlan describes the analog Fourier-plane topology")
        key = tuple(int(v) for v in input_shape)
        if len(key) != 4 or key[1] != self.in_channels:
            raise ValueError("Expected BCHW input matching layer input channels")
        if key not in self._shot_plans:
            plan = plan_convolution(
                input_shape=key,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                dilation=self.dilation,
                padding=self._padding_4tuple(),
                groups=self.groups,
                mapping=self.config.jtc_shot_mapping,
                nonnegative=self.assume_nonnegative_input,
                total_field=self.config.jtc_total_field,
                separation=self.config.jtc_separation,
                geometry_source=self.config.jtc_rowwise_geometry,
                readout=(
                    "valid_lags"
                    if self.config.conv_backend == "jtc_analytic"
                    else "full_window"
                ),
            )
            if self.config.conv_backend == "jtc_analytic" and plan.geometry is not None:
                if not all(
                    item["clean"]
                    for item in plan.geometry.contamination(plan.selected_offsets)
                ):
                    raise ValueError(
                        "jtc_analytic requires contamination-free selected lags; use jtc_analog_fourier"
                    )
            self._shot_plans[key] = plan
        return self._shot_plans[key]

    def _forward_optical(self, x: torch.Tensor, backend: str) -> torch.Tensor:
        if backend in {"jtc_analytic", "jtc_analog_fourier"}:
            plan = self.shot_plan(x.shape)
            if plan.mapping == "dot":
                return self._optical_conv2d(x, backend)
            if 1 < self.kernel_size[1] <= 7 and self.kernel_size[0] <= 7:
                if backend == "jtc_analytic":
                    return self._analytic_conv2d(x)
                if self.config.jtc_fourier_closed_form:
                    return self._fourier_conv2d(x)
            return self._rowwise_conv2d(x, backend)
        if self._can_use_rowwise_conv2d(x):
            return self._rowwise_conv2d(x, backend)
        return self._optical_conv2d(x, backend)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        backend = self.config.conv_backend
        if self.config.jtc_remodulation and backend not in {
            "jtc_analytic",
            "jtc_analog_fourier",
        }:
            raise ValueError(
                "jtc_remodulation requires an analog Fourier-plane backend"
            )
        if backend == "pytorch":
            return self._conv2d_native(x)
        if backend not in {
            "jtc_ideal",
            "jtc_emulation",
            "jtc_analytic",
            "jtc_analog_fourier",
        }:
            raise ValueError(
                f"Unknown conv_backend: {backend}. Must be one of: 'pytorch', "
                "'jtc_ideal', 'jtc_emulation', 'jtc_analytic', 'jtc_analog_fourier'"
            )
        if backend in {"jtc_analytic", "jtc_analog_fourier"}:
            self.shot.validate()
        if self.kernel_size == (1, 1):
            if backend in {"jtc_analytic", "jtc_analog_fourier"}:
                self.shot_plan(x.shape)
            return self._conv2d_native(x)
        # Activation checkpointing happens per shot chunk (see
        # _run_shot_chunk), not per layer: a layer-level checkpoint would
        # re-materialize every chunk of the layer at once during backward.
        return self._forward_optical(x, backend)


def configure_jtc_runtime(module: nn.Module, config: AppConfig) -> None:
    """Apply validated execution overrides by exact model module name."""
    layers = {
        name: layer
        for name, layer in module.named_modules()
        if isinstance(layer, JTCConv2d)
    }
    unknown = set(config.jtc_runtime_overrides) - set(layers)
    if unknown:
        raise ValueError(
            f"Runtime overrides name unknown JTC layers: {sorted(unknown)}"
        )
    for name, settings in config.jtc_runtime_overrides.items():
        layer = layers[name]
        layer.max_jtc_shots = settings.get("max_shots", layer.max_jtc_shots)
        layer.checkpoint_policy = settings.get(
            "checkpoint_policy", layer.checkpoint_policy
        )


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
