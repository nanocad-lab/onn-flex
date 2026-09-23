from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Integral

import yaml


def load_app_config_from_yaml(path: str) -> "AppConfig":
    """Load AppConfig from a plain YAML mapping."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    if raw is None:
        return AppConfig()
    if not isinstance(raw, dict):
        raise TypeError(f"Expected mapping in config YAML, got {type(raw).__name__}")
    valid_fields = set(AppConfig.__dataclass_fields__)
    unknown = sorted(str(k) for k in raw if k not in valid_fields)
    if unknown:
        raise ValueError(f"Unknown config field(s): {', '.join(unknown)}")
    return AppConfig(**raw)


@dataclass
class AppConfig:
    """Application configuration class."""

    # These control the config loading/saving but aren't part of the actual app config
    config_file: str | None = None

    # These values will be overridable from CLI or YAML
    output_dir: str = "./output"

    # JTC parameters
    input_length: int = 8  # length of input signal
    kernel_length: int = 8  # length of kernel/weight
    output_length: int | None = None  # length of output (auto-calculated if None)
    jtc_separation: int = 0  # separation between kernel and signal
    jtc_total_field: int = 16  # total size of the JTC plane (lens size)
    # Lens model (optional)
    # Models a unitary lens operator using a 1-D Legendre basis and coefficients.
    # Sweeping `lens_distortion_strength` in [0,1] interpolates between ideal lens
    # (all-zero coefficients) and the configured `lens_coefs`.
    lens_distortion_strength: float = 0.0
    lens_legendre_order: int = 2
    lens_coefs: list[float] = field(default_factory=lambda: [-2.772300, -1.266519])

    # Quantization parameters
    dac_bits: int | None = 4
    adc_bits: int | None = 6
    scale_output: str = "none"
    # Backward surrogate for hard converter saturation. Forward/inference always
    # clamps to [0,1]; "pwl" uses true clamp gradients, "mad" uses a decaying
    # nonzero gradient outside the rails.
    converter_clamp_grad: str = "pwl"
    # Linear reference used when distortion_strength == 0 for characterized transfer
    # functions. "endpoint" preserves the sampled endpoint range, "least_squares"
    # uses the unconstrained linear regression fit, and "midpoint_ls" uses the
    # regression slope anchored at the midpoint of the sampled endpoint range.
    transfer_linearization: str = "endpoint"

    # Driver parameters
    driver_distortion_strength: float = 0.0
    driver_distortion_data_path: str = "./component_data/driver_sim_data.csv"
    driver_distortion_polyfit_order: int | None = None

    pd_distortion_data_path: str = "./component_data/pd_sim_data.csv"
    pd_distortion_polyfit_order: int | None = None
    pd_distortion_strength: float = 0.0

    tia_distortion_data_path: str = "./component_data/tia_sim_data.csv"
    tia_distortion_polyfit_order: int | None = None
    tia_distortion_strength: float = 0.0
    # DC bias added to the TIA input (PD-output units) before the transfer —
    # physically a PD->TIA bias/offset stage that sets the TIA operating
    # point. Needed because the PD output at correlation-plane intensities
    # (~0.068 at dark) sits BELOW the characterized TIA input domain
    # (0.105–0.438, tia_sim_data.csv): unbiased, the TIA is cut off and any
    # nonzero strength collapses the layer to a threshold detector. The
    # offset-null/CDS output stage removes the bias, so it is a no-op at
    # strength 0. ~0.08 puts dark at ~0.15 (linear region).
    tia_input_bias: float = 0.0

    # MRM amplitude parameters
    mrm_amplitude_distortion_strength: float = 0.0
    mrm_amplitude_data_path: str = "./component_data/mrm_amp_w_sim_data.csv"
    mrm_amplitude_polyfit_order: int | None = None
    # Optional laser source power gain. The MRM transfer outputs field
    # amplitude, so source power gain is applied as sqrt(gain) on the field.
    laser_power_gain: float = 1.0

    # MRM phase parameters
    mrm_phase_distortion_strength: float = 0.0
    mrm_phase_data_path: str = "./component_data/mrm_phase_sim_data.csv"
    mrm_phase_polyfit_order: int | None = None

    # LER process variation
    ler_std_dev: float = 0.0

    # Noise terms
    # - laser_rin_db: per-shot global laser relative intensity noise (RIN) in dB.
    #   This is applied as a single scale factor shared across all MRM channels
    #   in a shot. It is defined on optical intensity/power; MRM transfer
    #   outputs field amplitude, so the field receives sqrt(scale).
    # - pd_noise_w: additive PD input-referred noise (Watts), sampled independently
    #   per input sample at each PD evaluation.
    laser_rin_db: float | None = None
    pd_noise_w: float = 0.0
    # PD input clamp (Watts). The default is an upper soft guard at the sampled
    # PD calibration max; the lower side is left unconstrained so dark/low-power
    # bins are not forced up to the calibration floor.
    pd_input_clamp_min_w: float | None = None
    pd_input_clamp_max_w: float | None = 1e-5
    pd_input_clamp_mode: str = "soft"
    pd_input_soft_clamp_width_w: float = 1e-6
    # Optional training penalty for detector power outside the characterized PD fit
    # domain. This is a soft constraint; the physical PD input clamp above still
    # controls the transfer-function input used in forward inference.
    pd_range_regularization_weight: float = 0.0
    pd_range_regularization_min_w: float | None = None
    pd_range_regularization_max_w: float | None = None

    conv_method: str = "patch"  # "patch" or "dot_product" or "tile"

    # Convolution backend selection
    # Options: pytorch, jtc_ideal, jtc_emulation, jtc_analytic, jtc_analog_fourier
    # Future-design standard: jtc_analog_fourier with an analog Fourier plane
    # and one final ADC stage per shot (fourier_plane_bits=None; see README).
    # - 'pytorch': Standard PyTorch conv2d
    # - 'jtc_ideal': Ideal JTC optical model without hardware distortions
    # - 'jtc_emulation': Full hardware JTC emulation pipeline
    # - 'jtc_analytic': Exact closed form of the JTC with an analog (never
    #   quantized/clamped) Fourier plane and output ADC gain-scaled to the
    #   useful correlation lags. Requires affine PD/TIA, clean valid lags,
    #   and no lens/MRM phase distortion. Input amplitude distortion and
    #   ADC-input noise are supported (JTCConv2d models only).
    conv_backend: str = "jtc_emulation"
    # Physical plane geometry for JTCConv2d:
    # - auto: row uses field 256, separation=row_width-kernel_width;
    #   dot grows the field to fit its flattened apertures and guard.
    # - config: fixed jtc_total_field / jtc_separation for either mapping.
    # Analog mappings use one row per shot; analytic requires clean lags.
    jtc_rowwise_geometry: str = "auto"
    # Physical mapping is independent of the numerical solver.
    jtc_shot_mapping: str = "row"  # row or flattened dot; never an implicit fallback
    # Continuous interstage component. See RemodulationSpec for YAML keys.
    jtc_remodulation: dict = field(default_factory=dict)

    # Output ADC gain handling for 'jtc_analytic':
    # - 'per_shot': divide each shot's useful outputs by their own max (AGC)
    # - 'calibrated': per-layer running estimate of the useful-output max,
    #   updated during training, frozen for eval
    # - 'calibrate_freeze': like 'calibrated' for the first
    #   jtc_gain_freeze_batches layer-forwards, then frozen — a stationary
    #   per-layer gain with hands-free setup (A/B: stationary gains train
    #   far better than adaptive ones)
    # - 'fixed': multiply by jtc_output_gain from the config
    jtc_output_gain_mode: str = "per_shot"
    jtc_output_gain: float = 1.0
    # Count full layer forwards, independent of shot chunks/checkpointing.
    # Every shot uses the gain at forward entry; the global batch maximum
    # (across DDP ranks) updates the EMA for the next forward.
    jtc_gain_freeze_batches: int = 50
    # Epoch-wise gain ranging for calibrate_freeze: every N epochs the
    # per-layer gain calibration re-opens for jtc_gain_freeze_batches, then
    # freezes again — stationary within an epoch (trainable), tracking
    # weight growth between epochs. 0 disables (freeze once, legacy).
    jtc_gain_recal_epochs: int = 0
    # Calibrated gains target full-scale/headroom instead of full-scale:
    # gain = 1 / (headroom * observed_max). Railing costs far more than the
    # equivalent loss of ADC codes (see the g8922 vs g2728 vs g1364 A/Bs).
    jtc_gain_headroom: float = 1.0
    # ADC-input-referred noise for every case-2 readout: Gaussian noise at
    # the ADC input with rms = 10^(-SNR/20) of full scale, independent of the
    # gain code. This is an output noise budget, not a fixed pre-PGA noise
    # source amplified by the gain. None disables (ideal front end).
    jtc_frontend_snr_db: float | None = None
    # Carrier stop-band for the full-transfer fourier path: null this many
    # bins on each side of DC in the JPS before the detector transfer
    # (physically a carrier-suppression spot / DC block). Keeps the huge
    # autocorrelation pedestal out of the detector's fitted input domain so
    # nonlinear PD/TIA curves act on the correlation fringes they were
    # characterized for. 0 disables. A nonzero value requires
    # jtc_analog_fourier with jtc_fourier_closed_form=false.
    jtc_carrier_stopband_bins: int = 0
    # GEMM precision for the jtc_analytic correlation. bfloat16 has fp32
    # range (no overflow) but ~0.4% mantissa precision, which can flip ADC
    # codes near quantization boundaries — comparable to an analog noise
    # floor, but not bitwise-faithful to the fp32 model.
    jtc_analytic_gemm_dtype: str = "float32"
    # jtc_analog_fourier evaluation strategy. True (default) uses the
    # closed-form lag decomposition (cross + mirror + autocorrelation terms
    # aliased mod-field), which is exact for the all-affine plane and runs at
    # GEMM speed like jtc_analytic. False selects per-shot FFT/DFT simulation
    # with full transfers; jtc_fourier_lag_gemm chooses its numerical solver.
    jtc_fourier_closed_form: bool = True
    # Full-transfer fast path: aperture-pixel DFT GEMMs compute the JPS,
    # followed by raw analog PD/TIA and a second DFT at extraction bins.
    # Supports nonlinear transfers, stopband, lens and MRM phase. False
    # selects the FFT plane implementation of the same physics.
    jtc_fourier_lag_gemm: bool = True

    fourier_plane_bits: int | None = 6  # Bits for Fourier plane quantization

    # ------------------------------------------------------------------
    #  Training / model-related CLI overrides
    # ------------------------------------------------------------------
    # Training dataset. MNIST images are padded 28->32 and replicated to 3
    # channels so every model architecture works unchanged on both datasets.
    dataset: str = "cifar10"
    model_arch: str = "fftconvnet"
    num_identical_layers: int = 5
    num_epochs: int = 20
    learning_rate: float = 1e-3
    # Optimizer recipe. The historical default is AdamW + cosine; the tuned
    # high-accuracy recipe for resnet references is SGD + momentum with
    # label smoothing and a linear warmup into the cosine schedule.
    optimizer: str = "adamw"
    momentum: float = 0.9
    weight_decay: float = 0.0
    label_smoothing: float = 0.0
    lr_warmup_epochs: int = 0
    # Max gradient norm; 0 disables clipping. STE-surrogate gradients
    # through quantized optical paths need clipping at SGD learning rates.
    grad_clip_norm: float = 0.0
    batch_size: int = 128
    seed: int | None = None
    eval_only: bool = False
    pretrained_weights: str = ""
    resume_checkpoint: str = ""
    checkpoint_interval: int = 0
    checkpoint_time_interval_minutes: float = 0.0
    enable_ddp: bool = True
    run_pretrain_tests: bool = True
    pretrain_tests_only: bool = False
    loss: float = 0.96
    # Debug helpers to shorten smoke tests without changing datasets
    max_train_batches: int | None = None
    max_eval_batches: int | None = None
    dataloader_num_workers: int = 8
    dataloader_pin_memory: bool = True
    dataloader_persistent_workers: bool = True
    dataloader_prefetch_factor: int | None = 2
    # Print memory snapshots every N training batches; 0 disables batch reports.
    memory_report_interval: int = 0
    # Whether to normalize activations after each identical block
    normalize_blocks: bool = False
    # Batch independent JTC shots to reduce Python/CUDA launch overhead.
    enable_jtc_batched_fast_path: bool = True
    # Wrap the JTC paired-shot pipeline in torch.compile (requires triton).
    # Off by default; enable per run for a large training speedup at the cost
    # of a one-time compile at startup.
    compile_jtc: bool = False
    # Fuse CUDA transfer polynomials and scalar-gain ADC readout. Independent
    # of whole-shot compilation; CPU uses the ordinary PyTorch expressions.
    jtc_fuse_pointwise: bool = True
    # Evaluate train-set metrics every N epochs (a full extra pass over the
    # training set). 1 keeps the historical every-epoch behavior; 0 disables.
    # Test-set evaluation and best-checkpoint tracking always run every epoch.
    train_eval_interval: int = 1
    # Show tqdm training progress bars. Disable for quieter logs.
    show_progress: bool = True
    # Run a second inference pass with all distortion strengths set to 1.0 after
    # training. Disable for memory-heavy distributed ResNet runs.
    run_full_strength_inference: bool = True
    # Only set this when every JTCConv2d input is known nonnegative.
    jtc_assume_nonnegative_input: bool = False
    # Maximum optical shots processed in one JTCConv2d chunk.
    jtc_max_shots: int = 65536
    # Recompute JTCConv2d optical activations during backward, one shot chunk
    # at a time: peak memory is then bounded by jtc_max_shots worth of
    # pipeline intermediates instead of a whole layer's.
    enable_jtc_activation_checkpointing: bool = False
    # "spectra" retains compact aperture preparation graphs and checkpoints
    # expanded detection planes. Other solvers still checkpoint whole chunks.
    jtc_checkpoint_policy: str = "recompute"
    # Exact model module names -> {max_shots: int, checkpoint_policy: str}.
    # Execution only; the physical shot plan and training batch stay fixed.
    jtc_runtime_overrides: dict = field(default_factory=dict)
    # Normalization mode for the built-in `x /= max(x)` steps in FFTConvNet.
    # Options: "max", "second_largest" (robust to single outliers).
    max_norm_mode: str = "max"
    # Fixed scalar applied only to FFTConvNet input images before the first
    # optical stem layer. This lets the raw pixel-domain stem use a different
    # optical operating point than BN-conditioned internal activations.
    fftconvnet_input_gain: float = 1.0

    def __post_init__(self):
        """Validate configuration parameters."""
        from onn_remodulation import RemodulationSpec

        if self.jtc_shot_mapping not in {"row", "dot"}:
            raise ValueError("jtc_shot_mapping must be 'row' or 'dot'")
        if not isinstance(self.jtc_remodulation, dict):
            raise ValueError("jtc_remodulation must be a mapping")
        RemodulationSpec(**self.jtc_remodulation)
        valid_backends = [
            "pytorch",
            "jtc_ideal",
            "jtc_emulation",
            "jtc_analytic",
            "jtc_analog_fourier",
        ]
        if self.conv_backend not in valid_backends:
            raise ValueError(
                f"Invalid conv_backend '{self.conv_backend}'. "
                f"Must be one of {valid_backends}"
            )
        self.jtc_rowwise_geometry = str(self.jtc_rowwise_geometry or "auto").lower()
        if self.jtc_rowwise_geometry not in {"auto", "config"}:
            raise ValueError("jtc_rowwise_geometry must be 'auto' or 'config'")
        self.jtc_output_gain_mode = str(self.jtc_output_gain_mode or "per_shot").lower()
        if self.jtc_output_gain_mode not in {
            "per_shot",
            "calibrated",
            "calibrate_freeze",
            "fixed",
        }:
            raise ValueError(
                "jtc_output_gain_mode must be 'per_shot', 'calibrated', "
                "'calibrate_freeze', or 'fixed'"
            )
        self.jtc_gain_freeze_batches = int(self.jtc_gain_freeze_batches or 0)
        if self.jtc_gain_freeze_batches < 1:
            raise ValueError("jtc_gain_freeze_batches must be >= 1")
        self.jtc_output_gain = float(self.jtc_output_gain)
        if not math.isfinite(self.jtc_output_gain) or self.jtc_output_gain <= 0:
            raise ValueError("jtc_output_gain must be finite and > 0")
        self.jtc_gain_recal_epochs = int(self.jtc_gain_recal_epochs or 0)
        if self.jtc_gain_recal_epochs < 0:
            raise ValueError("jtc_gain_recal_epochs must be >= 0")
        self.jtc_gain_headroom = float(self.jtc_gain_headroom)
        if not math.isfinite(self.jtc_gain_headroom) or self.jtc_gain_headroom < 1.0:
            raise ValueError("jtc_gain_headroom must be finite and >= 1")
        self.jtc_analytic_gemm_dtype = str(
            self.jtc_analytic_gemm_dtype or "float32"
        ).lower()
        if self.jtc_analytic_gemm_dtype not in {"float32", "bfloat16"}:
            raise ValueError("jtc_analytic_gemm_dtype must be 'float32' or 'bfloat16'")
        self.model_arch = str(self.model_arch or "fftconvnet").lower()
        if self.model_arch not in {"fftconvnet", "resnet11", "resnet18"}:
            raise ValueError(
                "model_arch must be 'fftconvnet', 'resnet11', or 'resnet18'"
            )
        self.dataset = str(self.dataset or "cifar10").lower()
        if self.dataset not in {"cifar10", "mnist"}:
            raise ValueError("dataset must be 'cifar10' or 'mnist'")
        self.optimizer = str(self.optimizer or "adamw").lower()
        if self.optimizer not in {"adamw", "sgd"}:
            raise ValueError("optimizer must be 'adamw' or 'sgd'")
        self.momentum = float(self.momentum)
        self.weight_decay = float(self.weight_decay)
        self.label_smoothing = float(self.label_smoothing)
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1)")
        self.lr_warmup_epochs = int(self.lr_warmup_epochs or 0)
        if self.lr_warmup_epochs < 0:
            raise ValueError("lr_warmup_epochs must be >= 0")
        self.grad_clip_norm = float(self.grad_clip_norm or 0.0)
        if self.grad_clip_norm < 0:
            raise ValueError("grad_clip_norm must be >= 0")
        # Normalize/validate noise params (allow YAML null -> None)
        if self.laser_rin_db is not None:
            self.laser_rin_db = float(self.laser_rin_db)
            if not math.isfinite(self.laser_rin_db):
                raise ValueError("laser_rin_db must be finite or null")
        self.pd_noise_w = float(self.pd_noise_w or 0.0)
        if not math.isfinite(self.pd_noise_w) or self.pd_noise_w < 0:
            raise ValueError("pd_noise_w must be finite and >= 0")

        for name in (
            "driver_distortion_strength",
            "mrm_amplitude_distortion_strength",
            "mrm_phase_distortion_strength",
            "pd_distortion_strength",
            "tia_distortion_strength",
            "lens_distortion_strength",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
            setattr(self, name, value)

        # PD clamp bounds (allow YAML null -> None)
        self.pd_input_clamp_min_w = (
            None
            if self.pd_input_clamp_min_w is None
            else float(self.pd_input_clamp_min_w)
        )
        self.pd_input_clamp_max_w = (
            None
            if self.pd_input_clamp_max_w is None
            else float(self.pd_input_clamp_max_w)
        )
        if self.pd_input_clamp_min_w is not None:
            if (
                not math.isfinite(self.pd_input_clamp_min_w)
                or self.pd_input_clamp_min_w < 0
            ):
                raise ValueError("pd_input_clamp_min_w must be finite and >= 0 or null")
        if self.pd_input_clamp_max_w is not None:
            if (
                not math.isfinite(self.pd_input_clamp_max_w)
                or self.pd_input_clamp_max_w < 0
            ):
                raise ValueError("pd_input_clamp_max_w must be finite and >= 0 or null")
        if (
            self.pd_input_clamp_min_w is not None
            and self.pd_input_clamp_max_w is not None
            and self.pd_input_clamp_max_w <= self.pd_input_clamp_min_w
        ):
            raise ValueError("pd_input_clamp_max_w must be > pd_input_clamp_min_w")
        self.pd_input_clamp_mode = str(self.pd_input_clamp_mode or "hard").lower()
        if self.pd_input_clamp_mode not in {"hard", "soft"}:
            raise ValueError("pd_input_clamp_mode must be 'hard' or 'soft'")
        self.pd_input_soft_clamp_width_w = float(
            self.pd_input_soft_clamp_width_w or 0.0
        )
        if (
            not math.isfinite(self.pd_input_soft_clamp_width_w)
            or self.pd_input_soft_clamp_width_w < 0
        ):
            raise ValueError("pd_input_soft_clamp_width_w must be finite and >= 0")
        self.pd_range_regularization_weight = float(
            self.pd_range_regularization_weight or 0.0
        )
        if (
            not math.isfinite(self.pd_range_regularization_weight)
            or self.pd_range_regularization_weight < 0
        ):
            raise ValueError("pd_range_regularization_weight must be finite and >= 0")
        self.pd_range_regularization_min_w = (
            None
            if self.pd_range_regularization_min_w is None
            else float(self.pd_range_regularization_min_w)
        )
        self.pd_range_regularization_max_w = (
            None
            if self.pd_range_regularization_max_w is None
            else float(self.pd_range_regularization_max_w)
        )
        if self.pd_range_regularization_min_w is not None:
            if (
                not math.isfinite(self.pd_range_regularization_min_w)
                or self.pd_range_regularization_min_w < 0
            ):
                raise ValueError(
                    "pd_range_regularization_min_w must be finite and >= 0 or null"
                )
        if self.pd_range_regularization_max_w is not None:
            if (
                not math.isfinite(self.pd_range_regularization_max_w)
                or self.pd_range_regularization_max_w < 0
            ):
                raise ValueError(
                    "pd_range_regularization_max_w must be finite and >= 0 or null"
                )
        if (
            self.pd_range_regularization_min_w is not None
            and self.pd_range_regularization_max_w is not None
            and self.pd_range_regularization_max_w <= self.pd_range_regularization_min_w
        ):
            raise ValueError(
                "pd_range_regularization_max_w must be > pd_range_regularization_min_w"
            )

        # Lens params
        self.lens_distortion_strength = float(self.lens_distortion_strength or 0.0)
        self.lens_legendre_order = int(self.lens_legendre_order or 0)

        if self.lens_legendre_order < 0:
            raise ValueError("lens_legendre_order must be >= 0")
        if self.lens_coefs is None:
            self.lens_coefs = []
        elif isinstance(self.lens_coefs, (int, float, str)):
            self.lens_coefs = [self.lens_coefs]
        else:
            self.lens_coefs = list(self.lens_coefs)
        self.lens_coefs = [float(value) for value in self.lens_coefs]

        # Treat non-positive debug limits as "no limit"
        self.seed = None if self.seed is None else int(self.seed)
        if self.max_train_batches is not None and self.max_train_batches <= 0:
            self.max_train_batches = None
        if self.max_eval_batches is not None and self.max_eval_batches <= 0:
            self.max_eval_batches = None
        self.dataloader_num_workers = int(self.dataloader_num_workers or 0)
        if self.dataloader_num_workers < 0:
            raise ValueError("dataloader_num_workers must be >= 0")
        self.dataloader_pin_memory = bool(self.dataloader_pin_memory)
        self.dataloader_persistent_workers = bool(self.dataloader_persistent_workers)
        if self.dataloader_num_workers == 0:
            self.dataloader_persistent_workers = False
            self.dataloader_prefetch_factor = None
        elif self.dataloader_prefetch_factor is not None:
            self.dataloader_prefetch_factor = int(self.dataloader_prefetch_factor)
            if self.dataloader_prefetch_factor < 1:
                self.dataloader_prefetch_factor = None
        self.memory_report_interval = int(self.memory_report_interval or 0)
        if self.memory_report_interval < 0:
            raise ValueError("memory_report_interval must be >= 0")
        self.dac_bits = self._validate_quant_bits(self.dac_bits, "dac_bits")
        self.adc_bits = self._validate_quant_bits(self.adc_bits, "adc_bits")
        self.fourier_plane_bits = self._validate_quant_bits(
            self.fourier_plane_bits,
            "fourier_plane_bits",
        )
        self.converter_clamp_grad = str(self.converter_clamp_grad or "pwl").lower()
        if self.converter_clamp_grad not in {"pwl", "mad"}:
            raise ValueError("converter_clamp_grad must be 'pwl' or 'mad'")
        self.transfer_linearization = str(
            self.transfer_linearization or "endpoint"
        ).lower()
        if self.transfer_linearization not in {
            "endpoint",
            "least_squares",
            "midpoint_ls",
        }:
            raise ValueError(
                "transfer_linearization must be 'endpoint', 'least_squares', "
                "or 'midpoint_ls'"
            )
        self.checkpoint_interval = int(self.checkpoint_interval or 0)
        if self.checkpoint_interval < 0:
            raise ValueError("checkpoint_interval must be >= 0")
        self.checkpoint_time_interval_minutes = float(
            self.checkpoint_time_interval_minutes or 0.0
        )
        if self.checkpoint_time_interval_minutes < 0:
            raise ValueError("checkpoint_time_interval_minutes must be >= 0")
        self.tia_input_bias = float(self.tia_input_bias or 0.0)
        if not math.isfinite(self.tia_input_bias):
            raise ValueError("tia_input_bias must be finite")
        self.pretrained_weights = str(self.pretrained_weights or "")
        self.resume_checkpoint = str(self.resume_checkpoint or "")
        self.enable_ddp = bool(self.enable_ddp)
        self.jtc_assume_nonnegative_input = bool(self.jtc_assume_nonnegative_input)
        self.jtc_max_shots = int(self.jtc_max_shots or 0)
        if self.jtc_max_shots < 1:
            raise ValueError("jtc_max_shots must be >= 1")
        self.enable_jtc_activation_checkpointing = bool(
            self.enable_jtc_activation_checkpointing
        )
        self.jtc_fuse_pointwise = bool(self.jtc_fuse_pointwise)
        policies = {"recompute", "spectra"}
        if self.jtc_checkpoint_policy not in policies:
            raise ValueError("jtc_checkpoint_policy must be 'recompute' or 'spectra'")
        if not isinstance(self.jtc_runtime_overrides, dict):
            raise ValueError("jtc_runtime_overrides must be a mapping of module names")
        for name, settings in self.jtc_runtime_overrides.items():
            if not isinstance(name, str) or not name or not isinstance(settings, dict):
                raise ValueError("Runtime overrides require module names and mappings")
            if not settings or set(settings) - {"max_shots", "checkpoint_policy"}:
                raise ValueError(f"Unknown or empty runtime override for {name}")
            if "max_shots" in settings and (
                type(settings["max_shots"]) is not int or settings["max_shots"] < 1
            ):
                raise ValueError(
                    f"Runtime max_shots for {name} must be a positive integer"
                )
            if settings.get("checkpoint_policy", "recompute") not in policies:
                raise ValueError(f"Invalid runtime checkpoint_policy for {name}")
        self.run_full_strength_inference = bool(self.run_full_strength_inference)
        self.compile_jtc = bool(self.compile_jtc)
        self.train_eval_interval = int(self.train_eval_interval or 0)
        if self.train_eval_interval < 0:
            raise ValueError("train_eval_interval must be >= 0")

        self.laser_power_gain = float(
            1.0 if self.laser_power_gain is None else self.laser_power_gain
        )
        if not math.isfinite(self.laser_power_gain) or self.laser_power_gain <= 0:
            raise ValueError("laser_power_gain must be finite and > 0")

        self.max_norm_mode = str(self.max_norm_mode or "max")
        if self.max_norm_mode not in {"max", "second_largest"}:
            raise ValueError("max_norm_mode must be 'max' or 'second_largest'")

        self.fftconvnet_input_gain = float(self.fftconvnet_input_gain)
        if (
            not math.isfinite(self.fftconvnet_input_gain)
            or self.fftconvnet_input_gain <= 0
        ):
            raise ValueError("fftconvnet_input_gain must be finite and > 0")

    @staticmethod
    def _validate_quant_bits(value: int | None, field_name: str) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"{field_name} must be an integer >= 1 or null")
        bits = int(value)
        if bits < 1:
            raise ValueError(f"{field_name} must be >= 1 or null")
        return bits
