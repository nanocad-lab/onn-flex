import math
from dataclasses import dataclass, field

import yaml


def load_app_config_from_yaml(path: str) -> "AppConfig":
    """Load AppConfig from a plain YAML mapping."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    if raw is None:
        return AppConfig()
    if not isinstance(raw, dict):
        raise TypeError(f"Expected mapping in config YAML, got {type(raw).__name__}")
    valid = {k: v for k, v in raw.items() if k in AppConfig.__dataclass_fields__}
    return AppConfig(**valid)


@dataclass
class AppConfig:
    """Application configuration class."""

    @classmethod
    def from_yaml(cls, path: str) -> "AppConfig":
        """Load an AppConfig from YAML."""
        return load_app_config_from_yaml(path)

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

    # MRM power parameters
    mrm_power_distortion_strength: float = 0.0
    mrm_power_data_path: str = "./component_data/mrm_pwr_w_sim_data.csv"
    mrm_power_polyfit_order: int | None = None
    # Optional gain applied to the MRM power output (Watts). This is a simple
    # global scaling knob to explore dynamic-range recovery effects.
    mrm_power_gain: float = 1.0

    # MRM phase parameters
    mrm_phase_distortion_strength: float = 0.0
    mrm_phase_data_path: str = "./component_data/mrm_phase_sim_data.csv"
    mrm_phase_polyfit_order: int | None = None

    # LER process variation
    ler_std_dev: float = 0.0

    # Noise terms
    # - laser_rin_db: per-shot global laser relative intensity noise (RIN) in dB.
    #   This is applied as a single scale factor shared across all MRM channels
    #   in a shot; LER splitter variation then distributes the noisy power unevenly.
    # - pd_noise_w: additive PD input-referred noise (Watts), sampled independently
    #   per input sample at each PD evaluation.
    laser_rin_db: float | None = None
    pd_noise_w: float = 0.0
    # PD input clamp (Watts). Historically we clamped the PD input power to
    # the calibration range [1e-6, 1e-5] W to avoid extrapolating the fitted
    # polynomial too far. Set either bound to null to disable it.
    pd_input_clamp_min_w: float | None = 1e-6
    pd_input_clamp_max_w: float | None = 1e-5

    conv_method: str = "patch"  # "patch" or "dot_product" or "tile"

    # Convolution backend selection
    # Options: 'pytorch', 'fourier', 'jtc_emulation'
    # - 'pytorch': Standard PyTorch conv2d
    # - 'fourier': FFT-based convolution (software JTC)
    # - 'jtc_emulation': Full hardware JTC emulation pipeline
    conv_backend: str | None = "jtc_emulation"

    fourier_plane_bits: int | None = 6  # Bits for Fourier plane quantization

    # Quantizer selection (single QAT block for all steps)
    # Options: 'ste_clipped', 'ste_maxscale', 'ios', 'mad', 'mph', 'pwl'
    quantizer: str = "ste_clipped"

    # ------------------------------------------------------------------
    #  Training / model-related CLI overrides
    # ------------------------------------------------------------------
    num_identical_layers: int = 5
    num_epochs: int = 20
    learning_rate: float = 1e-3
    batch_size: int = 128
    eval_only: bool = False
    pretrained_weights: str = ""
    run_pretrain_tests: bool = True
    pretrain_tests_only: bool = False
    loss: float = 0.96
    # Debug helpers to shorten smoke tests without changing datasets
    max_train_batches: int | None = None
    max_eval_batches: int | None = None
    # Whether to normalize activations after each identical block
    normalize_blocks: bool = False
    # Normalization mode for the built-in `x /= max(x)` steps in FFTConvNet.
    # Options: "max", "second_largest" (robust to single outliers).
    max_norm_mode: str = "max"

    def __post_init__(self):
        """Validate configuration parameters."""
        valid_backends = ["pytorch", "fourier", "jtc_emulation"]
        if self.conv_backend is not None and self.conv_backend not in valid_backends:
            raise ValueError(
                f"Invalid conv_backend '{self.conv_backend}'. "
                f"Must be one of {valid_backends}"
            )

        # Normalize/validate noise params (allow YAML null -> None)
        if self.laser_rin_db is not None:
            self.laser_rin_db = float(self.laser_rin_db)
            if not math.isfinite(self.laser_rin_db):
                raise ValueError("laser_rin_db must be finite or null")
        self.pd_noise_w = float(self.pd_noise_w or 0.0)
        if self.pd_noise_w < 0:
            raise ValueError("pd_noise_w must be >= 0")

        # PD clamp bounds (allow YAML null -> None)
        self.pd_input_clamp_min_w = (
            None
            if getattr(self, "pd_input_clamp_min_w", 1e-6) is None
            else float(getattr(self, "pd_input_clamp_min_w"))
        )
        self.pd_input_clamp_max_w = (
            None
            if getattr(self, "pd_input_clamp_max_w", 1e-5) is None
            else float(getattr(self, "pd_input_clamp_max_w"))
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
        if self.max_train_batches is not None and self.max_train_batches <= 0:
            self.max_train_batches = None
        if self.max_eval_batches is not None and self.max_eval_batches <= 0:
            self.max_eval_batches = None

        self.mrm_power_gain = float(getattr(self, "mrm_power_gain", 1.0) or 1.0)
        if not math.isfinite(self.mrm_power_gain) or self.mrm_power_gain <= 0:
            raise ValueError("mrm_power_gain must be finite and > 0")

        self.max_norm_mode = str(getattr(self, "max_norm_mode", "max") or "max")
        if self.max_norm_mode not in {"max", "second_largest"}:
            raise ValueError("max_norm_mode must be 'max' or 'second_largest'")
