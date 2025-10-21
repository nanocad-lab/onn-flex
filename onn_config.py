from dataclasses import dataclass
from typing import Optional


@dataclass
class AppConfig:
    """Application configuration class."""

    # These control the config loading/saving but aren't part of the actual app config
    config_file: Optional[str] = None

    # These values will be overridable from CLI or YAML
    output_dir: str = "./output"

    # JTC parameters
    jtc_half_size: int = 8  # half of the total size of the JTC
    jtc_separation: int = 0  # separation between kernel and weight
    jtc_total_field: int = 16  # total size of the JTC plane

    # Quantization parameters
    dac_bits: int = 4
    adc_bits: int = 6
    scale_output: str = "none"

    # Driver parameters
    driver_distortion_strength: float = 0.0
    driver_distortion_data_path: str = "./data/driver_distortion_data.csv"
    driver_distortion_polyfit_order: Optional[int] = None

    # PD/TIA parameters
    pd_tia_distortion_strength: float = 0.0
    pd_tia_distortion_data_path: str = "./data/pd_tia_distortion_data.csv"
    pd_tia_distortion_polyfit_order: Optional[int] = None

    pd_distortion_data_path: str = "./data/pd_distortion_data.csv"
    pd_distortion_polyfit_order: Optional[int] = None
    pd_distortion_strength: float = 0.0

    tia_distortion_data_path: str = "./data/tia_distortion_data.csv"
    tia_distortion_polyfit_order: Optional[int] = None
    tia_distortion_strength: float = 0.0

    # MRM power parameters
    mrm_power_distortion_strength: float = 0.0
    mrm_power_data_path: str = "./data/mrm_power_data.csv"
    mrm_power_polyfit_order: Optional[int] = None

    # MRM phase parameters
    mrm_phase_distortion_strength: float = 0.0
    mrm_phase_data_path: str = "./data/mrm_phase_data.csv"
    mrm_phase_polyfit_order: Optional[int] = None

    # LER process variation
    ler_std_dev: float = 0.0

    conv_method: str = "patch"  # "patch" or "dot_product" or "tile"
    use_pytorch_conv: bool = (
        False  # Use PyTorch conv2d with 'same' padding instead of JTC
    )
    use_fourier_conv: bool = False  # Use FFT-based convolution instead of JTC
    quantize_fourier_plane: bool = False  # Quantize FFT plane (real/imag)
    fourier_plane_bits: int = 6  # Bits for Fourier plane quantization

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
    # Whether to normalize activations after each identical block
    normalize_blocks: bool = False
