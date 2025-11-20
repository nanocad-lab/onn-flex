from dataclasses import dataclass
from typing import Optional

import yaml


@dataclass
class AppConfig:
    """Application configuration class."""

    # These control the config loading/saving but aren't part of the actual app config
    config_file: Optional[str] = None

    # These values will be overridable from CLI or YAML
    output_dir: str = "./output"

    # JTC parameters
    input_length: int = 8  # length of input signal
    kernel_length: int = 8  # length of kernel/weight
    output_length: Optional[int] = None  # length of output (auto-calculated if None)
    jtc_separation: int = 0  # separation between kernel and signal
    jtc_total_field: int = 16  # total size of the JTC plane (lens size)
    geometry_autopick_max_lens: int = 64  # cap when auto-searching geometry

    # Quantization parameters
    dac_bits: Optional[int] = 4
    adc_bits: Optional[int] = 6
    scale_output: str = "none"

    # Driver parameters
    driver_distortion_strength: float = 0.0
    driver_distortion_data_path: str = "./component_data/driver_sim_data.csv"
    driver_distortion_polyfit_order: Optional[int] = None

    # PD/TIA parameters
    pd_tia_distortion_strength: float = 0.0
    pd_tia_distortion_data_path: str = "./component_data/pd_tia_sim_data.csv"
    pd_tia_distortion_polyfit_order: Optional[int] = None

    pd_distortion_data_path: str = "./component_data/pd_sim_data.csv"
    pd_distortion_polyfit_order: Optional[int] = None
    pd_distortion_strength: float = 0.0

    tia_distortion_data_path: str = "./component_data/tia_sim_data.csv"
    tia_distortion_polyfit_order: Optional[int] = None
    tia_distortion_strength: float = 0.0

    # MRM power parameters
    mrm_power_distortion_strength: float = 0.0
    mrm_power_data_path: str = "./component_data/mrm_pwr_w_sim_data.csv"
    mrm_power_polyfit_order: Optional[int] = None

    # MRM phase parameters
    mrm_phase_distortion_strength: float = 0.0
    mrm_phase_data_path: str = "./component_data/mrm_phase_sim_data.csv"
    mrm_phase_polyfit_order: Optional[int] = None

    # LER process variation
    ler_std_dev: float = 0.0

    conv_method: str = "patch"  # "patch" or "dot_product" or "tile"

    # Convolution backend selection
    # Options: 'pytorch', 'fourier', 'jtc_emulation', 'jtc_fast'
    # - 'pytorch': Standard PyTorch conv2d
    # - 'fourier': FFT-based convolution (software JTC)
    # - 'jtc_emulation': Full hardware JTC emulation pipeline
    # - 'jtc_fast': Fourier backend + JTC distortions (Fast approximation)
    conv_backend: Optional[str] = "jtc_emulation"

    fourier_plane_bits: Optional[int] = 6  # Bits for Fourier plane quantization

    # Quantizer selection (single QAT block for all steps)
    # Options: 'ste_clipped', 'ste_maxscale', 'ios', 'mad', 'mph', 'pwl'
    quantizer: str = "ste_clipped"

    # Transfer-function simplification (optional)
    simplify_transfer_functions: bool = False
    transfer_fit_samples: int = 256
    transfer_fit_tolerance: float = 1e-4
    transfer_fit_max_degree: int = 10
    transfer_fit_min_r2: float = 0.999

    # Dataset / model selection
    dataset: str = "cifar10"  # Options: 'cifar10', 'mnist'
    model_arch: str = "fftconvnet"  # Options: 'fftconvnet', 'ftvgg11/13/16/19'
    vgg_variant: str = "vgg11"
    
    override_effective_stride: Optional[int] = None

    # ------------------------------------------------------------------
        # Training / model-related CLI overrides
        # ------------------------------------------------------------------
        num_identical_layers: int = 5
        num_epochs: int = 20
        pretrain_epochs: int = 0  # Number of epochs to pre-train using PyTorch backend
        learning_rate: float = 1e-3
        batch_size: int = 128
    eval_only: bool = False
    pretrained_weights: str = ""
    run_pretrain_tests: bool = True
    pretrain_tests_only: bool = False
    loss: float = 0.96
    # Whether to normalize activations after each identical block
    normalize_blocks: bool = False
    max_train_steps: Optional[int] = None
    max_eval_batches: Optional[int] = None
    skip_eval: bool = False
    dump_memory_summary: bool = False

    def __post_init__(self):
        """Validate configuration parameters."""
        valid_backends = ['pytorch', 'fourier', 'jtc_emulation', 'jtc_fast']
        if self.conv_backend is not None and self.conv_backend not in valid_backends:
            raise ValueError(
                f"Invalid conv_backend '{self.conv_backend}'. "
                f"Must be one of {valid_backends}"
            )
        valid_datasets = {'cifar10', 'mnist'}
        if self.dataset.lower() not in valid_datasets:
            raise ValueError(
                f"Invalid dataset '{self.dataset}'. Must be one of {sorted(valid_datasets)}"
            )
        valid_models = {'fftconvnet', 'ftvgg', 'ftvgg11', 'ftvgg13', 'ftvgg16', 'ftvgg19'}
        if self.model_arch.lower() not in valid_models:
            raise ValueError(
                f"Invalid model_arch '{self.model_arch}'. Must be one of {sorted(valid_models)}"
            )
        valid_variants = {'vgg3', 'vgg5', 'vgg9', 'vgg11', 'vgg13', 'vgg16', 'vgg19'}
        if self.vgg_variant.lower() not in valid_variants:
            raise ValueError(
                f"Invalid vgg_variant '{self.vgg_variant}'. Must be one of {sorted(valid_variants)}"
            )

    @classmethod
    def from_yaml(cls, path: str) -> "AppConfig":
        """Load an AppConfig from a YAML file."""
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        return cls(**(data or {}))
