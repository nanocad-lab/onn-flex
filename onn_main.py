import argparse
import os
from dataclasses import replace

import yaml

from onn_config import AppConfig, load_app_config_from_yaml
from onn_train import train_onn_model


def parse_initial_args() -> tuple[str | None, str | None]:
    """Parse just the config file and output directory from command line."""
    parser = argparse.ArgumentParser(
        description="Parse config location and output directory", add_help=False
    )
    parser.add_argument(
        "-c", "--config-file", type=str, help="Path to YAML config file"
    )
    parser.add_argument(
        "-o", "--output-dir", type=str, help="Output directory for saving config"
    )

    args, _ = parser.parse_known_args()
    return args.config_file, args.output_dir


def _str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    val = v.strip().lower()
    if val in {"yes", "true", "t", "1", "y"}:
        return True
    if val in {"no", "false", "f", "0", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def _optional_int(v: str | int | None) -> int | None:
    if v is None or isinstance(v, int):
        return v
    val = v.strip().lower()
    if val in {"none", "null"}:
        return None
    return int(v)


def _optional_float(v: str | float | None) -> float | None:
    if v is None or isinstance(v, float):
        return v
    val = v.strip().lower()
    if val in {"none", "null"}:
        return None
    return float(v)


def parse_cli_args(yaml_config: AppConfig) -> AppConfig:
    """Parse command line arguments that can override YAML values."""
    parser = argparse.ArgumentParser(description="Application with YAML config")

    parser.add_argument(
        "-c",
        "--config-file",
        type=str,
        default=yaml_config.config_file,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=yaml_config.output_dir,
        help="Output directory for saving config",
    )

    # JTC parameters
    parser.add_argument(
        "--input-length",
        type=int,
        default=yaml_config.input_length,
        help="Length of input signal",
    )
    parser.add_argument(
        "--kernel-length",
        type=int,
        default=yaml_config.kernel_length,
        help="Length of kernel/weight",
    )
    parser.add_argument(
        "--output-length",
        type=int,
        default=yaml_config.output_length,
        help="Length of output (auto-calculated if not specified)",
    )
    parser.add_argument(
        "--jtc-separation",
        type=int,
        default=yaml_config.jtc_separation,
        help="Separation between kernel and signal",
    )
    parser.add_argument(
        "--jtc-total-field",
        type=int,
        default=yaml_config.jtc_total_field,
        help="Total size of the JTC plane (lens size)",
    )
    parser.add_argument(
        "--lens-distortion-strength",
        type=float,
        default=yaml_config.lens_distortion_strength,
        help="Lens distortion strength in [0,1] (0=ideal, 1=full lens_coefs).",
    )
    parser.add_argument(
        "--lens-legendre-order",
        type=int,
        default=yaml_config.lens_legendre_order,
        help="Max Legendre order for the lens model (uses orders 1..N).",
    )
    parser.add_argument(
        "--lens-coefs",
        type=float,
        nargs="+",
        default=yaml_config.lens_coefs,
        help="Lens Legendre coefficients (for orders 1..lens_legendre_order).",
    )

    # Quantization parameters
    parser.add_argument(
        "--dac-bits",
        type=_optional_int,
        default=yaml_config.dac_bits,
        help="DAC bits for quantization; use 'none' to disable DAC quantization",
    )
    parser.add_argument(
        "--adc-bits",
        type=_optional_int,
        default=yaml_config.adc_bits,
        help="ADC bits for quantization; use 'none' to disable ADC quantization",
    )
    parser.add_argument(
        "--scale-output",
        type=str,
        default=yaml_config.scale_output,
        help="Where to scale output ['none', 'pd', 'adc']",
    )
    parser.add_argument(
        "--converter-clamp-grad",
        type=str,
        choices=["pwl", "mad"],
        default=yaml_config.converter_clamp_grad,
        help=(
            "Backward surrogate for hard converter clamps: 'pwl' uses true "
            "piecewise-linear clamp gradients, 'mad' uses a decaying nonzero "
            "gradient outside [0,1]. Forward remains a hard clamp."
        ),
    )
    parser.add_argument(
        "--transfer-linearization",
        type=str,
        choices=["endpoint", "least_squares", "midpoint_ls"],
        default=yaml_config.transfer_linearization,
        help=(
            "Linear reference for distortion_strength=0 transfer functions: "
            "'endpoint' preserves measured endpoints, 'least_squares' uses the "
            "free linear fit, and 'midpoint_ls' anchors the regression slope at "
            "the endpoint midpoint."
        ),
    )

    # Driver parameters
    parser.add_argument(
        "--driver-distortion-strength",
        type=float,
        default=yaml_config.driver_distortion_strength,
        help="Driver distortion strength",
    )
    parser.add_argument(
        "--driver-distortion-data-path",
        type=str,
        default=yaml_config.driver_distortion_data_path,
        help="Path to driver distortion data CSV",
    )
    parser.add_argument(
        "--driver-distortion-polyfit-order",
        type=int,
        default=yaml_config.driver_distortion_polyfit_order,
        help="Polynomial fit order for driver distortion",
    )

    # PD parameters
    parser.add_argument(
        "--pd-distortion-strength",
        type=float,
        default=yaml_config.pd_distortion_strength,
        help="PD distortion strength",
    )
    parser.add_argument(
        "--pd-noise-w",
        type=float,
        default=yaml_config.pd_noise_w,
        help="Additive PD input-referred noise (Watts), sampled per input sample.",
    )
    parser.add_argument(
        "--pd-distortion-data-path",
        type=str,
        default=yaml_config.pd_distortion_data_path,
        help="Path to PD distortion data CSV",
    )
    parser.add_argument(
        "--pd-distortion-polyfit-order",
        type=int,
        default=yaml_config.pd_distortion_polyfit_order,
        help="Polynomial fit order for PD distortion",
    )
    parser.add_argument(
        "--pd-input-clamp-min-w",
        type=_optional_float,
        default=yaml_config.pd_input_clamp_min_w,
        help="Lower PD input power clamp in Watts; use 'none' to disable",
    )
    parser.add_argument(
        "--pd-input-clamp-max-w",
        type=_optional_float,
        default=yaml_config.pd_input_clamp_max_w,
        help="Upper PD input power clamp in Watts; use 'none' to disable",
    )
    parser.add_argument(
        "--pd-input-clamp-mode",
        choices=["hard", "soft"],
        default=yaml_config.pd_input_clamp_mode,
        help="PD input guard mode before evaluating the PD transfer polynomial",
    )
    parser.add_argument(
        "--pd-input-soft-clamp-width-w",
        type=float,
        default=yaml_config.pd_input_soft_clamp_width_w,
        help="Soft PD input clamp feather width in Watts",
    )
    parser.add_argument(
        "--pd-range-regularization-weight",
        type=float,
        default=yaml_config.pd_range_regularization_weight,
        help="Training loss weight for detector power outside the PD fit range",
    )
    parser.add_argument(
        "--pd-range-regularization-min-w",
        type=_optional_float,
        default=yaml_config.pd_range_regularization_min_w,
        help="Lower range regularization bound in Watts; default uses PD fit min",
    )
    parser.add_argument(
        "--pd-range-regularization-max-w",
        type=_optional_float,
        default=yaml_config.pd_range_regularization_max_w,
        help="Upper range regularization bound in Watts; default uses PD fit max",
    )

    # TIA parameters
    parser.add_argument(
        "--tia-distortion-strength",
        type=float,
        default=yaml_config.tia_distortion_strength,
        help="TIA distortion strength",
    )
    parser.add_argument(
        "--tia-input-bias",
        type=float,
        default=yaml_config.tia_input_bias,
        help=(
            "DC bias added to the TIA input (PD-output units) to place the "
            "operating point inside the characterized TIA domain; removed by "
            "the offset-null output stage (no-op at strength 0)."
        ),
    )
    parser.add_argument(
        "--tia-distortion-data-path",
        type=str,
        default=yaml_config.tia_distortion_data_path,
        help="Path to TIA distortion data CSV",
    )
    parser.add_argument(
        "--tia-distortion-polyfit-order",
        type=int,
        default=yaml_config.tia_distortion_polyfit_order,
        help="Polynomial fit order for TIA distortion",
    )

    # MRM amplitude parameters
    parser.add_argument(
        "--mrm-amplitude-distortion-strength",
        type=float,
        default=yaml_config.mrm_amplitude_distortion_strength,
        help="MRM amplitude distortion strength",
    )
    parser.add_argument(
        "--mrm-amplitude-data-path",
        type=str,
        default=yaml_config.mrm_amplitude_data_path,
        help="Path to MRM amplitude data CSV",
    )
    parser.add_argument(
        "--mrm-amplitude-polyfit-order",
        type=int,
        default=yaml_config.mrm_amplitude_polyfit_order,
        help="Polynomial fit order for MRM amplitude",
    )
    parser.add_argument(
        "--laser-power-gain",
        type=float,
        default=yaml_config.laser_power_gain,
        help=(
            "Laser source power gain; MRM field amplitude is scaled by "
            "sqrt(laser_power_gain)"
        ),
    )

    # MRM phase parameters
    parser.add_argument(
        "--mrm-phase-distortion-strength",
        type=float,
        default=yaml_config.mrm_phase_distortion_strength,
        help="MRM phase distortion strength",
    )
    parser.add_argument(
        "--mrm-phase-data-path",
        type=str,
        default=yaml_config.mrm_phase_data_path,
        help="Path to MRM phase data CSV",
    )
    parser.add_argument(
        "--mrm-phase-polyfit-order",
        type=int,
        default=yaml_config.mrm_phase_polyfit_order,
        help="Polynomial fit order for MRM phase",
    )
    parser.add_argument(
        "--laser-rin-db",
        type=float,
        default=yaml_config.laser_rin_db,
        help="Per-shot global laser relative intensity noise (RIN) in dB.",
    )

    # Conv backend selection
    parser.add_argument(
        "--conv-backend",
        type=str,
        default=yaml_config.conv_backend,
        choices=[
            "pytorch",
            "jtc_ideal",
            "jtc_emulation",
            "jtc_analytic",
            "jtc_analog_fourier",
        ],
        help=(
            "Convolution backend: 'pytorch' (PyTorch conv2d), "
            "'jtc_ideal' (ideal JTC optical model), "
            "'jtc_emulation' (full hardware JTC emulation), or "
            "'jtc_analytic' (exact conv-form JTC with analog Fourier plane)"
        ),
    )
    parser.add_argument(
        "--jtc-output-gain-mode",
        dest="jtc_output_gain_mode",
        type=str,
        choices=["per_shot", "calibrated", "calibrate_freeze", "fixed"],
        default=yaml_config.jtc_output_gain_mode,
        help="Output ADC gain handling for the jtc_analytic backend",
    )
    parser.add_argument(
        "--jtc-output-gain",
        dest="jtc_output_gain",
        type=float,
        default=yaml_config.jtc_output_gain,
        help="Output ADC gain used when jtc-output-gain-mode is 'fixed'",
    )
    parser.add_argument(
        "--jtc-gain-freeze-batches",
        dest="jtc_gain_freeze_batches",
        type=int,
        default=yaml_config.jtc_gain_freeze_batches,
        help="Layer-forwards of gain calibration before freezing (calibrate_freeze)",
    )
    parser.add_argument(
        "--jtc-gain-recal-epochs",
        dest="jtc_gain_recal_epochs",
        type=int,
        default=yaml_config.jtc_gain_recal_epochs,
        help="Re-open calibrate_freeze gain ranging every N epochs (0=off)",
    )
    parser.add_argument(
        "--jtc-gain-headroom",
        dest="jtc_gain_headroom",
        type=float,
        default=yaml_config.jtc_gain_headroom,
        help="Calibrated gains target full-scale/headroom (>=1)",
    )
    parser.add_argument(
        "--jtc-frontend-snr-db",
        dest="jtc_frontend_snr_db",
        type=_optional_float,
        default=yaml_config.jtc_frontend_snr_db,
        help="Gain-referred front-end SNR at the output ADC; 'none' = ideal",
    )
    parser.add_argument(
        "--jtc-carrier-stopband-bins",
        dest="jtc_carrier_stopband_bins",
        type=int,
        default=yaml_config.jtc_carrier_stopband_bins,
        help="Null this many JPS bins each side of DC (carrier block); 0=off",
    )
    parser.add_argument(
        "--jtc-fourier-lag-gemm",
        dest="jtc_fourier_lag_gemm",
        type=_str2bool,
        nargs="?",
        const=True,
        default=yaml_config.jtc_fourier_lag_gemm,
        help=(
            "Full-transfer fast path via sparse-autocorrelation cosine GEMMs "
            "(exact for real planes; false forces the FFT plane path)"
        ),
    )
    parser.add_argument(
        "--fourier-plane-bits",
        type=_optional_int,
        default=yaml_config.fourier_plane_bits,
        help="Quantization bits used in the Fourier plane; use 'none' to disable it",
    )
    parser.add_argument(
        "--jtc-shot-mapping",
        choices=["row", "dot"],
        default=yaml_config.jtc_shot_mapping,
        help="Physical convolution mapping; independent of FFT/DFT solver",
    )
    parser.add_argument(
        "--jtc-remodulation",
        type=yaml.safe_load,
        default=yaml_config.jtc_remodulation,
        help="YAML mapping of continuous interstage remodulator settings",
    )
    parser.add_argument(
        "--jtc-rowwise-geometry",
        dest="jtc_rowwise_geometry",
        type=str,
        choices=["auto", "config"],
        default=yaml_config.jtc_rowwise_geometry,
        help=(
            "Shot geometry: 'auto' (mapping-specific field sizing) or "
            "'config' (use jtc_total_field/jtc_separation; lens studies)"
        ),
    )
    parser.add_argument(
        "--jtc-fourier-closed-form",
        dest="jtc_fourier_closed_form",
        type=_str2bool,
        nargs="?",
        const=True,
        default=yaml_config.jtc_fourier_closed_form,
        help=(
            "Use the closed-form lag decomposition for jtc_analog_fourier "
            "(false forces the two-FFT plane simulation)"
        ),
    )

    # Optional activation normalization inside identical blocks
    parser.add_argument(
        "--normalize-blocks",
        dest="normalize_blocks",
        action="store_true",
        default=yaml_config.normalize_blocks,
        help="Normalize activations after each identical block",
    )
    parser.add_argument(
        "--fftconvnet-input-gain",
        dest="fftconvnet_input_gain",
        type=float,
        default=yaml_config.fftconvnet_input_gain,
        help="Scalar applied before FFTConvNet's first optical stem layer",
    )
    parser.add_argument(
        "--enable-jtc-batched-fast-path",
        dest="enable_jtc_batched_fast_path",
        type=_str2bool,
        nargs="?",
        const=True,
        default=yaml_config.enable_jtc_batched_fast_path,
        help=("Enable batched JTC calls to reduce launch overhead (true/false)"),
    )
    parser.add_argument(
        "--compile-jtc",
        dest="compile_jtc",
        type=_str2bool,
        nargs="?",
        const=True,
        default=yaml_config.compile_jtc,
        help=(
            "Fuse the JTC paired-shot pipeline with torch.compile (true/false); "
            "adds a one-time compile at startup"
        ),
    )
    parser.add_argument(
        "--train-eval-interval",
        dest="train_eval_interval",
        type=int,
        default=yaml_config.train_eval_interval,
        help=(
            "Evaluate train-set metrics every N epochs (full extra pass over "
            "the training set); 0 disables, test eval still runs every epoch"
        ),
    )
    parser.add_argument(
        "--show-progress",
        dest="show_progress",
        type=_str2bool,
        nargs="?",
        const=True,
        default=yaml_config.show_progress,
        help="Show tqdm training progress bars (true/false)",
    )
    parser.add_argument(
        "--run-full-strength-inference",
        dest="run_full_strength_inference",
        type=_str2bool,
        nargs="?",
        const=True,
        default=yaml_config.run_full_strength_inference,
        help=(
            "Run post-training inference with all distortion strengths set to 1.0 "
            "(true/false)"
        ),
    )

    # ------------------------------------------------------------------
    #  Training / model-related CLI overrides
    # ------------------------------------------------------------------
    parser.add_argument(
        "--dataset",
        dest="dataset",
        type=str,
        choices=["cifar10", "mnist"],
        default=yaml_config.dataset,
        help="Training dataset (MNIST is padded to 32x32 and 3 channels)",
    )
    parser.add_argument(
        "--model-arch",
        dest="model_arch",
        type=str,
        choices=["fftconvnet", "resnet11", "resnet18"],
        default=yaml_config.model_arch,
        help="Model architecture to train",
    )
    parser.add_argument(
        "--num-identical-layers",
        dest="num_identical_layers",
        type=int,
        default=yaml_config.num_identical_layers,
        help="Number of identical conv blocks after the 2nd layer",
    )
    parser.add_argument(
        "--num-epochs",
        dest="num_epochs",
        type=int,
        default=yaml_config.num_epochs,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--learning-rate",
        dest="learning_rate",
        type=float,
        default=yaml_config.learning_rate,
        help="Learning rate",
    )
    parser.add_argument(
        "--optimizer",
        dest="optimizer",
        type=str,
        choices=["adamw", "sgd"],
        default=yaml_config.optimizer,
        help="Optimizer (sgd uses momentum/nesterov)",
    )
    parser.add_argument(
        "--momentum",
        dest="momentum",
        type=float,
        default=yaml_config.momentum,
        help="SGD momentum",
    )
    parser.add_argument(
        "--weight-decay",
        dest="weight_decay",
        type=float,
        default=yaml_config.weight_decay,
        help="Weight decay",
    )
    parser.add_argument(
        "--label-smoothing",
        dest="label_smoothing",
        type=float,
        default=yaml_config.label_smoothing,
        help="Cross-entropy label smoothing",
    )
    parser.add_argument(
        "--grad-clip-norm",
        dest="grad_clip_norm",
        type=float,
        default=yaml_config.grad_clip_norm,
        help="Max gradient norm; 0 disables clipping",
    )
    parser.add_argument(
        "--lr-warmup-epochs",
        dest="lr_warmup_epochs",
        type=int,
        default=yaml_config.lr_warmup_epochs,
        help="Linear LR warmup epochs before the cosine schedule",
    )
    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        type=int,
        default=yaml_config.batch_size,
        help="Training batch size",
    )
    parser.add_argument(
        "--seed",
        dest="seed",
        type=int,
        default=yaml_config.seed,
        help="Optional random seed for model init, transforms, and DataLoader workers",
    )
    parser.add_argument(
        "--max-train-batches",
        dest="max_train_batches",
        type=int,
        default=yaml_config.max_train_batches,
        help="Debug: limit number of training batches per epoch",
    )
    parser.add_argument(
        "--max-eval-batches",
        dest="max_eval_batches",
        type=int,
        default=yaml_config.max_eval_batches,
        help="Debug: limit number of eval/inference batches",
    )
    parser.add_argument(
        "--dataloader-num-workers",
        dest="dataloader_num_workers",
        type=int,
        default=yaml_config.dataloader_num_workers,
        help="DataLoader worker processes per DDP rank",
    )
    parser.add_argument(
        "--dataloader-pin-memory",
        dest="dataloader_pin_memory",
        type=_str2bool,
        nargs="?",
        const=True,
        default=yaml_config.dataloader_pin_memory,
        help="Use pinned host memory in DataLoader workers (true/false)",
    )
    parser.add_argument(
        "--dataloader-persistent-workers",
        dest="dataloader_persistent_workers",
        type=_str2bool,
        nargs="?",
        const=True,
        default=yaml_config.dataloader_persistent_workers,
        help="Keep DataLoader workers alive between epochs (true/false)",
    )
    parser.add_argument(
        "--dataloader-prefetch-factor",
        dest="dataloader_prefetch_factor",
        type=int,
        default=yaml_config.dataloader_prefetch_factor,
        help="Batches prefetched per DataLoader worker; non-positive disables it",
    )
    parser.add_argument(
        "--memory-report-interval",
        dest="memory_report_interval",
        type=int,
        default=yaml_config.memory_report_interval,
        help="Print per-rank memory snapshots every N training batches; 0 disables",
    )
    parser.add_argument(
        "--eval-only",
        dest="eval_only",
        action="store_true",
        default=yaml_config.eval_only,
        help="Skip training and only run evaluation using --pretrained-weights",
    )
    parser.add_argument(
        "--pretrained-weights",
        dest="pretrained_weights",
        type=str,
        default=yaml_config.pretrained_weights,
        help="Path to pretrained model weights (.pth) for evaluation or fine-tuning",
    )
    parser.add_argument(
        "--resume-checkpoint",
        dest="resume_checkpoint",
        type=str,
        default=yaml_config.resume_checkpoint,
        help="Path to a full training checkpoint for exact optimizer/scheduler resume",
    )
    parser.add_argument(
        "--checkpoint-interval",
        dest="checkpoint_interval",
        type=int,
        default=yaml_config.checkpoint_interval,
        help=(
            "Save checkpoint_epoch_XXXX.pth every N epochs in addition to latest/best; "
            "0 disables numbered epoch checkpoints"
        ),
    )
    parser.add_argument(
        "--checkpoint-time-interval-minutes",
        dest="checkpoint_time_interval_minutes",
        type=float,
        default=yaml_config.checkpoint_time_interval_minutes,
        help=(
            "Overwrite time_checkpoint.pth every N minutes during training; "
            "0 disables time-based checkpoints"
        ),
    )
    parser.add_argument(
        "--enable-ddp",
        dest="enable_ddp",
        type=_str2bool,
        nargs="?",
        const=True,
        default=yaml_config.enable_ddp,
        help="Enable torchrun DistributedDataParallel when WORLD_SIZE > 1 (true/false)",
    )
    parser.add_argument(
        "--jtc-assume-nonnegative-input",
        dest="jtc_assume_nonnegative_input",
        type=_str2bool,
        nargs="?",
        const=True,
        default=yaml_config.jtc_assume_nonnegative_input,
        help="Skip signed JTC decomposition when all JTCConv2d inputs are nonnegative",
    )
    parser.add_argument(
        "--jtc-max-shots",
        dest="jtc_max_shots",
        type=int,
        default=yaml_config.jtc_max_shots,
        help="Maximum optical shots per JTCConv2d chunk",
    )
    parser.add_argument(
        "--jtc-fuse-pointwise",
        type=_str2bool,
        nargs="?",
        const=True,
        default=yaml_config.jtc_fuse_pointwise,
        help="Fuse CUDA transfer polynomials and final ADC readout",
    )
    parser.add_argument(
        "--jtc-checkpoint-policy",
        choices=["recompute", "spectra"],
        default=yaml_config.jtc_checkpoint_policy,
        help="Checkpoint whole shot chunks or retain compact aperture preparation",
    )
    parser.add_argument(
        "--enable-jtc-activation-checkpointing",
        dest="enable_jtc_activation_checkpointing",
        type=_str2bool,
        nargs="?",
        const=True,
        default=yaml_config.enable_jtc_activation_checkpointing,
        help="Recompute JTCConv2d optical activations during backward (true/false)",
    )
    parser.add_argument(
        "--run-pretrain-tests",
        dest="run_pretrain_tests",
        type=_str2bool,
        nargs="?",
        const=True,
        default=yaml_config.run_pretrain_tests,
        help="Run pretrain tests (true/false)",
    )
    parser.add_argument(
        "--pretrain-tests-only",
        dest="pretrain_tests_only",
        action="store_true",
        default=yaml_config.pretrain_tests_only,
        help="Run only the pretrain tests/plots and exit",
    )

    args = parser.parse_args()

    return replace(yaml_config, **vars(args))


def save_config(config: AppConfig, output_dir: str) -> str:
    """Save the final configuration to output_dir/config.yaml."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "config.yaml")

    with open(output_path, "w") as f:
        yaml.safe_dump(vars(config), f, default_flow_style=False)

    return output_path


def main():
    config_path, output_dir = parse_initial_args()

    # Step 2: Load config from YAML if provided, otherwise use defaults
    yaml_config = AppConfig()
    if config_path:
        yaml_config = load_app_config_from_yaml(config_path)
        yaml_config.config_file = config_path

    # Step 3: Parse CLI args using YAML values as defaults
    final_config = parse_cli_args(yaml_config)
    if output_dir is None:
        raise ValueError("Output directory is required")

    # Step 4: Save the final config to output_dir/config.yaml. Under torchrun,
    # only rank 0 writes shared files.
    if int(os.environ.get("RANK", "0")) == 0:
        saved_path = save_config(final_config, final_config.output_dir)
        print(f"Saved config to output directory: {saved_path}")

    train_onn_model(final_config)


if __name__ == "__main__":
    main()
