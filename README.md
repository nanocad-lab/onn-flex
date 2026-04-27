# ONN-Flex

ONN-Flex models photonic joint-transform-correlator (JTC) convolution layers
inside small neural-network experiments. The main entry point is `onn_main.py`;
YAML files in `configs/` define optical geometry, quantization, hardware
distortions, and training settings.

## Quick Start

Activate the project environment, then run a baseline JTC training job:

```bash
conda activate onn-torch

python onn_main.py \
  --config-file configs/config_ideal.yaml \
  --output-dir runs/runs_ideal \
  --batch-size 128
```

## Training Modes

### JTC Emulation

```bash
python onn_main.py \
  --config-file configs/config_ideal.yaml \
  --output-dir runs/runs_ideal \
  --conv-backend jtc_emulation \
  --batch-size 128
```

### PyTorch Reference Convolution

```bash
python onn_main.py \
  --config-file configs/config_ideal.yaml \
  --output-dir runs/runs_pytorch \
  --conv-backend pytorch \
  --batch-size 128
```

### Fourier Reference Convolution

```bash
python onn_main.py \
  --config-file configs/config_ideal.yaml \
  --output-dir runs/runs_fft \
  --conv-backend fourier \
  --fourier-plane-bits 6 \
  --batch-size 128
```

### Evaluation Only

```bash
python onn_main.py \
  --config-file runs/runs_ideal/final_config.yaml \
  --output-dir runs/runs_ideal_eval \
  --eval-only \
  --pretrained-weights runs/runs_ideal/fftconv_checkpoint.pth
```

### Pretrain Diagnostics Only

```bash
python onn_main.py \
  --config-file configs/config_ideal.yaml \
  --output-dir runs/pretrain_only \
  --pretrain-tests-only
```

## Gen 2.5 FPGA

The FPGA path is enabled by `use_fpga_accelerator: true` in
`configs/config_fpga_g2p5.yaml`:

```bash
python onn_main.py \
  --config-file configs/config_fpga_g2p5.yaml \
  --output-dir runs/runs_fpga_g2p5
```

The FPGA config uses `input_length=8`, `kernel_length=8`,
`jtc_total_field=16`, and `jtc_separation=0`. For 32x32 inputs, the largest
`FFTConvNet` JTC calls produce `256 * batch_size` shots, so `batch_size=64`
keeps the accelerator input within `16384 x 16`.

The hardware integration boundary is `JTC.call_fpga_accelerator_uint16()` in
`onn_component.py`. It receives `accelerator_input_uint16` with shape
`[shots, 16]` and must return `accelerator_output_uint16` with shape
`[shots, 16]`. The wrapper converts the returned `uint16` codes back to
`float32` model-domain values before correlation-index selection. Until this
function is wired to hardware, FPGA runs intentionally raise
`NotImplementedError`.

## Analysis Scripts

### Quantization-Level Sweep

```bash
python scripts/sweep_quantlevel_ste_maxscale.py --gpus 0,1,2,3,4,5,6,7
```

### Distortion Sweep

Run inference sweeps and generate plots:

```bash
python scripts/distortion_sweep.py \
  --config runs/runs_ideal/final_config.yaml \
  --weights runs/runs_ideal/fftconv_checkpoint.pth \
  --output-dir sweep_results
```

Regenerate plots from existing sweep data:

```bash
python scripts/distortion_sweep.py --plot-only --output-dir sweep_results
```

Distortion sweep outputs include `{param}_results.txt`, `{param}_sweep.pdf`,
`jtc_2d_accuracy.npy/.txt`, `jtc_2d_sndr_output.npy/.txt`, and 2D JTC geometry
heatmaps.

### One-Hot Distortion Runs

One-hot runs evaluate or train with one distortion source enabled at a time, plus
all-zeros and all-ones cases. A base run directory must contain
`final_config.yaml` and `fftconv_checkpoint.pth`.

```bash
python scripts/one_hot_distortion_runs.py \
  --base-run runs/runs_ideal_0825 \
  --do-infer

python scripts/one_hot_distortion_runs.py \
  --base-run runs/runs_ideal_0825 \
  --do-train \
  --epochs 20

python scripts/one_hot_distortion_runs.py \
  --base-run runs/runs_ideal_0825 \
  --do-finetune \
  --finetune-additional-epochs 5 \
  --finetune-lr 5e-4
```

Results are written under `runs/<base>_onehots/` by default and include per-case
configs, metrics, checkpoints for train/finetune modes, and `summary.txt`.
