# ONN-Flex
A standard interface for modeling photonic JTC based neural networks.

# Training

## Standard Training
```bash
python onn_main.py --config-file configs/config_ideal.yaml --output-dir runs/runs_ideal/ --jtc-separation 8 --jtc-total-field 48 --batch-size 128
```

### Reduce GPU memory (activation checkpointing)
```bash
python onn_main.py --config-file configs/config_ideal.yaml --output-dir runs/runs_ideal_ckpt/ --jtc-checkpoint true
```

## Training with PyTorch Conv (Alternative to JTC)
```bash
python onn_main.py --config-file configs/config_ideal.yaml --output-dir runs/runs_pytorch/ --use-pytorch-conv --batch-size 128
```

## Training with Fourier Conv (FFT-based alternative to JTC)
```bash
python onn_main.py \
  --config-file configs/config_ideal.yaml \
  --output-dir runs/runs_fft/ \
  --use-fourier-conv \
  --batch-size 128
```

### Optional: Quantize the Fourier plane
```bash
python onn_main.py \
  --config-file configs/config_ideal.yaml \
  --output-dir runs/runs_fft_q/ \
  --use-fourier-conv \
  --quantize-fourier-plane \
  --fourier-plane-bits 6 \
  --batch-size 128
```

## Run only pretrain tests
```bash
python onn_main.py --config-file configs/config_ideal.yaml --output-dir runs/pretrain_only/ --pretrain-tests-only
```

# Checkpointing & Resume
- Checkpoints are saved under `OUTPUT_DIR/checkpoints/epoch_XXX.pth` and a rolling `OUTPUT_DIR/checkpoints/latest.pth`.
- Training also writes `OUTPUT_DIR/progress.csv` with `epoch, train_acc, test_acc, best_acc, loss`.
- Resume is automatic: if `latest.pth` exists in the chosen `--output-dir`, `onn_main.py` loads model/optimizer/scheduler/best_acc and continues from the next epoch.
- To start fresh, point `--output-dir` to a new folder or remove `latest.pth`.

# Distortion Sweep Analysis

The distortion sweep script analyzes the impact of various hardware distortions on model performance, generating both accuracy and SNDR (Signal-to-Noise and Distortion Ratio) plots.

## Full Distortion Sweep (Run inference and generate plots)
```bash
python distortion_sweep.py --config runs/runs_ideal/final_config.yaml --weights runs/runs_ideal/fftconv_checkpoint.pth --output-dir sweep_results/
```

## Plot-Only Mode (Generate plots from existing data)
```bash
python distortion_sweep.py --plot-only --output-dir sweep_results/
```

## Features

### Generated Outputs
- **1D Parameter Sweeps**: Plots showing accuracy and SNDR vs. distortion strength for:
  - Driver distortion
  - PD-TIA distortion
  - MRM power distortion
  - MRM phase distortion
- **2D JTC Geometry Sweep**: Heatmaps showing accuracy and SNDR vs. JTC separation and total field size
- **Digital Reference Line**: All accuracy plots include a reference line at 60.13% showing ideal digital performance

### Plot-Only Mode Benefits
- **Fast plot regeneration**: Create updated plots without expensive re-inference
- **Styling updates**: Apply new plot features (reference lines, SNDR terminology) to existing data
- **Backward compatibility**: Works with data from previous runs
- **Selective plotting**: Generate only the plots you need

### Data Files Generated
- `{param}_results.txt`: Tabulated results for each distortion parameter
- `{param}_sweep.pdf`: 1D plots showing accuracy and SNDR vs. distortion strength
- `jtc_2d_accuracy.npy/.txt`: 2D accuracy matrix for JTC geometry sweep
- `jtc_2d_sndr_output.npy/.txt`: 2D SNDR matrix for JTC geometry sweep
- `jtc_2d_sweep_accuracy.pdf`: 2D accuracy heatmap
- `jtc_2d_sweep_output_sndr.pdf`: 2D SNDR heatmap

### Example Workflow
```bash
# 1. Run full sweep (takes time)
python distortion_sweep.py --config config.yaml --weights model.pth --output-dir results/

# 2. Later, regenerate plots with updated styling (fast)
python distortion_sweep.py --plot-only --output-dir results/
```

# One‑Hot Distortion Runs

Iterate through each distortion strength by setting one parameter to 1.0 at a time (others 0.0), then run a case with all distortion strengths at 1.0. A baseline all‑zeros case is also run in the selected modes. Supports inference (using a pretrained ideal run), training from scratch, and fine‑tuning.

## Requirements
- A pretrained ideal run directory (e.g., `runs/runs_ideal_0825`) containing:
  - `final_config.yaml`
  - `fftconv_checkpoint.pth`

## Inference only
```bash
python scripts/one_hot_distortion_runs.py --base-run runs/runs_ideal_0825 --do-infer
```
Runs: `infer/all-zeros`, each one‑hot case, and `infer/all-ones`.

## Training only
```bash
python scripts/one_hot_distortion_runs.py --base-run runs/runs_ideal_0825 --do-train --epochs 20
```
Runs: `train/all-zeros` (baseline), each one‑hot case, and `train/all-ones`.

## Inference and training
```bash
python scripts/one_hot_distortion_runs.py --base-run runs/runs_ideal_0825 --do-infer --do-train --epochs 20
```
Runs all the above inference and training baselines and sweeps.

## Fine‑tuning from the checkpoint
Fine‑tune the all‑zeros baseline, each one‑hot case, and the all‑ones case initialized from the base run checkpoint. The additional epochs apply equally to keep baseline epoch parity:
```bash
python scripts/one_hot_distortion_runs.py --base-run runs/runs_ideal_0825 --do-finetune --finetune-additional-epochs 5 --finetune-lr 5e-4
```

## Optional: include PD‑TIA strength
By default, the sweep covers `driver`, `pd`, `tia`, `mrm_power`, and `mrm_phase`. To also include `pd_tia_distortion_strength`:
```bash
python scripts/one_hot_distortion_runs.py --base-run runs/runs_ideal_0825 --do-infer --include-pd-tia
```

## Outputs
- Results are written under `runs/<base>_onehots/` by default (override with `--output-root`).
- Structure:
  - `infer/all-zeros/{config.yaml, metrics.txt}`
  - `infer/<param-case>/{config.yaml, metrics.txt}`
  - `infer/all-ones/{config.yaml, metrics.txt}`
  - `train/all-zeros/{config.yaml, fftconv_checkpoint.pth, ...}`
  - `train/<param-case>/{config.yaml, fftconv_checkpoint.pth, ...}`
  - `train/all-ones/{config.yaml, fftconv_checkpoint.pth, ...}`
  - `finetune/all-zeros/{config.yaml, fftconv_checkpoint.pth, ...}`
  - `finetune/<param-case>/{config.yaml, fftconv_checkpoint.pth, ...}`
  - `finetune/all-ones/{config.yaml, fftconv_checkpoint.pth, ...}`
  - `summary.txt` with a concise per‑case summary
