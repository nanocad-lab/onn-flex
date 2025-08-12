# ONN-Flex
A standard interface for modeling photonic JTC based neural networks.

# Training

## Standard Training
```bash
python onn_main.py --config-file configs/config_ideal.yaml --output-dir runs/runs_ideal/ --jtc-separation 8 --jtc-total-field 48 --batch-size 128
```

## Training with PyTorch Conv (Alternative to JTC)
```bash
python onn_main.py --config-file configs/config_ideal.yaml --output-dir runs/runs_pytorch/ --use-pytorch-conv --batch-size 128
```

## Run only pretrain tests
```bash
python onn_main.py --config-file configs/config_ideal.yaml --output-dir runs/pretrain_only/ --pretrain-tests-only
```

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
- **Digital Reference Line**: All accuracy plots include a reference line at 67.51% showing ideal digital performance

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
