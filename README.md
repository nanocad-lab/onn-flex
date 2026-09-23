# ONN-Flex

ONN-Flex is a differentiable, steady-state simulator for photonic joint
transform correlator (pJTC) accelerators. It connects component transfer
curves, optical propagation, converters, shot mapping and PyTorch workloads.
It supports sensitivity studies and hardware-aware training; it does not
establish measured silicon accuracy or full-chip timing and energy.

## Quick start

Use Linux and Python 3.12. Run commands from the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-dev.txt
python onn_shotreport.py --config configs/config_analog_shotplan.yaml \
  --input-shape 1 3 32 32 --output /tmp/shotplan.json
pre-commit run --all-files
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -m pytest -q
```

For GPU work, install matching PyTorch/torchvision CUDA wheels first, then
the same requirements. CUDA fusion uses Triton from the Linux CUDA PyTorch
installation. CPU tests need neither datasets nor cluster access.
Run `pre-commit install` to enable the style hooks locally.

## Training

```bash
python onn_main.py --config-file configs/config_pytorch_conv.yaml \
  --model-arch resnet18 --output-dir runs/native --num-epochs 20
python onn_main.py --config-file configs/config_analog_shotplan.yaml \
  --output-dir runs/analog --num-epochs 20
```

`python onn_main.py --help` lists the supported flags. YAML paths are relative
to the repository root; unknown fields are errors. `--pretrained-weights`
loads model weights; `--resume-checkpoint` restores training, optimizer,
scheduler, calibration, RNG and data-order state. The standard trainer selects
checkpoints using its test loader; these best-test scores are not independent
held-out estimates. The averaging study below used a separate validation split.

Exact resume requires the same experiment configuration, including physical
parameters, loss/optimizer recipe, epoch budget and execution strategy. Output
paths, checkpoint/logging frequency and data-loader worker settings may change.
Use `--pretrained-weights` to start a new recipe from an existing model.
Keep source, characterization files and device/software environment fixed for
exact replay; configuration equality does not verify external file contents.

`--jtc-max-shots` bounds intermediate memory; activation checkpointing uses
`--enable-jtc-activation-checkpointing true`. `--jtc-fuse-pointwise true`
enables eligible CUDA fusion. For DDP, use `torchrun --standalone
--nproc_per_node=N onn_main.py ... --enable-ddp true`; batch size is per rank.
[The Slurm template](slurm/template.sbatch) accepts site settings through the
submit environment, including optional `ONN_CONDA_MODULE` and `CONDA_ENV`.

## Physical model

The future-design standard is an **analog Fourier plane with one final ADC
stage per shot**. One stage may digitize multiple output channels:

```text
input DAC → modulation → lens → PD/TIA → continuous remodulation
          → lens → output detector → offset/gain → ADC → digital decode/sum
```

| Backend | Contract |
|---|---|
| `pytorch` | Electronic convolution control. |
| `jtc_ideal` | Idealized stitched correlator with configurable converters. |
| `jtc_emulation` | Digitized intermediate plane and component transfers. |
| `jtc_analytic` | Clean-lag affine analog closed form; rejects unsupported settings. |
| `jtc_analog_fourier` | Analog remodulation, overlap and full component transfers. |

Use [the analog config](configs/config_analog_shotplan.yaml) explicitly;
`AppConfig()` defaults to the manuscript-related emulation topology.
`onn_shotplan.py` defines mapping/counts, `onn_analog_shot.py` applies shot
physics, `onn_remodulation.py` defines the interstage response, and
`onn_jtc_conv2d.py` handles signed branches, accumulation and calibration.
`onn_fused.py` implements CUDA kernels for the same model.

## Paper basis and model limits

- Yen et al., *Flexible Error Modeling and Analysis of SiPho JTC Accelerators
  for CNNs*, supplied 2026 manuscript: component fits, α interpolation and JTC
  geometry. The supplied copy has placeholder publication metadata. It models
  intermediate digitization; our analog topology is a separate design choice.
- Shurui Li, [*Architecture, Modeling, and Optimization of Photonic Neural
  Network Accelerators*](https://escholarship.org/uc/item/8pq9f0jz), UCLA, 2024,
  Chapter 4: PhotoFourier's direct analog remodulation and convolution mapping.
- Yang et al., [*Near-energy-free photonic Fourier transformation for convolution
  operation acceleration*](https://doi.org/10.1117/1.AP.7.5.056007), 2025,
  Eqs. (1)–(2): joint power spectrum and correlation lobes.

[Units and physical conventions](units.md) define the detector chain, transfer
data and sampling assumptions. Tests compare independent FFT/DFT references,
closed forms, component fits, signed mapping and converter behavior. These
validate the declared discrete model. Measured channel calibration, settling,
thermal drift, lane timing and chip energy remain outside its scope. Structural
overlap counts alone do not determine contamination magnitude or CIFAR accuracy.

## Retained findings

Spectrum reuse computes distinct signal/kernel apertures within a forward;
it does not assume signals repeat later. Symmetric DFT execution, bounded
checkpointing and CUDA fusion retain the physical shot plan. Regression tests
cover α=0 and individual/combined α=1, converter boundaries, gradients, noise
and replay. Coherent DFT sums retain FP64 because FP32 failed the fidelity gates;
simulator precision is separate from ADC precision.

In the archived B200 ResNet-18 batch-64 experiment, added boundary fusion
reduced forward/backward time from 52.14 to 25.41 s at TIA α=1, bias 0.08,
stopband 3 (two alternating timing repeats; optimizer/data loading excluded).
Outputs, gains and gradients were identical. Speedup depends on the operating
point and workload; GPU time is not accelerator latency.

The matched CIFAR ResNet-18 study found **94.84 ± 0.10%** accuracy with
individual outputs and **92.91 ± 0.32%** with adjacent pair means copied
back to both positions: **1.93 ± 0.41 percentage points** loss across three
seeds. It measures ideal arithmetic averaging with retraining; physical
power averaging, noise, quantization and mirrored sampling remain separate
questions. The planner's selected-readout checks remain in source; the matched
network wrapper, training recipes and predictions are archived together.

Expanded research documentation and study-only source live in Git-ignored
`context/`; the local index is `context/README.md`. Intermediate probes, campaign
launchers, logs and reproduction snippets remain in ignored `experiments/`.
Datasets/results stay in ignored `data/` and `runs/`, reference PDFs in ignored
`papers/`. Installation, tests and the shot-plan example are self-contained
without those local archives.
