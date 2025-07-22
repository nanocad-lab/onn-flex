# ONN-Flex
A standard interface for modeling photonic JTC based neural networks.

# Training
```
python onn_main.py --config-file configs/config_ideal.yaml --output-dir runs/runs_ideal/ --jtc-separation 8 --jtc-total-field 48 --batch-size 128
```

# Inference/SNR alpha sweep generation
```
python distortion_sweep.py --config runs/runs_ideal/final_config.yaml --weights runs/runs_ideal/fftconv_checkpoint.pth
```