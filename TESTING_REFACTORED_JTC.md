# Testing the Refactored JTC Implementation

This document outlines how to verify that the refactored JTC implementation works correctly.

## Quick Tests (Recommended)

### 1. Simple Output Comparison (< 1 minute)
```bash
python compare_jtc_outputs.py
```
**What it tests:** Verifies that the new forward() method produces identical outputs to the old implementation via backward-compatible wrappers.

**Expected output:**
```
✓ PASS: Outputs match within tolerance (1e-05)
The refactored JTC implementation is working correctly!
```

### 2. Comprehensive Test Suite (< 5 minutes)
```bash
python test_jtc_refactor.py
```
**What it tests:**
- Backward compatibility with old methods
- Individual helper methods (fft_and_magnitude, compute_correlation_indices, build_input_plane)
- 7-step pipeline execution
- Deterministic behavior
- Gradient flow during backpropagation

**Expected output:**
```
Total: 5/5 tests passed
🎉 All tests passed! The refactored JTC is working correctly.
```

## Existing Diagnostic Tests

### 3. Pretrain Tests (1-2 minutes)
```bash
./run_pretrain_tests.sh
# Or manually:
python onn_main.py --config-file configs/config_ideal.yaml \
  --output-dir testout/jtc_refactor_validation \
  --pretrain-tests-only \
  --jtc-separation 8 \
  --jtc-total-field 48
```
**What it tests:**
- Component distortion curves (Driver, PD, TIA, MRM)
- JTC range checks
- Stage-by-stage visualization

**What to look for:**
- No errors during execution
- Plots generated in `testout/jtc_refactor_validation/`
- Check stage plots to ensure they look reasonable

### 4. Quick Training Test (2-5 minutes)
```bash
./quick_training_test.sh
# Or manually:
python onn_main.py --config-file configs/config_ideal.yaml \
  --output-dir testout/quick_train_test \
  --epochs 1 \
  --batch-size 64 \
  --jtc-separation 8 \
  --jtc-total-field 48
```
**What it tests:** Full training pipeline with the refactored JTC

**What to look for:**
- Training completes without errors
- Loss decreases during the epoch
- Checkpoint saved successfully
- Accuracy reported at the end

## Full Validation Tests

### 5. Compare Training Results (30-60 minutes)

Run a full training session and compare with previous results:

```bash
# Before refactoring (if you have old code/branch)
git checkout <old-branch>
python onn_main.py --config-file configs/config_ideal.yaml \
  --output-dir runs/old_jtc --epochs 10

# After refactoring
git checkout claude/refactor-jtc-implementation-011CV32rTuvX1Sh5BPF8bVAE
python onn_main.py --config-file configs/config_ideal.yaml \
  --output-dir runs/new_jtc --epochs 10
```

**What to compare:**
- Final test accuracy should be within ±0.5%
- Loss curves should be nearly identical
- Training time should be similar

### 6. Distortion Sweep Test (variable time)

Run the full distortion sweep:

```bash
# Train model
python onn_main.py --config-file configs/config_ideal.yaml \
  --output-dir runs/refactor_test --epochs 20

# Run distortion sweep
python scripts/distortion_sweep.py \
  --config runs/refactor_test/final_config.yaml \
  --weights runs/refactor_test/fftconv_checkpoint.pth \
  --output-dir sweep_results/refactor_test/
```

**What to look for:**
- All distortion configurations run successfully
- SNDR plots generated
- Accuracy degradation curves look reasonable

## What Changed in the Refactoring

### New Helper Methods

1. **`fft_and_magnitude(x)`** - Unified FFT operation
   - Performs: FFT → fftshift → abs
   - Replaces inline `torch.abs(torch.fft.fftshift(torch.fft.fft(x)))`

2. **`compute_correlation_indices(device)`** - Centralized index calculation
   - Returns indices for extracting correlation output
   - Used in both forward() and compute_stage_tensors()

3. **`build_input_plane(signal, kernel)`** - Separated placement logic
   - Builds JTC input plane from already-distorted inputs
   - No longer combines distortion + placement in one method

### Refactored Forward Method

The forward() method now clearly shows the 7-step pipeline:

```python
# Step 1: Input distortion (signal & kernel)
signal_distorted = self.input_distortion(signal_reshaped)
kernel_distorted = self.input_distortion(kernel_reshaped)
input_plane = self.build_input_plane(signal_distorted, kernel_distorted)

# Step 2: FFT
jft = self.fft_and_magnitude(input_plane)

# Step 3: Output distortion
jps = self.output_distortion(jft)

# Step 4: Input distortion (again)
jps_distorted = self.input_distortion(jps)

# Step 5: FFT (again)
output_plane = self.fft_and_magnitude(jps_distorted)

# Step 6: Output distortion (again)
output_plane = self.output_distortion(output_plane)

# Step 7: Index selection
indices = self.compute_correlation_indices(output_plane.device)
output = output_plane[..., indices]
```

### Backward Compatibility

Old methods are preserved as deprecated wrappers:
- `generate_input_plane()` → calls `input_distortion()` + `build_input_plane()`
- `post_fft()` → calls inline FFT operations
- `post_output_distortion()` → calls `output_distortion(torch.abs(jft))`
- `inverse_output()` → uses new helper methods internally

This means existing code (diagnostics, inference) continues to work without modification.

## Common Issues and Solutions

### Issue: Import errors or module not found
**Solution:** Ensure you have PyTorch and other dependencies installed:
```bash
pip install torch torchvision numpy pandas matplotlib scikit-learn pyyaml
```

### Issue: Tests pass but training fails
**Possible causes:**
- CUDA/GPU issues (try with CPU: add `--device cpu`)
- Memory issues (reduce batch size)
- Configuration issues (check config file)

### Issue: Numerical differences in outputs
**Expected:** Differences < 1e-5 are acceptable (floating point precision)
**Problem:** Differences > 1e-3 indicate a bug

### Issue: Gradients are zero or NaN
**Check:**
- Model is in train mode: `jtc.train()`
- Inputs have `requires_grad=True`
- No in-place operations breaking gradient flow

## Verification Checklist

Use this checklist to verify the refactored implementation:

- [ ] `compare_jtc_outputs.py` passes
- [ ] `test_jtc_refactor.py` all tests pass
- [ ] Pretrain tests run without errors
- [ ] Quick training test completes successfully
- [ ] Training for 10 epochs produces reasonable accuracy
- [ ] Gradients flow correctly during backpropagation
- [ ] Distortion sweep runs successfully
- [ ] Existing diagnostic scripts still work

## Summary

The refactoring simplifies and unifies the JTC implementation while maintaining full backward compatibility. The new structure makes the 7-step pipeline explicit and eliminates code duplication.

**Key benefits:**
- Clearer code structure
- Easier to maintain and modify
- No functional changes (outputs are identical)
- Backward compatible with existing code

If all tests pass, you can be confident the refactored implementation works correctly!
