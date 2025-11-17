# Memory Optimization Features

This document describes the memory optimization features implemented to reduce GPU memory usage during ONN model training.

## Overview

Training optical neural network (ONN) models can be memory-intensive due to:
1. Multiple transfer function evaluations per forward pass
2. Retention of intermediate activations for backpropagation
3. FFT operations creating complex intermediate tensors
4. Large batch sizes required for effective training

This implementation provides several optimizations to address these issues while maintaining model accuracy.

## Features

### 1. Transfer Function Algebraic Simplification

**Problem:** Each JTC forward pass applies transfer functions twice:
- Input distortion (Driver → MRM): 2 applications per forward pass
- Output distortion (PD → TIA): 2 applications per forward pass

This creates multiple intermediate tensors that must be retained for gradient computation.

**Solution:** Algebraically compose cascaded polynomial transfer functions into simplified forms.

**Implementation:**
- `PD(x)` and `TIA(x)` are composed into a single polynomial `TIA(PD(x))`
- Polynomial order is automatically reduced while maintaining fidelity within acceptable error bounds
- Reduces computation by ~50% for these operations
- Reduces intermediate tensor creation and gradient tracking overhead

**Files:**
- `transfer_function_composer.py`: Polynomial composition utilities
- `simplified_transfer_functions.py`: Composed transfer function classes

**Configuration:**
```python
config.simplify_transfer_functions = True  # Enable simplification (default: True)
config.tf_simplification_max_error = 1e-4  # Max RMSE for order reduction (default: 1e-4)
```

**Technical Details:**

The composition works by algebraically combining polynomials:

For PD-TIA cascade:
```
PD(x) = strength_pd * poly_pd(x) + (1-strength_pd) * (a*x^2 + b)
TIA(y) = strength_tia * poly_tia(y) + (1-strength_tia) * (c*y + d)
Composed: TIA(PD(x)) = single polynomial h(x)
```

The order reduction algorithm:
1. Computes full composed polynomial
2. Fits lower-order polynomials to match the composed function
3. Measures fidelity using RMSE over test range
4. Selects minimum order that meets error threshold

**Benefits:**
- Reduced number of forward operations
- Fewer intermediate tensors stored
- Less gradient tracking overhead
- Maintains numerical accuracy (RMSE < 1e-4 by default)

### 2. Gradient Checkpointing

**Problem:** PyTorch retains all intermediate activations during the forward pass to compute gradients in the backward pass. For deep optical networks with many layers and complex JTC operations, this can consume significant memory.

**Solution:** Use activation checkpointing to trade compute for memory. Intermediate activations are recomputed during the backward pass rather than stored.

**Implementation:**
- Checkpointing added at two levels:
  - JTC forward pass (`checkpoint_jtc`)
  - FTconvlayer forward pass (`checkpoint_layers`)
- Uses PyTorch's `torch.utils.checkpoint` with `use_reentrant=False` for better memory efficiency

**Configuration:**
```python
config.use_gradient_checkpointing = True  # Master switch (default: False)
config.checkpoint_jtc = True              # Checkpoint JTC operations (default: False)
config.checkpoint_layers = True           # Checkpoint layer operations (default: False)
```

**Benefits:**
- Can reduce memory usage by 30-40% during training
- Trade-off: ~15-25% increase in training time due to recomputation

**When to Use:**
- Large batch sizes that don't fit in GPU memory
- Deep networks with many layers
- When training time is less critical than memory usage

### 3. Transfer Function Sharing (Future Enhancement)

**Note:** This feature is prepared but not yet fully implemented.

**Concept:** Since all layers use the same configuration, transfer function instances could be shared across layers rather than each layer having its own Driver, MRM, PD, TIA instances.

**Potential Benefits:**
- Reduce from 7×4=28 transfer function objects to 4 shared objects
- Reduces parameter storage overhead
- Further simplifies memory footprint

## Usage Guide

### Basic Usage with Defaults

The transfer function simplification is enabled by default. Just train normally:

```python
from onn_config import AppConfig
from onn_train import train_onn_model

config = AppConfig()
# Simplification is already enabled by default
model = train_onn_model(config)
```

### Enable All Optimizations

For maximum memory savings:

```python
config = AppConfig()

# Transfer function simplification (already on by default)
config.simplify_transfer_functions = True
config.tf_simplification_max_error = 1e-4

# Gradient checkpointing
config.checkpoint_jtc = True      # Checkpoint JTC operations
config.checkpoint_layers = True   # Checkpoint layer operations

model = train_onn_model(config)
```

### Conservative Approach (Accuracy-Critical)

For applications where numerical precision is critical:

```python
config = AppConfig()

# Use very tight error bounds
config.simplify_transfer_functions = True
config.tf_simplification_max_error = 1e-6  # Tighter bound

# Skip checkpointing if you have enough memory
config.checkpoint_jtc = False
config.checkpoint_layers = False
```

### Test and Validate

Run the validation script to measure improvements:

```bash
python test_memory_optimizations.py
```

This will:
1. Validate transfer function composition fidelity
2. Measure memory usage with/without optimizations
3. Test gradient checkpointing impact
4. Show combined optimization benefits

### YAML Configuration

You can also configure via YAML:

```yaml
# config.yaml

# Memory optimizations
simplify_transfer_functions: true
tf_simplification_max_error: 1.0e-4
checkpoint_jtc: true
checkpoint_layers: true

# Training parameters
batch_size: 128
num_epochs: 20
learning_rate: 0.001
```

Then load:
```python
config = AppConfig(config_file="config.yaml")
```

## Performance Characteristics

### Transfer Function Simplification

**Expected Results:**
- Memory reduction: 5-15% (varies with batch size and model architecture)
- Computation reduction: ~50% for PD-TIA evaluations
- Accuracy impact: Negligible (RMSE < 1e-4)
- Training time: Slightly faster due to fewer operations

**Fidelity Metrics:**
When enabled, you'll see output like:
```
ComposedPDTIA: Reduced order from 15 to 8 (RMSE: 3.45e-05)
```

This shows the polynomial order was reduced from 15 to 8 while maintaining RMSE below threshold.

### Gradient Checkpointing

**Expected Results:**
- Memory reduction: 30-40% during training
- Training time increase: 15-25%
- No accuracy impact (exact gradients recomputed)

**Trade-off Analysis:**
```
               Memory Usage    Training Time
Baseline       100%            100%
+ TF Simplify  90%             95%
+ Checkpoint   60%             120%
Combined       55%             115%
```

## Implementation Details

### File Structure

```
onn-flex/
├── onn_config.py                    # Added memory optimization config options
├── onn_component.py                 # Modified JTC with simplification & checkpointing
├── onn_layers.py                    # Modified FTconvlayer with checkpointing
├── transfer_function_composer.py    # NEW: Polynomial composition utilities
├── simplified_transfer_functions.py # NEW: Composed transfer function classes
├── test_memory_optimizations.py     # NEW: Validation and testing script
└── MEMORY_OPTIMIZATIONS.md          # This file
```

### Code Flow for Transfer Function Simplification

1. **Initialization** (onn_component.py:461-480):
   ```python
   if config.simplify_transfer_functions:
       self.pd_tia_composed = ComposedPDTIA(self.pd, self.tia, ...)
   ```

2. **Composition** (simplified_transfer_functions.py):
   - Algebraically compose polynomials using Horner's rule
   - Fit lower-order approximations
   - Measure fidelity (RMSE)
   - Select minimum order meeting threshold

3. **Usage** (onn_component.py:562-580):
   ```python
   if self.use_simplified_tf:
       x = self.pd_tia_composed(x)  # Single polynomial evaluation
   else:
       x = self.pd(x)
       x = self.tia(x)              # Two separate evaluations
   ```

### Code Flow for Gradient Checkpointing

1. **JTC Level** (onn_component.py:929-980):
   ```python
   def forward(self, signal, kernel):
       if config.checkpoint_jtc and self.training:
           output = checkpoint.checkpoint(self._forward_impl, ...)
       else:
           output = self._forward_impl(...)
   ```

2. **Layer Level** (onn_layers.py:536-547):
   ```python
   def forward(self, x):
       if config.checkpoint_layers and self.training:
           return checkpoint.checkpoint(self._forward_impl, x, ...)
       else:
           return self._forward_impl(x)
   ```

## Troubleshooting

### Issue: "Cannot reduce polynomial order"

**Symptom:**
```
Warning: Cannot reduce polynomial order while maintaining error < 1e-4.
Full order 20 has RMSE 2.5e-03
```

**Solution:**
- Increase error tolerance: `config.tf_simplification_max_error = 1e-3`
- Or disable simplification: `config.simplify_transfer_functions = False`

### Issue: Training slower with checkpointing

**Expected Behavior:** Checkpointing trades compute for memory, so training will be slower.

**Solutions:**
- Only enable if you're running out of memory
- Disable layer checkpointing, keep only JTC: `config.checkpoint_layers = False`
- Reduce batch size instead if memory allows

### Issue: Numerical differences with simplification

**Symptom:** Training metrics differ slightly from baseline.

**Solution:**
- Check fidelity metrics - should be < 1e-4 RMSE
- If critical, tighten threshold: `config.tf_simplification_max_error = 1e-6`
- Or disable: `config.simplify_transfer_functions = False`

## Future Work

### Potential Enhancements

1. **Transfer Function Sharing**
   - Share TF instances across all layers
   - Reduce from 28 to 4 TF objects

2. **Mixed Precision Training**
   - Use float16 for activations, float32 for weights
   - Could halve activation memory

3. **Driver-MRM Composition**
   - Currently only PD-TIA is composed
   - Could also compose Driver with MRM (complex-valued)

4. **Operator Fusion**
   - Fuse loss multiplication, scaling, and TF application
   - Reduce intermediate tensor allocations

5. **Batch Streaming**
   - Process mini-batches sequentially
   - Trade memory for slightly longer training

## References

- PyTorch Gradient Checkpointing: https://pytorch.org/docs/stable/checkpoint.html
- Polynomial Composition: Horner's method for efficient evaluation
- Memory-Efficient Training: Activation checkpointing techniques

## Contact

For issues or questions about memory optimizations, please open an issue on the GitHub repository.
