#!/bin/bash
# Quick training test to verify the refactored JTC works in actual training
echo "Running quick training test (1 epoch) with refactored JTC..."
python onn_main.py \
  --config-file configs/config_ideal.yaml \
  --output-dir testout/quick_train_test \
  --epochs 1 \
  --batch-size 64 \
  --jtc-separation 8 \
  --jtc-total-field 48

echo ""
echo "If training completes without errors, the refactored JTC is working correctly!"
