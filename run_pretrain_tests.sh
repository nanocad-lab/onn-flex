#!/bin/bash
# Run pretrain tests to verify JTC implementation
python onn_main.py \
  --config-file configs/config_ideal.yaml \
  --output-dir testout/jtc_refactor_validation \
  --pretrain-tests-only \
  --jtc-separation 8 \
  --jtc-total-field 48
