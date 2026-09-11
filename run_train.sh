#!/bin/bash
set -e

export HF_HUB_DISABLE_XET_DOWNLOAD=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /home/pavitra/satquery/scripts
exec python3 -u train_lora.py
