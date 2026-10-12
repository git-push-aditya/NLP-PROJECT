#!/usr/bin/env bash
# Run fine-tuning runs 1+2 directly on a machine with an NVIDIA GPU (no Colab). From finetune/: bash run_gpu.sh
set -euo pipefail
cd "$(dirname "$0")/src"
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA GPU visible'; print(torch.cuda.get_device_name(0), round(torch.cuda.get_device_properties(0).total_memory/2**30), 'GB')"
python s11_translate_train.py   # run 1: resumable, about 75 min on a T4-class GPU
python s12_finetune.py          # run 2: train + encode + save model to ../models/indicsbert-ft
