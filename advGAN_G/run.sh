#!/bin/bash
# ════════════════════════════════════════════════════════════════════════
#  run.sh - Optimized for H100 & AdvGAN-G (RICAI 2024)
# ════════════════════════════════════════════════════════════════════════

# 1. ตั้งค่า Environment สำหรับ Python
export PYTHONUNBUFFERED=1
export CUDA_LAUNCH_BLOCKING=0

echo "Starting Python script inside PyTorch container..."

python3 -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('GPU Model:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

python3 /nfs-share-stgnode/home/663380266-5/ADV_Project/advGAN_G.py

echo "Python script finished."