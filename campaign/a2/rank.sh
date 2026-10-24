#!/bin/bash
set -euo pipefail
export RANK="$SLURM_PROCID" LOCAL_RANK=0
hostname
date -u
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv
free -h
df -PT "$PWD" /data/scratch-fast/kwen1 /data/vision/torralba/datasets/imagenet_pytorch_new/train
exec python "$@"
