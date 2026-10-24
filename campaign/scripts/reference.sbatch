#!/bin/bash
#SBATCH --job-name=joint-a1-reference
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/data/vision/torralba/u/kwen1/image-tokenization-20261023/joint-adaptive-tokenization-20261024/outputs/reference-%j.out
#SBATCH --error=/data/vision/torralba/u/kwen1/image-tokenization-20261023/joint-adaptive-tokenization-20261024/outputs/reference-%j.err
printf 'FIRST_SCRIPT_STDOUT job=%s cwd=%s\n' "$SLURM_JOB_ID" "$PWD"
printf 'FIRST_SCRIPT_STDERR job=%s\n' "$SLURM_JOB_ID" >&2
set -euo pipefail
export PATH=/data/scratch-fast/kwen1/envs/joint-adaptive-tokenization-20261024/bin:/usr/local/cuda/bin:/usr/bin:/bin
export TZ=UTC SLURM_EXPORT_ENV=ALL PYTHONUNBUFFERED=1 PYTHONHASHSEED=20261023
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TORCH_HOME=/data/vision/torralba/u/kwen1/image-tokenization-20261023/joint-adaptive-tokenization-20261024/outputs/assets/torch
export LPIPS_CACHE_DIR=/data/vision/torralba/u/kwen1/image-tokenization-20261023/UNITE-released-20261023-inputs/lpips
export XDG_CACHE_HOME=/data/vision/torralba/u/kwen1/image-tokenization-20261023/joint-adaptive-tokenization-20261024/outputs/assets/cache
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/inductor-$SLURM_JOB_ID"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton-$SLURM_JOB_ID"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline
export INCEPTION_WEIGHTS=/data/vision/torralba/u/kwen1/image-tokenization-20261023/UNITE-released-20261023-inputs/weights-inception-2015-12-05-6726825d.pth
export IN256_FID_STATS=/data/vision/torralba/u/kwen1/image-tokenization-20261023/UNITE-released-20261023-inputs/jit_in256_stats.npz
[ "$#" -eq 1 ]
[ -f STAGING_MANIFEST.json ]
hostname
date -u
free -h
df -PT "$PWD" /data/scratch-fast/kwen1 /data/vision/torralba/datasets/imagenet_pytorch_new/val
/data/scratch-fast/kwen1/envs/joint-adaptive-tokenization-20261024/bin/python campaign/scripts/evaluate_a1.py --config "$1"
