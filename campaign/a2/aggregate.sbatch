#!/bin/bash
#SBATCH --job-name=jat-a2-aggregate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --export=NIL
set -euo pipefail
export PATH=/data/scratch-fast/kwen1/envs/jat-a2-postprocess-20261024-r1/bin:/usr/bin:/bin
export TZ=UTC SLURM_EXPORT_ENV=ALL PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:?}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline
export TORCH_HOME=/data/vision/torralba/u/kwen1/image-tokenization-20261023/joint-adaptive-tokenization-a2-20261024/outputs/a2-assets/torch
export LPIPS_CACHE_DIR=/data/vision/torralba/u/kwen1/image-tokenization-20261023/UNITE-released-20261023-inputs/lpips
export MPLCONFIGDIR="/data/scratch-fast/kwen1/jat-a2-20261024-mplconfig/${SLURM_JOB_ID:?}"
[ "$#" -eq 2 ]
[ -f "$1/STAGING_MANIFEST.json" ]
mkdir -p "$MPLCONFIGDIR"
cd -- "$1"
printf 'FIRST_SCRIPT_STDOUT job=%s stage=%s\n' "$SLURM_JOB_ID" "$PWD"
printf 'FIRST_SCRIPT_STDERR job=%s stage=%s\n' "$SLURM_JOB_ID" "$PWD" >&2
hostname
date -u
free -h
df -PT "$PWD" /data/scratch-fast/kwen1
srun --ntasks=1 python campaign/a2/aggregate.py --config "$2"
