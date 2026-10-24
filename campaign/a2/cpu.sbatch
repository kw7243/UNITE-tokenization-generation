#!/bin/bash
#SBATCH --job-name=jat-a2-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/data/vision/torralba/u/kwen1/image-tokenization-20261023/joint-adaptive-tokenization-a2-20261024/outputs/a2-preparation/cpu-%j.out
#SBATCH --error=/data/vision/torralba/u/kwen1/image-tokenization-20261023/joint-adaptive-tokenization-a2-20261024/outputs/a2-preparation/cpu-%j.err
printf 'FIRST_SCRIPT_STDOUT job=%s cwd=%s\n' "$SLURM_JOB_ID" "$PWD"
printf 'FIRST_SCRIPT_STDERR job=%s\n' "$SLURM_JOB_ID" >&2
set -euo pipefail
export PATH=/data/scratch-fast/kwen1/envs/joint-adaptive-tokenization-20261024-r1/bin:/usr/bin:/bin
export TZ=UTC SLURM_EXPORT_ENV=ALL PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=''
export PIP_NO_CACHE_DIR=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline
[ "$#" -eq 1 ]
[ -f STAGING_MANIFEST.json ]
hostname
date -u
free -h
df -PT "$PWD" /data/scratch-fast/kwen1
python campaign/a2/tests/check_cpu.py --output "$1"
