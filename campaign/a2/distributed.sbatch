#!/bin/bash
#SBATCH --job-name=jat-a2-distributed
#SBATCH --nodes=2
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=2
printf 'FIRST_SCRIPT_STDOUT job=%s cwd=%s\n' "$SLURM_JOB_ID" "$PWD"
printf 'FIRST_SCRIPT_STDERR job=%s\n' "$SLURM_JOB_ID" >&2
set -euo pipefail
export PATH=/data/scratch-fast/kwen1/envs/joint-adaptive-tokenization-20261024-r1/bin:/usr/bin:/bin
export TZ=UTC SLURM_EXPORT_ENV=ALL PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TORCH_HOME=/data/vision/torralba/u/kwen1/image-tokenization-20261023/joint-adaptive-tokenization-a2-20261024/outputs/a2-assets/torch
export LPIPS_CACHE_DIR=/data/vision/torralba/u/kwen1/image-tokenization-20261023/UNITE-released-20261023-inputs/lpips
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline
[ -f STAGING_MANIFEST.json ]
export MASTER_ADDR
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((20000+SLURM_JOB_ID%30000))
export WORLD_SIZE="$SLURM_NTASKS"
srun --ntasks=4 --ntasks-per-node=2 --gpus-per-task=1 --gpu-bind=single:1 bash campaign/a2/rank.sh "$@"
