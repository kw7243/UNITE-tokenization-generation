#!/bin/bash
#SBATCH --job-name=joint-a1-environment
#SBATCH --account=csail
#SBATCH --partition=tig-cpu
#SBATCH --qos=tig-main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/data/vision/torralba/u/kwen1/image-tokenization-20261023/joint-adaptive-tokenization-20261024/outputs/environment-%j.out
#SBATCH --error=/data/vision/torralba/u/kwen1/image-tokenization-20261023/joint-adaptive-tokenization-20261024/outputs/environment-%j.err
printf 'FIRST_SCRIPT_STDOUT job=%s cwd=%s\n' "$SLURM_JOB_ID" "$PWD"
printf 'FIRST_SCRIPT_STDERR job=%s\n' "$SLURM_JOB_ID" >&2
set -euo pipefail
export PATH=/usr/bin:/bin
export TZ=UTC SLURM_EXPORT_ENV=ALL
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PIP_CACHE_DIR=/data/scratch-fast/kwen1/.cache/unite-released-20261023/pip
export TORCH_HOME=/data/vision/torralba/u/kwen1/image-tokenization-20261023/joint-adaptive-tokenization-20261024/outputs/assets/torch
export LPIPS_CACHE_DIR=/data/vision/torralba/u/kwen1/image-tokenization-20261023/UNITE-released-20261023-inputs/lpips
export INCEPTION_WEIGHTS=/data/vision/torralba/u/kwen1/image-tokenization-20261023/UNITE-released-20261023-inputs/weights-inception-2015-12-05-6726825d.pth
export IN256_FID_STATS=/data/vision/torralba/u/kwen1/image-tokenization-20261023/UNITE-released-20261023-inputs/jit_in256_stats.npz
hostname
free -h
df -h "$PWD" /data/scratch-fast/kwen1 /data/vision/torralba/datasets/imagenet_pytorch_new/val
/usr/bin/python3 campaign/scripts/prepare_environment.py campaign/configs/environment.yaml
