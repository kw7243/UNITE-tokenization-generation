#!/bin/bash
#SBATCH --job-name=jat-a2-shared-visible
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
printf 'FIRST_SCRIPT_STDOUT job=%s cwd=%s\n' "$SLURM_JOB_ID" "$PWD"
printf 'FIRST_SCRIPT_STDERR job=%s\n' "$SLURM_JOB_ID" >&2
set -euo pipefail
export PATH=/data/scratch-fast/kwen1/envs/joint-adaptive-tokenization-20261024-r1/bin:/usr/bin:/bin
export TZ=UTC SLURM_EXPORT_ENV=ALL PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TORCH_HOME=/data/vision/torralba/u/kwen1/image-tokenization-20261023/joint-adaptive-tokenization-a2-20261024/outputs/a2-assets/torch
export LPIPS_CACHE_DIR=/data/vision/torralba/u/kwen1/image-tokenization-20261023/UNITE-released-20261023-inputs/lpips
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline
export NCCL_DEBUG=INFO
[ "$#" -eq 1 ]
[ -f STAGING_MANIFEST.json ]
[ ! -e "$1" ]
mkdir -p "$1"
export A2_MASTER_ADDR
A2_MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export A2_MASTER_PORT=$((20000+SLURM_JOB_ID%20000))
hostname
date -u
free -h
df -PT "$PWD" /data/scratch-fast/kwen1 /data/vision/torralba/datasets/imagenet_pytorch_new/train
srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --cpus-per-task=8 --gres=gpu:rtx_3090:2 --gpu-bind=none --kill-on-bad-exit=1 \
  bash campaign/a2/node-launch.sh campaign/a2/tests/check_collective.py --output "$1/collective"
# set -e prevents any model execution after a failed communication probe.
# Reuse passed unchanged direct denoiser predictions from job2834145.
for count in 64 16; do
  export A2_MASTER_PORT=$((A2_MASTER_PORT+1))
  srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --cpus-per-task=8 --gres=gpu:rtx_3090:2 --gpu-bind=none --kill-on-bad-exit=1 \
    bash campaign/a2/node-launch.sh campaign/a2/tests/check_ddp_gpu.py \
      --config "campaign/a2/configs/r6-source-ddp-unused/NATIVE${count}_JOINT.yaml" --output "$1/native${count}"
done
