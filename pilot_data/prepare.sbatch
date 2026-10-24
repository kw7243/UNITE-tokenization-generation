#!/bin/bash
#SBATCH --job-name=jat-pilot-data-v1
#SBATCH --account=csail
#SBATCH --partition=tig-cpu
#SBATCH --qos=tig-main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=/data/vision/torralba/u/kwen1/image-tokenization-20261023/joint-adaptive-tokenization-20261024-data/pilot-splits-v1/20261024T065816Z/attempt-1/stdout-%j.log
#SBATCH --error=/data/vision/torralba/u/kwen1/image-tokenization-20261023/joint-adaptive-tokenization-20261024-data/pilot-splits-v1/20261024T065816Z/attempt-1/stderr-%j.log
printf 'FIRST_SCRIPT_STDOUT job=%s host=%s cwd=%s utc=%s\n' "$SLURM_JOB_ID" "$(hostname)" "$PWD" "$(date -u +%FT%TZ)"
printf 'FIRST_SCRIPT_STDERR job=%s\n' "$SLURM_JOB_ID" >&2
set -euo pipefail
export PATH=/usr/bin:/bin TZ=UTC SLURM_EXPORT_ENV=ALL PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
trap 'rc=$?; printf "TERMINAL_SCRIPT job=%s exit=%s utc=%s\n" "$SLURM_JOB_ID" "$rc" "$(date -u +%FT%TZ)"; exit "$rc"' EXIT
hostname
free -h
df -PT "$PWD" /data/vision/torralba/datasets/imagenet_pytorch_new/train /data/scratch-fast/kwen1
test -f STAGING_MANIFEST.json
/usr/bin/python3 pilot_data/prepare.py --config pilot_data/resolved.yaml
