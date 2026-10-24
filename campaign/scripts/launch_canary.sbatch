#!/bin/bash
#SBATCH --job-name=joint-a1-nil-canary
#SBATCH --account=csail
#SBATCH --partition=tig-cpu
#SBATCH --qos=tig-main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:02:00
#SBATCH --output=/data/vision/torralba/u/kwen1/image-tokenization-20261023/joint-adaptive-tokenization-20261024/outputs/nil-canary-%j.out
#SBATCH --error=/data/vision/torralba/u/kwen1/image-tokenization-20261023/joint-adaptive-tokenization-20261024/outputs/nil-canary-%j.err
printf 'FIRST_SCRIPT_STDOUT job=%s sentinel=%s export=%s cwd=%s\n' "$SLURM_JOB_ID" "${CAMPAIGN_SUBMIT_SENTINEL-ABSENT}" "${SBATCH_EXPORT-ABSENT}" "$PWD"
printf 'FIRST_SCRIPT_STDERR job=%s\n' "$SLURM_JOB_ID" >&2
set -euo pipefail
export PATH=/usr/bin:/bin
export TZ=UTC SLURM_EXPORT_ENV=ALL
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
test -z "${CAMPAIGN_SUBMIT_SENTINEL-}"
test -d .git
/usr/bin/git fsck --full
/usr/bin/python3 campaign/scripts/launch_canary.py campaign/configs/launch-nil-cpu.yaml
