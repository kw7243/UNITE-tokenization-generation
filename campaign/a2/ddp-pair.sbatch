#!/bin/bash
#SBATCH --job-name=jat-a2-ddp-check
#SBATCH --nodes=2
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=2
printf 'FIRST_SCRIPT_STDOUT job=%s cwd=%s\n' "$SLURM_JOB_ID" "$PWD"
set -euo pipefail
[ "$#" -eq 1 ]
# Each invocation runs all four ranks from this immutable full repository stage.
for count in 64 16; do
  bash campaign/a2/distributed.sbatch campaign/a2/tests/check_ddp_gpu.py \
    --config "campaign/a2/configs/r3-explicit-device/NATIVE${count}_JOINT.yaml" \
    --output "$1/native${count}"
done
