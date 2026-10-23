#!/bin/bash
#SBATCH --job-name=unite-cpu-readiness
#SBATCH --account=csail
#SBATCH --partition=tig-cpu
#SBATCH --qos=tig-main
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/data/vision/torralba/u/kwen1/image-tokenization-20261023/UNITE-released-20261023-results/cpu-readiness-%j.log
set -euo pipefail
export TZ=UTC OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
UNITE_PYTHON=/data/scratch-fast/kwen1/envs/unite-released-20261023/bin/python
UNITE_INPUTS=/data/vision/torralba/u/kwen1/image-tokenization-20261023/UNITE-released-20261023-inputs
UNITE_RESULTS=/data/vision/torralba/u/kwen1/image-tokenization-20261023/UNITE-released-20261023-results
export TORCH_HOME="$UNITE_INPUTS/torch"
export LPIPS_CACHE_DIR="$UNITE_INPUTS/lpips"
export INCEPTION_WEIGHTS="$UNITE_INPUTS/weights-inception-2015-12-05-6726825d.pth"
export IN256_FID_STATS="$UNITE_INPUTS/jit_in256_stats.npz"
hostname
free -h
df -h "$PWD" "$UNITE_INPUTS" /data/vision/torralba/datasets/imagenet_pytorch_new/val
"$UNITE_PYTHON" -m pip check
for unite_pair in B:Base L:Large; do
  unite_short=${unite_pair%:*}
  unite_encoder=${unite_pair#*:}
  unite_sha=$("$UNITE_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["files"][sys.argv[2]]["sha256"])' "$UNITE_RESULTS/cpu-setup-2826271/checkpoint-metadata.json" "UNITE-$unite_short.pt")
  "$UNITE_PYTHON" reproduction/evaluate_released.py --checkpoint "$UNITE_INPUTS/UNITE-$unite_short.pt" --checkpoint-sha256 "$unite_sha" --encoder "$unite_encoder" --output "$UNITE_RESULTS/cpu-readiness-${SLURM_JOB_ID}-$unite_short" --mode check --protocol release
done
