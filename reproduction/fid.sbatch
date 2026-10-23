#!/bin/bash
#SBATCH --job-name=unite-fid
#SBATCH --account=vision-torralba-urops-meng
#SBATCH --partition=vision-torralba-rtx3090
#SBATCH --qos=vision-torralba-urops-meng
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/data/vision/torralba/u/kwen1/image-tokenization-20261023/UNITE-released-20261023-results/%x-%j.log
set -euo pipefail
export TZ=UTC OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TORCHINDUCTOR_COMPILE_THREADS=4
UNITE_ROOT=/data/vision/torralba/u/kwen1/image-tokenization-20261023
UNITE_PYTHON=/data/scratch-fast/kwen1/envs/unite-released-20261023/bin/python
UNITE_INPUTS="$UNITE_ROOT/UNITE-released-20261023-inputs"
UNITE_OUTPUT="$UNITE_ROOT/UNITE-released-20261023-results/fid-$UNITE_ENCODER-$UNITE_PROTOCOL-$SLURM_JOB_ID"
test -n "$UNITE_NPROC"
export TORCH_HOME="$UNITE_INPUTS/torch" LPIPS_CACHE_DIR="$UNITE_INPUTS/lpips"
export INCEPTION_WEIGHTS="$UNITE_INPUTS/weights-inception-2015-12-05-6726825d.pth"
export IN256_FID_STATS="$UNITE_INPUTS/jit_in256_stats.npz"
export TORCHINDUCTOR_CACHE_DIR=/data/scratch-fast/kwen1/.cache/unite-released-20261023/inductor
export TRITON_CACHE_DIR=/data/scratch-fast/kwen1/.cache/unite-released-20261023/triton
mkdir -p "$UNITE_OUTPUT"
printf 'owner=unite-reproduce-20261023\njob=%s\nstage=%s\noutput=%s\n' "$SLURM_JOB_ID" "$PWD" "$UNITE_OUTPUT" > "$UNITE_OUTPUT/receipt.txt"
srun --nodes=1 --ntasks=1 bash reproduction/gpu_preflight.sh > "$UNITE_OUTPUT/preflight.log" 2>&1
case "$UNITE_ENCODER" in
 Base) unite_short=B; unite_cfg=2.0; unite_sha=7b7f40dd37111196cad6e4401bf9737156268612e001c29cb0a760ae416bff4b ;;
 Large) unite_short=L; unite_cfg=1.8; unite_sha=5b15fe343c4b288d1e0231d04af4c38de126bd1609ac8b4155a5c147720d7a40 ;;
 *) exit 64 ;;
esac
srun --nodes=1 --ntasks=1 "$UNITE_PYTHON" -u -m torch.distributed.run --standalone --nproc-per-node="$UNITE_NPROC" reproduction/evaluate_released.py --checkpoint "$UNITE_INPUTS/UNITE-$unite_short.pt" --checkpoint-sha256 "$unite_sha" --encoder "$UNITE_ENCODER" --output "$UNITE_OUTPUT" --mode fid --protocol "$UNITE_PROTOCOL" --cfg-scale "$unite_cfg" --seed 0 --batch-size 16
