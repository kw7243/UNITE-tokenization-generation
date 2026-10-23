#!/bin/bash
#SBATCH --job-name=unite-cpu-setup
#SBATCH --account=csail
#SBATCH --partition=tig-cpu
#SBATCH --qos=tig-main
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/data/vision/torralba/u/kwen1/image-tokenization-20261023/UNITE-released-20261023-results/cpu-setup-%j.log
set -euo pipefail
export TZ=UTC
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PIP_CACHE_DIR=/data/scratch-fast/kwen1/.cache/unite-released-20261023/pip
export MAMBA_ROOT_PREFIX=/data/scratch-fast/kwen1/micromamba/root
UNITE_ENV=/data/scratch-fast/kwen1/envs/unite-released-20261023
UNITE_INPUTS=/data/vision/torralba/u/kwen1/image-tokenization-20261023/UNITE-released-20261023-inputs
UNITE_RESULT=/data/vision/torralba/u/kwen1/image-tokenization-20261023/UNITE-released-20261023-results/cpu-setup-${SLURM_JOB_ID}
mkdir -p "$UNITE_RESULT"
printf 'job=%s\nnode=%s\nstage=%s\nowner=unite-reproduce-20261023\n' "$SLURM_JOB_ID" "$(hostname)" "$PWD" > "$UNITE_RESULT/receipt.txt"
date -u
hostname
free -h
df -h "$PWD" /data/scratch-fast/kwen1 /data/vision/torralba/datasets/imagenet_pytorch_new/val
if [ ! -x "$UNITE_ENV/bin/python" ]; then
  /data/scratch-fast/kwen1/micromamba/bin/micromamba create -y -p "$UNITE_ENV" -c conda-forge python=3.10 pip
fi
"$UNITE_ENV/bin/python" -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
"$UNITE_ENV/bin/python" -m pip install -r requirements.txt
"$UNITE_ENV/bin/python" -m pip check
"$UNITE_ENV/bin/python" -m pip freeze > "$UNITE_RESULT/packages.txt"
curl -fL --max-time 900 'https://www.dropbox.com/scl/fi/6vbzwqiu9rqnmwm4cj5my/UNITE-B.pt?rlkey=yrqoyiw1zymqe6pn8rf0ss1d6&st=xt8hmz46&dl=1' -o "$UNITE_INPUTS/UNITE-B.pt"
curl -fL --max-time 1500 'https://www.dropbox.com/scl/fi/mvqbhq3oubog4gtswnthw/UNITE-L.pt?rlkey=ekn3m0b5wb26512dc901rj9w7&st=x91h020h&dl=1' -o "$UNITE_INPUTS/UNITE-L.pt"
curl -fL --max-time 300 https://github.com/toshas/torch-fidelity/releases/download/v0.2.0/weights-inception-2015-12-05-6726825d.pth -o "$UNITE_INPUTS/weights-inception-2015-12-05-6726825d.pth"
curl -fL --max-time 300 https://raw.githubusercontent.com/LTH14/JiT/main/fid_stats/jit_in256_stats.npz -o "$UNITE_INPUTS/jit_in256_stats.npz"
"$UNITE_ENV/bin/python" reproduction/inspect_inputs.py "$UNITE_INPUTS" "$UNITE_RESULT"
