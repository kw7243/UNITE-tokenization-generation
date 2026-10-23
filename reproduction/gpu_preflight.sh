#!/bin/bash
set -euo pipefail
export TZ=UTC OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
UNITE_PYTHON=/data/scratch-fast/kwen1/envs/unite-released-20261023/bin/python
printf 'owner=unite-reproduce-20261023\njob=%s\nstage=%s\n' "${SLURM_JOB_ID:?}" "$PWD"
hostname
date -u
uname -a
nvidia-smi --query-gpu=name,uuid,memory.total,memory.used,driver_version --format=csv
free -h
grep Cpus_allowed_list /proc/self/status
df -h "$PWD" /data/vision/torralba/u/kwen1/image-tokenization-20261023/UNITE-released-20261023-inputs /data/scratch-fast/kwen1/envs/unite-released-20261023 /data/vision/torralba/datasets/imagenet_pytorch_new/val
gcc --version | head -1
g++ --version | head -1
scontrol show job "$SLURM_JOB_ID" -o
"$UNITE_PYTHON" - <<'PY'
import json
import torch
assert torch.cuda.is_available(), "No CUDA in the allocated step"
assert torch.cuda.device_count() == 1, "Unexpected GPU visibility"
assert torch.cuda.get_device_capability(0)[0] >= 8, "Native TF32 architecture required"
print(json.dumps({"torch":torch.__version__,"torch_cuda":torch.version.cuda,
                  "cuda_available":torch.cuda.is_available(),"device_count":torch.cuda.device_count(),
                  "gpu_name":torch.cuda.get_device_name(0),"vram_bytes":torch.cuda.get_device_properties(0).total_memory,
                  "capability":torch.cuda.get_device_capability(0),"compiled_arches":torch.cuda.get_arch_list()},indent=2))
PY
