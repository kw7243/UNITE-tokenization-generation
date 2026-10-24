#!/bin/bash
set -euo pipefail
printf 'NODE_LAUNCH node=%s node_rank=%s CUDA_VISIBLE_DEVICES=%s\n' "$(hostname)" "$SLURM_NODEID" "${CUDA_VISIBLE_DEVICES-unset}"
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv
exec python -m torch.distributed.run \
  --rdzv-backend=static --nnodes=2 --nproc-per-node=2 --node-rank="$SLURM_NODEID" \
  --master-addr="$A2_MASTER_ADDR" --master-port="$A2_MASTER_PORT" \
  --max-restarts=0 "$@"
