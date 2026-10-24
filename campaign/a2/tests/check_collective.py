"""Minimal four-GPU NCCL counterfactual; no model, optimizer or scoring."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import sys
import time
import traceback

MASK_AT_IMPORT = os.environ.get('CUDA_VISIBLE_DEVICES')
import torch
import torch.distributed as dist

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
rank, local = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
record = dict(rank=rank, local_rank=local, group_rank=os.environ.get('GROUP_RANK'),
    node_rank=os.environ.get('SLURM_NODEID'), slurm_launcher_rank=os.environ.get('SLURM_PROCID'),
    slurm_launcher_local_rank=os.environ.get('SLURM_LOCALID'), world_size=int(os.environ['WORLD_SIZE']),
    local_world_size=int(os.environ['LOCAL_WORLD_SIZE']), node=os.uname().nodename,
    job_id=os.environ.get('SLURM_JOB_ID'), stage=str(Path.cwd()),
    cuda_visible_devices_at_import=MASK_AT_IMPORT,
    slurm_job_gpus=os.environ.get('SLURM_JOB_GPUS'), slurm_step_gpus=os.environ.get('SLURM_STEP_GPUS'),
    torch=torch.__version__, torch_cuda=torch.version.cuda,
    nccl=list(torch.cuda.nccl.version()), phase='PREFLIGHT', passed=False)
started = time.perf_counter()
try:
    assert (Path.cwd()/'STAGING_MANIFEST.json').is_file() and record['job_id']
    assert record['world_size'] == 4 and record['local_world_size'] == 2 and local in (0,1)
    assert torch.cuda.device_count() == 2
    record['visible_gpu_uuids'] = [str(torch.cuda.get_device_properties(i).uuid) for i in range(2)]
    record['visible_gpu_names'] = [torch.cuda.get_device_name(i) for i in range(2)]
    assert len(set(record['visible_gpu_uuids'])) == 2
    device = torch.device('cuda', local)
    torch.cuda.set_device(device)
    props = torch.cuda.get_device_properties(device)
    assert 'RTX 3090' in props.name and props.total_memory >= 24_000_000_000
    record.update(selected_uuid=str(props.uuid), current_device=torch.cuda.current_device(),
        selected_vram_bytes=props.total_memory, capability=list(torch.cuda.get_device_capability(device)),
        cuda_visible_devices_after_initialization=os.environ.get('CUDA_VISIBLE_DEVICES'))
    assert MASK_AT_IMPORT == record['cuda_visible_devices_after_initialization']
    record['phase'] = 'CUDA_ALLOCATION'
    value = torch.tensor([float(rank+1)],device=device)
    assert (value+1).item() == rank+2
    print(json.dumps(record), flush=True)
    record['phase'] = 'NCCL_INITIALIZATION'
    dist.init_process_group('nccl',device_id=device,timeout=timedelta(seconds=90))
    record['phase'] = 'FIRST_CUDA_ALL_REDUCE'
    dist.all_reduce(value)
    torch.cuda.synchronize(device)
    record['actual_sum'] = value.item()
    record['expected_sum'] = 10.
    assert value.item() == 10.
    record['phase'] = 'BARRIER'
    dist.barrier(device_ids=[local])
    record['cuda_visible_devices_at_end'] = os.environ.get('CUDA_VISIBLE_DEVICES')
    assert record['cuda_visible_devices_at_end'] == MASK_AT_IMPORT
    record.update(phase='COMPLETE', passed=True)
except BaseException as exc:
    record.update(error=repr(exc), traceback=traceback.format_exc())
    raise
finally:
    record['elapsed_seconds'] = time.perf_counter()-started
    with (a.output/f'rank{rank}.json').open('x') as f:
        json.dump(record,f,indent=2);f.write('\n')
    print(json.dumps(record),flush=True)
    if dist.is_initialized():
        dist.destroy_process_group()
